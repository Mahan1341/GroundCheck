"""CPU-only DiaSumFact SAMSum preparation; inference inputs never contain gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "models/claim_grounding_20261026/external_raw/diasumfact"
BASE = ROOT / "models/claim_grounding_20261026/external"
REPORT = ROOT / "reports/claim_grounding_20261026/samsum_preparation.json"
TOKENIZER = ROOT / "models/phase2b/span_inner_s42/best"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value)
            if jsonl else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def whitespace_view(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse whitespace only, retaining reversible original character boundaries."""
    chars, offsets = [], []
    for match in re.finditer(r"\S|\s+", text):
        value = match.group()
        if value.isspace():
            if not chars or match.end() == len(text):
                continue
            value = " "
        chars.append(value)
        offsets.append((match.start(), match.end()))
    return "".join(chars), offsets


def match_fragment(fragment: str, text: str) -> dict:
    """Never resolve repeated spans by arbitrary first occurrence or case folding."""
    if not fragment.strip():
        return {"status": "unmatched", "method": "empty", "candidates": []}
    exact = [[m.start(), m.start() + len(fragment)]
             for m in re.finditer("(?=" + re.escape(fragment) + ")", text)]
    if exact:
        return {"status": "unique" if len(exact) == 1 else "ambiguous",
                "method": "exact", "candidates": exact}
    needle, _ = whitespace_view(fragment)
    haystack, offsets = whitespace_view(text)
    candidates = [[offsets[m.start()][0], offsets[m.start() + len(needle) - 1][1]]
                  for m in re.finditer("(?=" + re.escape(needle) + ")", haystack)]
    return {"status": "unique" if len(candidates) == 1 else "ambiguous"
            if candidates else "unmatched", "method": "whitespace_only", "candidates": candidates}


def align_summary(response: str, sentence_annotations: list[dict]) -> dict:
    """Derive author labels; missing/ambiguous gold makes the whole span task unscorable."""
    issues, spans, sentence_ranges, sentence_labels, annotation_rows = [], [], [], [], []
    for sentence in sentence_annotations:
        label_rows = sentence["annotation"]
        labels = [annotation["error_class"] for annotation in label_rows]
        label = int(any(value != "No Error" for value in labels)) if labels else None
        sentence_labels.append(label)
        alignment = match_fragment(sentence["sentence"], response)
        if alignment["status"] != "unique":
            issues.append({"sentence_index": sentence["index"],
                           "reason": "sentence_" + alignment["status"]})
        if label is None:
            issues.append({"sentence_index": sentence["index"], "reason": "missing_annotation"})
        if alignment["status"] == "unique":
            start, end = alignment["candidates"][0]
            sentence_ranges.append((start, end))
            actual = response[start:end]
        else:
            start, actual = None, None
        for annotation in label_rows:
            if annotation["error_class"] == "No Error":
                continue
            local = (match_fragment(annotation["error_span"], actual) if actual is not None
                     else {"status": "unmatched", "method": "sentence_unresolved",
                           "candidates": []})
            original = [[start + a, start + b] for a, b in local["candidates"]]
            annotation_rows.append({**annotation, "sentence_index": sentence["index"],
                                    "alignment_status": local["status"],
                                    "alignment_method": local["method"],
                                    "candidate_summary_offsets": original})
            if local["status"] == "unique":
                spans.append(original[0])
            else:
                issues.append({"sentence_index": sentence["index"],
                               "reason": "error_span_" + local["status"]})
    covered = {i for start, end in sentence_ranges for i in range(start, end)}
    uncovered = [i for i, char in enumerate(response) if not char.isspace() and i not in covered]
    if uncovered:
        issues.append({"reason": "nonwhitespace_summary_without_aligned_human_sentence",
                       "n_characters": len(uncovered)})
    label = (1 if any(y == 1 for y in sentence_labels) else
             None if not sentence_labels or any(y is None for y in sentence_labels) else 0)
    scorable = not issues and label is not None
    return {"label": label, "localization_scorable": scorable,
            "gold_char_spans": sorted(spans) if scorable else None,
            "unscorable_reasons": issues, "error_annotations": annotation_rows,
            "human_sentence_labels": sentence_labels,
            "uncovered_nonwhitespace_characters": len(uncovered),
            "zero_token_semantics": "not annotated hallucination, never proof of entailment"}


def prepare() -> None:
    if (BASE / "samsum_manifest.json").exists():
        raise FileExistsError("Existing evaluation is frozen; use verify, do not prepare again")
    from transformers import AutoTokenizer

    from grounded_hallucination_lab.research.segments import clean_segments

    manifest = json.loads((RAW / "acquisition_manifest.json").read_text(encoding="utf-8"))
    for name, details in manifest["files"].items():
        if sha(RAW / name) != details["sha256"]:
            raise ValueError("Raw author artifact changed")
    freeze = json.loads((RAW / "evaluation_source_freeze.json").read_text(encoding="utf-8"))
    if sha(RAW / "source_records.jsonl") != freeze["source_manifest_sha256"]:
        raise ValueError("Frozen source-only manifest changed")
    selected = set(freeze["source_ids"])
    sources = {row["source_id"]: row for row in
               (json.loads(line) for line in (RAW / "source_records.jsonl")
                .read_text(encoding="utf-8").splitlines()) if row["source_id"] in selected}
    raw = json.loads((RAW / "annotations.json").read_text(encoding="utf-8"))
    if len(sources) != 59 or any(s["dataset"] != "SAMSum" for s in sources.values()):
        raise ValueError("Expected exactly the frozen 59 SAMSum sources")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True,
                                               trust_remote_code=False)
    inputs, labels, lengths = [], [], []
    for source_id in sorted(sources):
        source = sources[source_id]
        author = raw[source["original_source_id"]]
        prefix = f"Question: {source['question']}\nEvidence: {source['evidence']}\nAnswer:"
        prefix_length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        lengths.append({"source_id": source_id, "prefix_tokens": prefix_length,
                        "b7_full_context_fits_2048": prefix_length <= 2013})
        for generator, summary in sorted(author["annotations"].items()):
            response = summary["model_prediction"]
            record_id = source_id + ":" + generator
            tokens = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
            segments, dropped = clean_segments(response)
            inputs.append({"id": record_id, "source_id": source_id,
                           "parent_id": source["parent_id"], "question": source["question"],
                           "evidence": source["evidence"], "response": response,
                           "offsets": tokens["offset_mapping"],
                           "segments": [asdict(s) for s in segments], "dropped": dropped})
            labels.append({"id": record_id, "source_id": source_id,
                           "parent_id": source["parent_id"], "generator": generator,
                           "annotation_provenance": "DiaSumFact independent human adjudication",
                           **align_summary(response, summary["sentence_annotations"])})
    if len(inputs) != 295 or any(row["label"] is None for row in labels):
        raise ValueError("Expected 295 detection examples with explicit human-derived labels")
    frozen_paths = [Path(__file__), ROOT / "docs/CLAIM_GROUNDING_DATA.md",
                    ROOT / "tests/test_claim_grounding_external.py",
                    ROOT / "src/grounded_hallucination_lab/research/segments.py",
                    ROOT / "src/grounded_hallucination_lab/baselines/lexical.py",
                    RAW / "annotations.json", RAW / "evaluation_source_freeze.json",
                    RAW / "source_records.jsonl", RAW / "overlap_audit.json"]
    frozen_paths += [p for p in TOKENIZER.glob("*token*") if p.is_file()]
    frozen = {path.relative_to(ROOT).as_posix(): sha(path) for path in frozen_paths}
    write(BASE / "samsum_inputs.jsonl", inputs, jsonl=True)
    write(BASE / "samsum_labels.jsonl", labels, jsonl=True)
    unscorable = [row["id"] for row in labels if not row["localization_scorable"]]
    result = {"source_count": len(sources), "summary_count": len(inputs),
              "labels": {str(y): sum(r["label"] == y for r in labels) for y in (0, 1)},
              "localization_scorable": len(inputs) - len(unscorable),
              "localization_unscorable_ids": unscorable,
              "localization_coverage": (len(inputs) - len(unscorable)) / len(inputs),
              "source_token_capacity": lengths, "detection_exclusions": [],
              "gold_features_in_inference_inputs": False, "gpu_forwards": 0,
              "selection_rule": freeze["selection_rule"],
              "metric_policy": "Primary exact/token/character localization uses complete "
              "scorable summaries only; detection retains all295; denominators must be reported.",
              "policy_limit": "Error strings require unambiguous mapping; alignment exclusions "
              "are not evidence of model errors or annotation precision/recall."}
    write(REPORT, result)
    for path in (BASE / "samsum_inputs.jsonl", BASE / "samsum_labels.jsonl", REPORT):
        frozen[path.relative_to(ROOT).as_posix()] = sha(path)
    snapshot = BASE / "source_snapshot"
    snapshot.mkdir(exist_ok=False)
    for path in frozen_paths[:3]:
        with (snapshot / path.name).open("xb") as stream:
            stream.write(path.read_bytes())
    write(BASE / "samsum_manifest.json", {"files_sha256": frozen,
          "frozen_before_external_predictions": True, "new_model_scores_seen": False,
          "original_partitions_changed": False})
    print(json.dumps({key: result[key] for key in (
        "source_count", "summary_count", "labels", "localization_scorable",
        "localization_coverage", "gpu_forwards",
    )}))


def verify() -> None:
    manifest = json.loads((BASE / "samsum_manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest["files_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")
    print(json.dumps({"verified_files": len(manifest["files_sha256"]),
                      "manifest_sha256": sha(BASE / "samsum_manifest.json")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
