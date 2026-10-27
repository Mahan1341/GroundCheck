"""Prepare source-grouped RefChecker transfer data without opening locked labels.

The source split is frozen before annotation files are downloaded. The JSON scanner
decodes only top-level IDs for locked objects, never their annotation values.
Dataset text and original annotation bytes stay under ignored models/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/discovery_loop_20261026/external"
REPORT = ROOT / "reports/discovery_loop_20261026/external_preparation.json"
REVISION = "1df1b25cee792ba2b171302e31ca4f768bd67703"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
REMOTE = f"https://raw.githubusercontent.com/amazon-science/RefChecker/{REVISION}"
SEED = "GroundCheckDiscoveryExternal20261026:42"
ANNOTATIONS = {
    "alpaca_7B": (171986, "c518778972742da5c5fc9a8d703c8af48cde9535"),
    "chatgpt": (181486, "c8c59cbc2e697fe7d1d3927220eec1de93cd5227"),
    "claude2": (190475, "9b34b555a880df5808259e0939993ce9048f02d1"),
    "davinci001": (160810, "97fafdac7d737ab9acf7b4db428f111f9bf3d65d"),
    "falcon_40B_instruct": (174768, "112f16b725575c2b8999740592ee99255de10542"),
    "gpt4": (136021, "4d7b985a6fdc37573b3a09e2116252dca79eae57"),
    "llama2_70b_chat": (212938, "83b98bc98543453aded8e8967565964819864853"),
}
ALLOWED_LOCAL = (
    ROOT / "models/phase2b_r10/train_records.jsonl",
    ROOT / "models/phase2b_r10/natural_records.jsonl",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def source_group(text: str) -> str:
    return digest(normalized(text).encode("utf-8"))


def source_partitions(
    sources: list[dict[str, Any]], excluded_source_ids: set[str],
) -> dict[str, str]:
    """Assign whole normalized-context groups, independently of input order or labels."""
    excluded_groups = {s["group_id"] for s in sources if s["source_id"] in excluded_source_ids}
    groups = sorted({s["group_id"] for s in sources} - excluded_groups,
                    key=lambda group: digest(f"{SEED}:{group}".encode()))
    dev_count = math.floor(0.4 * len(groups))
    if not 0 < dev_count < len(groups):
        raise ValueError("Not enough source groups")
    dev_groups = set(groups[:dev_count])
    return {s["source_id"]: ("excluded_overlap" if s["group_id"] in excluded_groups
                             else "development" if s["group_id"] in dev_groups
                             else "locked_confirmation") for s in sources}


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_bytes(path, data.encode("utf-8"))


def download(name: str, url: str, expected: tuple[int, str] | None = None) -> dict[str, Any]:
    path = BASE / "raw" / name
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if expected:
            size, git_sha = expected
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if len(data) != size or blob != git_sha:
                raise ValueError(f"Pinned Git blob verification failed: {name}")
        write_bytes(path, data)
    if expected:
        size, git_sha = expected
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if len(data) != size or blob != git_sha:
            raise ValueError(f"Existing pinned Git blob mismatch: {name}")
    return {"path": str(path.relative_to(ROOT)), "url": url,
            "bytes": len(data), "sha256": digest(data)}


def local_context_audit(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Read only the two explicitly authorized train-side R10 record files."""
    local: dict[str, set[str]] = {}
    result: dict[str, Any] = {}
    for path in ALLOWED_LOCAL:
        texts: set[str] = set()
        raw_texts: set[str] = set()
        source_ids: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            evidence = row["evidence"]
            source_ids.add(row["source_id"])
            raw_texts.add(evidence)
            texts.add(normalized(evidence))
            for passage in re.split(r"(?im)^\s*passage\s+\d+\s*:\s*", evidence):
                if len(normalized(passage)) >= 100:
                    texts.add(normalized(passage))
        local[path.name] = texts
        result[path.name] = {
            "sha256": digest(path.read_bytes()), "sources": len(source_ids),
            "exact_full_matches": sorted(s["source_id"] for s in sources
                                         if s["evidence"] in raw_texts),
            "normalized_full_or_passage_matches": sorted(
                s["source_id"] for s in sources if normalized(s["evidence"]) in texts
            ),
        }
    result["excluded_source_ids"] = sorted({
        s["source_id"] for s in sources
        if any(normalized(s["evidence"]) in texts for texts in local.values())
    })
    result["limitation"] = (
        "Exact/NFKC-casefold-whitespace context and full-passage equality only. "
        "No semantic or pretraining contamination guarantee; no outer data accessed."
    )
    return result


def prepare_sources() -> None:
    if (BASE / "frozen_split.json").exists():
        raise FileExistsError("Source split already frozen; do not regenerate it")
    files = [download("dolly_example_ids.json", f"{REMOTE}/benchmark/data/accurate_context/"
                      "dolly_example_ids.json"),
             download("databricks-dolly-15k.jsonl", "https://huggingface.co/datasets/"
                      f"databricks/databricks-dolly-15k/resolve/{DOLLY_REVISION}/"
                      "databricks-dolly-15k.jsonl")]
    selected = json.loads((BASE / "raw/dolly_example_ids.json").read_text(encoding="utf-8"))
    if not isinstance(selected, list) or len(set(selected)) != len(selected):
        raise ValueError("Expected unique author-provided Dolly row-index strings")
    selected = set(selected)
    sources = []
    for index, line in enumerate((BASE / "raw/databricks-dolly-15k.jsonl")
                                 .read_text(encoding="utf-8").splitlines()):
        if str(index) not in selected:
            continue
        row = json.loads(line)
        evidence = row["context"]
        if not normalized(evidence):
            raise ValueError(f"Empty source context: {index}")
        sources.append({"source_id": str(index), "question": row["instruction"],
                        "evidence": evidence, "category": row["category"],
                        "group_id": source_group(evidence)})
    if {s["source_id"] for s in sources} != selected:
        raise ValueError("Could not reproduce author's source row-index mapping")
    audit = local_context_audit(sources)
    partitions = source_partitions(sources, set(audit["excluded_source_ids"]))
    for source in sources:
        source["partition"] = partitions[source["source_id"]]
    source_path = BASE / "source_contexts.jsonl"
    write_jsonl(source_path, sources)
    split = {
        "version": 1, "refchecker_revision": REVISION, "dolly_revision": DOLLY_REVISION,
        "seed": SEED, "grouping": "SHA256(NFKC-casefold-whitespace-normalized context)",
        "rule": "sort SHA256(seed:group_id), first floor(0.4*N_groups) development",
        "annotation_values_read_before_split": False,
        "source_contexts_sha256": digest(source_path.read_bytes()),
        "source_files": files, "overlap_audit": audit,
        "sources": [{k: s[k] for k in ("source_id", "group_id", "category", "partition")}
                    for s in sources],
        "source_counts": dict(Counter(s["partition"] for s in sources)),
        "group_counts": {part: len({s["group_id"] for s in sources if s["partition"] == part})
                         for part in ("development", "locked_confirmation", "excluded_overlap")},
    }
    write_json(BASE / "frozen_split.json", split)
    print(json.dumps({"source_counts": split["source_counts"],
                      "group_counts": split["group_counts"],
                      "split_sha256": digest((BASE / "frozen_split.json").read_bytes())}))


def skip_value(text: str, start: int) -> int:
    """Lexically skip a value; deliberately do not deserialize its contents."""
    index = start
    while text[index].isspace():
        index += 1
    if text[index] == '"':
        index += 1
        while index < len(text):
            if text[index] == "\\":
                index += 2
            elif text[index] == '"':
                return index + 1
            else:
                index += 1
        raise ValueError("Unterminated string")
    if text[index] in "[{":
        stack = [text[index]]
        index += 1
        while stack and index < len(text):
            char = text[index]
            if char == '"':
                index = skip_value(text, index)
                continue
            if char in "[{":
                stack.append(char)
            elif char in "]}":
                if stack.pop() != ("]" == char and "[" or "{"):
                    raise ValueError("Mismatched brackets")
            index += 1
        if stack:
            raise ValueError("Unterminated container")
        return index
    while index < len(text) and text[index] not in ",]} \r\n\t":
        index += 1
    return index


def top_level_id(obj: str) -> str:
    decoder = json.JSONDecoder()
    index = 1
    found = None
    while index < len(obj):
        while obj[index].isspace() or obj[index] == ",":
            index += 1
        if obj[index] == "}":
            break
        key, index = decoder.raw_decode(obj, index)
        while obj[index].isspace():
            index += 1
        if obj[index] != ":":
            raise ValueError("Missing key colon")
        index += 1
        while obj[index].isspace():
            index += 1
        end = skip_value(obj, index)
        if key == "id":
            if found is not None:
                raise ValueError("Duplicate top-level id")
            if obj[index] != '"':
                raise ValueError("Top-level ID must be a string, never a nested value")
            found = str(json.loads(obj[index:end]))
        index = end
    if found is None:
        raise ValueError("Missing top-level id")
    return found


def selected_objects(text: str, allowed: set[str]) -> list[dict[str, Any]]:
    index = 0
    while text[index].isspace():
        index += 1
    if text[index] != "[":
        raise ValueError("Expected annotation top-level JSON array")
    index += 1
    selected = []
    seen: set[str] = set()
    while index < len(text):
        while text[index].isspace() or text[index] == ",":
            index += 1
        if text[index] == "]":
            break
        if text[index] != "{":
            raise ValueError("Expected record object")
        end = skip_value(text, index)
        raw = text[index:end]
        source_id = top_level_id(raw)
        if source_id in seen:
            raise ValueError("Repeated source ID in annotation file")
        seen.add(source_id)
        if source_id in allowed:
            selected.append(json.loads(raw))
        index = end
    if not allowed.issubset(seen):
        raise ValueError("Missing allowed source IDs")
    return selected


def load_sources() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = json.loads((BASE / "frozen_split.json").read_text(encoding="utf-8"))
    source_bytes = (BASE / "source_contexts.jsonl").read_bytes()
    if digest(source_bytes) != split["source_contexts_sha256"]:
        raise ValueError("Frozen source contexts changed")
    return split, [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()]


def seal_and_inspect_dev_schema() -> None:
    split, sources = load_sources()
    allowed = {s["source_id"] for s in sources if s["partition"] == "development"}
    files = []
    schema = {}
    for generator, expected in ANNOTATIONS.items():
        name = f"dolly_{generator}_answers.json"
        files.append(download(name, f"{REMOTE}/benchmark/human_annotations_v1/"
                              f"accurate_context/{name}", expected))
        rows = selected_objects((BASE / "raw" / name).read_text(encoding="utf-8"), allowed)
        example = rows[0]
        schema[generator] = {k: type(v).__name__ for k, v in example.items()}
        for key, value in example.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                schema[generator][f"{key}[0]"] = {
                    k: type(v).__name__ for k, v in value[0].items()
                }
    write_json(BASE / "sealed_annotation_manifest.json", {
        "split_sha256": digest((BASE / "frozen_split.json").read_bytes()),
        "files": files, "locked_annotation_values_deserialized": False,
        "schema_from_development_only": schema,
        "development_sources": split["source_counts"]["development"],
    })
    print(json.dumps(schema))


def verify_sealed_files() -> dict[str, Any]:
    manifest = json.loads((BASE / "sealed_annotation_manifest.json").read_text(encoding="utf-8"))
    if manifest["split_sha256"] != digest((BASE / "frozen_split.json").read_bytes()):
        raise ValueError("Frozen split differs from sealed annotation manifest")
    for entry in manifest["files"]:
        if digest((ROOT / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Sealed bytes changed: {entry['path']}")
    return manifest


def export_partition(
    partition: str, freeze_path: Path | None = None, freeze_sha256: str | None = None,
) -> None:
    split, sources = load_sources()
    sealed = verify_sealed_files()
    if partition == "locked_confirmation":
        if freeze_path is None:
            raise ValueError("Confirmation export requires an explicit final candidate freeze")
        if freeze_sha256 is None or digest(freeze_path.read_bytes()) != freeze_sha256:
            raise ValueError("Explicit final freeze SHA256 is required and must match")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (freeze.get("stage") != "external_confirmation"
                or freeze.get("selection_complete") is not True):
            raise ValueError("Candidate selection must be complete")
        if freeze.get("external_split_sha256") != sealed["split_sha256"]:
            raise ValueError("Confirmation freeze must name this precise external split")
        eligibility_path = BASE / "frozen_eligibility.json"
        if freeze.get("external_eligibility_sha256") != digest(eligibility_path.read_bytes()):
            raise ValueError("Confirmation freeze must name the input-only eligibility mask")
        if freeze.get("no_adaptation_after_confirmation") is not True:
            raise ValueError("Confirmation must not feed further model/threshold selection")
        if not freeze.get("candidates") or not freeze.get("metrics"):
            raise ValueError("Freeze candidates and metrics before confirmation")
        for candidate in freeze["candidates"]:
            threshold = candidate.get("threshold")
            if isinstance(threshold, bool) or not isinstance(threshold, int | float):
                raise ValueError("Every candidate must have a frozen numeric threshold")
            if not math.isfinite(threshold):
                raise ValueError("Threshold must be finite")
            if not candidate.get("artifacts"):
                raise ValueError("Freeze model/configuration artifact hashes")
            for item in candidate["artifacts"]:
                if digest((ROOT / item["path"]).read_bytes()) != item["sha256"]:
                    raise ValueError("Frozen candidate artifact does not match")
        write_json(BASE / "confirmation_unlock.json", {
            "freeze_path": str(freeze_path), "freeze_sha256": digest(freeze_path.read_bytes()),
            "external_split_sha256": sealed["split_sha256"],
            "authorized_command": "export-confirmation --authorize-locked-export --freeze PATH "
            "--freeze-sha256 SHA256",
        })
    lookup = {s["source_id"]: s for s in sources if s["partition"] == partition}
    if partition == "locked_confirmation":
        eligible = json.loads((BASE / "frozen_eligibility.json").read_text(encoding="utf-8"))
        allowed = set(eligible["eligible_source_ids"][partition])
        lookup = {key: value for key, value in lookup.items() if key in allowed}
    records = []
    exclusions = []
    observed_claim_labels: Counter[str] = Counter()
    for generator in ANNOTATIONS:
        path = BASE / "raw" / f"dolly_{generator}_answers.json"
        for row in selected_objects(path.read_text(encoding="utf-8"), set(lookup)):
            source = lookup[row["id"]]
            record_id = f"refchecker:dolly:{row['id']}:{generator}"
            claims = row["claude2_response_kg"]
            labels = [claim.get("human_label") for claim in claims]
            if not labels:
                exclusions.append({"id": record_id, "source_id": source["source_id"],
                                   "reason": "no_human_annotated_triplets"})
                continue
            if any(label not in {"Entailment", "Neutral", "Contradiction"} for label in labels):
                raise ValueError(f"Unrecognized/missing human label in allowed record {record_id}")
            observed_claim_labels.update(labels)
            records.append({
                "id": record_id, "source_id": f"refchecker:dolly:{source['source_id']}",
                "original_source_id": source["source_id"], "group_id": source["group_id"],
                "question": source["question"], "evidence": source["evidence"],
                "response": row["response"],
                "label": int(any(label in {"Neutral", "Contradiction"} for label in labels)),
                "partition": partition, "category": source["category"], "generator": generator,
                "label_origin": "human_labels_of_Claude2_extracted_triplets",
                "human_annotated_triplets": len(labels), "span_labels_available": False,
            })
    filename = ("development_records.jsonl" if partition == "development"
                else "confirmation_records.jsonl")
    destination = BASE / filename
    write_jsonl(destination, records)
    write_json(BASE / f"{partition}_exclusions.json", exclusions)
    details = {
        "partition": partition, "eligible_responses": len(records),
        "eligible_sources": len({r["source_id"] for r in records}),
        "eligible_groups": len({r["group_id"] for r in records}),
        "labels": dict(Counter(r["label"] for r in records)),
        "categories": dict(Counter(r["category"] for r in records)),
        "human_claim_labels": dict(observed_claim_labels),
        "excluded_empty_claim_responses": len(exclusions),
        "records_sha256": digest(destination.read_bytes()),
        "records_path": str(destination.relative_to(ROOT)),
    }
    if partition != "development":
        write_json(BASE / "confirmation_export_report.json", details)
        print(json.dumps({"partition": partition, "exported": True}))
        return
    write_json(REPORT, {
        "dataset": "RefChecker human_annotations_v1 / accurate_context / Dolly",
        "refchecker_revision": REVISION, "dolly_revision": DOLLY_REVISION,
        "source_mapping": "Original zero-based Dolly JSONL row index, as in author gather script",
        "split_sha256": sealed["split_sha256"],
        "split_counts": split["source_counts"], "split_group_counts": split["group_counts"],
        "split_rule": split["rule"], "split_seed": SEED, "grouping": split["grouping"],
        "source_files": split["source_files"], "annotation_files": sealed["files"],
        "local_overlap_audit": split["overlap_audit"], "development": details,
        "label_mapping": {
            "positive": "Any included human-labeled triplet is Neutral or Contradiction",
            "negative": "At least one triplet, and all included human labels are Entailment",
            "empty_triplets": "Excluded as unknown, never coerced to supported",
            "unknown_label": "Fail closed with error",
            "inference_features": ["question", "evidence", "response"],
            "gold_triplets_as_inference_features": False,
            "character_span_labels": "Unavailable; no invented token/span gold",
        },
        "record_schema": {k: type(v).__name__ for k, v in records[0].items()},
        "licenses": {
            "RefChecker_code": {"license": "Apache-2.0", "url": f"{REMOTE}/LICENSE"},
            "RefChecker_human_annotations": {
                "license": "CC-BY-4.0",
                "url": f"{REMOTE}/benchmark/human_annotations_v1/LICENSE.txt"},
            "Dolly_contexts": {"license": "CC-BY-SA-3.0", "url":
                "https://huggingface.co/datasets/databricks/databricks-dolly-15k"},
        },
        "limitations": [
            "Human triplet labels are genuine; decomposition is Claude2-generated and filtered. "
            "response labels are derived over included claims, not exhaustive response annotation.",
            "Author reports 23% double annotation, 95% agreement; others single-annotated.",
            "Dolly accurate_context includes QA, summarization and information extraction; report "
            "category-specific transfer, do not label the full suite pure QA.",
            "Natural contexts/LLM answers are not synthetic hallucination perturbations, but this "
            "small source-grouped set is not representative of all natural documents.",
            "Absence of exact local overlap does not establish absence of semantic or pretraining "
            "contamination. Backbone pretraining provenance cannot guarantee this.",
            "All responses from one normalized context share a partition; bootstrap unit is group.",
            "Source-only split was frozen before annotation downloads. Locked objects are "
            "scanned for top-level ID only; their annotation values have not been deserialized.",
        ],
        "locked_access": {
            "unlocked": False, "locked_label_values_inspected": False,
            "locked_response_labels_exported": False,
            "mechanism": "export-confirmation --authorize-locked-export --freeze PATH",
            "required_freeze_schema": {
                "stage": "external_confirmation", "selection_complete": True,
                "no_adaptation_after_confirmation": True,
                "external_split_sha256": sealed["split_sha256"],
                "metrics": ["predeclared metric names"],
                "candidates": [{"name": "name", "threshold": "number",
                                "artifacts": [{"path": "repo-relative artifact path",
                                               "sha256": "frozen SHA256"}]}],
            },
            "policy": "Only run after final candidates/thresholds/metrics are frozen; no model "
            "selection using confirmation. This command is not an official RAGTruth test unlock.",
        },
    })
    print(json.dumps(details))


def factcg_overlap() -> None:
    """Audit source text only against authorized downloaded external training files."""
    _, sources = load_sources()
    for source in sources:
        source["dataset"] = "refchecker"
        source["source_key"] = "refchecker:" + source["source_id"]
    local_manifests = []
    for path in ALLOWED_LOCAL:
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["source_id"] in seen:
                continue
            seen.add(row["source_id"])
            sources.append({
                "source_id": row["source_id"], "evidence": row["evidence"],
                "dataset": "ragtruth", "partition": path.stem,
                "source_key": f"ragtruth:{path.stem}:{row['source_id']}",
            })
        local_manifests.append({"path": str(path.relative_to(ROOT)),
                                "sha256": digest(path.read_bytes()), "sources": len(seen)})
    by_key = {s["source_key"]: s for s in sources}
    width = 16
    source_tokens = {s["source_key"]: normalized(s["evidence"]).split() for s in sources}
    inverted: dict[tuple[str, ...], set[str]] = {}
    for source_id, tokens in source_tokens.items():
        for index in range(len(tokens) - width + 1):
            inverted.setdefault(tuple(tokens[index:index + width]), set()).add(source_id)
    training_root = ROOT / "models/discovery_loop_20261026/factcg_training"
    paths = sorted(training_root.glob("training_stage*/*.json"))
    if len(paths) != 6:
        raise ValueError("Expected exactly six authorized FactCG training files")
    results = []
    for path in paths:
        matches = []
        empty_evidence = 0
        count = 0
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            count += 1
            empty_evidence += not bool(row["text_a"].strip())
            fields = [("text_a", row["text_a"])]
            claims = row["text_b"] if isinstance(row["text_b"], list) else [row["text_b"]]
            fields += [(f"text_b[{n}]", claim) for n, claim in enumerate(claims)]
            for field, text in fields:
                tokens = normalized(text).split()
                overlaps: Counter[str] = Counter()
                for offset in range(len(tokens) - width + 1):
                    overlaps.update(inverted.get(tuple(tokens[offset:offset + width]), set()))
                for source_id, common in overlaps.items():
                    source = by_key[source_id]
                    norm_source = normalized(source["evidence"])
                    norm_train = normalized(text)
                    matches.append({
                        "source_id": source["source_id"], "source_key": source_id,
                        "dataset": source["dataset"], "partition": source["partition"],
                        "training_row": index, "training_field": field,
                        "exact_text": source["evidence"] == text,
                        "normalized_exact": norm_source == norm_train,
                        "normalized_full_source_in_training": norm_source in norm_train,
                        "normalized_full_training_in_source": norm_train in norm_source,
                        "matching_16word_windows": common,
                    })
        results.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes()),
                        "rows": count, "empty_text_a": empty_evidence, "matches": matches})
    report_path = ROOT / "reports/discovery_loop_20261026/external_factcg_overlap.json"
    write_json(report_path, {
        "source_contexts_sha256": digest((BASE / "source_contexts.jsonl").read_bytes()),
        "local_manifests": local_manifests,
        "width_words": width, "fields": ["text_a", "text_b"],
        "training_labels_used": False, "external_response_labels_used": False,
        "method": "NFKC-casefold-whitespace tokenization; any shared contiguous 16-word window; "
        "then exact, normalized exact, and full-context containment checks on candidates.",
        "limitation": "Lexical overlap is a contamination warning, not proof of memorization. "
        "No hit does not exclude paraphrase, shorter shared text, or backbone pretraining overlap. "
        "Published Musique data have 5391/6433 empty evidence text_a fields; claims-only matching "
        "cannot rule out overlap with their unavailable source documents. All reports concern "
        "published text only, without inspecting any training or locked response label values.",
        "files": results,
    })
    print(json.dumps({"factcg_overlap_report": str(report_path.relative_to(ROOT)),
                      "matching_sources_by_dataset": {
                          dataset: len({m["source_key"] for r in results for m in r["matches"]
                                        if m["dataset"] == dataset})
                          for dataset in ("refchecker", "ragtruth")},
                      "matched_training_fields": sum(len(r["matches"]) for r in results)}))


def eligibility_capacity() -> None:
    """Source-only common eligibility: full B7 context plus known FactCG text overlap."""
    from transformers import AutoTokenizer

    split, sources = load_sources()
    path = ROOT / "models/phase2b/span_inner_s42/best"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    details = []
    for source in sources:
        prefix = f"Question: {source['question']}\nEvidence: {source['evidence']}\nAnswer:"
        length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        reasons = []
        if source["source_id"] == "4070":
            reasons.append("published_FactCG_Hotpot_text_overlap_6_contiguous_16word_windows")
        if length > 2013:
            reasons.append("B7_complete_prefix_exceeds_2013_tokens_at_2048_limit")
        details.append({"source_id": source["source_id"], "group_id": source["group_id"],
                        "partition": source["partition"], "prefix_tokens": length,
                        "answer_capacity": min(512, 2048 - length - 3), "reasons": reasons})
    excluded_groups = {row["group_id"] for row in details if row["reasons"]}
    for row in details:
        if row["group_id"] in excluded_groups and not row["reasons"]:
            row["reasons"].append("same_normalized_context_group_as_ineligible_source")
    eligible = {partition: sorted(row["source_id"] for row in details
                                 if row["partition"] == partition and not row["reasons"])
                for partition in ("development", "locked_confirmation")}
    tokenizer_files = sorted(path.glob("*token*"))
    mask = {
        "split_sha256": digest((BASE / "frozen_split.json").read_bytes()),
        "unchanged_source_counts": split["source_counts"],
        "tokenizer_path": str(path.relative_to(ROOT)),
        "tokenizer_files": {str(p.relative_to(ROOT)): digest(p.read_bytes())
                            for p in tokenizer_files if p.is_file()},
        "rule": "Question/evidence prefix exactly as B7 windows(); full prefix <=2013 tokens, "
        "retaining >=32 answer tokens and 3 special tokens within2048. No truncation. "
        "Exclude source4070 for published FactCG training text overlap for all methods.",
        "frozen_before_any_locked_prediction_or_label_access": True,
        "locked_excluded_labels_must_stay_sealed": True,
        "eligible_source_ids": eligible, "source_details": details,
        "factcg_overlap_report_sha256": digest(
            (ROOT / "reports/discovery_loop_20261026/external_factcg_overlap.json").read_bytes()
        ),
    }
    destination = BASE / "frozen_eligibility.json"
    write_json(destination, mask)
    rows = [json.loads(line) for line in (BASE / "development_records.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["original_source_id"] in eligible["development"]]
    records_path = BASE / "development_eligible_records.jsonl"
    write_jsonl(records_path, rows)
    report = {
        "frozen_eligibility_sha256": digest(destination.read_bytes()),
        "source_counts": {key: len(value) for key, value in eligible.items()},
        "excluded_sources": [row for row in details if row["reasons"]],
        "prefix_token_range": [min(row["prefix_tokens"] for row in details),
                               max(row["prefix_tokens"] for row in details)],
        "development_eligible_responses": len(rows),
        "development_labels": dict(Counter(row["label"] for row in rows)),
        "development_records_path": str(records_path.relative_to(ROOT)),
        "development_records_sha256": digest(records_path.read_bytes()),
        "confirmation_response_counts_or_labels": "Not read; count only source contexts",
        "unlock": {
            "command": "export-confirmation --authorize-locked-export --freeze PATH "
            "--freeze-sha256 SHA256",
            "required_final_freeze_additions": {
                "external_eligibility_sha256": digest(destination.read_bytes()),
                "external_split_sha256": mask["split_sha256"],
            },
            "policy": "Same source eligibility for all models. Original split immutable. "
            "Excluded locked source labels remain sealed even during future confirmation export.",
        },
    }
    write_json(ROOT / "reports/discovery_loop_20261026/external_eligibility.json", report)
    print(json.dumps({key: report[key] for key in (
        "frozen_eligibility_sha256", "source_counts", "excluded_sources", "prefix_token_range",
        "development_eligible_responses", "development_labels",
    )}))


def self_test() -> None:
    # A locked value is invalid JSON: a full-file decoder would fail this test.
    text = '[{"nested":{"id":"dev"},"id":"locked","labels":INVALID},'
    text += '{"id":"dev","nested":["escaped \\\" }"],"labels":["Neutral"]}]'
    assert selected_objects(text, {"dev"})[0]["labels"] == ["Neutral"]
    assert top_level_id('{"nested":{"id":"wrong"},"id":"right"}') == "right"
    assert normalized("ＦＡＣＴ  One\n") == normalized("fact one")
    nested = '{"s":"}\\\"", "a":[1,2]}'
    assert skip_value(nested, 0) == len(nested)
    print("scanner/grouping self-tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sources", "seal", "development", "factcg-overlap",
                                            "eligibility", "export-confirmation", "self-test"))
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--authorize-locked-export", action="store_true")
    args = parser.parse_args()
    if args.command == "export-confirmation":
        if not args.authorize_locked_export:
            parser.error("Explicit --authorize-locked-export is required")
        export_partition("locked_confirmation", args.freeze, args.freeze_sha256)
        return
    {"sources": prepare_sources, "seal": seal_and_inspect_dev_schema,
     "development": lambda: export_partition("development"), "factcg-overlap": factcg_overlap,
     "eligibility": eligibility_capacity,
     "self-test": self_test}[args.command]()


if __name__ == "__main__":
    main()
