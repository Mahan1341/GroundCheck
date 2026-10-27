"""Matched extractive claim policies and strictly scoped cache reuse."""

import argparse
import json
import re
from dataclasses import asdict

from discovery_verifier import chunks, pair_key
from research_r10 import records

from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    read_rows,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.segments import clean_segments

BASE = ROOT / "models/scope_diagnosis_20261026"
PRED = ROOT / "predictions/scope_diagnosis_20261026"
REPORT = ROOT / "reports/scope_diagnosis_20261026"
OLD = ROOT / "models/discovery_loop_20261026"
OLDP = ROOT / "predictions/discovery_loop_20261026"
KEY = re.compile(r'^\{"key":\s*"([0-9a-f]{64})"')


def units(text, filtered):
    """Keep facts in mixed refusals; only historical standalone full matches removed."""
    kept, dropped = clean_segments(text)
    spans = [asdict(s) for s in kept]
    if not filtered:
        spans += [
            {"start": d["start"], "end": d["end"], "text": text[d["start"] : d["end"]]}
            for d in dropped
            if d["reason"] == "standalone_meta_or_refusal"
        ]
    spans.sort(key=lambda s: s["start"])
    assert all(s["text"] == text[s["start"] : s["end"]] for s in spans)
    assert all(a["end"] <= b["start"] for a, b in zip(spans, spans[1:], strict=False))
    return spans, dropped


def compose(spans):
    """Newline joins only exact substrings, with reversible piecewise provenance."""
    text, mapping = "", []
    for s in spans:
        if text:
            text += "\n"
        start = len(text)
        text += s["text"]
        mapping.append(
            {
                "view_start": start,
                "view_end": len(text),
                "original_start": s["start"],
                "original_end": s["end"],
            }
        )
    return text, mapping


def scoped_cache(path, allowed):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = KEY.match(line)
            if not match:
                raise ValueError("cache schema must expose key before any values")
            key = match[1]
            if key not in allowed:
                continue  # In particular, no unrelated former-confirmation value decoding.
            row = json.loads(line)
            if row["key"] != key or key in result:
                raise ValueError("invalid or duplicate allowed cache entry")
            result[key] = row
    return result


def build_layout(views, old, filtered, whole):
    pairs, rows = {}, []
    for v, oldrow in zip(views, old, strict=True):
        assert v["id"] == oldrow["id"]
        spans, dropped = units(v["response"], filtered)
        text, mapping = compose(spans)
        contexts = chunks(v["evidence"])
        claims = (
            [{"text": text, "pieces": [[s["start"], s["end"]] for s in spans]}]
            if whole and spans
            else [{"text": s["text"], "pieces": [[s["start"], s["end"]]]} for s in spans]
        )
        for claim in claims:
            evidence = []
            for c in contexts:
                key = pair_key(c["text"], claim["text"])
                pairs.setdefault(key, {"document": c["text"], "claim": claim["text"]})
                evidence.append({"key": key, "source_pieces": c["pieces"]})
            claim["evidence"] = evidence
        rows.append(
            {
                "id": v["id"],
                "source_id": v["source_id"],
                "offsets": oldrow["offsets"],
                "claims": claims,
                "retained": [[s["start"], s["end"]] for s in spans],
                "mapping": mapping,
                "dropped": dropped,
                "no_claim": not spans,
            }
        )
    return pairs, rows


def project(layout, cache):
    result = []
    for row in layout:
        claims = [
            {**c, "risk": 1 - max(cache[e["key"]]["support"] for e in c["evidence"])}
            for c in row["claims"]
        ]
        probabilities = [
            max(
                (c["risk"] for c in claims if any(a < y and b > x for x, y in c["pieces"])),
                default=0,
            )
            for a, b in row["offsets"]
        ]
        result.append(
            {
                **row,
                "claims": claims,
                "token_probabilities": probabilities,
                "score": max((c["risk"] for c in claims), default=0),
            }
        )
    return result


def datasets():
    data = {}
    for label, path in {
        "natural": ROOT / "models/phase2b_r10/natural_records.jsonl",
        "r1": ROOT / "models/phase2b_r10/r1_records.jsonl",
        "r6": ROOT / "predictions/phase2b_r6/transfer_records.jsonl",
    }.items():
        rr = records(path)
        data[label] = (
            [{k: getattr(r, k) for k in ("id", "source_id", "response", "evidence")} for r in rr],
            read_rows(ROOT / "predictions/phase2b_r10" / f"B7_{label}.jsonl"),
            path,
        )
    ext = OLD / "external/development_eligible_records.jsonl"
    data["external_dev"] = (read_rows(ext), read_rows(OLDP / "external_development_B7.jsonl"), ext)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "project"])
    args = parser.parse_args()
    methods = [
        (f"{u}_{p}", p == "filtered", u == "whole")
        for p in ("all", "filtered")
        for u in ("seg", "whole")
    ]
    if args.command == "prepare":
        pairs, files, counts = {}, {}, {}
        for name, (views, old, path) in datasets().items():
            files[str(path.relative_to(ROOT))] = digest(path)
            counts[name] = {}
            for method, filtered, whole in methods:
                pp, ll = build_layout(views, old, filtered, whole)
                pairs.update(pp)
                write_json(BASE / f"{name}_{method}_layout.json", ll)
                counts[name][method] = {
                    "n": len(ll),
                    "no_claim": sum(r["no_claim"] for r in ll),
                    "claims": sum(len(r["claims"]) for r in ll),
                }
        write_json(BASE / "pairs.json", pairs)
        oldcache = OLDP / "factcg_pair_cache.jsonl"
        reused = scoped_cache(oldcache, set(pairs))
        write_rows(PRED / "pair_cache.jsonl", list(reused.values()))
        source = [
            ROOT / "scripts/scope_inputs.py",
            ROOT / "scripts/scope_infer.py",
            ROOT / "docs/SCOPE_DIAGNOSIS_PLAN.md",
            ROOT / "scripts/discovery_verifier.py",
        ]
        write_json(
            BASE / "freeze.json",
            {
                "input_files": files,
                "source": {str(p.relative_to(ROOT)): digest(p) for p in source},
                "pairs_sha256": digest(BASE / "pairs.json"),
                "old_cache_sha256": digest(oldcache),
                "unique_pairs": len(pairs),
                "reused_pairs": len(reused),
                "missing_pairs": len(pairs) - len(reused),
                "counts": counts,
                "forbidden": "external confirmation and RAGTruth outer; no old cache writes",
            },
        )
        print({"unique": len(pairs), "reused": len(reused), "missing": len(pairs) - len(reused)})
    else:
        from grounded_hallucination_lab.research.protocol import read_json

        cache = scoped_cache(PRED / "pair_cache.jsonl", set(read_json(BASE / "pairs.json")))
        for name in ("natural", "external_dev", "r1", "r6"):
            for method, _, _ in methods:
                rows = project(read_json(BASE / f"{name}_{method}_layout.json"), cache)
                write_rows(PRED / f"{name}_{method}.jsonl", rows)


if __name__ == "__main__":
    main()
