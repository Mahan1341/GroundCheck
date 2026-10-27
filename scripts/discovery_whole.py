"""Cycle4 isolated hypothesis-unit ablation; no learned parameters or label features."""

import argparse

from discovery_verifier import BASE, PRED, chunks, pair_key, project, read_cache
from research_r10 import records

from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


def whole_prepare(views, old):
    pairs, layouts = {}, []
    for r, row in zip(views, old, strict=True):
        assert r["id"] == row["id"]
        contexts = chunks(r["evidence"])
        keys = []
        for c in contexts:
            key = pair_key(c["text"], r["response"])
            pairs.setdefault(key, {"document": c["text"], "claim": r["response"]})
            keys.append({"key": key, "pieces": c["pieces"]})
        layouts.append(
            {
                "id": r["id"],
                "source_id": r["source_id"],
                "offsets": row["offsets"],
                "dropped": [],
                "segments": [
                    {"start": 0, "end": len(r["response"]), "text": r["response"], "evidence": keys}
                ]
                if r["response"].strip()
                else [],
            }
        )
    return pairs, layouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "project"])
    args = parser.parse_args()
    if args.mode == "prepare":
        pairs, inputs = {}, {}
        for label, path in {
            "natural": ROOT / "models/phase2b_r10/natural_records.jsonl",
            "r1": ROOT / "models/phase2b_r10/r1_records.jsonl",
            "r6": ROOT / "predictions/phase2b_r6/transfer_records.jsonl",
        }.items():
            rr = records(path)
            old = read_rows(ROOT / "predictions/phase2b_r10" / f"B7_{label}.jsonl")
            views = [
                {k: getattr(r, k) for k in ("id", "source_id", "response", "evidence")} for r in rr
            ]
            pp, ll = whole_prepare(views, old)
            pairs.update(pp)
            write_json(BASE / f"cycle4_{label}_layout.json", ll)
            inputs[str(path.relative_to(ROOT))] = digest(path)
        write_json(BASE / "cycle4_pairs.json", pairs)
        write_json(
            BASE / "cycle4_freeze.json",
            {
                "inputs": inputs,
                "pairs_sha256": digest(BASE / "cycle4_pairs.json"),
                "source_sha256": digest(ROOT / "scripts/discovery_whole.py"),
                "prereg_sha256": digest(ROOT / "docs/DISCOVERY_CYCLE4_PREREG.md"),
                "verifier_source_sha256": digest(ROOT / "scripts/discovery_verifier.py"),
                "threshold": 0.5,
            },
        )
        print({"unique_pairs": len(pairs)}, flush=True)
    else:
        cache = read_cache(PRED / "factcg_pair_cache.jsonl")
        for label in ("natural", "r1", "r6"):
            result = project(read_json(BASE / f"cycle4_{label}_layout.json"), cache)
            write_rows(PRED / f"cycle4_FactCG_{label}.jsonl", result)


if __name__ == "__main__":
    main()
