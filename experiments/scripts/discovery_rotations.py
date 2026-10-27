"""Previously screened R10 invariance cases; same inputs for both FactCG adapters."""

import argparse

import numpy as np
from discovery_verifier import BASE, PRED, prepare, project, read_cache, run
from discovery_whole import whole_prepare
from research_r10 import records

from grounded_hallucination_lab.research.passage_probe import evidence_view
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "project", "analyze"])
    args = parser.parse_args()
    name = "safe_rotations"
    screen = read_json(ROOT / "models/phase2b_r10/eligibility_frozen.json")["natural"]
    rr = [
        r
        for r in records(ROOT / "models/phase2b_r10/natural_records.jsonl")
        if screen[r.source_id]["eligible"]
    ]
    assert len(rr) == 204 and len({r.source_id for r in rr}) == 34
    if args.mode == "prepare":
        pairs = {}
        for rotation in ("rotate1", "rotate2"):
            old = read_rows(ROOT / "predictions/phase2b_r10" / f"B7_{rotation}.jsonl")
            views = [
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "response": r.response,
                    "evidence": evidence_view(r.evidence, rotation),
                }
                for r in rr
            ]
            for method, fn in (("FactCG", prepare), ("Whole", whole_prepare)):
                pp, layout = fn(views, old)
                pairs.update(pp)
                write_json(BASE / f"{name}_{method}_{rotation}_layout.json", layout)
        write_json(BASE / f"{name}_pairs.json", pairs)
        write_json(
            BASE / f"{name}_freeze.json",
            {
                "pairs_sha256": digest(BASE / f"{name}_pairs.json"),
                "screen_sha256": digest(ROOT / "models/phase2b_r10/eligibility_frozen.json"),
                "source_sha256": digest(ROOT / "scripts/discovery_rotations.py"),
                "prereg_sha256": digest(ROOT / "docs/DISCOVERY_CYCLE4_PREREG.md"),
            },
        )
    elif args.mode == "run":
        run(name, BASE / f"{name}_pairs.json", minutes=10)
    elif args.mode == "project":
        cache = read_cache(PRED / "factcg_pair_cache.jsonl")
        for method in ("FactCG", "Whole"):
            for rotation in ("rotate1", "rotate2"):
                write_rows(
                    PRED / f"{name}_{method}_{rotation}.jsonl",
                    project(read_json(BASE / f"{name}_{method}_{rotation}_layout.json"), cache),
                )
    else:
        result = {}
        ids = {r.id for r in rr}
        for method, threshold in (("B7", 0.37612172961235046), ("FactCG", 0.5), ("Whole", 0.5)):
            baseline = (
                ROOT / "predictions/phase2b_r10/B7_natural.jsonl"
                if method == "B7"
                else PRED / f"{'cycle2' if method == 'FactCG' else 'cycle4'}_FactCG_natural.jsonl"
            )
            old = [p for p in read_rows(baseline) if p["id"] in ids]
            changed = []
            for rotation in ("rotate1", "rotate2"):
                path = (
                    ROOT / "predictions/phase2b_r10" / f"B7_{rotation}.jsonl"
                    if method == "B7"
                    else PRED / f"{name}_{method}_{rotation}.jsonl"
                )
                rows = read_rows(path)
                assert [p["id"] for p in old] == [p["id"] for p in rows]
                changed.append(rows)
            flips = np.array(
                [
                    any((a["score"] >= threshold) != (b[i]["score"] >= threshold) for b in changed)
                    for i, a in enumerate(old)
                ]
            )
            deltas = [abs(a["score"] - b[i]["score"]) for i, a in enumerate(old) for b in changed]
            result[method] = {
                "n": 204,
                "sources": 34,
                "any_rotation_flips": int(flips.sum()),
                "mean_absolute_score_delta": float(np.mean(deltas)),
                "max_absolute_score_delta": max(deltas),
            }
        write_json(ROOT / "reports/discovery_loop_20261026/invariance.json", result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
