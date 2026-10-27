"""Smaller source-only C2 request after measured resource gate; no outcome adaptation."""

import argparse
import hashlib

import claim_grounding_c2 as run
from claim_grounding_run import cluster, write_rows
from scope_analysis import read, rows, write

from grounded_hallucination_lab.research.protocol import digest

ORIGINAL_BASE, ORIGINAL_PRED, ORIGINAL_REPORT = run.BASE, run.PRED, run.REPORT
BASE, PRED, REPORT = run.BASE / "c2_ready", run.PRED / "c2_ready", run.REPORT / "c2_ready"


def prepare():
    for folder in (BASE, PRED, REPORT):
        folder.mkdir(exist_ok=False)
    all_pairs = read(ORIGINAL_BASE / "c2_pairs.json")
    pairs, counts = {}, {}
    for dataset in read(ORIGINAL_BASE / "c2_freeze.json")["counts"]:
        ll = rows(ORIGINAL_BASE / f"c2_{dataset}_layout.jsonl")
        if dataset in ("natural", "r6"):
            groups = sorted({cluster(r["source_id"]) for r in ll}, key=lambda g: hashlib.sha256(
                ("claim-grounding-c1:42:" + g).encode()).hexdigest())
            selected = set(groups[:12 if dataset == "natural" else 8])
            ll = [r for r in ll if cluster(r["source_id"]) in selected]
        for row in ll:
            for claim in row["claims"]:
                for hypothesis in claim["hypotheses"].values():
                    if hypothesis is not None:
                        for e in hypothesis["evidence"]:
                            pairs[e["key"]] = all_pairs[e["key"]]
        write_rows(BASE / f"c2_{dataset}_layout.jsonl", ll)
        counts[dataset] = len(ll)
    write(BASE / "c2_pairs.json", pairs)
    cache = [r for r in rows(ORIGINAL_PRED / "c2_pair_cache.jsonl") if r["key"] in pairs]
    write_rows(PRED / "c2_pair_cache.jsonl", cache)
    original_profile = read(ORIGINAL_REPORT / "c2_profile_resources.json")
    full_pending = read(ORIGINAL_BASE / "c2_freeze.json")["pairs"] - read(
        ORIGINAL_BASE / "c2_freeze.json")["reused"] - original_profile["new_pairs"]
    estimate = (original_profile["remaining_estimate_seconds"] / full_pending
                * (len(pairs) - len(cache)))
    write(REPORT / "c2_profile_resources.json", {
        "profile_reused": str(
            (ORIGINAL_REPORT / "c2_profile_resources.json").relative_to(run.ROOT)),
        "seconds_with_load": 0, "new_pairs": 0, "remaining_estimate_seconds": estimate,
        "passed": estimate < 840,
        "measurement": "estimate from original actual profile, not new run"})
    files = [run.ROOT / p for p in ("scripts/claim_grounding_c2_bounded.py",
                                    "scripts/claim_grounding_c2.py",
                                    "scripts/claim_grounding_context.py",
                                    "docs/CLAIM_GROUNDING_C2_PLAN.md",
                                    "docs/CLAIM_GROUNDING_C2_RESOURCE.md")]
    files += [BASE / "c2_pairs.json", REPORT / "c2_profile_resources.json"]
    files += [BASE / f"c2_{d}_layout.jsonl" for d in counts]
    write(BASE / "c2_freeze.json", {
        "files": {str(p.relative_to(run.ROOT)): digest(p) for p in files},
        "counts": counts, "pairs": len(pairs), "reused": len(cache)})
    print({"counts": counts, "pairs": len(pairs), "cached": len(cache), "estimate": estimate})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "project"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        run.BASE, run.PRED, run.REPORT = BASE, PRED, REPORT
        run.infer("run") if args.mode == "run" else run.project_all()
