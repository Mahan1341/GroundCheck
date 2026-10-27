"""Resource-only C1 subset caller; preserve the original failed profile and code."""

import argparse
import hashlib

import claim_grounding_run as run
import numpy as np
from scope_analysis import read, rows, write

from grounded_hallucination_lab.research.protocol import digest

ORIGINAL_BASE, ORIGINAL_PRED = run.BASE, run.PRED
BASE, PRED, REPORT = run.BASE / "c1_bounded", run.PRED / "c1_bounded", run.REPORT / "c1_bounded"


def native_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: native_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native_json(v) for v in value]
    return value


def prepare():
    for folder in (BASE, PRED, REPORT):
        folder.mkdir(exist_ok=False)
    pairs, membership = {}, {}
    original = read(ORIGINAL_BASE / "c1_pairs.json")
    for dataset in ("natural", "r1", "r6"):
        ll = rows(ORIGINAL_BASE / f"c1_{dataset}_layout.jsonl")
        groups = sorted({run.cluster(r["source_id"]) for r in ll}, key=lambda g: hashlib.sha256(
            ("claim-grounding-c1:42:" + g).encode()).hexdigest())[:4]
        ll = [r for r in ll if run.cluster(r["source_id"]) in groups]
        run.write_rows(BASE / f"c1_{dataset}_layout.jsonl", ll)
        membership[dataset] = {"ids": [r["id"] for r in ll], "clusters": groups, "n": len(ll)}
        for row in ll:
            pairs.update({k: original[k] for k in row["winnerkeys"]})
    run.write_rows(PRED / "c1_gradients.jsonl", [r for r in rows(
        ORIGINAL_PRED / "c1_gradients.jsonl") if r["key"] in pairs])
    write(BASE / "c1_pairs.json", pairs)
    write(BASE / "c1_membership.json", membership)
    files = [run.ROOT / p for p in ("scripts/claim_grounding_bounded.py",
                                    "scripts/claim_grounding_run.py",
                                    "scripts/claim_grounding_projection.py",
                                    "docs/CLAIM_GROUNDING_C1_PLAN.md",
                                    "docs/CLAIM_GROUNDING_RESOURCE_AMENDMENT.md")]
    files += [BASE / "c1_pairs.json", BASE / "c1_membership.json"]
    files += [BASE / f"c1_{d}_layout.jsonl" for d in membership]
    write(BASE / "c1_freeze.json", {
        "files": {str(p.relative_to(run.ROOT)): digest(p) for p in files},
        "unique_pairs": len(pairs), "resource_only_amendment": True})
    print({"pairs": len(pairs), "samples": {k: v["n"] for k, v in membership.items()}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile", "run"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        run.BASE, run.PRED, run.REPORT = BASE, PRED, REPORT
        run.write = lambda path, value: write(path, native_json(value))
        run.infer(args.mode)


if __name__ == "__main__":
    main()
