"""Preregistered CPU-only falsification of the non-content max explanation."""

from collections import defaultdict
from pathlib import Path

from discovery_core import content_readout, short_error
from research_r10 import records

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.metrics import bootstrap, metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    read_json,
    read_rows,
    write_json,
    write_rows,
)

BASE = ROOT / "models/discovery_loop_20261026"
OUT = ROOT / "reports/discovery_loop_20261026/cycle1"
PRED = ROOT / "predictions/discovery_loop_20261026/cycle1"
T = 0.37612172961235046


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    paths = {
        "natural": ROOT / "models/phase2b_r10/natural_records.jsonl",
        "r1": ROOT / "models/phase2b_r10/r1_records.jsonl",
        "r6": ROOT / "predictions/phase2b_r6/transfer_records.jsonl",
    }
    sources = [
        Path(__file__),
        ROOT / "scripts/discovery_core.py",
        ROOT / "docs/R11_PLUS_RESEARCH_LOG.md",
        ROOT / "docs/RESEARCH_DISCOVERY_LOOP.md",
    ]
    sources += list(paths.values())
    sources += [ROOT / "predictions/phase2b_r10" / f"B7_{name}.jsonl" for name in paths]
    frozen = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
    write_json(
        BASE / "cycle1_freeze.json",
        {
            "files": frozen,
            "threshold": T,
            "gate": (
                "natural gain>=.01 and lowerCI>0; no lost short TP/hit; "
                "paired accuracies nondecreasing; identical localization"
            ),
            "gpu_minutes": 0,
            "training": False,
        },
    )
    # Keep executed code/preregistration recoverable even when later cycles append the log.
    snap = BASE / "cycle1_source"
    for p in sources[:4]:
        target = snap / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    all_results = {}
    for name, path in paths.items():
        rr = records(path)
        old = read_rows(ROOT / "predictions/phase2b_r10" / f"B7_{name}.jsonl")
        assert [r.id for r in rr] == [p["id"] for p in old]
        new = [content_readout(r.response, p) for r, p in zip(rr, old, strict=True)]
        write_rows(PRED / f"content_{name}.jsonl", new)
        y = [gold_label(r) for r in rr]
        systems = {"B7": old, "content": new}
        result = {
            "n": len(rr),
            "sources": len({r.source_id for r in rr}),
            "positive": sum(y),
            "metrics": {
                m: metrics(y, [p["score"] for p in rows], T) for m, rows in systems.items()
            },
            "localization": {m: span_metrics(rr, rows) for m, rows in systems.items()},
        }
        assert result["localization"]["B7"] == result["localization"]["content"]
        result["bootstrap"] = bootstrap(
            y,
            [r.source_id for r in rr],
            {
                m: ([int(p["score"] >= T) for p in rows], [p["score"] for p in rows])
                for m, rows in systems.items()
            },
            "B7",
        )
        result["lost_short_tp"] = [
            r.id
            for r, a, b in zip(rr, old, new, strict=True)
            if short_error(r) and a["score"] >= T > b["score"]
        ]
        result["changed_ids"] = [
            r.id for r, a, b in zip(rr, old, new, strict=True) if a["score"] != b["score"]
        ]
        if name != "natural":
            design = read_json(
                ROOT / "models/phase2b_r10/r1_design.json"
                if name == "r1"
                else ROOT / "predictions/phase2b_r6/transfer_design.json"
            )
            pairs = defaultdict(list)
            for i, d in enumerate(design):
                assert d["id"] == rr[i].id
                pairs[d["pair_id"]].append(i)
            result["pair_accuracy"] = {}
            for m, rows in systems.items():
                correct = 0
                for indices in pairs.values():
                    assert len(indices) == 2 and sorted(y[i] for i in indices) == [0, 1]
                    correct += all(int(rows[i]["score"] >= T) == y[i] for i in indices)
                result["pair_accuracy"][m] = {
                    "correct": correct,
                    "n": len(pairs),
                    "accuracy": correct / len(pairs),
                }
            result["local_gap_unchanged"] = True  # Token probabilities identical by construction.
        all_results[name] = result
        print(name, result["metrics"]["content"]["macro_f1"], flush=True)
    n = all_results["natural"]
    gate = {
        "natural_gain": n["metrics"]["content"]["macro_f1"] - n["metrics"]["B7"]["macro_f1"]
        >= 0.01,
        "natural_lower_ci": n["bootstrap"]["difference_from_reference"]["content"]["macro_f1"][
            "low"
        ]
        > 0,
        "short_retained": not n["lost_short_tp"],
        "controlled_retained": all(
            all_results[k]["pair_accuracy"]["content"]["correct"]
            >= all_results[k]["pair_accuracy"]["B7"]["correct"]
            for k in ("r1", "r6")
        ),
        "localization_identical": True,
    }
    write_json(
        OUT / "analysis.json",
        {"datasets": all_results, "gate": gate, "passed": all(gate.values()), "gpu_minutes": 0},
    )
    for name, sha in frozen.items():
        assert digest(ROOT / name) == sha, "source/input changed during cycle1"
    print({"passed": all(gate.values()), "gate": gate}, flush=True)


if __name__ == "__main__":
    main()
