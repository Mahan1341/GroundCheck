"""Frozen R9 natural passage competition pilot; no training or outer-data reader."""

import argparse
import json
import time
from collections import Counter
from dataclasses import replace

import numpy as np
from research_r7 import THRESHOLD
from research_r7 import data as inner_data

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.metrics import bootstrap, metrics, span_metrics
from grounded_hallucination_lab.research.passage_probe import (
    evidence_view,
    passage_ranges,
    pool_passages,
    sentence_ranges,
)
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    preserve_code,
    read_json,
    read_rows,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.relations import content_hash

BASE = ROOT / "models/phase2b_r9"
FREEZE = BASE / "freeze_v4.json"
PROFILE = BASE / "profile_v2.json"
PRED = ROOT / "predictions/phase2b_r9_final"
REPORT = ROOT / "reports/phase2b_r9_final"
MODELS = {
    "B7": (
        "models/phase2b/span_inner_s42/best",
        "7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4",
    ),
    "R6B": (
        "models/phase2b_r6/R6Bs42/best",
        "86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e",
    ),
}
CONDITIONS = ("full", "rotate1", "rotate2", "single0", "single1", "single2")
ARMS = ("full", "rotate1", "rotate2", "sentence_pool", "token_min")


def data():
    return [r for r in inner_data("natural")[0] if r.source_id != "15181"]


def state():
    return {
        "source": preserve_code(),
        "proposal": digest(ROOT / "docs/R9_RESEARCH_PROPOSAL.md"),
        "correction": digest(ROOT / "docs/R9_ALIGNMENT_CORRECTION.md"),
        "punctuation_correction": digest(ROOT / "docs/R9_ALIGNMENT_CORRECTION_V4.md"),
        "saved_forward_manifest": digest(BASE / "v2_execution_manifest.json"),
    }


def old_integrity():
    counts = {}
    for stage in (6, 7, 8):
        manifest = read_json(ROOT / f"models/phase2b_r{stage}/R{stage}_artifact_manifest.json")
        for name, spec in manifest["files"].items():
            assert digest(ROOT / name) == spec["sha256"], name
        counts[str(stage)] = len(manifest["files"])
    return counts


def freeze():
    old = old_integrity()
    records = data()
    sources = {r.source_id for r in records}
    assert len(records) == 642 and len(sources) == 107
    assert all(r.split == "train" and r.labels_available for r in records)
    for path, sha in MODELS.values():
        assert digest(ROOT / path / "model.safetensors") == sha
        fit = read_json(ROOT / path / "../training_config.json")["train_sources"]
        assert not sources & set(fit)
    design = [
        {
            "id": r.id,
            "source_id": r.source_id,
            "passages": passage_ranges(r.evidence),
            "sentences": sentence_ranges(r.response),
        }
        for r in records
    ]
    if (BASE / "design.json").exists():
        assert read_json(BASE / "design.json") == json.loads(json.dumps(design))
    else:
        write_json(BASE / "design.json", design)
    write_json(
        FREEZE,
        {
            "state": state(),
            "old_artifacts": old,
            "models": MODELS,
            "data_sha256": content_hash(records, []),
            "design_sha256": digest(BASE / "design.json"),
            "n": len(records),
            "sources": sorted(sources),
            "positive": sum(map(gold_label, records)),
            "excluded_documented_duplicate": "15181",
            "outer_access": False,
        },
    )
    print({"frozen": len(records), "sources": len(sources), "old_verified": old}, flush=True)


def checked():
    frozen = read_json(FREEZE)
    assert state() == frozen["state"], "R9 source/plan mismatch; do not bypass"
    assert content_hash(data(), []) == frozen["data_sha256"]
    assert digest(BASE / "design.json") == frozen["design_sha256"]
    for path, sha in MODELS.values():
        assert digest(ROOT / path / "model.safetensors") == sha
    return frozen


def profile():
    from grounded_hallucination_lab.research.span_model import encode_records, load_model
    from grounded_hallucination_lab.research.stable_inference import score

    checked()
    records = data()
    result = {}
    for name, (path, _) in MODELS.items():
        model, tok = load_model(ROOT / path)
        enc, meta = encode_records(records, tok, 2048, False)
        longest = max(meta, key=lambda m: m["max_window_tokens"])
        examples = [e for e in enc if e["record_id"] == longest["id"]]
        _, warm = score(model, tok, examples, [longest])
        _, timed = score(model, tok, examples, [longest])
        result[name] = {
            "metadata": longest,
            "warm": warm,
            "timed": timed,
            "six_full_length_upper_estimate_s": timed["pipeline_seconds"] * 6 * 642,
        }
        assert all(m["answer_coverage"] == m["evidence_coverage"] == 1 for m in meta)
        del model
    estimated = sum(v["six_full_length_upper_estimate_s"] for v in result.values())
    write_json(
        PROFILE,
        {"models": result, "estimated_seconds": estimated, "passed": estimated <= 900},
    )
    assert estimated <= 900, "15 minute inference resource gate"
    print({"profile_passed": True, "estimated_seconds": estimated}, flush=True)


def run(name):
    from grounded_hallucination_lab.research.span_model import encode_records, load_model
    from grounded_hallucination_lab.research.stable_inference import score

    frozen = checked()
    assert read_json(PROFILE)["passed"]
    model, tok = load_model(ROOT / MODELS[name][0])
    records = data()
    predictions = {}
    for condition in CONDITIONS:
        # Do not consume GPU only to fail the exclusive-write guard afterwards.
        assert not (PRED / f"{name}_{condition}.jsonl").exists()
        started = time.perf_counter()
        modified = [replace(r, evidence=evidence_view(r.evidence, condition)) for r in records]
        enc, meta = encode_records(modified, tok, 2048, False)
        assert all(m["answer_coverage"] == m["evidence_coverage"] == 1 for m in meta)
        rows, timing = score(model, tok, enc, meta)
        timing["pipeline_including_encoding_s"] = time.perf_counter() - started
        timing["evidence_coverage_meaning"] = "relative to supplied view, no truncation"
        predictions[condition] = rows
        if condition == "full":
            old_name = "B7" if name == "B7" else "R6Bs42"
            old = {
                r["id"]: r
                for r in read_rows(ROOT / f"predictions/phase2b_r6/{old_name}_natural.jsonl")
            }
            assert all(
                r["token_probabilities"] == old[r["id"]]["token_probabilities"]
                and [list(v) for v in r["offsets"]] == old[r["id"]]["offsets"]
                for r in rows
            )
            timing["historical_token_replay_exact"] = True
        else:
            assert all(
                a["offsets"] == b["offsets"] for a, b in zip(rows, predictions["full"], strict=True)
            )
        write_rows(PRED / f"{name}_{condition}.jsonl", rows)
        write_json(
            REPORT / f"{name}_{condition}_runtime.json",
            {**timing, "state": frozen["state"], "checkpoint_sha256": MODELS[name][1]},
        )
        print(
            {
                "model": name,
                "condition": condition,
                "seconds": timing["pipeline_including_encoding_s"],
            },
            flush=True,
        )
    coherent, token, provenance = [], [], []
    for i, design in enumerate(read_json(BASE / "design.json")):
        a, b, choices = pool_passages(
            [predictions[f"single{j}"][i] for j in range(3)],
            design["sentences"],
            text=records[i].response,
        )
        coherent.append(a)
        token.append(b)
        provenance.append(
            {"id": design["id"], "choices": choices, "original_passages": design["passages"]}
        )
    write_rows(PRED / f"{name}_sentence_pool.jsonl", coherent)
    write_rows(PRED / f"{name}_token_min.jsonl", token)
    write_rows(PRED / f"{name}_choices.jsonl", provenance)


def repool():
    """Correct only derived pooling from hash-verified completed neural forwards."""
    checked()
    manifest = read_json(BASE / "v2_execution_manifest.json")
    for name, sha in manifest["files"].items():
        assert digest(ROOT / name) == sha, name
    records = data()
    design = read_json(BASE / "design.json")
    for model in MODELS:
        predictions = {}
        for condition in CONDITIONS:
            original = ROOT / f"predictions/phase2b_r9/{model}_{condition}.jsonl"
            target = PRED / original.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(original.read_bytes())
            assert digest(target) == digest(original)
            predictions[condition] = read_rows(target)
            runtime = ROOT / f"reports/phase2b_r9/{model}_{condition}_runtime.json"
            copied = REPORT / runtime.name
            copied.parent.mkdir(parents=True, exist_ok=True)
            with copied.open("xb") as stream:
                stream.write(runtime.read_bytes())
        coherent, token, choices = [], [], []
        for i, d in enumerate(design):
            a, b, refs = pool_passages(
                [predictions[f"single{j}"][i] for j in range(3)],
                d["sentences"],
                text=records[i].response,
            )
            coherent.append(a)
            token.append(b)
            choices.append({"id": d["id"], "choices": refs, "original_passages": d["passages"]})
        old = read_rows(ROOT / f"predictions/phase2b_r9/{model}_token_min.jsonl")
        assert token == old, "token-min control must remain completely unchanged"
        write_rows(PRED / f"{model}_sentence_pool.jsonl", coherent)
        write_rows(PRED / f"{model}_token_min.jsonl", token)
        write_rows(PRED / f"{model}_choices.jsonl", choices)
    print("Corrected pooling completed from verified saved probabilities; no GPU", flush=True)


def short(record):
    chars = {
        j
        for a in record.annotations
        for j in range(a.start, a.end)
        if not record.response[j].isspace()
    }
    return (
        bool(chars)
        and len(record.response) >= 500
        and len(chars) <= 0.05 * sum(not c.isspace() for c in record.response)
    )


def interval(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return {
        "low": float(np.percentile(v, 2.5)) if len(v) else None,
        "high": float(np.percentile(v, 97.5)) if len(v) else None,
        "valid": len(v),
    }


def localization(records, systems):
    """Same source bootstrap draws as response metrics; gold is analysis-only."""
    from grounded_hallucination_lab.research.span_model import token_targets

    groups = sorted({r.source_id for r in records})
    index = {s: i for i, s in enumerate(groups)}
    arrays, points, details = {}, {}, {}
    for key, rows in systems.items():
        counts = np.zeros((len(groups), 14))
        for r, row in zip(records, rows, strict=True):
            loc = span_metrics([r], [row])
            c = counts[index[r.source_id]]
            c[:9] += [
                loc[k][v] for k in ("token", "character", "exact_span") for v in ("tp", "fp", "fn")
            ]
            target = np.array(
                token_targets(row["offsets"], [(a.start, a.end) for a in r.annotations])
            )
            probs = np.array(row["token_probabilities"])
            if short(r):
                c[9:12] += [
                    1,
                    row["score"] >= THRESHOLD,
                    bool(np.any((target == 1) & (probs >= 0.5))),
                ]
            if gold_label(r) and row["score"] >= THRESHOLD:
                c[12:] += [1, target[int(np.argmax(probs))] == 1]

        def values(c):
            out = [2 * c[j] / max(1, 2 * c[j] + c[j + 1] + c[j + 2]) for j in (0, 3, 6)]
            out += [c[j] / c[9] if c[9] else np.nan for j in (10, 11)]
            out += [c[13] / c[12] if c[12] else np.nan]
            return np.array(out)

        points[key] = values(counts.sum(axis=0)).tolist()
        details[key] = {
            "short_n": int(counts[:, 9].sum()),
            "short_tp": int(counts[:, 10].sum()),
            "short_gold_hit": int(counts[:, 11].sum()),
            "positive_detected": int(counts[:, 12].sum()),
            "positive_peak_on_gold": int(counts[:, 13].sum()),
        }
        rng = np.random.default_rng(42)
        arrays[key] = np.array(
            [
                values(
                    np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                    @ counts
                )
                for _ in range(2000)
            ]
        )
    names = (
        "token_f1",
        "character_f1",
        "exact_span_f1",
        "short_recall",
        "short_gold_hit",
        "positive_detected_peak_on_gold",
    )

    def intervals(a):
        return {n: interval(a[:, i]) for i, n in enumerate(names)}

    return {
        "points": {k: dict(zip(names, v, strict=True)) for k, v in points.items()},
        "counts": details,
        "absolute": {k: intervals(v) for k, v in arrays.items()},
        "difference_vs_own_full": {
            k: intervals(v - arrays[k.split("/")[0] + "/full"]) for k, v in arrays.items()
        },
    }


def analyze():
    checked()
    records = data()
    y = [gold_label(r) for r in records]
    sources = [r.source_id for r in records]
    systems = {f"{m}/{a}": read_rows(PRED / f"{m}_{a}.jsonl") for m in MODELS for a in ARMS}
    assert all([p["id"] for p in rows] == [r.id for r in records] for rows in systems.values())
    metrics_all = {k: metrics(y, [p["score"] for p in v], THRESHOLD) for k, v in systems.items()}
    bp = {
        k: ([int(p["score"] >= THRESHOLD) for p in v], [p["score"] for p in v])
        for k, v in systems.items()
    }
    boot = bootstrap(y, sources, bp, "B7/full")
    r6boot = bootstrap(
        y, sources, {k: v for k, v in bp.items() if k.startswith("R6B/")}, "R6B/full"
    )
    loc = localization(records, systems)
    changes, slices, rotations = {}, {}, {}
    for model in MODELS:
        full = systems[model + "/full"]
        pooled = systems[model + "/sentence_pool"]
        change = Counter()
        cases = []
        for r, label, old, new in zip(records, y, full, pooled, strict=True):
            a, b = old["score"] >= THRESHOLD, new["score"] >= THRESHOLD
            if a != b:
                kind = "fixed" if int(b) == label else "new_error"
                change[kind] += 1
                cases.append(
                    {
                        "id": r.id,
                        "source_id": r.source_id,
                        "gold": label,
                        "kind": kind,
                        "old_score": old["score"],
                        "new_score": new["score"],
                        "response": r.response,
                        "evidence": r.evidence,
                        "annotations": [vars(a) for a in r.annotations],
                    }
                )
        write_rows(PRED / f"{model}_changed_errors.jsonl", cases)
        changes[model] = dict(change)
        for lower, upper in ((0, 100), (101, 200), (201, 1000000)):
            ids = [i for i, p in enumerate(full) if lower <= p["answer_tokens"] <= upper]
            slices[f"{model}/{lower}-{upper}"] = {
                "n": len(ids),
                "sources": len({sources[i] for i in ids}),
                "positive": sum(y[i] for i in ids),
                "arms": {
                    a: metrics(
                        [y[i] for i in ids],
                        [systems[model + "/" + a][i]["score"] for i in ids],
                        THRESHOLD,
                    )
                    for a in ("full", "sentence_pool", "token_min")
                },
            }
        # Response and token movement at the same complete evidence set.
        group = sorted(set(sources))
        vals = []
        for a, r1, r2 in zip(
            full, systems[model + "/rotate1"], systems[model + "/rotate2"], strict=True
        ):
            changes_p = [
                np.abs(np.array(b["token_probabilities"]) - a["token_probabilities"])
                for b in (r1, r2)
            ]
            vals.append(
                [
                    np.mean([abs(b["score"] - a["score"]) for b in (r1, r2)]),
                    np.mean([np.mean(v) for v in changes_p]),
                    any((b["score"] >= THRESHOLD) != (a["score"] >= THRESHOLD) for b in (r1, r2)),
                ]
            )
        vals = np.asarray(vals)
        sums = np.array([vals[np.array(sources) == s].sum(axis=0) for s in group])
        ns = np.array([sources.count(s) for s in group])
        rng = np.random.default_rng(42)
        draws = []
        for _ in range(2000):
            w = np.bincount(rng.integers(0, len(group), len(group)), minlength=len(group))
            draws.append(w @ sums / (w @ ns))
        keys = ("mean_abs_response_delta", "mean_abs_token_delta", "any_rotation_flip_rate")
        rotations[model] = {
            k: {"value": float(vals[:, i].mean()), "ci": interval(np.asarray(draws)[:, i])}
            for i, k in enumerate(keys)
        }
        rotations[model]["flip_count"] = int(vals[:, 2].sum())
    write_json(
        REPORT / "analysis.json",
        {
            "n": len(records),
            "sources": len(set(sources)),
            "positive": sum(y),
            "metrics": metrics_all,
            "bootstrap_vs_B7": boot,
            "bootstrap_vs_R6B": r6boot,
            "localization": loc,
            "changed_errors": changes,
            "length_slices": slices,
            "rotations": rotations,
            "outer_access": False,
        },
    )
    base, candidate = metrics_all["B7/full"], metrics_all["B7/sentence_pool"]
    count = loc["counts"]
    criteria = {
        "macro_gain_at_least_02": candidate["macro_f1"] - base["macro_f1"] >= 0.02,
        "macro_paired_ci_positive": boot["difference_from_reference"]["B7/sentence_pool"][
            "macro_f1"
        ]["low"]
        > 0,
        "positive_recall_loss_at_most_03": candidate["positive_class"]["recall"]
        >= base["positive_class"]["recall"] - 0.03,
        "token_f1_nondecreasing": loc["points"]["B7/sentence_pool"]["token_f1"]
        >= loc["points"]["B7/full"]["token_f1"],
        "short_gold_hit_retained_nonvacuous": count["B7/full"]["short_n"] > 0
        and count["B7/sentence_pool"]["short_gold_hit"] >= count["B7/full"]["short_gold_hit"],
    }
    write_json(
        REPORT / "gate.json",
        {"criteria": criteria, "passed": all(criteria.values()), "new_training": False},
    )
    print(
        {"metrics": {k: round(v["macro_f1"], 4) for k, v in metrics_all.items()}, "gate": criteria},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("freeze", "profile", "run", "analyze", "verify-old", "repool")
    )
    parser.add_argument("--model", choices=tuple(MODELS))
    args = parser.parse_args()
    if args.command == "run":
        if args.model is None:
            parser.error("--model required")
        run(args.model)
    else:
        {
            "freeze": freeze,
            "profile": profile,
            "analyze": analyze,
            "repool": repool,
            "verify-old": lambda: print(old_integrity()),
        }[args.command]()
