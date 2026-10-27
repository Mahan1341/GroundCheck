"""Frozen native context-erasure pairs, independent human transfer, bounded inference."""

import argparse
import hashlib
import itertools
import json
import math
import time

from claim_grounding_context import prepare_view, project
from claim_grounding_run import BASE, MODEL, PRED, REPORT, write_rows
from discovery_verifier import PROMPT
from scope_analysis import DATA, ROOT, read, rows, write

from grounded_hallucination_lab.research.protocol import digest
from grounded_hallucination_lab.research.segments import clean_segments


def factorial():
    """Combine disjoint preexisting controlled sources, without new factual statements."""
    rr = {r["id"]: r for r in rows(DATA["r6"])}
    design = read(ROOT / "predictions/phase2b_r6/transfer_design.json")
    candidates = {}
    for d in design:
        if d["padding"] == 0 and len(clean_segments(rr[d["id"]]["response"])[0]) == 1:
            candidates.setdefault(d["source_id"], {}).setdefault(d["pair_id"], {})[d["gold"]] = d
    chosen = {s: candidates[s][min(candidates[s])] for s in candidates}
    sources = sorted(chosen, key=lambda s: hashlib.sha256(
        ("c2-factorial:42:" + s).encode()).hexdigest())
    result, info = [], []
    for a, b in zip(sources[::2], sources[1::2], strict=True):
        entities_a = {f[0] for d in chosen[a].values() for f in d["facts"]}
        entities_b = {f[0] for d in chosen[b].values() for f in d["facts"]}
        assert not entities_a & entities_b
        parent = f"c2:{a}+{b}"
        for ya, yb, order in itertools.product((0, 1), repeat=3):
            da, db = chosen[a][ya], chosen[b][yb]
            ra, rb = rr[da["id"]], rr[db["id"]]
            ordered = [(ra, da), (rb, db)] if order == 0 else [(rb, db), (ra, da)]
            response, annotations, targets = "", [], []
            for row, d in ordered:
                if response:
                    response += "\n"
                start = len(response)
                response += row["response"]
                annotations.extend({**ann, "start": ann["start"] + start,
                                    "end": ann["end"] + start} for ann in row["annotations"])
                targets.append({"start": start, "end": len(response), "gold": d["gold"],
                                "original_id": row["id"], "parent": row["source_id"]})
            identity = f"{parent}:{ya}{yb}:o{order}"
            result.append({**ra, "id": identity, "source_id": parent, "response": response,
                           "evidence": ra["evidence"] + "\n" + rb["evidence"],
                           "annotations": annotations, "model": "composed_existing_controls"})
            info.append({"id": identity, "source_id": parent, "parents": [a, b],
                         "targets": targets, "world": [ya, yb], "order": order})
    return result, info


def prepare():
    from transformers import AutoTokenizer

    inputs, files = {}, []
    membership = read(BASE / "c1_membership.json")
    for dataset in ("natural", "r1", "r6"):
        source = rows(DATA[dataset])
        offset_path = ROOT / f"predictions/phase2b_r10/B7_{dataset}.jsonl"
        offsets = {r["id"]: r["offsets"] for r in rows(offset_path)}
        allowed = set(membership["natural"]["ids"]) if dataset == "natural" else set(offsets)
        inputs[dataset] = [{k: r[k] for k in ("id", "source_id", "response", "evidence")}
                           | {"offsets": offsets[r["id"]]} for r in source if r["id"] in allowed]
        files += [DATA[dataset], offset_path]
    external = BASE / "external/samsum_inputs.jsonl"
    inputs["external"] = rows(external)
    files.append(external)
    control, design = factorial()
    write_rows(BASE / "c2_factorial_records.jsonl", control)
    write(BASE / "c2_factorial_design.json", design)
    tok = AutoTokenizer.from_pretrained(ROOT / "models/phase2b/span_inner_s42/best",
                                       local_files_only=True)
    inputs["factorial"] = [{k: r[k] for k in ("id", "source_id", "response", "evidence")}
                           | {"offsets": tok(r["response"], add_special_tokens=False,
                                             return_offsets_mapping=True)["offset_mapping"]}
                           for r in control]
    pairs, counts = {}, {}
    for dataset, views in inputs.items():
        layouts = []
        for view in views:
            pp, layout = prepare_view(view, view["offsets"])
            pairs.update(pp)
            layouts.append(layout)
        write_rows(BASE / f"c2_{dataset}_layout.jsonl", layouts)
        counts[dataset] = len(layouts)
    write(BASE / "c2_pairs.json", pairs)
    oldcache = ROOT / "predictions/scope_diagnosis_20261026/pair_cache.jsonl"
    reused = []
    for row in rows(oldcache):
        if row["key"] in pairs and all(p > 0 for p in row["probabilities"]):
            reused.append({**row, "margin": math.log(row["probabilities"][0]) -
                           math.log(row["probabilities"][1]), "reused": True})
    write_rows(PRED / "c2_pair_cache.jsonl", reused)
    source = [ROOT / p for p in ("scripts/claim_grounding_c2.py",
                                 "scripts/claim_grounding_context.py",
                                 "docs/CLAIM_GROUNDING_C2_PLAN.md")]
    files += source + [BASE / "c2_pairs.json", BASE / "c2_factorial_records.jsonl",
                       BASE / "c2_factorial_design.json"]
    files += [BASE / f"c2_{dataset}_layout.jsonl" for dataset in inputs]
    snapshot = BASE / "c2_source"
    snapshot.mkdir(exist_ok=False)
    for path in source:
        (snapshot / path.name).write_bytes(path.read_bytes())
    write(BASE / "c2_freeze.json", {"files": {str(p.relative_to(ROOT)): digest(p) for p in files},
                                    "counts": counts, "pairs": len(pairs), "reused": len(reused)})
    print({"counts": counts, "pairs": len(pairs), "reused": len(reused)})


def infer(mode):
    from grounded_hallucination_lab.research.verifiers import memory_report, setup_runtime

    for name, sha in read(BASE / "c2_freeze.json")["files"].items():
        assert digest(ROOT / name) == sha, name
    assert digest(MODEL / "model.safetensors") == (
        "379a343d1ca25c9dfe24ae149f76cc958ea4f09c00dae88bd49bbbf58c749d5e")
    destination = REPORT / f"c2_{mode}_resources.json"
    assert not destination.exists()
    started = time.perf_counter()
    torch = setup_runtime()
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pairs = read(BASE / "c2_pairs.json")
    cache_path = PRED / "c2_pair_cache.jsonl"
    done = {r["key"] for r in rows(cache_path)}
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.pad_token = tok.eos_token
    lengths = {k: len(tok(PROMPT.format(text_a=v["document"], text_b=v["claim"]))["input_ids"])
               for k, v in pairs.items() if k not in done}
    assert max(lengths.values(), default=0) <= 2048, "no silent truncation"
    todo = sorted(lengths)
    if mode == "profile":
        todo = list(dict.fromkeys(todo[:16] + [max(lengths, key=lengths.get)]))
    else:
        assert read(REPORT / "c2_profile_resources.json")["passed"]
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    completed, durations, failure = 0, [], None
    try:
        with cache_path.open("a", encoding="utf-8") as stream:
            for key in todo:
                if time.perf_counter() - started > (60 if mode == "profile" else 840):
                    failure = "time_cap"
                    break
                pair = pairs[key]
                encoded = tok(PROMPT.format(text_a=pair["document"], text_b=pair["claim"]),
                              return_tensors="pt", truncation=False).to("cuda")
                t = time.perf_counter()
                with torch.inference_mode():
                    logits = model(**encoded).logits[0].float()
                    value = {"key": key, "logits": logits.cpu().tolist(),
                             "margin": float(logits[0] - logits[1]),
                             "probabilities": logits.softmax(-1).cpu().tolist(),
                             "input_tokens": lengths[key], "truncated_tokens": 0}
                durations.append(time.perf_counter() - t)
                stream.write(json.dumps(value) + "\n")
                stream.flush()
                completed += 1
                if completed % 200 == 0:
                    print(f"{mode} {completed}/{len(todo)}", flush=True)
                if torch.cuda.max_memory_allocated() > 10 * 1024**3:
                    failure = "VRAM_cap"
                    break
    except RuntimeError as exc:
        failure = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    estimate = (len(lengths) - completed) * sum(durations) / max(1, completed)
    write(destination, {"seconds_with_load": elapsed, "new_pairs": completed, "failure": failure,
                        "remaining_estimate_seconds": estimate,
                        "passed": failure is None and estimate < 840,
                        "max_input_tokens": max(lengths.values(), default=0), **memory_report()})
    print({"new_pairs": completed, "seconds": elapsed, "failure": failure, "estimate": estimate})
    if failure:
        raise SystemExit(1)


def project_all():
    cache = {r["key"]: r for r in rows(PRED / "c2_pair_cache.jsonl")}
    assert set(cache) == set(read(BASE / "c2_pairs.json"))
    for dataset in read(BASE / "c2_freeze.json")["counts"]:
        output = [project(row, cache) for row in rows(BASE / f"c2_{dataset}_layout.jsonl")]
        for arm in ("original", "joint_window", "delta"):
            write_rows(PRED / f"c2_{dataset}_{arm}.jsonl", [r[arm] for r in output])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile", "run", "project"))
    args = parser.parse_args()
    if args.mode in ("profile", "run"):
        infer(args.mode)
    else:
        {"prepare": prepare, "project": project_all}[args.mode]()
