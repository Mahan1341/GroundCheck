"""C1 frozen-model input-gradient pilot; explicit allowlists and process budgets."""

import argparse
import hashlib
import json
import time

import numpy as np
from discovery_verifier import PROMPT
from scope_analysis import ROOT, read, rows, write

from grounded_hallucination_lab.research.protocol import digest

BASE = ROOT / "models/claim_grounding_20261026"
PRED = ROOT / "predictions/claim_grounding_20261026"
REPORT = ROOT / "reports/claim_grounding_20261026"
OLD = ROOT / "models/scope_diagnosis_20261026"
OLDP = ROOT / "predictions/scope_diagnosis_20261026"
MODEL = ROOT / "models/discovery_loop_20261026/factcg"


def write_rows(path, values):
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def cluster(source):
    return "14325+15545" if source in {"14325", "15545"} else source


def prepare():
    from research_r10 import records
    from scope_analysis import DATA

    all_pairs = read(OLD / "pairs.json")  # Scope packet has no confirmation pairs.
    supports = {r["key"]: r["support"] for r in rows(OLDP / "pair_cache.jsonl")}
    selected, membership, input_files = {}, {}, []
    for dataset in ("natural", "r1", "r6"):
        rr = records(DATA[dataset])
        layout_path = OLDP / f"{dataset}_seg_filtered.jsonl"
        layouts = rows(layout_path)
        input_files += [DATA[dataset], layout_path]
        assert [r.id for r in rr] == [r["id"] for r in layouts]
        groups = sorted({cluster(r.source_id) for r in rr}, key=lambda g: hashlib.sha256(
            ("claim-grounding-c1:42:" + g).encode()).hexdigest())
        allowed = set(groups[:24] if dataset == "natural" else groups)
        output = []
        for record, layout in zip(rr, layouts, strict=True):
            if cluster(record.source_id) not in allowed:
                continue
            winners = [min((e["key"] for e in c["evidence"]),
                           key=lambda k: (-supports[k], k)) for c in layout["claims"]]
            for key in winners:
                selected[key] = {**all_pairs[key], "support": supports[key]}
            output.append({**layout, "response": record.response, "winnerkeys": winners})
        write_rows(BASE / f"c1_{dataset}_layout.jsonl", output)
        membership[dataset] = {"ids": [r["id"] for r in output],
                               "clusters": sorted(allowed), "n": len(output)}
    write(BASE / "c1_pairs.json", selected)
    write(BASE / "c1_membership.json", membership)
    source = [ROOT / p for p in ("scripts/claim_grounding_run.py",
                                 "scripts/claim_grounding_projection.py",
                                 "docs/CLAIM_GROUNDING_C1_PLAN.md",
                                 "scripts/discovery_verifier.py")]
    snapshot = BASE / "c1_source"
    snapshot.mkdir(exist_ok=False)
    for p in source:
        (snapshot / p.name).write_bytes(p.read_bytes())
    files = source + input_files + [BASE / "c1_pairs.json", BASE / "c1_membership.json"]
    files += [BASE / f"c1_{d}_layout.jsonl" for d in membership]
    write(BASE / "c1_freeze.json", {"files": {str(p.relative_to(ROOT)): digest(p) for p in files},
                                    "unique_pairs": len(selected), "gpu_cap_minutes": 18})
    print({"unique_pairs": len(selected), "samples": {k: v["n"] for k, v in membership.items()}})


def infer(mode):
    from grounded_hallucination_lab.research.verifiers import memory_report, setup_runtime

    for name, sha in read(BASE / "c1_freeze.json")["files"].items():
        assert digest(ROOT / name) == sha, name
    assert digest(MODEL / "model.safetensors") == (
        "379a343d1ca25c9dfe24ae149f76cc958ea4f09c00dae88bd49bbbf58c749d5e")
    resource_path = REPORT / f"c1_{mode}_resources.json"
    assert not resource_path.exists(), "completed run must not be repeated"
    started = time.perf_counter()
    torch = setup_runtime()
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pairs = read(BASE / "c1_pairs.json")
    cache_path = PRED / "c1_gradients.jsonl"
    cache = {r["key"]: r for r in rows(cache_path)} if cache_path.exists() else {}
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.pad_token = tok.eos_token
    lengths = {k: len(tok(PROMPT.format(text_a=v["document"], text_b=v["claim"]))["input_ids"])
               for k, v in pairs.items() if k not in cache}
    assert max(lengths.values(), default=0) <= 2048
    todo = sorted(lengths)
    if mode == "profile":
        todo = list(dict.fromkeys(todo[:16] + [max(lengths, key=lengths.get)]))
    else:
        assert read(REPORT / "c1_profile_resources.json")["passed"]
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32).to("cuda").eval()
    model.requires_grad_(False)
    captured = {}
    hook = model.classifier.register_forward_pre_hook(
        lambda module, args: captured.update(pooled=args[0]))
    generator = torch.Generator(device="cuda").manual_seed(42)
    direction = torch.randn(model.classifier.in_features, generator=generator, device="cuda")
    direction *= (model.classifier.weight[0] - model.classifier.weight[1]).std() / direction.std()
    torch.cuda.reset_peak_memory_stats()
    durations, completed, failure, max_delta = [], 0, None, 0.0
    limit = 90 if mode == "profile" else 990
    try:
        with cache_path.open("a", encoding="utf-8") as stream:
            for key in todo:
                if time.perf_counter() - started >= limit:
                    failure = "time_cap"
                    break
                pair = pairs[key]
                text = PROMPT.format(text_a=pair["document"], text_b=pair["claim"])
                start = len(PROMPT.split("{text_b}")[0].format(text_a=pair["document"]))
                end = start + len(pair["claim"])
                encoded = tok(text, return_offsets_mapping=True, return_tensors="pt")
                offsets = encoded.pop("offset_mapping")[0].tolist()
                encoded = {k: v.to("cuda") for k, v in encoded.items()}
                ids = encoded.pop("input_ids")
                emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                t = time.perf_counter()
                logits = model(inputs_embeds=emb, **encoded).logits[0]
                margin = logits[0] - logits[1]
                gradient = torch.autograd.grad(margin, emb, retain_graph=mode == "profile")[0]
                norm = gradient.norm(dim=-1)[0].detach().cpu().tolist()
                random_norm = None
                if mode == "profile":
                    random_margin = (captured["pooled"][0] * direction).sum()
                    random_norm = torch.autograd.grad(random_margin, emb)[0].norm(dim=-1)[0]
                    random_norm = random_norm.detach().cpu().tolist()
                support = float(logits.detach().softmax(-1)[1])
                max_delta = max(max_delta, abs(support - pair["support"]))
                assert abs(support - pair["support"]) < 1e-5, "native input path drift"
                indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
                value = {"key": key, "claim_offsets": [
                    [max(0, offsets[i][0] - start), min(end, offsets[i][1]) - start]
                    for i in indices],
                    "L2values": [norm[i] for i in indices], "support": support,
                    "input_tokens": len(offsets), "truncated_tokens": 0}
                if random_norm is not None:
                    value["random_head_L2values"] = [random_norm[i] for i in indices]
                stream.write(json.dumps(value, allow_nan=False) + "\n")
                stream.flush()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - t)
                completed += 1
                if torch.cuda.max_memory_allocated() > 10 * 1024**3:
                    failure = "VRAM_cap"
                    break
                if completed % 100 == 0:
                    print(f"{mode}: {completed}/{len(todo)}", flush=True)
    except (RuntimeError, AssertionError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        hook.remove()
        elapsed = time.perf_counter() - started
        estimate = (len(lengths) - completed) * np.mean(durations) if durations else None
        write(resource_path, {"seconds_with_load": elapsed, "new_pairs": completed,
                              "failure": failure, "remaining_estimate_seconds": estimate,
                              "max_forward_support_delta": max_delta,
                              "passed": failure is None and estimate is not None and estimate < 990,
                              "max_input_tokens": max(lengths.values(), default=0),
                              **memory_report()})
    print({"mode": mode, "completed": completed, "seconds": elapsed,
           "failure": failure, "remaining_estimate": estimate}, flush=True)
    if failure:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "profile", "run"))
    mode = parser.parse_args().mode
    prepare() if mode == "prepare" else infer(mode)


if __name__ == "__main__":
    main()
