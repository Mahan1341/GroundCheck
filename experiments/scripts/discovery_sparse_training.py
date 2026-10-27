"""Cycle 3: explicitly gated matched CE with train-only positive-mass redistribution.

Importing this module uses only the standard library. GPU work requires an explicit
profile or authorized train command; historical source and artifacts are immutable.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/discovery_loop_20261026"
FREEZE = BASE / "cycle3_freeze.json"
PREREG = ROOT / "docs/DISCOVERY_CYCLE3_PREREG.md"
WEIGHT = 3.370202392225359
SEED = 42


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def positive_constant(label_rows: list[list[int]], weight: float = WEIGHT) -> float:
    """Preserve the sum of positive CE coefficients over equal-exposure responses."""
    fractions = []
    for labels in label_rows:
        valid = [y for y in labels if y != -100]
        if not valid or any(y not in (0, 1) for y in valid):
            raise ValueError("nonempty binary supervised answer required")
        if sum(valid):
            fractions.append(sum(valid) / len(valid))
    if not fractions or not math.isfinite(weight) or weight <= 0:
        raise ValueError("positive targets and positive finite weight required")
    return weight * statistics.fmean(fractions)


def token_coefficients(labels: list[int], constant: float, arm: str) -> list[float]:
    """Pure CPU CE coefficients; negative terms are identical in both arms."""
    valid = [y for y in labels if y != -100]
    if arm not in ("A", "B") or not valid or any(y not in (0, 1) for y in valid):
        raise ValueError("arm A/B and nonempty binary answer targets required")
    if not math.isfinite(constant) or constant <= 0:
        raise ValueError("positive finite constant required")
    n, positives = len(valid), sum(valid)
    return [
        0.0
        if y == -100
        else 1 / n
        if y == 0
        else WEIGHT / n
        if arm == "A"
        else constant / positives
        for y in labels
    ]


def logit_gradient_mass(probabilities, labels, coefficients) -> dict:
    """Exact L1 gradient mass for binary-softmax logits; NOT encoder gradients."""
    if not (len(probabilities) == len(labels) == len(coefficients)):
        raise ValueError("unaligned inputs")
    mass = {"positive": 0.0, "negative": 0.0}
    for probability, label, coefficient in zip(probabilities, labels, coefficients, strict=True):
        if not 0 <= probability <= 1 or not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("invalid probability/coefficient")
        if label == -100:
            if coefficient != 0:
                raise ValueError("masked target has nonzero coefficient")
            continue
        if label not in (0, 1):
            raise ValueError("binary labels required")
        mass["positive" if label else "negative"] += 2 * abs(probability - label) * coefficient
    total = sum(mass.values())
    return {**mass, "positive_share": mass["positive"] / total if total else 0.0}


def derivative_gate(rows: list[dict]) -> dict:
    """Frozen gate uses median per-response shares, never pooled token mass."""
    historical = statistics.median(row["A"]["positive_share"] for row in rows)
    treatment = statistics.median(row["B"]["positive_share"] for row in rows)
    ratio = treatment / historical if historical > 0 else None
    return {
        "historical_median_positive_derivative_fraction": historical,
        "treatment_median_positive_derivative_fraction": treatment,
        "median_fraction_ratio": ratio,
        "passed": historical < 0.20 and ratio is not None and ratio >= 2,
    }


def checked() -> dict:
    from research_r10 import checked as old_checked

    frozen = read(FREEZE)
    assert frozen["runner_sha256"] == sha(Path(__file__)), "runner freeze mismatch"
    assert frozen["prereg_sha256"] == sha(PREREG), "prereg freeze mismatch"
    if "tests_sha256" in frozen:
        assert frozen["tests_sha256"] == sha(ROOT / "tests/test_discovery_sparse.py")
    old = old_checked()  # No replacement of the historical source/dependency guard.
    return {"cycle3": frozen, "historical_R10": old["state"]}


def prepare(tokenizer):
    from research_r10 import batches

    items, info = batches(tokenizer, SEED, "A")
    assert len(items) == 2568 and info["updates"] == 642
    assert len({item["examples"][0]["record_id"] for item in items}) == 2568
    for item in items:
        assert item["masks"] is None
        assert len(item["examples"]) == 2 and item["examples"][0] == item["examples"][1]
        for y, w in zip(
            item["examples"][0]["labels"], item["examples"][0]["loss_weights"], strict=True
        ):
            assert (y == -100 and w == 0) or (y in (0, 1) and w == 1)
    labels = [item["examples"][0]["labels"] for item in items]
    constant = positive_constant(labels)
    mass = {}
    for arm in ("A", "B"):
        mass[arm] = sum(
            c
            for row in labels
            for y, c in zip(row, token_coefficients(row, constant, arm), strict=True)
            if y == 1
        )
    assert math.isclose(mass["A"], mass["B"], abs_tol=1e-9)
    return items, {**info, "positive_constant": constant, "positive_coefficient_mass": mass}


def treatment_loss(model, tokenizer, item, constant):
    import torch
    import torch.nn.functional as F

    from grounded_hallucination_lab.research.span_model import collate

    assert item["examples"][0] == item["examples"][1] and item["masks"] is None
    batch = collate(item["examples"], tokenizer.pad_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"]
    tokens = F.cross_entropy(
        logits.float().reshape(-1, 2), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(labels)
    positive, negative = labels == 1, labels == 0
    count = (positive | negative).sum(1).clamp_min(1)
    per_row = (tokens * negative).sum(1) / count
    per_row += constant * (tokens * positive).sum(1) / positive.sum(1).clamp_min(1)
    return per_row.mean()


def loss_for(arm, model, tokenizer, item, constant):
    if arm == "A":
        from grounded_hallucination_lab.research.invariance_training import batch_loss

        return batch_loss(model, tokenizer, item, 0)[0]
    return treatment_loss(model, tokenizer, item, constant)


def config(state, info, arm) -> dict:
    return {
        "arm": arm,
        "seed": SEED,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "warmup": 64,
        "updates": 642,
        "microbatch": 2,
        "accumulation": 4,
        "epochs": 1,
        "positive_weight_A": WEIGHT,
        "precision": "BF16 autocast; FP32 parameters/AdamW/loss; TF32 off",
        "positive_redistribution": "constant per positive response, global mass preserved",
        "negative_terms": "identical historical coefficients; no synthetic data/feature changes",
        "state": state,
        **info,
    }


def is_short(record) -> bool:
    chars = {
        i
        for a in record.annotations
        for i in range(a.start, a.end)
        if not record.response[i].isspace()
    }
    return (
        bool(chars)
        and len(record.response) >= 500
        and len(chars) <= 0.05 * sum(not c.isspace() for c in record.response)
    )


def gradient_pilot(model, tokenizer, items, constant) -> dict:
    import torch
    import torch.nn.functional as F
    from research_r10 import BASE as R10_BASE
    from research_r10 import records

    from grounded_hallucination_lab.research.span_model import collate

    rr = records(R10_BASE / "train_records.jsonl")
    lookup = {item["examples"][0]["record_id"]: item["examples"][0] for item in items}
    groups = {
        "positive_long_short": [r for r in rr if is_short(r)],
        "positive_other": [r for r in rr if r.annotations and not is_short(r)],
        "negative": [r for r in rr if not r.annotations],
    }
    result = {}
    model.eval()
    for group, candidates in groups.items():
        chosen = sorted(
            candidates,
            key=lambda r: hashlib.sha256(f"cycle3:42:{r.source_id}:{r.id}".encode()).hexdigest(),
        )[:32]
        assert len(chosen) == 32
        rows = []
        for record in chosen:
            example = lookup[record.id]
            batch = collate([example], tokenizer.pad_token_id)
            with torch.inference_mode():  # Stable FP32 batch1; no autocast.
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                probabilities = output.logits.float().softmax(-1)[0, :, 1].cpu().tolist()
                token_losses = (
                    F.cross_entropy(
                        output.logits.float().reshape(-1, 2),
                        batch["labels"].reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    .cpu()
                    .tolist()
                )
            labels = example["labels"]
            arms = {}
            for arm in ("A", "B"):
                coefficients = token_coefficients(labels, constant, arm)
                arms[arm] = {
                    **logit_gradient_mass(probabilities, labels, coefficients),
                    "positive_ce": sum(
                        loss * c
                        for loss, c, y in zip(token_losses, coefficients, labels, strict=True)
                        if y == 1
                    ),
                    "negative_ce": sum(
                        loss * c
                        for loss, c, y in zip(token_losses, coefficients, labels, strict=True)
                        if y == 0
                    ),
                }
            rows.append(
                {
                    "id": record.id,
                    "source_id": record.source_id,
                    "answer_tokens": sum(y >= 0 for y in labels),
                    "positive_tokens": sum(y == 1 for y in labels),
                    **arms,
                }
            )
        summary = {}
        for arm in ("A", "B"):
            pos = sum(row[arm]["positive"] for row in rows)
            neg = sum(row[arm]["negative"] for row in rows)
            summary[arm] = {
                "positive": pos,
                "negative": neg,
                "positive_share": pos / (pos + neg) if pos + neg else 0.0,
                "positive_ce": sum(row[arm]["positive_ce"] for row in rows),
                "negative_ce": sum(row[arm]["negative_ce"] for row in rows),
            }
        result[group] = {"rows": rows, "sources": len({r.source_id for r in chosen}), **summary}
    short = result["positive_long_short"]
    gate = derivative_gate(short["rows"])
    return {
        "groups": result,
        "gradient": "2*abs(p-y)*coefficient: L1 of two output logits, NOT encoder gradient",
        "gate_definition": "median per-response short positive derivative share A<.20; B/A>=2",
        "pooled_short_positive_mass_ratio_secondary": short["B"]["positive"]
        / short["A"]["positive"]
        if short["A"]["positive"]
        else None,
        **gate,
    }


def profile() -> None:
    import torch

    from grounded_hallucination_lab.research.pair_training import INITIAL
    from grounded_hallucination_lab.research.span_model import load_model
    from grounded_hallucination_lab.research.verifiers import memory_report

    state = checked()
    destination = BASE / "cycle3_profile"
    destination.mkdir(exist_ok=False)
    model, tokenizer = load_model(INITIAL, SEED)
    items, info = prepare(tokenizer)
    write(destination / "config.json", config(state, info, "profile"))
    pilot = gradient_pilot(model, tokenizer, items, info["positive_constant"])
    write(destination / "gradient_pilot.json", pilot)
    if not pilot["passed"]:
        write(destination / "result.json", {"passed": False, "reason": "gradient gate closed"})
        print("Cycle3 gradient gate closed; no backward/profile/fit", flush=True)
        return
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    ordered = sorted(items, key=lambda item: len(item["examples"][0]["input_ids"]))
    resources = {}
    for arm in ("A", "B"):
        timings = []
        for quantile in (1.0, 0.25, 0.5, 0.75, 1.0):
            item = ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = loss_for(arm, model, tokenizer, item, info["positive_constant"])
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(
                {
                    "quantile": quantile,
                    "seconds": time.perf_counter() - start,
                    "loss": float(loss.detach()),
                    **memory_report(),
                }
            )
        resources[arm] = {
            "timings": timings,
            "projected_fit_seconds": statistics.fmean(t["seconds"] for t in timings[1:])
            * len(items),
        }
    assert checked() == state
    projected = sum(value["projected_fit_seconds"] for value in resources.values())
    passed = projected < 1800 and all(
        t["cuda_peak_allocated_bytes"] < 10 * 2**30
        for value in resources.values()
        for t in value["timings"]
    )
    result = {
        "passed": passed,
        "gradient_gate": pilot["passed"],
        "resources": resources,
        "projected_pair_seconds": projected,
        "warning": "discard-only resource steps; shared mutable model, NOT quality comparison",
        "checkpoint_saved": False,
    }
    write(destination / "result.json", result)
    print({"profile_passed": passed, "projected_pair_seconds": projected}, flush=True)


def train(arm: str, authorize: bool) -> None:
    if not authorize:
        raise ValueError("fit requires explicit --authorize-fit after parent approval")
    import torch
    from transformers import get_linear_schedule_with_warmup

    from grounded_hallucination_lab.research.pair_training import INITIAL, INITIAL_SHA
    from grounded_hallucination_lab.research.span_model import load_model
    from grounded_hallucination_lab.research.verifiers import memory_report

    state = checked()
    profile_path = BASE / "cycle3_profile/result.json"
    assert read(profile_path)["passed"], "profile gate closed"
    assert read(BASE / "cycle3_profile/config.json")["state"] == state
    destination = BASE / f"cycle3_{arm}s42"
    destination.mkdir(exist_ok=False)
    model, tokenizer = load_model(INITIAL, SEED)
    model.gradient_checkpointing_enable()
    items, info = prepare(tokenizer)
    cfg = {
        **config(state, info, arm),
        "initial_sha256": INITIAL_SHA,
        "profile_sha256": sha(profile_path),
    }
    write(destination / "training_config.json", cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, 64, 642)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with (destination / "batch_log.jsonl").open("x", encoding="utf-8") as stream:
        for step, item in enumerate(items, 1):
            loss = loss_for(arm, model, tokenizer, item, info["positive_constant"])
            assert torch.isfinite(loss)
            (loss / 4).backward()
            stream.write(json.dumps({"step": step, "ce": float(loss.detach())}) + "\n")
            if step % 4 == 0:
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                stream.flush()
                print(f"Cycle3 {arm} seed42 {step}/2568 CE={float(loss.detach()):.5f}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert checked() == state
    model.save_pretrained(destination / "final", safe_serialization=True)
    tokenizer.save_pretrained(destination / "final")
    result = {
        "training_seconds": seconds,
        "batches_completed": len(items),
        "updates_completed": 642,
        "checkpoint_sha256": sha(destination / "final/model.safetensors"),
        "batch_log_sha256": sha(destination / "batch_log.jsonl"),
        **memory_report(),
    }
    if arm == "A":
        old = read(ROOT / "models/phase2b_r10/R10As42/result.json")
        result["historical_R10A_checkpoint_sha256"] = old["checkpoint_sha256"]
        result["bitwise_matches_historical_R10A"] = (
            result["checkpoint_sha256"] == old["checkpoint_sha256"]
        )
        result["reproduction_note"] = "Hash mismatch requires evaluation; no equivalence assumed."
    write(destination / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("profile", "train"))
    parser.add_argument("--arm", choices=("A", "B"))
    parser.add_argument("--authorize-fit", action="store_true")
    args = parser.parse_args()
    if args.command == "profile":
        if args.arm or args.authorize_fit:
            parser.error("profile accepts neither arm nor fit authorization")
        profile()
    elif not args.arm:
        parser.error("train requires --arm")
    else:
        train(args.arm, args.authorize_fit)
