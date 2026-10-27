"""Opt-in frozen B7 inference on SAMSum inputs only; never deserialize gold labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/claim_grounding_20261026"
INPUT = BASE / "external/samsum_inputs.jsonl"
MANIFEST = BASE / "external/samsum_manifest.json"
FREEZE = BASE / "external_B7_freeze.json"
PREDICTIONS = ROOT / "predictions/claim_grounding_20261026/external_B7.jsonl"
RESOURCES = ROOT / "reports/claim_grounding_20261026/external_B7_resources.json"
CHECKPOINT = ROOT / "models/phase2b/span_inner_s42/best"
WEIGHT_SHA256 = "7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4"
BUDGET_SECONDS = 180.0
INPUT_FIELDS = {"id", "source_id", "parent_id", "question", "evidence", "response",
                "offsets", "segments", "dropped"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if jsonl:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def check_input_rows(rows: list[dict], expected_count: int = 295) -> None:
    if len(rows) != expected_count or len({r["id"] for r in rows}) != expected_count:
        raise ValueError("Expected unique complete frozen inference IDs")
    if any(set(row) != INPUT_FIELDS for row in rows):
        raise ValueError("Inference schema must have no gold/generator/annotation fields")
    if expected_count == 295 and len({r["parent_id"] for r in rows}) != 59:
        raise ValueError("Expected all 59 SAMSum parents")


def encode_inputs(rows: list[dict], tokenizer) -> tuple[list, list]:
    from grounded_hallucination_lab.research.protocol import InferenceInput
    from grounded_hallucination_lab.research.span_model import windows

    examples, metadata = [], []
    for row in rows:
        view = InferenceInput(**{k: row[k] for k in (
            "id", "source_id", "question", "evidence", "response",
        )})
        encoded, info = windows(view, tokenizer, max_length=2048, answer_only=False, spans=None)
        if [list(p) for p in info["offsets"]] != [list(p) for p in row["offsets"]]:
            raise ValueError(f"Original B7 token offsets changed: {row['id']}")
        if (info["answer_coverage"] != 1 or info["evidence_coverage"] != 1
                or info["truncated_tokens"] or info["max_window_tokens"] > 2048):
            raise ValueError("Complete answer/evidence coverage is mandatory")
        if any(any(y != -100 for y in example["labels"]) for example in encoded):
            raise ValueError("Inference must not contain supervised targets")
        info["parent_id"] = row["parent_id"]
        examples.extend(encoded)
        metadata.append(info)
    return examples, metadata


def verify_prepared_data() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"SAMSum freeze mismatch: {name}")
    # Label files appear only in the hash loop above, never a JSON loader.
    if digest(CHECKPOINT / "model.safetensors") != WEIGHT_SHA256:
        raise ValueError("B7 checkpoint hash changed")
    config = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    if config["id2label"] != {"0": "not_annotated", "1": "hallucination"}:
        raise ValueError("Unexpected B7 label mapping")
    if config.get("torch_dtype") != "float32":
        raise ValueError("Unexpected B7 weight dtype")


def prepare() -> None:
    if FREEZE.exists():
        raise FileExistsError("B7 inference source already frozen")
    verify_prepared_data()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True,
                                               trust_remote_code=False)
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines()]
    check_input_rows(rows)
    examples, metadata = encode_inputs(rows, tokenizer)
    sources = [Path(__file__), ROOT / "tests/test_claim_grounding_external_b7.py",
               ROOT / "src/grounded_hallucination_lab/research/span_model.py",
               ROOT / "src/grounded_hallucination_lab/research/stable_inference.py",
               ROOT / "src/grounded_hallucination_lab/research/protocol.py",
               ROOT / "src/grounded_hallucination_lab/research/verifiers.py",
               CHECKPOINT / "config.json"]
    paths = [*sources, MANIFEST, INPUT, CHECKPOINT / "model.safetensors"]
    paths += [p for p in CHECKPOINT.glob("*token*") if p.is_file()]
    snapshot = BASE / "external_B7_source"
    snapshot.mkdir(exist_ok=False)
    for path in sources:
        with (snapshot / path.name).open("xb") as stream:
            stream.write(path.read_bytes())
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "numpy")}
    write(FREEZE, {"files_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
                   "response_ids": [row["id"] for row in rows], "responses": len(rows),
                   "parents": 59, "windows": len(examples), "versions": versions,
                   "max_window_tokens": max(m["max_window_tokens"] for m in metadata),
                   "precision": "fp32", "batch_size": 1, "tf32": False, "seed": 42,
                   "max_length": 2048, "wall_budget_seconds": BUDGET_SECONDS,
                   "gold_labels_deserialized": False, "gpu_launched": False,
                   "protocol": "Existing windows and stable_inference.score; unchanged max risk",
                   "launch_requires": "--execute-gpu and explicit selected-C2 plan path+SHA256"})
    print(json.dumps({"prepared_responses": len(rows), "windows": len(examples),
                      "freeze_sha256": digest(FREEZE), "gpu_launched": False}))


def verify_freeze() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for name, expected in freeze["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"B7 inference freeze mismatch: {name}")
    for name, expected in freeze["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Pinned runtime version changed: {name}")
    return freeze


def validate_launch(execute_gpu: bool, plan: Path | None, plan_sha256: str | None) -> None:
    if not execute_gpu:
        raise ValueError("GPU execution requires explicit --execute-gpu")
    if plan is None or plan_sha256 is None or digest(plan) != plan_sha256:
        raise ValueError("Selected C2 evaluation plan path and matching SHA256 are required")


class Deadline:
    def __init__(self, started: float, budget: float = BUDGET_SECONDS):
        self.started = started
        self.budget = budget
        self.attempted_forwards = 0

    def check(self) -> None:
        if time.perf_counter() - self.started >= self.budget:
            raise TimeoutError("B7 external wall budget exhausted; no further forward permitted")

    def pre_forward(self, _model, _args) -> None:
        self.check()
        self.attempted_forwards += 1


def run(execute_gpu: bool, plan: Path | None, plan_sha256: str | None) -> None:
    started = time.perf_counter()
    validate_launch(execute_gpu, plan, plan_sha256)
    if PREDICTIONS.exists() or RESOURCES.exists() or (BASE / "external_B7_launch.json").exists():
        raise FileExistsError("Preserving previous B7 execution artifacts")
    freeze = verify_freeze()
    verify_prepared_data()
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines()]
    check_input_rows(rows)
    if [r["id"] for r in rows] != freeze["response_ids"]:
        raise ValueError("Frozen response IDs changed")
    write(BASE / "external_B7_launch.json", {"source_freeze_sha256": digest(FREEZE),
          "selected_c2_plan_path": str(plan), "selected_c2_plan_sha256": plan_sha256,
          "wall_budget_seconds": BUDGET_SECONDS, "gold_labels_deserialized": False})
    deadline = Deadline(started)
    runtime, status, failure, hook = {}, "incomplete", None, None
    try:
        deadline.check()
        from grounded_hallucination_lab.research.span_model import load_model
        from grounded_hallucination_lab.research.stable_inference import score

        model, tokenizer = load_model(CHECKPOINT, seed=42)
        model.float()
        deadline.check()
        examples, metadata = encode_inputs(rows, tokenizer)
        hook = model.register_forward_pre_hook(deadline.pre_forward)
        predictions, runtime = score(model, tokenizer, examples, metadata,
                                     precision="fp32", batch_size=1, pad_to=0)
        if [p["id"] for p in predictions] != freeze["response_ids"]:
            raise ValueError("Scorer dropped/reordered response IDs")
        write(PREDICTIONS, predictions, jsonl=True)
        status = "completed"
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if hook is not None:
            hook.remove()
        seconds = time.perf_counter() - started
        write(RESOURCES, {**runtime, "status": status, "failure": failure,
              "attempted_forwards": deadline.attempted_forwards,
              "wall_seconds_including_load_and_validation": seconds,
              "wall_budget_seconds": BUDGET_SECONDS, "budget_met": seconds <= BUDGET_SECONDS,
              "budget_enforcement": "before model load, after load, before each forward; a "
              "blocking load/forward cannot be preempted; any overrun is reported",
              "checkpoint_sha256": WEIGHT_SHA256, "freeze_sha256": digest(FREEZE),
              "gold_labels_deserialized": False, "new_training": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        freeze = verify_freeze()
        print(json.dumps({"verified_files": len(freeze["files_sha256"]),
                          "freeze_sha256": digest(FREEZE)}))
    else:
        run(args.execute_gpu, args.plan, args.plan_sha256)


if __name__ == "__main__":
    main()
