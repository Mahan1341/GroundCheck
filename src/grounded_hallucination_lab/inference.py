"""Local-only B7 inference with verified weights and original character offsets."""

import argparse
import hashlib
import json
import math
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

from grounded_hallucination_lab.research.protocol import InferenceInput
from grounded_hallucination_lab.research.span_model import windows


def load_profile(path: Path | None = None) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8")) if path else json.loads(
        files("grounded_hallucination_lab").joinpath("assets/b7_inner_seed42.json").read_text()
    )
    required = {"config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json"}
    if set(profile.get("files", {})) != required:
        raise ValueError("Profile must hash exactly the five documented checkpoint files.")
    for key in ("response_threshold", "token_threshold"):
        if not isinstance(profile.get(key), int | float) or not 0 < profile[key] <= 1:
            raise ValueError(f"Profile {key} must be in (0, 1].")
    if profile.get("max_length") != 2048 or not isinstance(profile.get("model_id"), str):
        raise ValueError("Profile must describe a named B7-compatible 2048-token model.")
    return profile


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(directory: Path, profile: dict) -> None:
    """Fail before model loading; never silently create or download random weights."""
    if not directory.is_dir():
        raise ValueError(
            "Checkpoint directory missing. See docs/CHECKPOINT.md; weights not bundled."
        )
    for name, expected in profile["files"].items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Checkpoint file missing: {name}. See docs/CHECKPOINT.md.")
        if file_hash(path) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {name}; expected the declared artifact.")


def validate_input(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Each input must be a JSON object.")
    for key in ("question", "evidence", "answer"):
        if key not in value or not isinstance(value[key], str):
            raise ValueError(f"Input field '{key}' must be a string.")
    if "id" in value and not isinstance(value["id"], str | int):
        raise ValueError("Optional id must be a string or integer.")
    return value


def predicted_spans(answer: str, offsets: list, scores: list, threshold: float) -> list[dict]:
    """Trim whitespace and merge overlapping/whitespace-adjacent positive token ranges."""
    if len(offsets) != len(scores):
        raise ValueError("Token offsets and scores are not aligned.")
    result = []
    for (start, end), score in zip(offsets, scores, strict=True):
        if not 0 <= start <= end <= len(answer) or not math.isfinite(score):
            raise ValueError("Invalid token offset or score.")
        if score < threshold:
            continue
        while start < end and answer[start].isspace():
            start += 1
        while end > start and answer[end - 1].isspace():
            end -= 1
        if end == start:
            continue
        if result and (start <= result[-1]["end"] or answer[result[-1]["end"]:start].isspace()):
            result[-1]["end"] = max(end, result[-1]["end"])
            result[-1]["score"] = max(score, result[-1]["score"])
        else:
            result.append({"start": start, "end": end, "score": score})
    return [{**span, "text": answer[span["start"]:span["end"]]} for span in result]


class B7Detector:
    """Stable FP32/batch-one adapter; checkpoint, inference and thresholds are explicit."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", profile: Path | None = None):
        self.profile = load_profile(profile)
        self.profile_origin = "user-supplied" if profile else "bundled-B7-reference"
        directory = Path(checkpoint)
        validate_checkpoint(directory, self.profile)
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise ValueError("Install the neural dependencies: pip install '.[neural]'.") from exc
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.float32, attn_implementation="sdpa", reference_compile=False,
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        if self.model.config.id2label.get(1) != "hallucination":
            raise ValueError("Checkpoint label mapping does not match the B7 contract.")

    def predict(self, value: dict, *, include_tokens: bool = False) -> dict:
        import torch

        value = validate_input(value)
        answer = value["answer"]
        view = InferenceInput("input", "input", value["question"], value["evidence"], answer)
        encoded, info = windows(view, self.tokenizer, max_length=self.profile["max_length"])
        sums = np.zeros(info["answer_tokens"], dtype=np.float64)
        counts = np.zeros(info["answer_tokens"], dtype=np.int64)
        if answer.strip():
            for window in encoded:
                inputs = {
                    key: torch.tensor([window[key]], dtype=torch.long, device=self.device)
                    for key in ("input_ids", "attention_mask")
                }
                with torch.inference_mode():
                    probabilities = self.model(**inputs).logits.float().softmax(-1)[0, :, 1]
                probabilities = probabilities.cpu().tolist()
                for index, score in zip(window["answer_indices"], probabilities, strict=True):
                    if index >= 0:
                        sums[index] += score
                        counts[index] += 1
            if not np.all(counts > 0):
                raise RuntimeError("An answer token was lost during window projection.")
        else:
            counts[:] = 1
        scores = (sums / counts).tolist() if len(counts) else []
        risk = max(scores, default=0.0)
        result = {
            "hallucination_risk": risk,
            "risk_is_calibrated_probability": False,
            "prediction": risk >= self.profile["response_threshold"],
            "status": "scored" if answer.strip() else "no_answer",
            "spans": predicted_spans(
                answer, info["offsets"], scores, self.profile["token_threshold"]
            ),
            "model": {
                "id": self.profile["model_id"],
                "profile_origin": self.profile_origin,
                "weights_sha256": self.profile["files"]["model.safetensors"],
                "response_threshold": self.profile["response_threshold"],
                "token_threshold": self.profile["token_threshold"],
                "decision_rule": "risk >= response_threshold",
                "precision": "float32", "batch_size": 1, "device": self.device, "tf32": False,
            },
            "coverage": {k: info[k] for k in (
                "answer_tokens", "prefix_tokens", "windows", "answer_coverage",
                "evidence_coverage", "truncated_tokens", "max_window_tokens",
            )},
            "warnings": ["Evidence is empty; no source-grounding assurance."]
            if not value["evidence"].strip() else [],
        }
        if "id" in value:
            result["id"] = value["id"]
        if include_tokens:
            result["tokens"] = [
                {"start": a, "end": b, "score": p}
                for (a, b), p in zip(info["offsets"], scores, strict=True)
            ]
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--profile", type=Path, help="Explicit profile for your retrained checkpoint"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "jsonl"), default="json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-tokens", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.exists():
            raise ValueError("Output already exists; choose a new path to preserve predictions.")
        content = args.input.read_text(encoding="utf-8-sig")
        values = [json.loads(content)] if args.format == "json" else [
            json.loads(line) for line in content.splitlines() if line.strip()
        ]
        if not values:
            raise ValueError("Input contains no examples.")
        for value in values:
            validate_input(value)
        model = B7Detector(args.checkpoint, args.device, args.profile)
        outputs = [model.predict(v, include_tokens=args.include_tokens) for v in values]
        text = json.dumps(outputs[0], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.format == "jsonl":
            text = "".join(
                json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in outputs
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"groundcheck: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
