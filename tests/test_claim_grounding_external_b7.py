"""CPU-only external B7 coverage, no-gold input and explicit launch guard tests."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/claim_grounding_external_b7.py"
SPEC = importlib.util.spec_from_file_location("claim_grounding_external_b7", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Tokenizer:
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [3] * len(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def row():
    return {"id": "fixture", "source_id": "source", "parent_id": "parent",
            "question": "", "evidence": "A has three boxes.", "response": "A has two boxes.",
            "offsets": [[i, i + 1] for i in range(len("A has two boxes."))],
            "segments": [], "dropped": []}


def test_encoder_uses_existing_full_evidence_windows_without_gold():
    value = row()
    runner.check_input_rows([value], expected_count=1)
    examples, metadata = runner.encode_inputs([value], Tokenizer())
    assert metadata[0]["parent_id"] == "parent"
    assert metadata[0]["evidence_coverage"] == metadata[0]["answer_coverage"] == 1
    assert all(all(label == -100 for label in ex["labels"]) for ex in examples)


def test_offsets_changed_or_evidence_too_long_fails_before_inference():
    value = row()
    value["offsets"][0] = [0, 2]
    with pytest.raises(ValueError, match="offsets"):
        runner.encode_inputs([value], Tokenizer())
    value = row()
    value["evidence"] = "x" * 2048
    with pytest.raises(ValueError, match="complete evidence"):
        runner.encode_inputs([value], Tokenizer())


def test_gold_fields_and_duplicate_ids_rejected():
    value = row()
    value["label"] = 1
    with pytest.raises(ValueError, match="no gold"):
        runner.check_input_rows([value], expected_count=1)
    with pytest.raises(ValueError, match="unique"):
        runner.check_input_rows([row(), row()], expected_count=2)


def test_explicit_gpu_opt_in_and_selected_plan_hash_required(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text("selected C2 before external evaluation", encoding="utf-8")
    with pytest.raises(ValueError, match="execute-gpu"):
        runner.validate_launch(False, plan, runner.digest(plan))
    with pytest.raises(ValueError, match="SHA256"):
        runner.validate_launch(True, plan, "wrong")
    runner.validate_launch(True, plan, runner.digest(plan))


def test_deadline_blocks_next_forward_without_cuda(monkeypatch):
    deadline = runner.Deadline(started=10, budget=180)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 189.99)
    deadline.pre_forward(None, None)
    assert deadline.attempted_forwards == 1
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 190)
    with pytest.raises(TimeoutError):
        deadline.pre_forward(None, None)
    assert deadline.attempted_forwards == 1
