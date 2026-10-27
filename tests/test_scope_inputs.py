"""Offline matched-content and closed-cache regressions."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "scope_inputs", Path(__file__).resolve().parents[1] / "scripts/scope_inputs.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "text",
    [
        "I cannot answer the question. The chamber holds 42 liters.",
        "I cannot answer the question because the chamber holds 42 liters.",
    ],
)
def test_mixed_refusal_keeps_facts(text):
    spans, _ = module.units(text, True)
    assert any("42 liters" in s["text"] for s in spans)
    assert all(text[s["start"] : s["end"]] == s["text"] for s in spans)


def test_standalone_policy_and_exact_mapping():
    text = "I cannot answer the question. The chamber holds 42 liters."
    raw, _ = module.units(text, False)
    clean, _ = module.units(text, True)
    assert len(raw) == len(clean) + 1
    joined, mapping = module.compose(clean)
    for p in mapping:
        assert (
            joined[p["view_start"] : p["view_end"]] == text[p["original_start"] : p["original_end"]]
        )
    assert not module.units("I cannot answer the question.", True)[0]
    assert module.units("I cannot answer the question.", False)[0]


def test_piecewise_localization_does_not_fill_dropped_gaps():
    row = {
        "offsets": [[0, 2], [3, 5], [6, 8]],
        "claims": [{"pieces": [[0, 2], [6, 8]], "evidence": [{"key": "a"}]}],
    }
    out = module.project([row], {"a": {"support": 0.1}})[0]
    assert out["token_probabilities"] == pytest.approx([0.9, 0, 0.9])


def test_unrelated_cache_values_are_never_decoded(tmp_path):
    good, closed = "a" * 64, "b" * 64
    path = tmp_path / "cache.jsonl"
    path.write_text(
        json.dumps({"key": good, "support": 0.8})
        + "\n"
        + '{"key": "'
        + closed
        + '", "support": INVALID}\n',
        encoding="utf-8",
    )
    assert module.scoped_cache(path, {good}) == {good: {"key": good, "support": 0.8}}
    with pytest.raises(json.JSONDecodeError):
        module.scoped_cache(path, {closed})


def test_duplicate_allowed_cache_fails_closed(tmp_path):
    key = "a" * 64
    path = tmp_path / "cache.jsonl"
    path.write_text((json.dumps({"key": key, "support": 0.8}) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        module.scoped_cache(path, {key})
