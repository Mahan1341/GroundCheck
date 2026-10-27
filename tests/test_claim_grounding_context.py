"""Offline checks of native context cancellation and original target provenance."""

import importlib.util
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "claim_grounding_context",
    Path(__file__).resolve().parents[1] / "scripts/claim_grounding_context.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def offline_chunks(monkeypatch):
    """No NLTK resources needed; native chunker is reused, not reimplemented here."""
    monkeypatch.setattr(module, "chunks", lambda text: [{"text": text, "pieces": [[0, len(text)]]}])


def view(response="Red. Blue. Green."):
    return {"id": "example", "source_id": "source", "response": response, "evidence": "Facts."}


def prepare(response="Red. Blue. Green."):
    value = view(response)
    return module.prepare_view(value, [[i, i + 1] for i in range(len(response))])


def test_immediate_kept_neighbors_original_order_and_pair_deduplication():
    pairs, layout = prepare()
    middle = layout["claims"][1]
    assert middle["text"] == "Blue."
    assert middle["neighbor_indices"] == [0, 2]
    assert middle["hypotheses"]["joint_window"]["text"] == "Red.\nBlue.\nGreen."
    assert middle["hypotheses"]["context_only"]["text"] == "Red.\nGreen."
    assert len(pairs) == 7  # 3 singletons, 2 pairs, full joint, noncontiguous context.
    assert all(key == module.pair_key(row["document"], row["claim"]) for key, row in pairs.items())


def test_kept_neighbor_does_not_reintroduce_filtered_refusal():
    pairs, layout = prepare("Red. I cannot answer the question. Blue.")
    assert len(layout["claims"]) == 2
    assert all("cannot" not in p["claim"] for p in pairs.values())
    assert layout["claims"][0]["hypotheses"]["joint_window"]["text"] == "Red.\nBlue."
    assert layout["dropped"][0]["reason"] == "standalone_meta_or_refusal"


def test_frozen_piecewise_units_exclude_gaps_and_annotation_fields():
    value = view("Red DROP Blue. Green.")
    value["claims"] = [
        {"text": "Red\nBlue.", "pieces": [[0, 3], [9, 14]]},
        {"text": "Green.", "pieces": [[15, 21]]},
    ]
    value["gold_spans"] = [[0, 3]]
    value["label"] = 1
    before = deepcopy(value)
    pairs, layout = module.prepare_view(value, [[0, 3], [4, 8], [9, 14], [15, 21]])
    assert value == before
    assert "gold_spans" not in layout and "label" not in layout
    assert all("DROP" not in p["claim"] for p in pairs.values())
    out = module.project(layout, {key: {"margin": 2} for key in pairs})
    assert out["original"]["token_probabilities"] == [1, 0, 1, 1]
    assert out["delta"]["token_probabilities"] == [0, 0, 0, 0]
    assert all(c["verdict"] == "abstain" for c in out["delta"]["claims"])


def test_delta_is_difference_not_joint_score_and_projects_only_target():
    pairs, layout = prepare("Red. Blue.")
    margins = {"Red.": -2, "Blue.": 3, "Red.\nBlue.": 2}
    cache = {key: {"margin": margins[p["claim"]]} for key, p in pairs.items()}
    before = deepcopy((layout, cache))
    out = module.project(layout, cache)
    assert [c["margin"] for c in out["original"]["claims"]] == [-2, 3]
    assert [c["margin"] for c in out["joint_window"]["claims"]] == [2, 2]
    assert [c["margin"] for c in out["delta"]["claims"]] == [-1, 4]
    assert out["delta"]["score"] == 4  # Not a probability and never sigmoid-normalized.
    assert out["delta"]["token_probabilities"] == [0] * 5 + [1] * 5
    assert (layout, cache) == before


def test_independent_best_support_branches_and_deterministic_key_tie(monkeypatch):
    monkeypatch.setattr(module, "chunks", lambda _: [
        {"text": "First", "pieces": [[0, 5]]},
        {"text": "Second", "pieces": [[6, 12]]},
    ])
    pairs, layout = prepare("Red. Blue.")
    margins = {
        ("First", "Red."): -1, ("Second", "Red."): -1,
        ("First", "Blue."): 3, ("Second", "Blue."): -3,
        ("First", "Red.\nBlue."): -2, ("Second", "Red.\nBlue."): 2,
    }
    cache = {key: {"margin": margins[p["document"], p["claim"]]} for key, p in pairs.items()}
    out = module.project(layout, cache)
    claim = out["delta"]["claims"][0]
    assert claim["margin"] == 1  # min joint(-2) - min context(-3), not min pairwise delta.
    assert claim["winning_evidence_keys"]["original"] == min(
        module.pair_key(doc, "Red.") for doc in ("First", "Second")
    )
    assert claim["winning_evidence_keys"]["context_only"] == module.pair_key("Second", "Blue.")


def test_singleton_fallback_no_empty_context_forward_and_native_log_odds():
    pairs, layout = prepare("Red.")
    assert len(pairs) == 1
    assert layout["claims"][0]["hypotheses"]["context_only"] is None
    cache = {key: {"probabilities": [0.8, 0.2]} for key in pairs}
    out = module.project(layout, cache)
    for row in out.values():
        assert row["score"] == pytest.approx(math.log(4))
        assert row["no_context_fallback_count"] == 1
        assert row["prediction"] == 1
        assert row["token_probabilities"] == [1] * 4


def test_exact_zero_abstains_negative_preserves_ranking_and_special_offsets():
    pairs, layout = module.prepare_view(view("Red."), [[0, 0], [0, 4]])
    for margin, verdict in ((0, "abstain"), (-0.1, "supported"), (0.1, "unsupported")):
        out = module.project(layout, {key: {"logits": [margin, 0]} for key in pairs})
        assert out["original"]["score"] == margin
        assert out["original"]["verdict"] == verdict
        assert out["original"]["token_probabilities"] == [0, int(margin > 0)]


def test_no_claim_has_no_forward_and_explicit_abstention():
    pairs, layout = prepare("I cannot answer the question.")
    assert pairs == {}
    for row in module.project(layout, {}).values():
        assert row["score"] == 0
        assert row["verdict"] == "abstain"
        assert row["no_claim"]
        assert not any(row["token_probabilities"])


@pytest.mark.parametrize("entry", [
    {"margin": float("nan")}, {"margin": float("inf")}, {"support": 0.9},
    {"probabilities": [0, 1]}, {"probabilities": [-0.1, 1.1]},
    {"probabilities": [0.4, 0.4]}, {"logits": [1, 2, 3]},
])
def test_invalid_or_saturated_cache_fails_without_clipping(entry):
    pairs, layout = prepare("Red.")
    with pytest.raises(ValueError):
        module.project(layout, {key: entry for key in pairs})


@pytest.mark.parametrize("fault", ["offset_id", "offset_range", "piece_text", "piece_order"])
def test_preparation_rejects_invalid_provenance(fault):
    value, offsets = view("Red. Blue."), [[0, 4], [5, 10]]
    if fault == "offset_id":
        offsets = {"id": "different", "offsets": offsets}
    elif fault == "offset_range":
        offsets = [[0, 999]]
    elif fault == "piece_text":
        value["claims"] = [{"text": "Green.", "pieces": [[0, 4]]}]
    else:
        value["claims"] = [
            {"text": "Blue.", "pieces": [[5, 10]]}, {"text": "Red.", "pieces": [[0, 4]]},
        ]
    with pytest.raises(ValueError):
        module.prepare_view(value, offsets)
