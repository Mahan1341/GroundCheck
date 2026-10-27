"""R3 semantics, isolation, normalization and aligned supervised loss guards."""

import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.invariance import (
    VALUES,
    normalize_evidence,
    number_words,
    parse_words,
    suite,
)
from grounded_hallucination_lab.research.protocol import InferenceInput


def test_r3_semantics_alignment_balance_and_isolation():
    seen, families, numbers = set(), set(), set()
    for split in VALUES:
        records, design = suite(split)
        sources = {r.source_id for r in records}
        renderers = {d["family"] for d in design}
        values = {v for pair in VALUES[split] for v in pair}
        assert not seen & sources and not families & renderers and not numbers & values
        seen |= sources
        families |= renderers
        numbers |= values
        pairs, marginal = defaultdict(list), defaultdict(list)
        by_key = {}
        for r, d in zip(records, design, strict=True):
            assert gold_label(r) == d["gold"] == int(d["answer_index"] != d["evidence_index"])
            assert r.response[d["claim_start"] : d["claim_end"]].endswith(".")
            for ann in r.annotations:
                assert r.response[ann.start : ann.end] == ann.text
            pairs[d["pair_id"]].append(r)
            marginal[(r.source_id, d["variant"], d["order"], r.response)].append(gold_label(r))
            assert set(vars(InferenceInput.from_record(r))) == {
                "id",
                "source_id",
                "question",
                "evidence",
                "response",
            }
            by_key[
                (d["source_id"], d["variant"], d["order"], d["answer_index"], d["evidence_index"])
            ] = r
        assert all(sorted(v) == [0, 1] for v in marginal.values())
        assert all(p[0].response == p[1].response and len(p) == 2 for p in pairs.values())
        # Normalization restores identical entire premises, not just a numeric bag.
        for d in design:
            if d["variant"] in ("words", "units"):
                key = (
                    d["source_id"],
                    d["variant"],
                    d["order"],
                    d["answer_index"],
                    d["evidence_index"],
                )
                original = by_key[key]
                canonical = by_key[(key[0], "exact", *key[2:])]
                assert normalize_evidence(original.evidence) == canonical.evidence


def test_conservative_numeric_normalization():
    assert parse_words("two thousand") == 2000
    assert parse_words("one one") is None
    assert parse_words("hundred") is None
    for n in (0, 12, 42, 100, 125, 2000, 2048, 9999):
        assert parse_words(number_words(n)) == n
    assert normalize_evidence("Exactly two thousand meters.") == "Exactly 2000 meters."
    assert normalize_evidence("A: 2 kilometers. B: 120 seconds.") == "A: 2000 meters. B: 2 minutes."
    for text in (
        "About 120 seconds.",
        "Between 2 and 4 kilometers.",
        "2-4 kilometers.",
        "1.5 kilometers.",
        "At least 120 seconds.",
        "One could walk.",
        "90 seconds.",
        "two to four kilometers.",
        "two or three kilometers.",
        "two and three kilometers respectively.",
    ):
        assert normalize_evidence(text) == text
    a = normalize_evidence("A: 120 seconds. B: 180 seconds.")
    b = normalize_evidence("A: 180 seconds. B: 120 seconds.")
    assert a != b


def test_invariance_loss_zero_direction_and_gradients():
    torch = pytest.importorskip("torch")
    from grounded_hallucination_lab.research.invariance_training import pair_objective

    x = torch.tensor([[[0.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]], requires_grad=True)
    masks = [[False, True], [False, True]]
    assert pair_objective(x, masks).item() == 0
    y = torch.tensor([[[9.0, -9.0], [2.0, 0.0]], [[-9.0, 9.0], [0.0, 2.0]]], requires_grad=True)
    loss = pair_objective(y, masks)
    loss.backward()
    assert loss.item() > 0
    assert y.grad[0, 1, 1] < 0 < y.grad[1, 1, 1]
    assert torch.equal(y.grad[:, 0, :], torch.zeros_like(y.grad[:, 0, :]))
    with pytest.raises(ValueError):
        pair_objective(y, [[False, False], [False, True]])


@pytest.mark.skipif(
    os.getenv("GROUNDCHECK_RESEARCH_MODELS") != "1", reason="local tokenizer opt-in"
)
def test_real_training_pairs_exclude_truth_flips_and_mask_prefix(monkeypatch):
    from transformers import AutoTokenizer

    from grounded_hallucination_lab.research.invariance_training import INITIAL, batches

    monkeypatch.setitem(sys.modules, "research_r3", SimpleNamespace(frozen=suite))
    tokenizer = AutoTokenizer.from_pretrained(INITIAL, local_files_only=True)
    items, info = batches(tokenizer, 42)
    assert info["natural_rows"] == 2568
    assert info["synthetic_unique_rows"] == 960
    assert info["synthetic_effective_rows"] == 1056
    constrained = 0
    for item in items:
        for e in item["examples"]:
            for idx, label, weight in zip(
                e["answer_indices"], e["labels"], e["loss_weights"], strict=True
            ):
                if idx < 0:
                    assert label == -100 and weight == 0
        if item["masks"] is None:
            continue
        constrained += 1
        for e, mask in zip(item["examples"], item["masks"], strict=True):
            assert all(label in (-100, 0) for label in e["labels"])
            assert all(not m or label == 0 for m, label in zip(mask, e["labels"], strict=True))
    assert constrained == 120
