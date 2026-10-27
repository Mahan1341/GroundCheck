"""Offline tests for R6 semantic truth, grouping and matched interventions."""

from collections import defaultdict

import pytest

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.binding import suite
from grounded_hallucination_lab.research.relations import content_hash


@pytest.mark.parametrize("split", ["fit", "transfer"])
def test_binding_truth_and_crossed_controls(split):
    rows, design = suite(split)
    assert len(rows) == 768 and len({r.source_id for r in rows}) == 16
    answers, evidence, pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    for r, d in zip(rows, design, strict=True):
        facts = {(e, p): v for e, p, v, _ in d["facts"]}
        assert d["actual_value"] == facts[d["target_entity"], d["target_property"]]
        assert gold_label(r) == d["gold"] == int(d["answer_value"] != d["actual_value"])
        assert len({v for _, _, v, _ in d["facts"]}) == 2
        assert str(d["answer_value"]) in r.evidence
        assert r.response[d["slot_start"] : d["slot_end"]].startswith(str(d["answer_value"]))
        for a in r.annotations:
            assert r.response[a.start : a.end] == a.text
        answers[r.response].append(d["gold"])
        evidence[r.evidence].append(d["gold"])
        pairs[d["pair_id"]].append(d["gold"])
    assert all(sum(v) * 2 == len(v) for v in list(answers.values()) + list(evidence.values()))
    assert all(sorted(v) == [0, 1] for v in pairs.values())


def test_binding_matched_targets_and_disjoint_parents_templates():
    a, da = suite("fit", "A")
    b, db = suite("fit", "B")
    held, dh = suite()
    assert content_hash(a, da) == content_hash(*suite("fit", "A"))
    assert {r.source_id for r in a}.isdisjoint(r.source_id for r in held)
    assert {d["family"] for d in db}.isdisjoint(d["family"] for d in dh)
    assert {d["answer_value"] for d in da}.isdisjoint(d["answer_value"] for d in dh)
    entities = defaultdict(set)
    for d in dh:
        for entity, _, _, _ in d["facts"]:
            entities[entity].add(d["source_id"])
    assert all(len(v) == 1 for v in entities.values())
    for x, y in zip(a, b, strict=True):
        assert (x.id, x.response, x.question, x.annotations) == (
            y.id,
            y.response,
            y.question,
            y.annotations,
        )
        assert x.evidence != y.evidence
    assert [d["facts"] for d in da] == [d["facts"] for d in db]


def test_binding_invalid_split_rejected():
    with pytest.raises(ValueError):
        suite("dev")
