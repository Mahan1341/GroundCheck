"""Offline semantic/alignment guards for authored R2 diagnostics."""

import re
from collections import Counter, defaultdict

import pytest

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.protocol import InferenceInput
from grounded_hallucination_lab.research.relations import content_hash, relation_suite


@pytest.mark.parametrize(
    "split,n,sources", [("fit", 768, 24), ("selection", 936, 9), ("diagnostic", 1248, 12)]
)
def test_suite_semantics(split, n, sources):
    records, design = relation_suite(split)
    assert len(records) == n
    assert len({r.source_id for r in records}) == sources
    answer_balance, evidence_balance, pair_groups = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for r, d in zip(records, design, strict=True):
        assert gold_label(r) == d["gold"] == int(d["answer_index"] != d["evidence_index"])
        assert d["values"][0] != d["values"][1]
        assert 2 * d["values"][0] != d["values"][1]
        assert r.response[d["slot_start"] : d["slot_end"]] == str(d["values"][d["answer_index"]])
        assert d["claim_start"] <= d["slot_start"] < d["slot_end"] <= d["claim_end"]
        if r.annotations:
            assert r.annotations[0].start == d["slot_start"]
            assert r.annotations[0].text == r.response[d["slot_start"] : d["slot_end"]]
        assert set(InferenceInput.from_record(r).__dict__) == {
            "id",
            "source_id",
            "question",
            "evidence",
            "response",
        }
        answer_balance[r.response].append(d["gold"])
        evidence_balance[r.evidence].append(d["gold"])
        pair_groups[d["pair_id"]].append((r, d))
    for values in [*answer_balance.values(), *evidence_balance.values()]:
        assert sum(values) * 2 == len(values)
    for pair in pair_groups.values():
        assert len(pair) == 2
        assert pair[0][0].response == pair[1][0].response
        assert Counter(re.findall(r"\w+", pair[0][0].evidence)) == Counter(
            re.findall(r"\w+", pair[1][0].evidence)
        )
    assert content_hash(records, design) == content_hash(*relation_suite(split))


def test_parent_family_isolation():
    sets = [{r.source_id for r in relation_suite(s)[0]} for s in ("fit", "selection", "diagnostic")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    fit_families = {d["family"] for d in relation_suite("fit")[1]}
    val = {d["family"] for d in relation_suite("selection")[1] if d["variant"] == "novel"}
    test = {d["family"] for d in relation_suite("diagnostic")[1] if d["variant"] == "novel"}
    assert not fit_families & val and not fit_families & test and not val & test
    with pytest.raises(ValueError):
        relation_suite("dev")
