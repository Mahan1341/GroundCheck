from collections import Counter, defaultdict
from dataclasses import replace

import pytest

from grounded_hallucination_lab.data.ragtruth import Annotation, Record, gold_label
from grounded_hallucination_lab.research.counterfactual import (
    make_pairs,
    text_hash,
    training_records,
)


@pytest.fixture
def parents():
    evidence = "Cover the pan and cook for 20 minutes."
    start = evidence.index("20")
    manifest = {
        "accepted": [
            {
                "source_id": "s",
                "text_sha256": text_hash(evidence),
                "start": start,
                "end": start + 2,
                "number": "20",
            }
        ]
    }
    base = Record(
        "r",
        "s",
        "QA",
        "fixture",
        "train",
        "good",
        "Natural answer.",
        "What is the instruction?",
        evidence,
        (),
        {},
        {},
    )
    records = [
        replace(
            base,
            id=f"r{i}",
            annotations=(Annotation(0, 7, "Natural", "Evident Baseless Info", False, False, {}),)
            if i % 2
            else (),
        )
        for i in range(12)
    ]
    return records, manifest


def test_cf_neither_answer_nor_premise_alone_identifies_label(parents):
    records, manifest = parents
    rows, audit = make_pairs(records, manifest)
    assert len(rows) == 4
    answers, premises = defaultdict(list), defaultdict(list)
    for row in rows:
        answers[row.response].append(gold_label(row))
        premises[row.evidence].append(gold_label(row))
        assert row.source_id == "s"
        for annotation in row.annotations:
            assert row.response[annotation.start : annotation.end] == annotation.text
            assert annotation.text in ("20", "40")
    assert all(sorted(v) == [0, 1] for v in answers.values())
    assert all(sorted(v) == [0, 1] for v in premises.values())
    assert audit[0]["source_start"] == 0


def test_cf_control_has_matched_size_and_response_balance(parents):
    records, manifest = parents
    c0, _, a0 = training_records(records, manifest, "C0")
    c1, _, a1 = training_records(records, manifest, "C1")
    assert len(c0) == len(c1) == 16
    assert a0["extra_rows"] == a1["extra_rows"] == 4
    assert Counter(map(gold_label, c0)) == Counter(map(gold_label, c1))
    assert len({r.id for r in c1}) == 16


def test_cf_rejects_wrong_parent_hash_source_and_test(parents):
    records, manifest = parents
    with pytest.raises(ValueError, match="outside supplied fit"):
        make_pairs([replace(r, source_id="heldout") for r in records], manifest)
    with pytest.raises(ValueError, match="hash not found"):
        make_pairs([replace(r, evidence="different") for r in records], manifest)
    with pytest.raises(ValueError, match="test/withheld"):
        make_pairs([replace(r, split="test") for r in records], manifest)


def test_cf_rejects_corrupted_numeric_offsets(parents):
    records, manifest = parents
    manifest["accepted"][0]["end"] += 1
    with pytest.raises(ValueError, match="offset/target mismatch"):
        make_pairs(records, manifest)


def test_probe_rejects_fitted_parent_sources(parents):
    from grounded_hallucination_lab.research.counterfactual import require_unseen_sources

    records, _ = parents
    require_unseen_sources(records, {"different_source"})
    with pytest.raises(ValueError, match="fit supervised weights"):
        require_unseen_sources(records, {"s"})


@pytest.mark.parametrize("changes", [{"split": "test"}, {"labels_available": False}])
def test_probe_never_opens_official_test(parents, changes):
    from grounded_hallucination_lab.research.counterfactual import require_unseen_sources

    records, _ = parents
    with pytest.raises(ValueError, match="not authorized"):
        require_unseen_sources([replace(r, **changes) for r in records], set())
