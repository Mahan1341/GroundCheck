"""Semantic hazards are screened before fitting, independently of labels/IDs."""

import pytest

from grounded_hallucination_lab.research.passage_probe import evidence_view
from grounded_hallucination_lab.research.permutation import eligibility, rotation_for_source


def text(middle="Liam arrived in 2021."):
    return f"passage 1:Mira arrived in 2020.\n\npassage 2:{middle}\n\npassage 3:Ada stayed.\n\n"


def test_explicit_dates_and_stable_ids_move_together():
    evidence = text()
    assert eligibility(evidence, "Who arrived?", ["Passage 1 says Mira arrived in 2020."])[
        "eligible"
    ]
    rotated = evidence_view(evidence, "rotate1")
    assert "passage 1:Mira arrived in 2020." in rotated
    assert len(rotated) == len(evidence)


@pytest.mark.parametrize(
    "lead",
    [
        "He stayed.",
        "Then Liam arrived.",
        "This was later.",
        "After that, Liam arrived.",
        "the party arrived.",
    ],
)
def test_cross_passage_dependency_abstains(lead):
    assert not eligibility(text(lead), "Who?", ["Mira."])["eligible"]


@pytest.mark.parametrize(
    "answer",
    [
        "The first passage says yes.",
        "As mentioned above, yes.",
        "The former arrived.",
        "They came in 2020 and 2021, respectively.",
    ],
)
def test_one_answer_hazard_blocks_whole_source(answer):
    assert not eligibility(text(), "Who?", ["Mira arrived.", answer])["eligible"]


def test_internal_chronology_kept_intact():
    assert eligibility(text("Liam arrived. Then Ada left."), "Who?", ["Liam arrived."])["eligible"]
    assert rotation_for_source("source") == rotation_for_source("source")
    assert rotation_for_source("source") in (1, 2)


def test_procedural_queries_keep_raw_replay_after_semantic_audit():
    assert not eligibility(text(), "How to bake bread?", ["Bake."])["eligible"]
    assert eligibility(text(), "When did Liam arrive?", ["2021."])["eligible"]
