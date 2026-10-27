"""Offline natural relation safety; local NLP inference is independently opt-in."""

import os
import re
from dataclasses import replace

import pytest

from grounded_hallucination_lab.research.evidence_graph import (
    NaturalFact,
    NaturalGraph,
    Provenance,
    natural_fact_from_tags,
    natural_match,
    natural_override,
)


def fact(entity="The lake", value="blue", **kwargs):
    return NaturalFact(
        entity, "be", value, "text", Provenance("doc", "p1", 0, 17), 12, 16, **kwargs
    )


@pytest.mark.parametrize("links", [False, True])
@pytest.mark.parametrize(
    "evidence,expected",
    [
        ([], "unknown"),
        ([fact()], "supported"),
        ([fact(value="green")], "unknown"),
        ([fact(negated=True)], "contradicted"),
        ([fact(), fact(negated=True)], "ambiguous"),
        ([fact(status="candidate")], "ambiguous"),
        ([fact(scope="may")], "unknown"),
        ([fact(time="2023")], "unknown"),
        ([fact(entity="Other lake")], "unknown"),
    ],
)
def test_shared_flat_graph_policy(links, evidence, expected):
    assert natural_match(fact(), NaturalGraph(facts=evidence), links=links)["status"] == expected


def test_graph_only_adds_explicit_evidence_alias_traversal():
    e = NaturalGraph(facts=[fact()], aliases=[("azure pool", "lake", Provenance("d", "p2", 0, 20))])
    claim = fact(entity="Azure Pool")
    assert natural_match(claim, e, links=False)["status"] == "unknown"
    assert natural_match(claim, e, links=True)["status"] == "supported"
    e.aliases.append(("azure pool", "other", Provenance("d", "p3", 30, 50)))
    assert natural_match(claim, e, links=True)["status"] == "ambiguous"


def test_candidate_and_modal_negation_abstain():
    assert (
        natural_match(fact(status="candidate"), NaturalGraph(facts=[fact()]), links=True)["status"]
        == "unknown"
    )
    e = NaturalGraph(facts=[fact(negated=True, scope="may")])
    assert natural_match(fact(scope="may"), e, links=True)["status"] == "unknown"


def test_override_is_local_and_does_not_trust_answer_aliases():
    row = {
        "offsets": [(0, 3), (13, 16), (20, 24)],
        "token_probabilities": [0.2, 0.8, 0.7],
        "score": 0.8,
    }
    e = NaturalGraph(facts=[fact()])
    c = NaturalGraph(facts=[fact()])
    out, _ = natural_override(row, c, e, links=True)
    assert out["token_probabilities"] == [0, 0, 0.7]
    c.facts = [fact(entity="Azure Pool")]
    c.aliases = [("azure pool", "lake", Provenance("answer", "p0", 0, 20))]
    assert natural_override(row, c, e, links=True)[0] == row


@pytest.mark.parametrize(
    "text,tags,issue",
    [
        ("The lake is blue.", ["DT", "NN", "VBZ", "JJ", "."], None),
        ("It is blue.", ["PRP", "VBZ", "JJ", "."], "unresolved_coreference"),
        ("Heat the oven.", ["VB", "DT", "NN", "."], "implicit_subject"),
        (
            "The cost is about 50 dollars.",
            ["DT", "NN", "VBZ", "RB", "CD", "NNS", "."],
            "inexact_value",
        ),
    ],
)
def test_offsets_and_conservative_candidates(text, tags, issue):
    offsets = [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]
    f = natural_fact_from_tags(
        text, offsets, tags, Provenance("doc", "p1", 0, len(text)), str.lower
    )
    assert text[f.value_start : f.value_end] == f.value
    if f.entity != "__implicit_subject__":
        assert text[f.entity_start : f.entity_end] == f.entity
    assert (issue in f.issues) if issue else f.status == "exact"
    assert f.confidence.endswith("unreviewed") if issue is None else f.confidence == "provisional"


@pytest.mark.skipif(os.getenv("GROUNDCHECK_RESEARCH_MODELS") != "1", reason="local pinned NLP")
def test_local_natural_prose_offsets_and_new_syntax():
    from grounded_hallucination_lab.research.evidence_graph import extract_natural

    text = "passage 1:The museum holds paintings.\n\npassage 2:The lake is blue."
    e = extract_natural(text, "doc")
    assert len(e.facts) == 2
    assert {f.provenance.passage_id for f in e.facts} == {"p1", "p2"}
    for f in e.facts:
        assert text[f.value_start : f.value_end] == f.value
        assert text[f.entity_start : f.entity_end] == f.entity
    copied = extract_natural("A lake is blue.", "answer").facts[0]
    assert natural_match(copied, e, links=True)["status"] == "supported"
    assert natural_match(replace(copied, value="green"), e, links=True)["status"] == "unknown"
    assert not extract_natural("Unable to answer based on given passages.", "answer").facts
