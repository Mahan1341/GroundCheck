"""Graph extraction, open-world safety, provenance, and oracle isolation."""

import inspect
from collections import defaultdict
from dataclasses import replace

import pytest

from grounded_hallucination_lab.research import evidence_graph as eg
from grounded_hallucination_lab.research.graph_suite import oracle_graphs, suite


def verdict(evidence, answer):
    e, c = eg.extract(evidence, "doc"), eg.extract(answer, "answer")
    return eg.match_fact(c.facts[0], e, c)["status"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The stock at Alden is 453 crates.", "supported"),
        ("The stock at Alden is 445 crates.", "contradicted"),
        ("The stock at Other is 453 crates.", "unknown"),
        ("In 2024, The stock at Alden is 445 crates.", "unknown"),
        ("The stock at Alden is not 453 crates.", "contradicted"),
        ("The stock at Alden is not 445 crates.", "unknown"),
        ("The stock at Alden is 445 crates. The stock at Alden is 453 crates.", "ambiguous"),
        ("The stock at Alden is 453 crates. It might hold 445 crates.", "unknown"),
        ("The stock at Alden is 453 minutes.", "unknown"),
    ],
)
def test_open_world_and_qualifier_safety(text, expected):
    assert verdict(text, "The stock at Alden is 453 crates.") == expected


def test_alias_resolution_and_cross_passage_provenance():
    text = "The stock at North Depot is 445 crates.\n\nThe label Elm Site refers to North Depot."
    graph = eg.extract(text, "doc")
    claims = eg.extract("The stock at Elm Site is 453 crates.", "answer")
    assert eg.match_fact(claims.facts[0], graph, claims)["status"] == "contradicted"
    retrieved, refs = eg.retrieval_text(text, graph.facts, graph)
    assert {p.passage_id for p in refs} == {"p0", "p1"}
    assert all(text[p.start : p.end] in retrieved for p in refs)
    conflict = text + " The label Elm Site refers to South Depot."
    assert verdict(conflict, "The stock at Elm Site is 453 crates.") == "ambiguous"


def test_fallback_preserves_neural_predictions_and_offsets():
    c = eg.extract("The stock at Elm is 453 crates.", "answer")
    row = {
        "offsets": [(i, i + 1) for i in range(31)],
        "token_probabilities": [0.2] * 31,
        "score": 0.2,
    }
    changed, _ = eg.graph_override(row, c, eg.extract("Unknown evidence.", "doc"))
    assert changed == row
    evidence = eg.extract("The stock at Elm is 445 crates.", "doc")
    changed, _ = eg.graph_override(row, c, evidence)
    assert changed["score"] == 1 and row["score"] == 0.2
    assert [i for i, v in enumerate(changed["token_probabilities"]) if v == 1] == list(
        range(c.facts[0].value_start, c.facts[0].value_end)
    )


def test_automatic_extractor_cannot_receive_design_or_annotations():
    assert list(inspect.signature(eg.extract).parameters) == ["text", "document_id"]
    assert "graph_suite" not in inspect.getsource(eg)
    assert "annotations" not in inspect.getsource(eg)


def test_suite_truth_offsets_and_length_matched_interventions():
    records, design = suite()
    assert len(records) == 1536 and len({r.source_id for r in records}) == 16
    groups = defaultdict(list)
    pairs = defaultdict(list)
    for r, d in zip(records, design, strict=True):
        e, c = oracle_graphs(d)
        assert eg.match_fact(c.facts[0], e, c)["status"] == (
            "contradicted" if d["gold"] else "supported"
        )
        for f in e.facts:
            assert r.evidence[f.value_start : f.value_end] == f"{f.value} {f.unit}"
        assert r.response[d["slot_start"] : d["slot_end"]] == f"{d['value']} crates"
        if d["variant"] in ("different", "one_same", "repeated"):
            key = (r.source_id, d["long"], d["order"], d["answer_index"], d["world"])
            groups[key].append((r, d))
        pairs[d["pair_id"]].append(d["gold"])
    assert all(sorted(v) == [0, 1] for v in pairs.values())
    for group in groups.values():
        assert len({r.response for r, d in group}) == 1
        assert len({len(r.evidence) for r, d in group}) == 1
        assert len({d["actual"] for r, d in group}) == 1
        assert len({d["gold"] for r, d in group}) == 1
        assert len({d["facts"][0]["provenance"]["start"] for r, d in group}) == 1


def test_ambiguous_qualifiers_are_not_merged():
    e = eg.extract(
        "In 2024, The stock at Elm is 453 crates. In 2025, The stock at Elm is 445 crates.", "doc"
    )
    c = eg.extract("In 2025, The stock at Elm is 453 crates.", "answer")
    assert eg.match_fact(c.facts[0], e, c)["status"] == "contradicted"
    c.facts[0] = replace(c.facts[0], time="")
    assert eg.match_fact(c.facts[0], e, c)["status"] == "unknown"
