"""C2 strict-margin and grouped diagnostic checks, entirely synthetic and offline."""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scripts"))

from claim_grounding_c2_analyze import (  # noqa: E402
    ARMS,
    coverage,
    factorial_metrics,
    paired_bootstrap,
    response_metrics,
)


def test_zero_margin_is_not_positive_and_rank_scores_may_be_negative():
    value, predicted = response_metrics([0, 0, 1], [-20.0, 0.0, 2.0])
    assert predicted == [0, 0, 1]
    assert value["macro_f1"] == 1
    assert value["average_precision"] == 1


def test_bootstrap_identical_methods_have_zero_difference():
    table = [
        {"source": source, "context": True, "label": label, "scores": dict.fromkeys(ARMS, score)}
        for source, label, score in (("a", 0, -1.0), ("a", 1, 1.0), ("b", 0, -0.5), ("b", 1, 2.0))
    ]
    out = paired_bootstrap(
        ["a", "b"],
        [0, 1],
        dict.fromkeys(ARMS, [0, 1]),
        dict.fromkeys(ARMS, [[0, 0, 0], [2, 1, 1]]),
        table,
        resamples=20,
    )
    assert out["differences_from_original"]["delta"]["token_f1"]["low"] == 0
    assert out["context_claim_ap_difference_from_original"]["delta"]["high"] == 0


def test_branch_switch_uses_evidence_provenance_not_hypothesis_pair_key():
    claim = {
        "margin": 0.0,
        "no_context_fallback": False,
        "winning_evidence_keys": {"original": "a", "joint_window": "j", "context_only": "c"},
        "hypotheses": {
            "original": {"evidence": [{"key": "a", "source_pieces": [[0, 10]]}]},
            "joint_window": {"evidence": [{"key": "j", "source_pieces": [[0, 10]]}]},
            "context_only": {"evidence": [{"key": "c", "source_pieces": [[20, 30]]}]},
        },
    }
    value = coverage(dict.fromkeys(ARMS, [{"claims": [claim]}]))
    assert value["evidence_chunk_switch_original_joint"] == 0
    assert value["evidence_chunk_switch_joint_context"] == 1
    assert value["delta_absolute_margin_lt_1e_minus_6"] == 1
    assert value["delta_tiny_positive_margin_lt_1e_minus_6"] == 0


def test_factorial_conditions_use_both_roles_and_both_answer_orders():
    design, predictions = [], []
    for a, b, order in itertools.product((0, 1), repeat=3):
        parents = [("p", a), ("q", b)] if order == 0 else [("q", b), ("p", a)]
        identity = f"{a}{b}{order}"
        target = [
            {"start": i * 2, "end": i * 2 + 1, "parent": parent, "gold": gold}
            for i, (parent, gold) in enumerate(parents)
        ]
        design.append({"id": identity, "source_id": "group", "order": order, "targets": target})
        predictions.append(
            {
                "id": identity,
                "response": "\n".join(p for p, _ in parents),
                "claims": [
                    {"pieces": [[t["start"], t["end"]]], "margin": 1.0 if t["gold"] else -1.0}
                    for t in target
                ],
            }
        )
    report = factorial_metrics(design, dict.fromkeys(ARMS, predictions))
    assert report["delta"]["target_role_observations"] == 16
    for neighbor in ("0", "1"):
        value = report["delta"]["neighbor_conditions"][neighbor]
        assert value["pairs"] == 4
        assert value["pair_accuracy"] == 1
        assert value["target_metrics"]["positive_class"]["recall"] == 1
