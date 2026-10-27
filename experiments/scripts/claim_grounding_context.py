"""Pure CPU native-context cancellation; no model, labels, files or calibration.

All three arms retain the same target pieces. Context is a hypothesis to be verified,
never trusted evidence. A raw log-odds difference is not a factuality probability.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

from discovery_verifier import chunks, pair_key

from grounded_hallucination_lab.research.segments import clean_segments

ARMS = ("original", "joint_window", "delta")


def _offsets(values: Sequence, limit: int, *, empty: bool) -> list[list[int]]:
    result = []
    for value in values:
        if len(value) != 2 or any(type(i) is not int for i in value):
            raise ValueError("Integer [start,end] offsets required")
        a, b = value
        if not 0 <= a <= b <= limit or (not empty and a == b):
            raise ValueError("Offsets outside text or empty retained piece")
        result.append([a, b])
    return result


def _text(response: str, pieces: Sequence[Sequence[int]]) -> str:
    return "\n".join(response[a:b] for a, b in pieces)


def prepare_view(
    view: Mapping[str, Any], offsets: Sequence[Sequence[int]] | Mapping[str, Any]
) -> tuple[dict[str, dict], dict]:
    """Prepare target, adjacent kept units, and their union in original order.

    Required view fields: id, source_id, response, evidence. Optional ``claims`` are
    frozen filtered units with exact text and original pieces; otherwise clean_segments
    is used. Offsets are original answer tokenizer offsets, or its id/offsets row.
    Evidence packing and exact pair keys reuse the existing author adapter unchanged.
    """
    response, evidence = view["response"], view["evidence"]
    if not isinstance(response, str) or not isinstance(evidence, str):
        raise ValueError("Original response and evidence strings required")
    if isinstance(offsets, Mapping):
        if offsets.get("id", view["id"]) != view["id"]:
            raise ValueError("Offset row ID differs from view ID")
        offsets = offsets["offsets"]
    offsets = _offsets(offsets, len(response), empty=True)
    if "claims" in view:
        units = [{"text": c["text"], "pieces": c["pieces"]} for c in view["claims"]]
        dropped = copy.deepcopy(view.get("dropped", []))
    else:
        segments, dropped = clean_segments(response)
        units = [{"text": s.text, "pieces": [[s.start, s.end]]} for s in segments]
    last_end = 0
    for unit in units:
        unit["pieces"] = _offsets(unit["pieces"], len(response), empty=False)
        if not unit["pieces"] or unit["text"] != _text(response, unit["pieces"]):
            raise ValueError("Unit must exactly reconstruct its original pieces")
        for a, b in unit["pieces"]:
            if a < last_end:
                raise ValueError("Kept units/pieces must be ordered and nonoverlapping")
            last_end = b
    contexts, pairs, claims = chunks(evidence), {}, []

    def hypothesis(indices: Sequence[int]) -> dict | None:
        if not indices:
            return None  # Never ask a verifier to score an empty context claim.
        pieces = [piece for index in indices for piece in units[index]["pieces"]]
        text = _text(response, pieces)
        refs = []
        for context in contexts:
            key = pair_key(context["text"], text)
            pairs.setdefault(key, {"document": context["text"], "claim": text})
            refs.append({"key": key, "source_pieces": copy.deepcopy(context["pieces"])})
        return {"text": text, "pieces": copy.deepcopy(pieces), "evidence": refs}

    for index, unit in enumerate(units):
        window = list(range(max(0, index - 1), min(len(units), index + 2)))
        neighbors = [i for i in window if i != index]
        claims.append({
            **copy.deepcopy(unit),
            "index": index,
            "neighbor_indices": neighbors,
            "no_context_fallback": not neighbors,
            "hypotheses": {
                "original": hypothesis([index]),
                "joint_window": hypothesis(window),
                "context_only": hypothesis(neighbors),
            },
        })
    layout = {
        "id": view["id"], "source_id": view["source_id"], "response": response,
        "offsets": offsets, "claims": claims, "dropped": dropped, "no_claim": not claims,
        "retained": [copy.deepcopy(p) for u in units for p in u["pieces"]],
    }
    return pairs, layout


def _margin(entry: Mapping[str, Any]) -> float:
    if "margin" in entry:
        value = float(entry["margin"])
    elif "logits" in entry:
        if len(entry["logits"]) != 2:
            raise ValueError("Expected [No,Yes] logits")
        value = float(entry["logits"][0]) - float(entry["logits"][1])
    elif "probabilities" in entry:
        probabilities = entry["probabilities"]
        if len(probabilities) != 2 or any(
            not math.isfinite(p) or not 0 < p <= 1 for p in probabilities
        ):
            raise ValueError("Positive finite [No,Yes] probabilities required; no clipping")
        if not math.isclose(sum(probabilities), 1, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to one")
        value = math.log(probabilities[0]) - math.log(probabilities[1])
    else:
        raise ValueError("Cache requires native margin, logits or [No,Yes] probabilities")
    if not math.isfinite(value):
        raise ValueError("Finite native risk margin required")
    return value


def _best(hypothesis: Mapping[str, Any], cache: Mapping[str, Mapping]) -> tuple[float, str]:
    candidates = [(_margin(cache[e["key"]]), e["key"]) for e in hypothesis["evidence"]]
    if not candidates:
        raise ValueError("At least one evidence chunk required")
    return min(candidates)  # Best support; key gives deterministic exact-tie handling.


def _verdict(margin: float) -> str:
    return "unsupported" if margin > 0 else "supported" if margin < 0 else "abstain"


def project(layout: Mapping[str, Any], cache: Mapping[str, Mapping]) -> dict[str, dict]:
    """Return original/joint_window/delta rows; strict margin>0, exact zero abstains.

    Each hypothesis independently chooses minimum margin over identical evidence chunks.
    Delta = best joint margin minus best context-only margin. A singleton target uses
    its original margin, explicitly flagged. No averaging, sigmoid or threshold fitting
    is applied. Response score is max target margin; the legacy token_probabilities key
    contains binary target flags, not probabilities. Only target pieces are projected.
    """
    response = layout["response"]
    offsets = _offsets(layout["offsets"], len(response), empty=True)
    result = {
        arm: {
            **copy.deepcopy(dict(layout)), "arm": arm, "claims": [],
            "score_semantics": "raw risk margin; delta is a contextual margin difference",
            "decision_rule": "margin > 0; exact zero abstains and predicts no error",
            "token_value_semantics": "binary target error flags; not probabilities",
        }
        for arm in ARMS
    }
    for claim in layout["claims"]:
        pieces = _offsets(claim["pieces"], len(response), empty=False)
        if not pieces or _text(response, pieces) != claim["text"]:
            raise ValueError("Target provenance changed after preparation")
        values, winners = {}, {}
        for name in ("original", "joint_window", "context_only"):
            hypothesis = claim["hypotheses"][name]
            if hypothesis is not None:
                values[name], winners[name] = _best(hypothesis, cache)
        fallback = claim["hypotheses"]["context_only"] is None
        values["delta"] = (
            values["original"] if fallback else values["joint_window"] - values["context_only"]
        )
        for arm in ARMS:
            result[arm]["claims"].append({
                **copy.deepcopy(claim), "margin": values[arm], "margins": dict(values),
                "winning_evidence_keys": dict(winners), "no_context_fallback": fallback,
                "verdict": _verdict(values[arm]), "prediction": int(values[arm] > 0),
            })
    for row in result.values():
        row["score"] = max((c["margin"] for c in row["claims"]), default=0.0)
        row["prediction"] = int(row["score"] > 0)
        row["verdict"] = _verdict(row["score"])
        row["token_probabilities"] = [
            int(a < b and any(
                c["margin"] > 0 and any(a < y and b > x for x, y in c["pieces"])
                for c in row["claims"]
            ))
            for a, b in offsets
        ]
        row["no_context_fallback_count"] = sum(c["no_context_fallback"] for c in row["claims"])
    return result
