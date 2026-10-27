"""Pure CPU projection of frozen verifier sensitivity; no labels, models or file access.

Gradient magnitudes are score sensitivities, not calibrated error probabilities.
This module changes token readout only; original claim and response verdicts survive.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _interval(value: Sequence[int], limit: int, *, empty: bool = False) -> tuple[int, int]:
    if len(value) != 2 or any(type(x) is not int for x in value):
        raise ValueError("Offsets must be integer [start,end] pairs")
    a, b = value
    if not 0 <= a <= b <= limit or (a == b and not empty):
        raise ValueError("Offset outside original text or empty retained piece")
    return a, b


def stratified_permutation(
    values: Sequence[float], token_texts: Sequence[str], key: str, *, seed: int = 42
) -> tuple[list[float], dict[str, int]]:
    """Hash-order null preserving digit/boundary bins and each bin's value multiset.

    Boundary means the first or last original token intersecting the claim. Values are
    permuted AFTER tokenizer projection. Singleton/tied bins can make the null weak;
    the counts below expose this instead of rerolling until a desired change occurs.
    """
    if len(values) != len(token_texts) or not values:
        raise ValueError("Nonempty aligned saliencies/token texts required")
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Finite nonnegative saliencies required")
    bins: dict[tuple[bool, bool], list[int]] = defaultdict(list)
    for index, text in enumerate(token_texts):
        bins[(any(c.isdigit() for c in text), index in (0, len(values) - 1))].append(index)
    result = list(values)
    moved, eligible = 0, 0
    for category, indices in bins.items():
        eligible += len(indices) if len(indices) > 1 else 0
        shuffled = sorted(
            indices,
            key=lambda i: hashlib.sha256(
                f"claim-gradient-null:{seed}:{key}:{category}:{i}".encode()
            ).hexdigest(),
        )
        for destination, source in zip(indices, shuffled, strict=True):
            result[destination] = values[source]
            moved += int(destination != source)
    assert sorted(result) == sorted(values)
    return result, {
        "eligible_tokens": eligible,
        "permuted_positions": moved,
        "changed_values": sum(a != b for a, b in zip(values, result, strict=True)),
        "singleton_bins": sum(len(v) == 1 for v in bins.values()),
    }


def project(
    layoutrow: Mapping[str, Any],
    cache: Mapping[str, Mapping[str, Any]],
    winnerkeys: Sequence[str],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Project selected verifier gradients and a position-stratified null.

    Input row: response, original B7 offsets, claims with text/risk/pieces/evidence keys.
    Cache entries: claim_offsets relative to claim.text and aligned nonnegative L2values.
    For multi-piece claims, text must be the newline join of exact original substrings.
    The caller selects the already frozen best-support evidence key; no labels are read.

    For each B7 token, average magnitudes of intersecting verifier tokens; normalize by
    the claim maximum and multiply by unchanged claim risk. Overlapping claims take max.
    Missing nonspace tokenizer coverage is an error. A genuine <=1e-12 gradient maximum
    uses uniform claim risk and is counted. Whitespace-only unmapped tokens get zero.
    """
    response = layoutrow["response"]
    if not isinstance(response, str):
        raise ValueError("Original response text required")
    offsets = [_interval(pair, len(response), empty=True) for pair in layoutrow["offsets"]]
    claims = layoutrow["claims"]
    if len(winnerkeys) != len(claims):
        raise ValueError("Exactly one selected evidence key per claim required")
    native, null = [0.0] * len(offsets), [0.0] * len(offsets)
    stats: dict[str, Any] = {
        "claims": len(claims),
        "zero_denominator_fallbacks": 0,
        "unmapped_whitespace_tokens": 0,
        "tokens_crossing_removed_nonspace": 0,
        "null_eligible_tokens": 0,
        "null_permuted_positions": 0,
        "null_changed_values": 0,
        "claim_stats": [],
        "projection": "arithmetic mean of intersecting verifier-token L2 input gradients",
        "null": "post-projection permutation within digit/non-digit x boundary/interior bins",
        "seed": seed,
    }
    for claim, key in zip(claims, winnerkeys, strict=True):
        risk = float(claim["risk"])
        if not math.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Finite claim risk in [0,1] required")
        if key not in {e["key"] for e in claim["evidence"]}:
            raise ValueError("Selected key is not evidence for this claim")
        entry = cache[key]
        pieces = [_interval(pair, len(response)) for pair in claim["pieces"]]
        if not pieces or any(a[1] > b[0] for a, b in zip(pieces, pieces[1:], strict=False)):
            raise ValueError("Sorted nonoverlapping retained original pieces required")
        text = "\n".join(response[a:b] for a, b in pieces)
        if text != claim["text"]:
            raise ValueError("Claim text does not match exact piecewise original provenance")
        gradient_offsets = [_interval(p, len(text), empty=True) for p in entry["claim_offsets"]]
        magnitudes = list(entry["L2values"])
        if len(gradient_offsets) != len(magnitudes):
            raise ValueError("Gradient token offsets/magnitudes must align")
        if any(not math.isfinite(v) or v < 0 for v in magnitudes):
            raise ValueError("Finite nonnegative L2 values required")
        mapping, cursor = [], 0
        for a, b in pieces:
            mapping.append((a, b, cursor))
            cursor += b - a + 1
        indices, values, texts = [], [], []
        for index, (start, end) in enumerate(offsets):
            if start == end:
                continue
            intersections = [
                (max(start, a), min(end, b), a, view)
                for a, b, view in mapping
                if start < b and end > a
            ]
            if not intersections:
                continue
            selected = {
                j
                for left, right, original, view in intersections
                for j, (a, b) in enumerate(gradient_offsets)
                if a < b and a < view + right - original and b > view + left - original
            }
            retained_chars = {i for a, b, _, _ in intersections for i in range(a, b)}
            crossing = any(
                not response[i].isspace() and i not in retained_chars for i in range(start, end)
            )
            stats["tokens_crossing_removed_nonspace"] += int(crossing)
            if not selected:
                if any(not response[i].isspace() for i in retained_chars):
                    raise ValueError(
                        "Missing verifier tokenizer coverage of retained nonspace token"
                    )
                stats["unmapped_whitespace_tokens"] += 1
            value = math.fsum(magnitudes[j] for j in selected) / len(selected) if selected else 0.0
            indices.append(index)
            values.append(value)
            texts.append(response[start:end])
        if not indices:
            raise ValueError("Nonempty claim has no original token coverage")
        shuffled, null_stats = stratified_permutation(values, texts, key, seed=seed)
        denominator = max(values)
        fallback = denominator <= 1e-12
        stats["zero_denominator_fallbacks"] += int(fallback)
        for label in ("eligible_tokens", "permuted_positions", "changed_values"):
            stats[f"null_{label}"] += null_stats[label]
        for index, value, noise in zip(indices, values, shuffled, strict=True):
            native[index] = max(native[index], risk if fallback else risk * value / denominator)
            null[index] = max(null[index], risk if fallback else risk * noise / denominator)
        stats["claim_stats"].append(
            {
                "winner_key": key,
                "original_tokens": len(indices),
                "max_L2": denominator,
                "zero_denominator_fallback": fallback,
                "null": null_stats,
                "projected_L2_norm": math.sqrt(math.fsum(v * v for v in values)),
            }
        )
    native_row, null_row = copy.deepcopy(dict(layoutrow)), copy.deepcopy(dict(layoutrow))
    native_row["token_probabilities"] = native
    null_row["token_probabilities"] = null
    return {"native": native_row, "null": null_row, "stats": stats}
