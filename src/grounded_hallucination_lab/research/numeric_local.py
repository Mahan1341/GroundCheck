"""Inference-only R4 numeric views and extractive local claims, with original offsets."""

import re
from dataclasses import dataclass

from grounded_hallucination_lab.baselines.lexical import sentences
from grounded_hallucination_lab.research.invariance import SMALL, TENS, parse_words

WORDS = "|".join(SMALL + TENS[2:] + ["hundred", "thousand"])
MEASURE = re.compile(
    rf"\b(?P<number>\d+|(?:{WORDS})(?:[ -](?:{WORDS}))*)"
    r"(?P<space>\s+)(?P<unit>bolts?|minutes?|seconds?|meters?|kilometers?)\b",
    re.I,
)
UNCERTAIN = re.compile(
    r"\b(about|approximately|around|between|roughly|nearly|over|under|least|most"
    r"|up to|more than|less than|or|and|almost|circa|estimated|estimate"
    r"|million|billion|trillion|half|quarter|dozen|couple|minus|negative)\b"
    r"|\d\s*[-–—/±+]\s*\d|\d[.,]\d|[<>~≈]|(?<!\w)-\d",
    re.I,
)
ANAPHORA = re.compile(
    r"\b(it|its|they|them|their|theirs|he|she|him|her|his|hers|this|that|these|those"
    r"|former|latter|such|same|above|below|previous|respectively)\b",
    re.I,
)


@dataclass(frozen=True)
class MappedText:
    """Each output character maps to an original half-open character interval.

    Rewritten numeral/unit pieces map to the entire original numeral/unit. This
    set-valued alignment is exact provenance, not invented one-to-one alignment.
    """

    original: str
    text: str
    origins: tuple[tuple[int, int], ...]
    edits: tuple[dict, ...]

    def original_range(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start < end <= len(self.text):
            raise ValueError("invalid normalized interval")
        return min(a for a, _ in self.origins[start:end]), max(
            b for _, b in self.origins[start:end]
        )

    def normalized_range(self, start: int, end: int) -> tuple[int, int]:
        """Smallest normalized interval overlapping an original character span."""
        if not 0 <= start < end <= len(self.original):
            raise ValueError("invalid original interval")
        indices = [i for i, (a, b) in enumerate(self.origins) if a < end and b > start]
        if not indices:
            raise ValueError("original interval was not represented")
        return min(indices), max(indices) + 1


def ambiguous_numeric_range(text: str) -> bool:
    """Reject numeric range/fraction separators, allowing canonical twenty-four."""
    pattern = rf"(?=(\b(?:\d+|{WORDS})\s*(?:to\b|through\b|[-/–—±+]|\.{{2,}})\s*(?:\d+|{WORDS})\b))"
    return any(parse_words(m[1]) is None for m in re.finditer(pattern, text, re.I))


def normalize_numbers(text: str) -> MappedText:
    """Typed exact English integers only; canonical digits/minutes/meters.

    Preserve every untouched byte of text (including whitespace). Refuse an
    entire sentence containing uncertain/range/decimal/coordinated syntax.
    Seconds convert only when divisible by 60; no rounding or temperature math.
    """
    replacements = []
    for sentence in sentences(text):
        if UNCERTAIN.search(sentence.text) or ambiguous_numeric_range(sentence.text):
            continue
        for match in MEASURE.finditer(sentence.text):
            raw = match["number"]
            # Refuse suffix matches inside unsupported number expressions.
            before = sentence.text[: match.start()]
            if re.search(r"(?:\d[.,/]\s*|\w-)$", before):
                continue
            value = int(raw) if raw.isdigit() else parse_words(raw)
            if value is None or (raw.isdigit() and len(raw) > 1 and raw.startswith("0")):
                continue
            unit = match["unit"].lower().removesuffix("s")
            if unit == "second" and value % 60 == 0:
                value, unit = value // 60, "minute"
            elif unit == "kilometer":
                value, unit = value * 1000, "meter"
            canonical = {"number": str(value), "unit": unit + ("s" if value != 1 else "")}
            for part, replacement in canonical.items():
                start, end = match.span(part)
                start, end = start + sentence.start, end + sentence.start
                if text[start:end] != replacement:
                    replacements.append((start, end, replacement))
    output, mapping, edits = [], [], []
    cursor = 0
    for start, end, replacement in sorted(replacements):
        output.append(text[cursor:start])
        mapping.extend((i, i + 1) for i in range(cursor, start))
        normalized_start = len(mapping)
        output.append(replacement)
        mapping.extend([(start, end)] * len(replacement))
        edits.append(
            {
                "original": [start, end],
                "normalized": [normalized_start, len(mapping)],
                "before": text[start:end],
                "after": replacement,
            }
        )
        cursor = end
    output.append(text[cursor:])
    mapping.extend((i, i + 1) for i in range(cursor, len(text)))
    return MappedText(text, "".join(output), tuple(mapping), tuple(edits))


def project_probabilities(original_offsets, mapped: MappedText, offsets, probabilities):
    """Max-overlap projection onto the SAME original-token grid for all views.

    Expansion of a rewritten numeral may enlarge predicted spans; this is a
    conservative projection, explicitly measured separately from native scores.
    """
    spans = [
        (mapped.original_range(a, b), p)
        for (a, b), p in zip(offsets, probabilities, strict=True)
        if b > a
    ]
    return [
        max((p for (a, b), p in spans if a < end and b > start), default=0.0)
        for start, end in original_offsets
    ]


def local_claims(text: str) -> tuple[list[dict], str | None]:
    """Every sentence is central once, with one neighbor on each side.

    Explicit anaphora anywhere in the answer triggers a full-answer fallback.
    This conservative heuristic does not claim general coreference resolution.
    No labels, error spans or question/evidence features choose the windows.
    """
    pieces = sentences(text)
    if len(pieces) <= 1 or ANAPHORA.search(text):
        return [
            {"start": 0, "end": len(text), "central_start": 0, "central_end": len(text)}
        ], "anaphora" if ANAPHORA.search(text) else "single"
    result = []
    for i, piece in enumerate(pieces):
        result.append(
            {
                "start": pieces[max(0, i - 1)].start,
                "end": pieces[min(len(pieces) - 1, i + 1)].end,
                "central_start": piece.start,
                "central_end": piece.end,
            }
        )
    return result, None
