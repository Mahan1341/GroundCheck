"""R9 inference-only passage controls; no labels or learned retrieval."""

import re

import numpy as np


def passage_ranges(text: str) -> list[tuple[int, int]]:
    """Partition at original RAGTruth headers without discarding any characters."""
    starts = [m.start() for m in re.finditer(r"(?im)^passage\s+\d+:", text)]
    if len(starts) != 3 or starts[0] != 0:
        raise ValueError("R9 requires exactly three complete header-delimited passages")
    return list(zip(starts, [*starts[1:], len(text)], strict=True))


def evidence_view(text: str, condition: str) -> str:
    parts = [text[a:b] for a, b in passage_ranges(text)]
    if condition == "full":
        return text
    if condition in ("rotate1", "rotate2"):
        k = int(condition[-1])
        return "".join(parts[k:] + parts[:k])
    if condition in ("single0", "single1", "single2"):
        return parts[int(condition[-1])]
    raise ValueError(condition)


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Keep complete original answer; sentence ranges also own intervening spaces."""
    from grounded_hallucination_lab.research.verifiers import _punkt_tokenizer

    spans = list(_punkt_tokenizer().span_tokenize(text))
    starts = [0, *[a for a, _ in spans[1:]]]
    return list(zip(starts, [*starts[1:], len(text)], strict=True))


def pool_passages(
    rows: list[dict], sentences: list[tuple[int, int]], *, text: str
) -> tuple[dict, dict, list]:
    """Return coherent sentence pooling, token-min control and selected provenance.

    Every row must score the same complete answer in original token coordinates.
    Coherent selection keeps each sentence's scores from exactly one passage.
    """
    if len(rows) != 3:
        raise ValueError("three passage predictions required")
    offsets = rows[0]["offsets"]
    if any(r["id"] != rows[0]["id"] or r["offsets"] != offsets for r in rows):
        raise ValueError("answer IDs and token offsets must match")
    values = np.asarray([r["token_probabilities"] for r in rows], dtype=float)
    if values.shape != (3, len(offsets)) or not np.all(np.isfinite(values)):
        raise ValueError("invalid token probabilities")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("probabilities outside [0,1]")
    coherent = np.zeros(len(offsets))
    covered = np.zeros(len(offsets), dtype=int)
    choices = []
    # BPE offsets may start in the preceding sentence's trailing whitespace.
    contents = [[j for j in range(a, b) if not text[j].isspace()] for a, b in offsets]
    lexical = [[j for j in chars if text[j].isalnum()] for chars in contents]
    anchors = [words or chars for words, chars in zip(lexical, contents, strict=True)]
    for a, b in sentences:
        indexes = [
            i
            for i, (start, _) in enumerate(offsets)
            if a <= (anchors[i][0] if anchors[i] else start) < b
        ]
        if not indexes:
            continue
        if any(any(j < a or j >= b for j in lexical[i]) for i in indexes):
            raise ValueError("token content crosses sentence boundary")
        passage = int(np.argmin(values[:, indexes].max(axis=1)))
        coherent[indexes] = values[passage, indexes]
        covered[indexes] += 1
        choices.append({"start": a, "end": b, "passage_index": passage})
    if not np.all(covered == 1):
        raise ValueError("sentence partition must assign every answer token exactly once")
    token_min = values.min(axis=0)

    def row(probabilities):
        return {
            **rows[0],
            "token_probabilities": probabilities.tolist(),
            "score": float(max(probabilities, default=0)),
            "evidence_coverage": 1.0,
            "coverage_definition": "union of all three complete passage forwards",
        }

    return row(coherent), row(token_min), choices
