"""Frozen sentence-to-context TF-IDF baseline with original-text provenance."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from grounded_hallucination_lab.data.ragtruth import Record
from grounded_hallucination_lab.data.splits import Partitions

TOKEN = re.compile(r"\d+(?:[.,]\d+)*%?|[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
PASSAGE = re.compile(r"(?i)(?<!\S)passage\s+(\d+)\s*:")
BOUNDARY = re.compile(r"""[.!?]+["”’')\]]*(?=\s|$)|\n+""")
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "e.g.", "i.e.", "u.s."}
NEGATIONS = {"no", "not", "never", "neither", "nor", "without"}


def tokenize(text: str) -> list[str]:
    """Retain numbers/negations; normalize English contractions and digit grouping."""
    text = text.casefold().replace("’", "'")
    for old, new in (("cannot", "can not"), ("can't", "can not"), ("won't", "will not")):
        text = re.sub(rf"\b{old}\b", new, text)
    text = text.replace("n't", " not")
    result = []
    for match in TOKEN.finditer(text):
        token = match.group()
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?", token):
            token = token.replace(",", "")
        result.append(token)
    return result


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str


def _span(text: str, start: int, end: int) -> TextSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return TextSpan(start, end, text[start:end]) if end > start else None


def sentences(text: str) -> list[TextSpan]:
    result = []
    start = 0
    for boundary in BOUNDARY.finditer(text):
        prefix = text[start : boundary.end()].strip().casefold()
        last = prefix.split()[-1] if prefix else ""
        if not boundary.group().startswith("\n") and (
            last in ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last)
        ):
            continue
        piece = _span(text, start, boundary.end())
        if piece:
            result.append(piece)
        start = boundary.end()
    tail = _span(text, start, len(text))
    if tail:
        result.append(tail)
    return result


@dataclass(frozen=True)
class Chunk:
    source_id: str
    chunk_id: str
    passage_ordinal: int
    passage_label: str | None
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class LexicalConfig:
    max_chunk_tokens: int = 100
    chunk_overlap_tokens: int = 20
    top_k: int = 3
    ngram_max: int = 2
    sublinear_tf: bool = True

    def __post_init__(self) -> None:
        for key in ("max_chunk_tokens", "chunk_overlap_tokens", "top_k", "ngram_max"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if not 0 <= self.chunk_overlap_tokens < self.max_chunk_tokens:
            raise ValueError("overlap must be nonnegative and smaller than chunk size")
        if self.top_k < 1 or self.ngram_max not in (1, 2) or type(self.sublinear_tf) is not bool:
            raise ValueError("invalid TF-IDF configuration")


def chunks(evidence: str, source_id: str, config: LexicalConfig) -> list[Chunk]:
    markers = list(PASSAGE.finditer(evidence))
    passages = []
    if markers:
        if evidence[: markers[0].start()].strip():
            passages.append((0, markers[0].start(), None))
        for i, marker in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(evidence)
            passages.append((marker.end(), end, marker.group(1)))
    else:
        passages.append((0, len(evidence), None))
    result = []
    for ordinal, (start, end, label) in enumerate(passages):
        words = list(re.finditer(r"\S+", evidence[start:end]))
        step = config.max_chunk_tokens - config.chunk_overlap_tokens
        for offset in range(0, len(words), step):
            stop = min(offset + config.max_chunk_tokens, len(words))
            left, right = start + words[offset].start(), start + words[stop - 1].end()
            result.append(
                Chunk(
                    source_id,
                    f"p{ordinal}:t{offset}",
                    ordinal,
                    label,
                    left,
                    right,
                    evidence[left:right],
                )
            )
            if stop == len(words):
                break
    return result


class LexicalDetector:
    def __init__(self, config: LexicalConfig | None = None) -> None:
        self.config = config or LexicalConfig()
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, self.config.ngram_max),
            sublinear_tf=self.config.sublinear_tf,
            smooth_idf=True,
            norm="l2",
            dtype=np.float64,
        )
        self.fit_source_ids: list[str] = []
        self.fit_chunk_count = 0
        self._cache: dict = {}

    def fit_train(self, parts: Partitions) -> "LexicalDetector":
        allowed = {r.source_id for r in parts.train}
        forbidden = {r.source_id for name, rows in parts.items() if name != "train" for r in rows}
        if not allowed or allowed & forbidden or any(r.split != "train" for r in parts.train):
            raise ValueError(
                "fit requires disjoint nonempty train; no calibration/dev/test sources"
            )
        evidence = {}
        for record in parts.train:
            if record.source_id in evidence and evidence[record.source_id] != record.evidence:
                raise ValueError("inconsistent train evidence")
            evidence[record.source_id] = record.evidence
        documents = [
            c.text for sid in sorted(evidence) for c in chunks(evidence[sid], sid, self.config)
        ]
        if not documents or not any(tokenize(d) for d in documents):
            raise ValueError("train evidence has no lexical tokens")
        self.vectorizer.fit(documents)
        self.fit_source_ids = sorted(allowed)
        self.fit_chunk_count = len(documents)
        self._cache.clear()
        return self

    def fingerprint(self) -> str:
        payload = {
            "config": asdict(self.config),
            "sources": self.fit_source_ids,
            "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
            "idf": self.vectorizer.idf_.tolist(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def clear_cache(self) -> None:
        self._cache.clear()

    def score(self, record: Record) -> dict:
        if record.split == "test":
            raise ValueError(
                "official test scoring requires separate final-evaluation authorization"
            )
        if not self.fit_source_ids:
            raise ValueError("detector is not fitted")
        key = (record.source_id, hashlib.sha256(record.evidence.encode()).hexdigest())
        if key not in self._cache:
            passages = chunks(record.evidence, record.source_id, self.config)
            matrix = self.vectorizer.transform([p.text for p in passages]) if passages else None
            self._cache[key] = (passages, matrix)
        passages, matrix = self._cache[key]
        spans = sentences(record.response)
        diagnostics = []
        vectors = self.vectorizer.transform([s.text for s in spans]) if spans else None
        context_numbers = {t for p in passages for t in tokenize(p.text) if t[0].isdigit()}
        for index, span in enumerate(spans):
            vector = vectors[index]
            similarities = (matrix @ vector.T).toarray().ravel() if passages else np.array([])
            ranked = sorted(range(len(passages)), key=lambda k: (-similarities[k], k))
            retrieved = [
                {**asdict(passages[k]), "similarity": float(np.clip(similarities[k], 0, 1))}
                for k in ranked[: self.config.top_k]
            ]
            support = retrieved[0]["similarity"] if retrieved else 0.0
            tokens = tokenize(span.text)
            numbers = {t for t in tokens if t[0].isdigit()}
            retrieved_tokens = {t for p in retrieved for t in tokenize(p["text"])}
            sentence_negation = bool(set(tokens) & NEGATIONS)
            best_negation = (
                bool(set(tokenize(retrieved[0]["text"])) & NEGATIONS) if retrieved else False
            )
            diagnostics.append(
                {
                    **asdict(span),
                    "support": support,
                    "risk": 1 - support,
                    "oov_only": bool(tokens) and vector.nnz == 0,
                    "tokenless": not tokens,
                    "numbers": sorted(numbers),
                    "numbers_absent_from_retrieved": sorted(numbers - retrieved_tokens),
                    "numbers_absent_from_context": sorted(numbers - context_numbers),
                    "negation_present": sentence_negation,
                    "negation_cue_mismatch_top1": sentence_negation != best_negation,
                    "retrieved": retrieved,
                }
            )
        return {
            "id": record.id,
            "source_id": record.source_id,
            "score": max((s["risk"] for s in diagnostics), default=0.0),
            "empty_response": not record.response.strip(),
            "empty_evidence": not passages,
            "context_chunk_count": len(passages),
            "sentences": diagnostics,
        }
