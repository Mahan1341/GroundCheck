"""R5 supervised views: conservative canonicalization with remapped span targets."""

from dataclasses import replace

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.numeric_local import normalize_numbers


def canonical_record(record):
    """Preserve annotation policy and map original intervals, never infer labels."""
    gold = gold_label(record)  # Also rejects sealed test records.
    answer = normalize_numbers(record.response)
    evidence = normalize_numbers(record.evidence)
    annotations, expansions = [], 0
    for span in record.annotations:
        if record.response[span.start : span.end] != span.text:
            raise ValueError("annotation text does not match original offsets")
        start, end = answer.normalized_range(span.start, span.end)
        back = answer.original_range(start, end)
        expansions += int(back != (span.start, span.end))
        annotations.append(replace(span, start=start, end=end, text=answer.text[start:end]))
    transformed = replace(
        record, response=answer.text, evidence=evidence.text, annotations=tuple(annotations)
    )
    assert gold_label(transformed) == gold
    return transformed, {
        "id": record.id,
        "answer_edits": list(answer.edits),
        "evidence_edits": list(evidence.edits),
        "expanded_annotation_intervals": expansions,
    }
