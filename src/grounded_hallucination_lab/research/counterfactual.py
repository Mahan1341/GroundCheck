"""Manually audited inner-fit numerical counterfactuals, with balanced answer priors."""

import hashlib
import re
from collections import Counter
from dataclasses import replace

from grounded_hallucination_lab.baselines.lexical import PASSAGE, sentences
from grounded_hallucination_lab.data.ragtruth import Annotation, Record, gold_label


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_pairs(records, manifest):
    """Four combinations per parent; source grouping and answer offsets are retained."""
    if any(r.split == "test" or not r.labels_available for r in records):
        raise ValueError("test/withheld records cannot supply counterfactual parents")
    sources = {}
    for record in sorted(records, key=lambda r: r.id):
        sources.setdefault(record.source_id, record)
    output, audit = [], []
    accepted = sorted(manifest["accepted"], key=lambda x: x["text_sha256"])
    if len({(x["source_id"], x["text_sha256"]) for x in accepted}) != len(accepted):
        raise ValueError("duplicate accepted counterfactual parent")
    for parent_index, item in enumerate(accepted):
        if item["source_id"] not in sources:
            raise ValueError("counterfactual parent is outside supplied fit sources")
        parent = sources[item["source_id"]]
        candidates = [s.text.strip() for s in sentences(PASSAGE.sub(" ", parent.evidence))]
        matches = [s for s in candidates if text_hash(s) == item["text_sha256"]]
        if not matches:
            raise ValueError("accepted parent text hash not found")
        original = matches[0]
        numbers = list(re.finditer(r"\d+", original))
        if len(numbers) != 1:
            raise ValueError("counterfactual parent must have exactly one integer")
        number = numbers[0]
        if (number.start(), number.end(), number.group()) != (
            item["start"],
            item["end"],
            item["number"],
        ):
            raise ValueError("numeric offset/target mismatch")
        original_start = parent.evidence.find(original)
        if original_start < 0:
            raise ValueError("parent statement must be an exact source substring")
        values = [number.group(), str(2 * int(number.group()))]
        if values[0] == values[1]:
            raise ValueError("counterfactual must change the integer")
        statements = [original[: number.start()] + v + original[number.end() :] for v in values]
        evidence = []
        for index, statement in enumerate(statements):
            identifier = f"Record identifier: {values[1 - index]}."
            evidence.append(
                f"{identifier} {statement}" if parent_index % 2 else f"{statement} {identifier}"
            )
        if Counter(re.findall(r"\w+", evidence[0])) != Counter(re.findall(r"\w+", evidence[1])):
            raise ValueError("paired evidence word multiset changed")
        key = f"cf:{parent.source_id}:{item['text_sha256'][:12]}"
        for answer_index, answer in enumerate(statements):
            for evidence_index, premise in enumerate(evidence):
                positive = answer_index != evidence_index
                start, end = number.start(), number.start() + len(values[answer_index])
                annotations = (
                    (
                        Annotation(
                            start, end, answer[start:end], "Evident Conflict", False, False, {}
                        ),
                    )
                    if positive
                    else ()
                )
                output.append(
                    replace(
                        parent,
                        id=f"{key}:a{answer_index}:e{evidence_index}",
                        response=answer,
                        evidence=premise,
                        annotations=annotations,
                        quality="good",
                        model="train_counterfactual",
                    )
                )
        audit.append(
            {
                "source_id": parent.source_id,
                "parent_response_id": parent.id,
                "text_sha256": item["text_sha256"],
                "source_start": original_start,
                "source_end": original_start + len(original),
                "values": values,
                "identifier_first": bool(parent_index % 2),
            }
        )
    return output, audit


def training_records(fit, manifest, mode):
    if mode not in ("C0", "C1"):
        raise ValueError(mode)
    fit = sorted(fit, key=lambda r: r.id)
    synthetic, audit = make_pairs(fit, manifest)
    size = 4 * (len(fit) // 12)
    if not synthetic or len(synthetic) % 4 or not size:
        raise ValueError("need complete balanced counterfactual blocks")
    if mode == "C1":
        extra = [
            replace(synthetic[i % len(synthetic)], id=f"{synthetic[i % len(synthetic)].id}:r{i}")
            for i in range(size)
        ]
    else:
        classes = {
            label: sorted(
                (r for r in fit if gold_label(r) == label),
                key=lambda r: text_hash(f"replay:42:{r.id}"),
            )
            for label in (0, 1)
        }
        if not all(classes.values()):
            raise ValueError("natural replay needs both response classes")
        extra = []
        for i in range(size):
            group = classes[i % 2]
            parent = group[(i // 2) % len(group)]
            extra.append(replace(parent, id=f"replay:{parent.id}:r{i}"))
    assert sum(gold_label(r) for r in extra) == size // 2
    assert {r.source_id for r in extra} <= {r.source_id for r in fit}
    return (
        fit + extra,
        synthetic,
        {
            "mode": mode,
            "natural_rows": len(fit),
            "extra_rows": size,
            "extra_positive": size // 2,
            "unique_synthetic_rows": len(synthetic),
            "synthetic_sources": len({r.source_id for r in synthetic}),
            "parents": audit,
        },
    )


def require_unseen_sources(records: list[Record], train_sources: set[str]) -> None:
    """Reject weight-fit parent overlap; this does not imply untouched model selection."""
    if any(r.split == "test" or not r.labels_available for r in records):
        raise ValueError("official test/withheld sources are not authorized for this probe")
    if {r.source_id for r in records} & train_sources:
        raise ValueError("probe source was used to fit supervised weights")
