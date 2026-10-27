"""R2 authored exact numeric worlds; parent and rendering-family isolation."""

import hashlib
import json

from grounded_hallucination_lab.data.ragtruth import Annotation, Record
from grounded_hallucination_lab.research.controls import FILLER

VALUES = {
    "fit": [(11, 17), (23, 29), (31, 37), (41, 47), (53, 59), (61, 67), (71, 79), (83, 89)],
    "selection": [(13, 19), (27, 33), (43, 51)],
    "diagnostic": [(21, 26), (34, 39), (46, 57), (63, 78)],
}
NAMES = [
    "Aster",
    "Birch",
    "Cedar",
    "Dahlia",
    "Elm",
    "Fir",
    "Grove",
    "Hazel",
    "Iris",
    "Juniper",
    "Kestrel",
    "Larch",
    "Maple",
    "Nettle",
    "Oak",
    "Pine",
    "Quartz",
    "Reed",
    "Spruce",
    "Thistle",
    "Umber",
    "Violet",
    "Willow",
    "Yarrow",
    "Acorn",
    "Bramble",
    "Coral",
    "Drift",
    "Ember",
    "Flint",
    "Garnet",
    "Heather",
    "Indigo",
    "Jasper",
    "Kelvin",
    "Linden",
    "Marble",
    "Nimbus",
    "Opal",
    "Pearl",
    "Robin",
    "Sage",
    "Tansy",
    "Vale",
    "Wren",
]


def words(n):
    small = (
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen"
    ).split()
    tens = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
    if n < 20:
        return small[n]
    return tens[n // 10] + ("-" + small[n % 10] if n % 10 else "")


def statement(kind, name, value, style="plain", expression="digits"):
    unit = "seconds" if expression == "seconds" else "minutes"
    value = value * 60 if expression == "seconds" else value
    v = words(value) if expression == "words" else str(value)
    if kind == "quantity":
        text = (
            f"The {name} batch contained exactly {v} bolts."
            if style == "plain"
            else f"Exactly {v} bolts made up the entire {name} batch."
        )
    elif kind == "duration":
        text = (
            f"The {name} trial lasted exactly {v} {unit}."
            if style == "plain"
            else f"The total elapsed time of the {name} trial was {v} {unit}."
        )
    else:
        text = (
            f"The {name} museum first opened in {v}."
            if style == "plain"
            else f"The first opening of the {name} museum took place in {v}."
        )
    return text, v


def premise(kind, name, other, a, b, family, expression="digits"):
    first = statement(kind, name, a, expression=expression)[0]
    if family == "record":
        return [first, f"Record identifier: {b}."]
    if family == "parallel":
        return [first, statement(kind, other, b, expression=expression)[0]]
    if family == "reported":
        return [
            statement(kind, name, a, "paraphrase", expression)[0],
            statement(kind, other, b, "paraphrase", expression)[0],
        ]
    field = {
        "quantity": "Exact bolt count",
        "duration": "Exact elapsed minutes",
        "date": "Year of first opening",
    }[kind]
    noun = {"quantity": "batch", "duration": "trial", "date": "museum"}[kind]
    return [
        f"{name} {noun} catalogue entry. {field}: {a}.",
        f"{other} {noun} catalogue entry. {field}: {b}.",
    ]


def relation_suite(split):
    """Return ordinary records plus separate privileged semantic/offset diagnostics."""
    if split not in VALUES:
        raise ValueError("only synthetic fit/selection/diagnostic; no outer partitions")
    records, design = [], []
    start_name = {"fit": 0, "selection": 24, "diagnostic": 33}[split]
    novel = "catalogue" if split == "selection" else "reported"
    for kind_index, kind in enumerate(("quantity", "duration", "date")):
        for j, pair in enumerate(VALUES[split]):
            idx = kind_index * len(VALUES[split]) + j
            name, other = NAMES[start_name + idx], NAMES[start_name + idx] + " Annex"
            source = f"r2:{split}:{kind}:{name}"
            variants = (
                [
                    ("record", "record", 0, "first"),
                    ("parallel", "parallel", 0, "first"),
                    ("record_pad8", "record", 8, "last"),
                    ("parallel_pad8", "parallel", 8, "last"),
                ]
                if split == "fit"
                else [
                    ("record", "record", 0, "first"),
                    ("parallel", "parallel", 0, "first"),
                    ("novel", novel, 0, "first"),
                    ("known_values_novel", novel, 0, "first"),
                    ("answer_paraphrase", "parallel", 0, "first"),
                    ("target_middle", "parallel", 0, "first"),
                    ("pad8_first", novel, 8, "first"),
                    ("pad8_last", novel, 8, "last"),
                    ("pad32_first", novel, 32, "first"),
                    ("pad32_middle", novel, 32, "middle"),
                    ("pad32_last", novel, 32, "last"),
                    ("multi_fact_sentence", novel, 8, "middle"),
                ]
                + ([("number_words", "parallel", 0, "first")] if kind != "date" else [])
                + ([("unit_conversion", "parallel", 0, "first")] if kind == "duration" else [])
            )
            for variant, family, padding, position in variants:
                values = VALUES["fit"][j] if variant == "known_values_novel" else pair
                values = [1900 + v if kind == "date" else v for v in values]
                for order in (0, 1):
                    for ai, value in enumerate(values):
                        claim, slot = statement(
                            kind,
                            name,
                            value,
                            "paraphrase" if variant == "answer_paraphrase" else "plain",
                        )
                        extras = [FILLER[k % len(FILLER)] for k in range(padding)]
                        at = {"first": 0, "middle": padding // 2, "last": padding}[position]
                        if variant == "multi_fact_sentence":
                            claim = (
                                claim[:-1]
                                + "; the archive contains maps and the station has a clock."
                            )
                        pieces = extras[:at] + [claim] + extras[at:]
                        answer = " ".join(pieces)
                        cs = sum(len(t) + 1 for t in extras[:at])
                        slot_start = cs + claim.index(slot)
                        pair_id = f"{source}:{variant}:o{order}:a{ai}"
                        for ei, evidence_value in enumerate(values):
                            expr = {"number_words": "words", "unit_conversion": "seconds"}.get(
                                variant, "digits"
                            )
                            facts = premise(
                                kind, name, other, evidence_value, values[1 - ei], family, expr
                            )
                            if order:
                                facts.reverse()
                            if variant == "target_middle":
                                evidence = " ".join([*FILLER[:6], *facts, *FILLER[6:]])
                            else:
                                evidence = " ".join([*facts, *FILLER])
                            gold = int(ai != ei)
                            identifier = f"{pair_id}:e{ei}"
                            ann = (
                                (
                                    Annotation(
                                        slot_start,
                                        slot_start + len(slot),
                                        slot,
                                        "Evident Conflict",
                                        False,
                                        False,
                                        {},
                                    ),
                                )
                                if gold
                                else ()
                            )
                            records.append(
                                Record(
                                    identifier,
                                    source,
                                    "QA",
                                    "authored_R2",
                                    "train",
                                    "good",
                                    answer,
                                    "What does the evidence state?",
                                    evidence,
                                    ann,
                                    {},
                                    {},
                                )
                            )
                            design.append(
                                {
                                    "id": identifier,
                                    "source_id": source,
                                    "split": split,
                                    "pair_id": pair_id,
                                    "kind": kind,
                                    "variant": variant,
                                    "family": family,
                                    "order": order,
                                    "answer_index": ai,
                                    "evidence_index": ei,
                                    "values": values,
                                    "gold": gold,
                                    "claim_start": cs,
                                    "claim_end": cs + len(claim),
                                    "slot_start": slot_start,
                                    "slot_end": slot_start + len(slot),
                                    "padding": padding,
                                    "claim_position": position,
                                    "provenance": "authored exact closed world; not RAGTruth",
                                }
                            )
    return records, design


def content_hash(records, design):
    from dataclasses import asdict

    return hashlib.sha256(
        json.dumps(
            {"records": [asdict(r) for r in records], "design": design}, sort_keys=True
        ).encode()
    ).hexdigest()
