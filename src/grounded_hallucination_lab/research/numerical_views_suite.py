"""R5 held-out fictional parents; exact quantities, crossed entity assignments."""

from grounded_hallucination_lab.data.ragtruth import Annotation, Record
from grounded_hallucination_lab.research.invariance import number_words

NAMES = (
    "Mossbrook Oakmere Pinehaven Quillford Reedwick Stonevale "
    "Thornbay Umberley Valeport Willowmarsh Yarrowgate Zephyrhill"
).split()
VARIANTS = (
    "digits",
    "spaced_words",
    "decimal_units",
    "exact_units",
    "paraphrase",
    "property",
    "long_short",
)
FILL = (
    "The depot has a copper roof.",
    "The courtyard contains a wooden bench.",
    "The notice is printed on white paper.",
)


def measure(kind, value, variant):
    unit = {"quantity": "bolts", "duration": "minutes", "distance": "meters"}[kind]
    if variant == "spaced_words":
        return f"{number_words(value).replace('-', ' ')} {unit}"
    if variant in ("exact_units", "decimal_units"):
        if kind == "duration":
            value, unit = value * 60, "seconds"
        elif kind == "distance":
            value, unit = value / 1000, "kilometers"
        if variant == "decimal_units":
            return f"{value:.2f} {unit}"
        return f"{value:g} {unit}"
    return f"{value} {unit}"


def suite():
    records, design = [], []
    for i, name in enumerate(NAMES):
        kind = ("quantity", "duration", "distance")[i // 4]
        values = (
            (213 + 16 * i, 227 + 16 * i) if kind != "distance" else (1000 * (i - 7), 1000 * (i - 6))
        )
        other = name + " Annex"
        source = f"r5:{name}"
        noun = {"quantity": "shipment", "duration": "inspection", "distance": "course"}[kind]
        for variant in VARIANTS:
            for order in (0, 1):
                for ai, value in enumerate(values):
                    av = variant if variant in ("spaced_words", "decimal_units") else "digits"
                    phrase = measure(kind, value, av)
                    claim = f"For {name}, the {noun} was recorded as {phrase}."
                    if variant == "paraphrase":
                        claim = f"The recorded {kind} for {name}'s {noun} comes to {phrase}."
                    if variant == "property":
                        claim = f"The first {noun} at {name} was recorded as {phrase}."
                    extras = list(FILL) * 12 if variant == "long_short" else []
                    answer = " ".join(extras[:18] + [claim] + extras[18:])
                    start = answer.index(phrase)
                    for ei, actual in enumerate(values):
                        ev = variant if variant in ("exact_units", "spaced_words") else "digits"
                        facts = []
                        for entity, number in ((name, actual), (other, values[1 - ei])):
                            p = measure(kind, number, ev)
                            first = "first " if variant == "property" else ""
                            if i % 2:
                                fact = (
                                    f"When asked about {entity}'s {first}{noun}, "
                                    f"the clerk stated: {p}."
                                )
                            else:
                                fact = f"Receipt for {entity}: {first}{noun}; recorded {kind}: {p}."
                            facts.append(fact)
                        if variant == "property":
                            facts.append(
                                f"The second {noun} at {name} was recorded as "
                                f"{measure(kind, values[1 - ei], 'digits')}."
                            )
                        if order:
                            facts.reverse()
                        evidence = " ".join(facts + list(FILL))
                        gold = int(ai != ei)
                        pair = f"{source}:{variant}:o{order}:a{ai}"
                        identifier = f"{pair}:e{ei}"
                        annotations = (
                            (
                                Annotation(
                                    start,
                                    start + len(phrase),
                                    phrase,
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
                                "authored_R5",
                                "train",
                                "good",
                                answer,
                                f"What was recorded for {name}?",
                                evidence,
                                annotations,
                                {},
                                {},
                            )
                        )
                        design.append(
                            {
                                "id": identifier,
                                "source_id": source,
                                "pair_id": pair,
                                "variant": variant,
                                "kind": kind,
                                "family": "interview" if i % 2 else "receipt",
                                "gold": gold,
                                "semantic_status": "contradicted" if gold else "supported",
                                "values": list(values),
                                "answer_index": ai,
                                "evidence_index": ei,
                                "slot_start": start,
                                "slot_end": start + len(phrase),
                                "padding": len(extras),
                            }
                        )
    return records, design
