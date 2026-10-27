"""Exact authored R3 worlds and conservative evidence-only numeric normalization."""

import re

from grounded_hallucination_lab.data.ragtruth import Annotation, Record

VALUES = {
    "fit": [(12, 18), (24, 32), (36, 42), (48, 54), (62, 68), (72, 84)],
    "selection": [(14, 22), (38, 52), (64, 76)],
    "diagnostic": [(16, 28), (44, 56), (66, 82), (86, 94)],
}
SMALL = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
FILL = {
    "fit": [
        "The laboratory has a blue door.",
        "The office contains a desk.",
        "The garden has a gate.",
        "The lobby contains a bench.",
    ],
    "selection": [
        "The depot has a green roof.",
        "The workshop contains a cabinet.",
        "The courtyard has a fence.",
    ],
    "diagnostic": [
        "The observatory has a white dome.",
        "The corridor contains a noticeboard.",
        "The storeroom has a window.",
    ],
}


def number_words(n: int) -> str:
    if not 0 <= n < 10000:
        raise ValueError("only exact nonnegative integers below 10000")
    if n < 20:
        return SMALL[n]
    if n < 100:
        return TENS[n // 10] + ("-" + SMALL[n % 10] if n % 10 else "")
    scale = 1000 if n >= 1000 else 100
    return (
        number_words(n // scale)
        + (" thousand" if scale == 1000 else " hundred")
        + (" " + number_words(n % scale) if n % scale else "")
    )


def parse_words(text: str) -> int | None:
    """Accept only our unambiguous canonical cardinal grammar, not free-form prose."""
    normalized = text.lower().replace("-", " ").strip()
    total, current = 0, 0
    for token in normalized.split():
        if token in SMALL:
            current += SMALL.index(token)
        elif token in TENS[2:]:
            current += TENS.index(token) * 10
        elif token == "hundred":
            current *= 100
        elif token == "thousand":
            total += current * 1000
            current = 0
        else:
            return None
    value = total + current
    if value >= 10000 or number_words(value).replace("-", " ") != normalized:
        return None
    return value


def normalize_evidence(text: str) -> str:
    """No relation extraction. Only explicit integer quantities; leave uncertain clauses alone."""
    vocab = "|".join(SMALL + TENS[2:] + ["hundred", "thousand"])
    pattern = (
        rf"\b((?:{vocab})(?:[ -](?:{vocab}))*) "
        r"(?=bolts\b|minutes\b|seconds\b|meters\b|kilometers\b)"
    )
    chunks = re.split(r"(?<=[.!?])\s+", text)
    output = []
    for chunk in chunks:
        if re.search(
            r"\b(about|approximately|around|between|roughly|nearly|over|under|at least|at most"
            r"|up to|more than|less than|to|or|and)\b|\d\s*(?:[-–/]|to)\s*\d|\d[.,]\d",
            chunk,
            re.I,
        ):
            output.append(chunk)
            continue

        def words_match(m):
            value = parse_words(m[1])
            return m[0] if value is None else f"{value} "

        chunk = re.sub(pattern, words_match, chunk, flags=re.I)

        def unit_match(m):
            value, unit = int(m[1]), m[2]
            if unit == "seconds" and value % 60 == 0:
                return f"{value // 60} minutes"
            if unit == "kilometers":
                return f"{value * 1000} meters"
            return m[0]

        chunk = re.sub(r"\b(\d+) (seconds|kilometers)\b", unit_match, chunk)
        output.append(chunk)
    return " ".join(output)


def measurement(kind, value, view):
    unit = {"quantity": "bolts", "duration": "minutes", "distance": "meters"}[kind]
    if kind == "distance":
        value *= 1000
    if view == "units" and kind == "duration":
        value, unit = value * 60, "seconds"
    if view == "units" and kind == "distance":
        value, unit = value // 1000, "kilometers"
    # Distance words use kilometer units to stay in the exact cardinal grammar.
    if view == "words" and kind == "distance":
        value, unit = value // 1000, "kilometers"
    return (number_words(value) if view == "words" else str(value)) + " " + unit


def fact(split, kind, name, value, view):
    measure = measurement(kind, value, view)
    noun, prop = {
        "quantity": ("batch", "contained"),
        "duration": ("trial", "lasted"),
        "distance": ("route", "measured"),
    }[kind]
    if split == "fit":
        return f"The {name} {noun} {prop} exactly {measure}."
    if split == "selection":
        return f"Ledger for {name} {noun}: exact {kind} = {measure}."
    return f"An inspection established {measure} as the exact {kind} of the {name} {noun}."


def suite(split):
    """Balanced numeric/attribution counterfactuals; all privileged metadata separate."""
    if split not in VALUES:
        raise ValueError("only R3 authored train-side splits")
    records, design = [], []
    names = {
        "fit": "Arden Bexley Carlow Dunbar Epping Fenton",
        "selection": "Gosport Hartley Ilford",
        "diagnostic": "Jarrow Kendal Lewes Marlow",
    }[split].split()
    family = {"fit": "declarative", "selection": "ledger", "diagnostic": "inspection"}[split]
    for kind in ("quantity", "duration", "distance"):
        variants = ["exact", "words", "answer_paraphrase", "answer_words", "long", "unsupported"]
        if kind != "quantity":
            variants.append("units")
        for j, values in enumerate(VALUES[split]):
            name, other = names[j], "Westhaven " + names[j]
            source = f"r3:{split}:{kind}:{name}"
            for variant in variants:
                for order in (0, 1):
                    for ai, value in enumerate(values):
                        phrase = measurement(
                            kind, value, "words" if variant == "answer_words" else "digits"
                        )
                        noun = {"quantity": "batch", "duration": "trial", "distance": "route"}[kind]
                        claim = f"The exact {kind} of the {name} {noun} was {phrase}."
                        if variant == "answer_paraphrase":
                            claim = f"For the {name} {noun}, the {kind} was exactly {phrase}."
                        extras = []
                        position = 0
                        if variant == "long":
                            count = {"fit": 4, "selection": 9, "diagnostic": 18}[split]
                            extras = [FILL[split][k % len(FILL[split])] for k in range(count)]
                            position = (
                                (count if order else 0)
                                if split == "fit"
                                else count * (1 if split == "selection" else 2) // 3
                            )
                        answer = " ".join(extras[:position] + [claim] + extras[position:])
                        start = sum(len(s) + 1 for s in extras[:position])
                        # Numeric phrase is the minimal changed factual component (includes unit).
                        slot_start = start + claim.index(phrase)
                        for ei, actual in enumerate(values):
                            gold = int(ai != ei)
                            view = variant if variant in ("words", "units") else "digits"
                            facts = [
                                fact(split, kind, name, actual, view),
                                fact(split, kind, other, values[1 - ei], view),
                            ]
                            if variant == "unsupported" and gold:
                                facts[0] = f"The {name} {noun} was inspected in the morning."
                            if order:
                                facts.reverse()
                            evidence = " ".join(facts + FILL[split])
                            pair_id = f"{source}:{variant}:o{order}:a{ai}"
                            identifier = f"{pair_id}:e{ei}"
                            annotations = (
                                (
                                    Annotation(
                                        slot_start,
                                        slot_start + len(phrase),
                                        phrase,
                                        "Evident Baseless Info"
                                        if variant == "unsupported"
                                        else "Evident Conflict",
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
                                    "authored_R3",
                                    "train",
                                    "good",
                                    answer,
                                    "What does the evidence state?",
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
                                    "split": split,
                                    "pair_id": pair_id,
                                    "kind": kind,
                                    "variant": variant,
                                    "family": family,
                                    "order": order,
                                    "answer_index": ai,
                                    "evidence_index": ei,
                                    "values": list(values),
                                    "gold": gold,
                                    "claim_start": start,
                                    "claim_end": start + len(claim),
                                    "slot_start": slot_start,
                                    "slot_end": slot_start + len(phrase),
                                    "padding": len(extras),
                                    "claim_position": position,
                                    "semantic_status": "unsupported"
                                    if variant == "unsupported" and gold
                                    else "contradicted"
                                    if gold
                                    else "supported",
                                }
                            )
    return records, design
