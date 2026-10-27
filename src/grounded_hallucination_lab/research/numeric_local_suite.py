"""Preregistered R4 fictional worlds, disjoint parents from supervised R1/R2/R3."""

from grounded_hallucination_lab.data.ragtruth import Annotation, Record
from grounded_hallucination_lab.research.invariance import number_words

NAMES = (
    "Alderwick",
    "Brackenford",
    "Cedarvale",
    "Dovemere",
    "Elmbridge",
    "Foxmere",
    "Glenhaven",
    "Hazelwick",
    "Ivybridge",
    "Juniperford",
    "Kingsmere",
    "Larchvale",
)
FILLER = (
    "The archive has a red ceiling.",
    "The entrance contains a brass bell.",
    "The terrace has a stone railing.",
)
VARIANTS = (
    "digits",
    "evidence_words",
    "answer_words",
    "both_words",
    "both_units",
    "long12",
    "long36",
    "coreference36",
    "clause36",
)


def phrase(kind, value, view):
    unit = {"quantity": "bolts", "duration": "minutes", "distance": "meters"}[kind]
    if kind == "distance":
        value *= 1000
    if view == "units":
        if kind == "duration":
            value, unit = value * 60, "seconds"
        elif kind == "distance":
            value, unit = value // 1000, "kilometers"
    return f"{number_words(value) if view == 'words' else value} {unit}"


def suite():
    records, design = [], []
    for i, name in enumerate(NAMES):
        kind = ("quantity", "duration", "distance")[i // 4]
        noun = {"quantity": "crate", "duration": "procedure", "distance": "trail"}[kind]
        values = (112 + 14 * i, 118 + 14 * i)
        source = f"r4:{name}"
        other = "Zincford"
        for variant in VARIANTS:
            for order in (0, 1):
                for ai, value in enumerate(values):
                    av = (
                        "words"
                        if variant in ("answer_words", "both_words")
                        else ("units" if variant == "both_units" else "digits")
                    )
                    # Distance words use kilometers to stay within exact cardinal grammar.
                    answer_phrase = phrase(
                        kind, value, "units" if kind == "distance" and av == "words" else av
                    )
                    if kind == "distance" and av == "words":
                        answer_phrase = f"{number_words(value)} kilometers"
                    claim = f"The {name} {noun} had a recorded {kind} of {answer_phrase}."
                    padding = (
                        12
                        if variant == "long12"
                        else 36
                        if variant in ("long36", "coreference36", "clause36")
                        else 0
                    )
                    extras = [FILLER[k % 3] for k in range(padding)]
                    if variant == "coreference36":
                        extras = ["It was registered.", "It was examined.", "It was recorded."] * 12
                        claim = f"Its recorded {kind} was {answer_phrase}."
                        answer = (
                            f"The {name} {noun} was documented. " + " ".join(extras) + " " + claim
                        )
                        # This extra factual sentence is explicitly supported in evidence.
                        claim_start = answer.rindex(claim)
                    elif variant == "clause36":
                        lead = "; ".join(s[:-1].lower() for s in extras)
                        answer = lead[:1].upper() + lead[1:] + "; " + claim[:1].lower() + claim[1:]
                        claim_start = len(lead) + 2
                    else:
                        before = " ".join(extras[: padding // 2])
                        after = " ".join(extras[padding // 2 :])
                        answer = " ".join(s for s in (before, claim, after) if s)
                        claim_start = len(before) + bool(before)
                    slot_start = answer.index(answer_phrase, claim_start)
                    for ei, actual in enumerate(values):
                        ev = (
                            "words"
                            if variant in ("evidence_words", "both_words")
                            else ("units" if variant == "both_units" else "digits")
                        )
                        facts = []
                        for entity, quantity in ((name, actual), (other, values[1 - ei])):
                            measure = phrase(
                                kind,
                                quantity,
                                "units" if kind == "distance" and ev == "words" else ev,
                            )
                            if kind == "distance" and ev == "words":
                                measure = f"{number_words(quantity)} kilometers"
                            facts.append(
                                f"According to the signed dispatch, the {entity} {noun} "
                                f"had a recorded {kind} of {measure}."
                            )
                        if order:
                            facts.reverse()
                        evidence = " ".join(
                            facts + list(FILLER) + [f"The {name} {noun} was documented."]
                        )
                        if variant == "coreference36":
                            evidence += " " + " ".join(
                                f"The {name} {noun} was {verb}."
                                for verb in ("registered", "examined", "recorded")
                            )
                        gold = int(ai != ei)
                        pair = f"{source}:{variant}:o{order}:a{ai}"
                        identifier = f"{pair}:e{ei}"
                        annotations = (
                            (
                                Annotation(
                                    slot_start,
                                    slot_start + len(answer_phrase),
                                    answer_phrase,
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
                                "authored_R4",
                                "train",
                                "good",
                                answer,
                                "What does the dispatch establish?",
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
                                "gold": gold,
                                "semantic_status": "contradicted" if gold else "supported",
                                "padding": padding,
                                "claim_start": claim_start,
                                "claim_end": len(answer)
                                if variant in ("coreference36", "clause36")
                                else claim_start + len(claim),
                                "slot_start": slot_start,
                                "slot_end": slot_start + len(answer_phrase),
                                "order": order,
                                "answer_index": ai,
                                "evidence_index": ei,
                                "values": list(values),
                            }
                        )
    return records, design
