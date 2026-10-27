"""Authored, verifiable evidence interventions; never relabel shuffled natural data."""

from grounded_hallucination_lab.data.ragtruth import Annotation, Record

FILLER = (
    "The archive contains maps.",
    "The library opens at nine.",
    "The courtyard has two gates.",
    "The annex is north of the library.",
    "The survey covered three rooms.",
    "The catalogue lists local buildings.",
    "The depot stores wooden crates.",
    "The workshop closes on Friday.",
    "The garden has stone paths.",
    "The station has a clock.",
    "The museum displays pottery.",
    "The hall contains six benches.",
)


def fact_cases():
    """All swapped values/entities remain somewhere in evidence: unigram matching fails."""
    for i, (a, b) in enumerate((("Mira", "Owen"), ("Nora", "Liam"), ("Ada", "Leon"))):
        year, other = 1990 + i * 7, 1991 + i * 7
        yield {
            "group": f"date:{i}",
            "kind": "date",
            "answer": f"The {a} museum first opened in {year}.",
            "error": str(year),
            "supported": (
                f"The {a} museum first opened in {year}. The {b} museum first opened in {other}."
            ),
            "contradicted": (
                f"The {a} museum first opened in {other}. The {b} museum first opened in {year}."
            ),
            "paraphrase": (
                f"The first opening of the {a} museum was in {year}. "
                f"The {b} museum first opened in {other}."
            ),
        }
        yield {
            "group": f"quantity:{i}",
            "kind": "quantity",
            "answer": f"{a} packed exactly {12 + i} crates.",
            "error": str(12 + i),
            "supported": f"{a} packed exactly {12 + i} crates. {b} packed exactly {17 + i} crates.",
            "contradicted": (
                f"{a} packed exactly {17 + i} crates. {b} packed exactly {12 + i} crates."
            ),
            "paraphrase": (
                f"The exact number of crates packed by {a} was {12 + i}. "
                f"{b} packed exactly {17 + i} crates."
            ),
        }
        yield {
            "group": f"unit:{i}",
            "kind": "unit",
            "answer": f"{a}'s tank has a capacity of exactly 10 litres.",
            "error": "litres",
            "supported": (
                f"{a}'s tank has a capacity of exactly 10 litres. "
                f"{b}'s tank has a capacity of exactly 10 gallons."
            ),
            "contradicted": (
                f"{a}'s tank has a capacity of exactly 10 gallons. "
                f"{b}'s tank has a capacity of exactly 10 litres."
            ),
            "paraphrase": (
                f"The capacity of {a}'s tank is 10 L. "
                f"{b}'s tank has a capacity of exactly 10 gallons."
            ),
        }
        yield {
            "group": f"negation:{i}",
            "kind": "negation",
            "answer": f"{a} signed the agreement.",
            "error": "signed the agreement",
            "supported": f"{a} signed the agreement. {b} did not sign the agreement.",
            "contradicted": f"{a} did not sign the agreement. {b} signed the agreement.",
            "paraphrase": f"The agreement was signed by {a}. {b} did not sign the agreement.",
        }
        yield {
            "group": f"attribution:{i}",
            "kind": "attribution",
            "answer": f"{a} was the sole designer of the bridge.",
            "error": a,
            "supported": (
                f"{a} was the sole designer of the bridge. {b} was the sole designer of the tower."
            ),
            "contradicted": (
                f"{a} was the sole designer of the tower. {b} was the sole designer of the bridge."
            ),
            "paraphrase": (
                f"The bridge had exactly one designer, {a}. {b} was the sole designer of the tower."
            ),
        }


def controls():
    """300 cases /15 underlying facts; changing padding does not create independent facts."""
    result, info = [], []
    for case in fact_cases():
        for padding in (0, 8, 32):
            correct = " ".join(FILLER[j % len(FILLER)] for j in range(padding))
            for position in ("only",) if not padding else ("first", "last"):
                answer = " ".join(
                    s
                    for s in (
                        (correct, case["answer"])
                        if position == "last"
                        else (case["answer"], correct)
                    )
                    if s
                )
                claim_start = answer.index(case["answer"])
                for condition in ("supported", "contradicted", "paraphrase", "reordered"):
                    evidence = case["supported" if condition == "reordered" else condition]
                    evidence = (
                        " ".join(reversed(FILLER)) + " " + evidence
                        if condition == "reordered"
                        else evidence + " " + " ".join(FILLER)
                    )
                    identifier = f"control:{case['group']}:{padding}:{position}:{condition}"
                    annotations = ()
                    if condition == "contradicted":
                        start = claim_start + case["answer"].index(case["error"])
                        annotations = (
                            Annotation(
                                start,
                                start + len(case["error"]),
                                case["error"],
                                "Evident Conflict",
                                False,
                                False,
                                {},
                            ),
                        )
                    result.append(
                        Record(
                            identifier,
                            "control:" + case["group"],
                            "QA",
                            "authored",
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
                    info.append(
                        {
                            "id": identifier,
                            "group": case["group"],
                            "kind": case["kind"],
                            "padding_sentences": padding,
                            "position": position,
                            "condition": condition,
                            "gold": int(bool(annotations)),
                            "claim_start": claim_start,
                            "claim_end": claim_start + len(case["answer"]),
                        }
                    )
    return result, info
