"""R7 independently authored worlds and oracle ledgers; never imported by extractor."""

from dataclasses import asdict

from grounded_hallucination_lab.data.ragtruth import Annotation, Record
from grounded_hallucination_lab.research.evidence_graph import Fact, Graph, Provenance

NAMES = (
    "Lunora Meriva Novali Orelia Parnel Quenby Rovana Selvia "
    "Torven Ulmora Valora Wenric Xandor Yelvia Zorlan Belora"
).split()
PEERS = (
    "Branor Celdan Dervin Elvara Farsen Galdor Hesper Ismera "
    "Jorlen Kelmar Lorven Morlen Nelric Perdan Reldan Seldon"
).split()
VARIANTS = ("different", "one_same", "repeated", "similar_names", "same_property", "split_passages")
FILL = (
    "The office has a tiled floor.",
    "The entrance has a glass door.",
    "The wall clock has a round face.",
)


def render(entity, prop, value, unit, family, prefix):
    templates = (
        "The {p} at {e} is {v} {u}.",
        "{e}'s {p} is {v} {u}.",
        "At {e}, the {p} totals {v} {u}.",
        "{e} has {v} {u} in its {p}.",
        "{e} carried a {p} tally of {v} {u}.",
        "An inventory credits {e} with {v} {u} under {p}.",
    )
    return prefix + templates[family].format(e=entity, p=prop, v=value, u=unit)


def suite():
    records, designs = [], []
    for i, name in enumerate(NAMES):
        source = f"r7:{name}"
        entity, alias = f"{name} HubA", f"{name} Site"
        qualifiers = ({}, {"time": "2024"}, {"scope": "alpha"}, {"condition": "sealed"})[i % 4]
        prefix = ("", "In 2024, ", "For batch alpha, ", "When sealed, ")[i % 4]
        values, third = (713 + 10 * i, 719 + 10 * i), 715 + 10 * i
        family = i % 6
        for variant in VARIANTS:
            for long in (False, True):
                for order in (0, 1):
                    for ai, value in enumerate(values):
                        phrase = f"{value} crates"
                        claim = prefix + f"The evening stock at {alias} is {phrase}."
                        fillers = list(FILL) * 8 if long else []
                        cut = (0, len(fillers) // 2, len(fillers))[(i + order) % 3]
                        answer = " ".join(fillers[:cut] + [claim] + fillers[cut:])
                        cs = answer.index(claim)
                        vs = answer.index(phrase)
                        pair = f"{source}:{variant}:l{int(long)}:o{order}:a{ai}"
                        for world in (0, 1):
                            actual = values[world]
                            entities = [f"{PEERS[i]} Hub{j}" for j in "BCDE"]
                            if variant == "similar_names":
                                entities = [f"{name} Hub{j}" for j in "BCDE"]
                            distractors = [values[0], values[1], third, third]
                            if variant == "one_same":
                                distractors[2] = value
                            if variant in (
                                "repeated",
                                "similar_names",
                                "same_property",
                                "split_passages",
                            ):
                                distractors[2:] = [value, value]
                            props = ["evening stock"] * 2 + [
                                ("evening stock" if variant == "same_property" else "morning stock")
                            ] * 2
                            specs = [(entity, "evening stock", actual)] + list(
                                zip(entities, props, distractors, strict=True)
                            )
                            if order:
                                specs.reverse()
                            separator = "\n\n" if variant == "split_passages" else " "
                            label_text = f"The label {alias} refers to {entity}."
                            texts = [render(e, p, v, "crates", family, prefix) for e, p, v in specs]
                            texts += [label_text, *FILL]
                            evidence = separator.join(texts)
                            facts = []
                            cursor = 0
                            for (e, p, v), text in zip(specs, texts, strict=False):
                                pos = evidence.index(text, cursor)
                                num = pos + text.index(f"{v} crates")
                                prov = Provenance(
                                    source,
                                    f"p{evidence[:pos].count(chr(10) + chr(10))}",
                                    pos,
                                    pos + len(text),
                                )
                                facts.append(
                                    Fact(
                                        e,
                                        p,
                                        str(v),
                                        "crates",
                                        prov,
                                        num,
                                        num + len(f"{v} crates"),
                                        **qualifiers,
                                    )
                                )
                                cursor = pos + len(text)
                            ap = evidence.index(label_text)
                            aliases = [
                                (
                                    alias,
                                    entity,
                                    Provenance(
                                        source,
                                        f"p{evidence[:ap].count(chr(10) + chr(10))}",
                                        ap,
                                        ap + len(label_text),
                                    ),
                                )
                            ]
                            claim_fact = Fact(
                                alias,
                                "evening stock",
                                str(value),
                                "crates",
                                Provenance("answer", "p0", cs, cs + len(claim)),
                                vs,
                                vs + len(phrase),
                                **qualifiers,
                            )
                            gold = int(value != actual)
                            ident = f"{pair}:w{world}"
                            annotations = (
                                (
                                    Annotation(
                                        vs,
                                        vs + len(phrase),
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
                                    ident,
                                    source,
                                    "QA",
                                    "authored_R7",
                                    "train",
                                    "good",
                                    answer,
                                    "What was the evening stock at " + alias + "?",
                                    evidence,
                                    annotations,
                                    {},
                                    {},
                                )
                            )
                            designs.append(
                                {
                                    "id": ident,
                                    "source_id": source,
                                    "pair_id": pair,
                                    "variant": variant,
                                    "family": str(family),
                                    "gold": gold,
                                    "order": order,
                                    "long": long,
                                    "claim_position": cut,
                                    "claim_start": cs,
                                    "claim_end": cs + len(claim),
                                    "slot_start": vs,
                                    "slot_end": vs + len(phrase),
                                    "actual": actual,
                                    "value": value,
                                    "world": world,
                                    "answer_index": ai,
                                    "facts": [asdict(f) for f in facts],
                                    "aliases": [(a, b, asdict(p)) for a, b, p in aliases],
                                    "claim": asdict(claim_fact),
                                }
                            )
    return records, designs


def fact_from_dict(value):
    return Fact(**{**value, "provenance": Provenance(**value["provenance"])})


def oracle_graphs(design):
    """Diagnostic-only adapter; must never be called on automatic inference paths."""
    evidence = Graph(
        [fact_from_dict(f) for f in design["facts"]],
        [(a, b, Provenance(**p)) for a, b, p in design["aliases"]],
    )
    return evidence, Graph([fact_from_dict(design["claim"])])
