"""R6 fictional relation worlds; renderers never receive a gold label."""

from grounded_hallucination_lab.data.ragtruth import Annotation, Record

VARIANTS = (
    "entity_swap",
    "property_swap",
    "shared_value",
    "similar_name",
    "prose_order",
    "long_short",
)
FIT = (
    "Ansel Birch Cobalt Dorian Esten Flint Garnet Harlan "
    "Indigo Jasper Keaton Linden Mercer Nolan Orson Palmer"
).split()
TRANSFER = (
    "Acacia Bellflower Camellia Dahlia Edelweiss Freesia Gardenia Hyacinth "
    "Iris Jasmine Kalmia Lilac Magnolia Narcissus Orchid Peony"
).split()
FIT_PEERS = (
    "Avon Bromley Chesham Deal Exmouth Fareham Grimsby Harwich "
    "Ipswich Kendal Leighton Maldon Newbury Otley Penrith Redcar"
).split()
TRANSFER_PEERS = (
    "Amble Bude Crediton Dawlish Emsworth Fowey Goole Hailsham "
    "Ingleton Jedburgh Kinross Lerwick Matlock Nairn Oban Penzance"
).split()
FILL = (
    "The office has a tiled floor.",
    "The entrance has a glass door.",
    "The wall clock has a round face.",
)


def render(facts, family):
    """Render a list of (entity, property, value, unit) without derived claims."""
    templates = {
        "old": "The {e} {p} contained exactly {v} {u}.",
        "fit0": "For {e}, {p} amounted to {v} {u}.",
        "fit1": "A total of {v} {u} was recorded under {p} for {e}.",
        "fit2": "{e}: {p} = {v} {u}.",
        "fit3": "The entry headed {p} assigns {v} {u} to {e}.",
        "fit4": "{e} had {v} {u} in its {p}.",
        "fit5": "Under {e}, the ledger lists {p} as {v} {u}.",
        "held0": "According to the inventory, {v} {u} belong to the {p} at {e}.",
        "held1": "{e}'s record reads as follows: {v} {u}, corresponding to {p}.",
        "held2": "The audited figure associated with {e} and {p} is {v} {u}.",
        "held3": "For the {p}, the auditor attributed {v} {u} to {e}.",
        "held4": "An audit of {p} at {e} established a count of {v} {u}.",
        "held5": "{v} {u} is the quantity registered against {p} in the account of {e}.",
    }
    return " ".join(templates[family].format(e=e, p=p, v=v, u=u) for e, p, v, u in facts)


def suite(split="transfer", arm="B"):
    if split not in ("fit", "transfer") or arm not in ("A", "B"):
        raise ValueError("unknown split/arm")
    records, design = [], []
    names = FIT if split == "fit" else TRANSFER
    for i, name in enumerate(names):
        values = [101 + 12 * i, 107 + 12 * i] if split == "fit" else [403 + 14 * i, 411 + 14 * i]
        properties, unit = (
            (("morning shipment", "evening shipment"), "bolts")
            if i % 2
            else (("reserve stock", "display stock"), "crates")
        )
        source = f"r6:{split}:{name}"
        for vi, variant in enumerate(VARIANTS):
            peer = (FIT_PEERS if split == "fit" else TRANSFER_PEERS)[i]
            entities = (
                [name + " North", name + " South"]
                if variant == "similar_name"
                else [name + " Depot", peer + " Warehouse"]
            )
            te, tp = (i // 2) % 2, i % 2
            family = (
                "old"
                if split == "fit" and arm == "A"
                else f"{'fit' if split == 'fit' else 'held'}{(vi + i) % 6}"
            )
            for order in (0, 1):
                for ai, answer_value in enumerate(values):
                    phrase = f"{answer_value} {unit}"
                    if split == "fit":
                        claim = f"The {entities[te]} {properties[tp]} contained exactly {phrase}."
                    else:
                        claim = (
                            f"The amount attributable to {properties[tp]} "
                            f"at {entities[te]} is {phrase}."
                        )
                    fillers = list(FILL) * (4, 8, 12)[i % 3] if variant == "long_short" else []
                    insertion = (0, len(fillers) // 2, len(fillers))[(i + order) % 3]
                    answer = " ".join(fillers[:insertion] + [claim] + fillers[insertion:])
                    start = answer.index(phrase)
                    for world in (0, 1):
                        facts = []
                        for e in (0, 1):
                            for p in (0, 1):
                                affected = (
                                    e == te
                                    if variant == "property_swap"
                                    else p == tp
                                    if variant == "entity_swap"
                                    else True
                                )
                                facts.append(
                                    (
                                        entities[e],
                                        properties[p],
                                        values[(e + p + world * affected) % 2],
                                        unit,
                                    )
                                )
                        if variant == "shared_value":
                            facts += [
                                (name + " Annex", properties[p], values[p], unit) for p in (0, 1)
                            ]
                        if order:
                            facts.reverse()
                        actual = values[(te + tp + world) % 2]
                        gold = int(answer_value != actual)
                        pair = f"{source}:{variant}:o{order}:a{ai}"
                        identifier = f"{pair}:w{world}"
                        anns = (
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
                        evidence = render(facts, family) + " " + " ".join(FILL)
                        records.append(
                            Record(
                                identifier,
                                source,
                                "QA",
                                "authored_R6",
                                "train",
                                "good",
                                answer,
                                f"How much was in the {properties[tp]} at {entities[te]}?",
                                evidence,
                                anns,
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
                                "family": family,
                                "gold": gold,
                                "world": world,
                                "answer_index": ai,
                                "order": order,
                                "target_entity": entities[te],
                                "target_property": properties[tp],
                                "actual_value": actual,
                                "answer_value": answer_value,
                                "facts": facts,
                                "slot_start": start,
                                "slot_end": start + len(phrase),
                                "claim_start": answer.index(claim),
                                "claim_end": answer.index(claim) + len(claim),
                                "padding": len(fillers),
                                "claim_position": insertion,
                            }
                        )
    return records, design
