"""Conservative scalar fact graph. Text-only extraction; no dataset/gold access."""

import re
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class Provenance:
    document_id: str
    passage_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Fact:
    entity: str
    property: str
    value: str
    unit: str
    provenance: Provenance
    value_start: int
    value_end: int
    time: str = ""
    scope: str = ""
    condition: str = ""
    negated: bool = False
    status: str = "exact"


@dataclass
class Graph:
    facts: list[Fact] = field(default_factory=list)
    aliases: list[tuple[str, str, Provenance]] = field(default_factory=list)
    unknown: list[Provenance] = field(default_factory=list)
    uncertain_numeric: bool = False

    def entity(self, text: str) -> str | None:
        """Only explicit directed identity links; conflicting links/cycles abstain."""
        key = clean(text)
        seen = set()
        while key not in seen:
            seen.add(key)
            targets = {clean(b) for a, b, _ in self.aliases if clean(a) == key}
            if not targets:
                return key
            if len(targets) != 1:
                return None
            key = targets.pop()
        return None

    def nodes(self) -> dict:
        """Equal values can share nodes; separate subject edges retain binding."""
        return {
            "entities": sorted({f.entity for f in self.facts}),
            "values": sorted({(f.value, f.unit) for f in self.facts}),
        }


def clean(text: str) -> str:
    return " ".join(text.casefold().strip().split())


# Open entity/property strings, deliberately narrow complete scalar clauses.
E = r"(?P<entity>[A-Za-z][A-Za-z0-9 '’-]*?)"
P = r"(?P<property>[a-z][a-z -]*?)"
V = r"(?P<value>\d+(?:\.\d+)?) (?P<unit>bolts|crates|meters|minutes|seconds|kilograms)"
N = r"(?P<neg>not )?"
PATTERNS = tuple(
    re.compile(p)
    for p in (
        rf"The {P} at {E} is {N}{V}\.",
        rf"{E}'s {P} is {N}{V}\.",
        rf"At {E}, the {P} totals {V}\.",
        rf"{E} has {V} in its {P}\.",
    )
)
ALIAS = re.compile(r"The label (?P<a>[A-Za-z0-9 '’-]+) refers to (?P<b>[A-Za-z0-9 '’-]+)\.")
QUALIFIERS = (
    ("time", r"In (\d{4}), "),
    ("scope", r"For batch ([a-z]+), "),
    ("condition", r"When ([a-z ]+), "),
)


def sentences(text: str):
    """Offsets include complete clauses; avoid splitting decimal numerals."""
    for match in re.finditer(r"\S[\s\S]*?(?:[.!?](?=\s|$)|$)", text):
        a, b = match.span()
        while a < b and text[a].isspace():
            a += 1
        if a < b:
            yield a, b, text[a:b]


def extract(text: str, document_id: str) -> Graph:
    graph = Graph()
    for start, end, clause in sentences(text):
        passage = f"p{text[:start].count(chr(10) + chr(10))}"
        provenance = Provenance(document_id, passage, start, end)
        alias = ALIAS.fullmatch(clause)
        if alias:
            graph.aliases.append((alias["a"], alias["b"], provenance))
            continue
        qualifiers = {}
        prefix = 0
        for key, pattern in QUALIFIERS:
            match = re.match(pattern, clause[prefix:])
            if match:
                qualifiers[key] = match[1]
                prefix += match.end()
        body = clause[prefix:]
        match = next((m for p in PATTERNS if (m := p.fullmatch(body))), None)
        if match:
            prop = match["property"]
            # Do not silently turn approximate/range/comparative quantities into exact facts.
            if re.search(r"\b(about|average|minimum|maximum|estimated|roughly|range)\b", prop):
                graph.unknown.append(provenance)
                graph.uncertain_numeric = True
                continue
            graph.facts.append(
                Fact(
                    match["entity"],
                    prop,
                    str(Decimal(match["value"])),
                    match["unit"],
                    provenance,
                    start + prefix + match.start("value"),
                    start + prefix + match.end("unit"),
                    negated=bool(match.groupdict().get("neg")),
                    **qualifiers,
                )
            )
        else:
            graph.unknown.append(provenance)
            graph.uncertain_numeric |= bool(re.search(r"\d", clause))
    return graph


def keyed(claim: Fact, evidence: Graph, claims: Graph) -> list[Fact]:
    """Entity+property lookup, deliberately leaves qualifier reasoning to verifier."""
    # Answer aliases must also be unambiguous; evidence identity declarations resolve names.
    entity = claims.entity(claim.entity)
    entity = evidence.entity(entity) if entity is not None else None
    if entity is None:
        return []
    return [
        f
        for f in evidence.facts
        if evidence.entity(f.entity) == entity
        and clean(f.property) == clean(claim.property)
        and f.status == "exact"
    ]


def match_fact(claim: Fact, evidence: Graph, claims: Graph) -> dict:
    """Open world: absent/ambiguous facts never imply contradiction."""
    if evidence.uncertain_numeric:
        return {"status": "unknown", "stage": "extraction", "references": []}
    if evidence.entity(claim.entity) is None or claims.entity(claim.entity) is None:
        return {"status": "ambiguous", "stage": "entity_linking", "references": []}
    candidates = keyed(claim, evidence, claims)
    if not candidates:
        return {"status": "unknown", "stage": "retrieval", "references": []}
    same = [
        f
        for f in candidates
        if (f.time, f.scope, f.condition, f.unit)
        == (claim.time, claim.scope, claim.condition, claim.unit)
    ]
    if not same:
        return {"status": "unknown", "stage": "qualifiers", "references": []}
    positive = {Decimal(f.value) for f in same if not f.negated}
    negative = {Decimal(f.value) for f in same if f.negated}
    if len(positive) > 1 or positive & negative:
        return {"status": "ambiguous", "stage": "relation_matching", "references": same}
    value = Decimal(claim.value)
    if positive:
        equal = value in positive
        supported = equal != claim.negated
    elif value in negative:
        supported = claim.negated
    else:
        return {"status": "unknown", "stage": "relation_matching", "references": same}
    return {
        "status": "supported" if supported else "contradicted",
        "stage": "relation_matching",
        "references": same,
    }


def retrieval_text(text: str, facts: list[Fact], graph: Graph) -> tuple[str, list[Provenance]]:
    references = {f.provenance for f in facts}
    references.update(p for _, _, p in graph.aliases)
    ordered = sorted(references, key=lambda p: p.start)
    return "\n\n".join(text[p.start : p.end] for p in ordered), ordered


def graph_override(row: dict, claims: Graph, evidence: Graph) -> tuple[dict, list[dict]]:
    probabilities = list(row["token_probabilities"])
    diagnostics = []
    for claim in claims.facts:
        verdict = match_fact(claim, evidence, claims)
        diagnostics.append({"claim": claim, **verdict})
        status = verdict["status"]
        if status == "supported":
            a, b, risk = claim.provenance.start, claim.provenance.end, 0.0
        elif status == "contradicted":
            a, b, risk = claim.value_start, claim.value_end, 1.0
        else:
            continue
        for i, (x, y) in enumerate(row["offsets"]):
            if x < b and y > a:
                probabilities[i] = risk
    return {
        **row,
        "token_probabilities": probabilities,
        "score": max(probabilities, default=0),
    }, diagnostics


# R8 extension. R7 entry points above remain unchanged; use the archived R7 source
# when replaying its freeze. Natural facts are proposals, never reviewed graph gold.
@dataclass(frozen=True)
class Mention:
    text: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class NaturalFact(Fact):
    entity_start: int = 0
    entity_end: int = 0
    property_start: int = 0
    property_end: int = 0
    confidence: str = "provisional"
    issues: tuple[str, ...] = ()
    numeric: bool = False


@dataclass
class NaturalGraph(Graph):
    mentions: list[Mention] = field(default_factory=list)
    clauses: list[Provenance] = field(default_factory=list)


def natural_key(text: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", clean(text).strip(" .;:!?"))


def natural_fact_from_tags(text, offsets, tags, provenance, stem) -> NaturalFact | None:
    """Extract one auditable subject/predicate/complement candidate per clause.

    Tagging and sentence splitting are injected: unit tests need no downloaded NLP.
    Complex clauses and unresolved subjects remain visible but cannot override.
    """
    if not offsets:
        return None
    words = [text[a:b] for a, b in offsets]
    verb = next((i for i, t in enumerate(tags) if t.startswith("VB") or t == "MD"), None)
    if verb is None or verb == len(words) - 1:
        return None
    end = len(words)
    while end and re.fullmatch(r"[.!?;]+", words[end - 1]):
        end -= 1
    j = verb
    negated = False
    modal = []
    chain = []
    while j < end:
        word = words[j].lower()
        if tags[j] == "MD":
            modal.append({"ca": "can", "wo": "will"}.get(word, word))
        elif word in ("not", "n't", "never"):
            negated = True
        elif tags[j].startswith("VB"):
            chain.append(word)
        else:
            break
        j += 1
    if not chain or j >= end:
        return None
    issues = []
    subject_end = offsets[verb][0]
    subject_start = offsets[0][0]
    while subject_end > subject_start and text[subject_end - 1].isspace():
        subject_end -= 1
    entity = text[subject_start:subject_end].strip()
    if not entity:
        entity = "__implicit_subject__"
        issues.append("implicit_subject")
        subject_end = subject_start
    pronouns = r"\b(it|they|this|these|that|those|he|she|we|you|there|its|their|your)\b"
    if re.search(pronouns, entity, re.I):
        issues.append("unresolved_coreference")
    if len(entity.split()) > 12 or any(t.startswith("VB") for t in tags[:verb]):
        issues.append("complex_subject")
    if "," in entity or re.search(
        r"\b(if|when|while|because|although|after|before)\b", entity, re.I
    ):
        issues.append("scoped_subject")
    if tags[:verb] and (tags[verb - 1] in ("CC", "RB", "TO") or "NN" not in " ".join(tags[:verb])):
        issues.append("non_nominal_subject")
    value_start, value_end = offsets[j][0], offsets[end - 1][1]
    value = text[value_start:value_end]
    # No silent dropping of subordinate clauses, ranges, approximation or coordination.
    if re.search(r"\b(if|when|while|because|unless|until|which|who|that)\b", value, re.I):
        issues.append("embedded_relation")
    if any(t.startswith("VB") for t in tags[j:]):
        issues.append("embedded_predicate")
    if re.search(
        r"\b(about|approximately|around|roughly|nearly|up to|at least|at most|between)\b",
        value,
        re.I,
    ):
        issues.append("inexact_value")
    if re.search(r"\d\s*(?:-|to|or)\s*\d", value):
        issues.append("range_value")
    if re.search(r"\bnot only\b", text[provenance.start : provenance.end], re.I):
        issues.append("negation_scope")
    if any(w in (";", ":") for w in words[j:]):
        issues.append("compound_complement")

    def normalize(word):
        return "be" if word in ("is", "are", "was", "were", "am", "be", "been") else stem(word)

    relation = " ".join(normalize(w) for w in chain if w not in ("do", "does", "did"))
    if not relation:
        return None
    sentence = text[provenance.start : provenance.end]
    # Literal qualifiers, no inferred dates, scope transfer or invented coreference.
    time = "past" if tags[verb] == "VBD" else "present"
    dates = re.findall(r"\b(?:in|since|during) (?:\d{4}|[A-Z][a-z]+ \d{4})\b", sentence)
    if dates:
        time += ":" + "|".join(dates)
    scope = "|".join(modal)
    conditions = re.findall(r"\b(?:if|when|unless|while)\b.*", sentence, re.I)
    status = "candidate" if issues else "exact"
    return NaturalFact(
        entity,
        relation,
        value,
        "text",
        provenance,
        value_start,
        value_end,
        time=time,
        scope=scope,
        condition="|".join(conditions),
        negated=negated,
        status=status,
        entity_start=subject_start,
        entity_end=subject_end,
        property_start=offsets[verb][0],
        property_end=offsets[j - 1][1],
        confidence="rule_high_unreviewed" if status == "exact" else "provisional",
        issues=tuple(issues),
        numeric=bool(re.search(r"\d", value)),
    )


def extract_natural(text: str, document_id: str) -> NaturalGraph:
    """Local pinned Punkt + English POS; no fitting, graph labels or model downloads."""
    from nltk.stem import PorterStemmer
    from nltk.tokenize import TreebankWordTokenizer

    from grounded_hallucination_lab.research.natural_nlp import tagger
    from grounded_hallucination_lab.research.segments import META
    from grounded_hallucination_lab.research.verifiers import punkt_spans

    graph = NaturalGraph()
    tokenizer = TreebankWordTokenizer()
    stem = PorterStemmer().stem
    # Split paragraphs before Punkt so source markers never join adjacent passages.
    for block in re.finditer(r"[^\n]+", text):
        for span in punkt_spans(block.group()):
            start, end = block.start() + span.start, block.start() + span.end
            header = re.match(r"(?:passage \d+:)?\s*(?:\d+[.)]?\s+)?", text[start:end], re.I)
            start += header.end()
            repeated_heading = re.match(r"([A-Z]+)-(?=\1\b)", text[start:end], re.I)
            if repeated_heading:
                start += repeated_heading.end()
            if start >= end:
                continue
            passage = list(re.finditer(r"passage (\d+):", text[:start], re.I))
            pid = "p" + passage[-1][1] if passage else "p0"
            provenance = Provenance(document_id, pid, start, end)
            graph.clauses.append(provenance)
            clause = text[start:end]
            if META.fullmatch(clause):
                graph.unknown.append(provenance)
                continue
            offsets = [(start + a, start + b) for a, b in tokenizer.span_tokenize(clause)]
            words = [text[a:b] for a, b in offsets]
            tags = [t for _, t in tagger().tag(words)]
            # Sentence-capitalized imperatives can be mistagged as names.
            if tags and tags[0] == "NNP":
                lower_tags = tagger().tag([words[0].lower(), *words[1:]])
                if lower_tags[0][1] == "VB":
                    tags[0] = "VB"
            for (a, b), t in zip(offsets, tags, strict=True):
                if t.startswith("NN") or t in ("PRP", "PRP$"):
                    graph.mentions.append(Mention(text[a:b], a, b, t))
            alias = re.fullmatch(
                r"(?P<e>[^,;]+?) (?:is|are) (?:also )?"
                r"(?:known as|called|referred to as) (?P<a>[^,;]+?)\.",
                clause,
            )
            if alias and not re.search(r"\b(it|they|this|that|he|she)\b", alias["e"], re.I):
                graph.aliases.append((natural_key(alias["a"]), natural_key(alias["e"]), provenance))
            fact = natural_fact_from_tags(text, offsets, tags, provenance, stem)
            if fact:
                graph.facts.append(fact)
                if fact.status != "exact":
                    graph.unknown.append(provenance)
            else:
                graph.unknown.append(provenance)
    return graph


def natural_match(claim: NaturalFact, evidence: NaturalGraph, *, links: bool) -> dict:
    """One verdict policy for flat and graph; only evidence alias traversal differs."""
    if claim.status != "exact":
        return {"status": "unknown", "stage": "claim_extraction", "references": []}

    def entity(value):
        key = natural_key(value)
        return evidence.entity(key) if links else key

    subject = entity(claim.entity)
    if subject is None:
        return {"status": "ambiguous", "stage": "entity_linking", "references": []}
    candidates = [
        f for f in evidence.facts if entity(f.entity) == subject and f.property == claim.property
    ]
    if not candidates:
        return {"status": "unknown", "stage": "retrieval", "references": []}
    if any(f.status != "exact" for f in candidates):
        return {"status": "ambiguous", "stage": "evidence_extraction", "references": candidates}
    same = [
        f
        for f in candidates
        if (f.time, f.scope, f.condition) == (claim.time, claim.scope, claim.condition)
    ]
    if not same:
        return {"status": "unknown", "stage": "qualifiers", "references": candidates}
    equal = [f for f in same if natural_key(f.value) == natural_key(claim.value)]
    polarities = {f.negated for f in equal}
    if len(polarities) > 1:
        return {"status": "ambiguous", "stage": "relation_matching", "references": equal}
    if equal:
        if claim.negated in polarities:
            return {"status": "supported", "stage": "relation_matching", "references": equal}
        if not claim.scope:
            return {"status": "contradicted", "stage": "relation_matching", "references": equal}
    # Different descriptions/values are NOT generally contradictory (open world).
    return {"status": "unknown", "stage": "relation_matching", "references": same}


def natural_override(row: dict, claims: NaturalGraph, evidence: NaturalGraph, *, links: bool):
    probabilities = list(row["token_probabilities"])
    decisions = []
    for claim in claims.facts:
        decision = natural_match(claim, evidence, links=links)
        decisions.append({"claim": claim, **decision})
        if decision["status"] not in ("supported", "contradicted"):
            continue
        # Only the extracted assertion, never another clause or unresolved pronoun.
        start, end = claim.provenance.start, claim.provenance.end
        for j, (a, b) in enumerate(row["offsets"]):
            if a < end and b > start:
                probabilities[j] = float(decision["status"] == "contradicted")
    result = {**row, "token_probabilities": probabilities, "score": max(probabilities, default=0)}
    return result, decisions
