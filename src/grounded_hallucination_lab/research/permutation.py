"""Conservative, label-free screening for whole-passage permutation experiments."""

import hashlib
import re

from grounded_hallucination_lab.research.passage_probe import passage_ranges

# Deliberately conservative: these are exclusions, not an entailment/coreference parser.
ORDER_REFERENCE = re.compile(
    r"\b(?:first|second|third|last|previous|next|preceding|following|earlier|later)\s+"
    r"(?:passage|paragraph|document|source|text|excerpt)s?\b|"
    r"\b(?:above|below|aforementioned|former|latter|respectively)\b|"
    r"\b(?:as\s+(?:noted|mentioned|described|stated)|see)\s+(?:earlier|previously|above|below)\b",
    re.I,
)
DEPENDENT_START = re.compile(
    r"^(?:this|these|those|that|it|its|they|their|he|his|she|her|such|then|next|finally|"
    r"afterwards|subsequently|previously|earlier|later|meanwhile|also|additionally|"
    r"however|but|and|similarly|therefore|thus|consequently|instead|otherwise)\b|"
    r"^(?:after|before|because of|despite)\s+(?:this|that|these|those)\b",
    re.I,
)
STABLE_REFERENCE = re.compile(r"\bpassage\s+[123]\b", re.I)
PROCEDURAL_QUERY = re.compile(
    r"^(?:how\s+to\b|how\s+(?:do|can|should|would)\b|best\s+way\b|ways?\s+to\b)|"
    r"\brecipes?\b|\bhow long\b.*\b(?:cook|bake|grill|fry|roast)\b",
    re.I,
)
FRAGMENT_CUE = re.compile(
    r"^(?:in addition|the same|the other|the remaining|as a result|for this reason)\b|"
    r"^(?:add|place|cook|preheat|heat|mix|dip|wash|cut|remove|pour|stir|start|select|"
    r"click|enter|disconnect|reattach|leave|drain|combine|reduce|bring|bake|put|take|"
    r"use|rinse|fill|return|serve|turn|flip|sprinkle|attach|insert|press|open|close|go|"
    r"load|clean|store)\b",
    re.I,
)


def eligibility(evidence: str, question: str, answers: list[str]) -> dict:
    """Screen a source and all its answers without accessing annotation metadata.

    Stable numbered IDs move with their passage; ordinal/positional references and
    possible cross-passage anaphora/continuations abstain. This is not proof of safety.
    """
    reasons, details = [], []
    if PROCEDURAL_QUERY.search(question.strip()):
        reasons.append("procedural_query_ambiguous_step_dependencies")
    try:
        ranges = passage_ranges(evidence)
    except ValueError:
        return {"eligible": False, "reasons": ["passage_structure"], "details": []}
    for field, text in [("question", question), *[("answer", a) for a in answers]]:
        for m in ORDER_REFERENCE.finditer(text):
            reasons.append("positional_reference")
            details.append({"field": field, "text": m.group(), "offset": m.start()})
    for i, (a, b) in enumerate(ranges):
        block = evidence[a:b]
        body = re.sub(r"^passage\s+\d+:\s*", "", block, flags=re.I)
        # A numbered procedural step still moves as part of the intact passage.
        body = re.sub(r"^\d+[.)]?\s+", "", body).lstrip(" \t\r\n\"'([{")
        for m in ORDER_REFERENCE.finditer(block):
            reasons.append("positional_reference")
            details.append({"field": f"passage{i + 1}", "text": m.group(), "offset": a + m.start()})
        if DEPENDENT_START.search(body) or FRAGMENT_CUE.search(body):
            reasons.append("dependent_passage_start")
            details.append({"field": f"passage{i + 1}", "text": body[:100], "offset": a})
        if body and body[0].islower():
            reasons.append("fragment_start")
            details.append({"field": f"passage{i + 1}", "text": body[:100], "offset": a})
    return {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "details": details,
        "stable_numbered_references": sum(len(STABLE_REFERENCE.findall(t)) for t in answers),
    }


def rotation_for_source(source_id: str) -> int:
    """One predeclared non-identity cyclic rotation, shared by related answers."""
    return 1 + int(hashlib.sha256(f"R10:42:{source_id}".encode()).hexdigest(), 16) % 2
