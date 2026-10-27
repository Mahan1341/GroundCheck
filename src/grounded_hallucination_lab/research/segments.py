"""Extractive segmentation. Never uses labels, quality flags or generator identity."""

import re

from grounded_hallucination_lab.baselines.lexical import TextSpan, sentences

BARE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*$")
LEADER = re.compile(r"^\s*(?:[-*•]+\s*|\d+[.)]\s+)")
META = re.compile(
    r"^(?:here(?:'s| is| are) (?:my |the |an? )?(?:answer|summary|information|steps|examples)"
    r"(?: based on (?:the )?(?:given |provided )?passages)?[.!:]?"
    r"|based on (?:the )?(?:given |provided )?passages,?"
    r"|that(?:'s| is) it[.!]?|hope (?:this|that) helps[.!]?"
    r"|(?:i am |i'm )?unable to answer"
    r"(?: (?:based on|using) (?:the )?(?:given |provided )?(?:passages|context))?[.!]?"
    r"|(?:i )?(?:cannot|can't|can not) answer (?:the |this )?question"
    r"(?: (?:based on|using) (?:the )?(?:given |provided )?(?:passages|context))?[.!]?)$",
    re.IGNORECASE,
)


def clean_segments(text: str) -> tuple[list[TextSpan], list[dict]]:
    kept, dropped = [], []
    for sentence in sentences(text):
        line_start = text.rfind("\n", 0, sentence.start) + 1
        line_end = text.find("\n", sentence.end)
        if line_end < 0:
            line_end = len(text)
        has_inline_content = bool(text[sentence.end : line_end].strip())
        at_line_start = not text[line_start : sentence.start].strip()
        if BARE.fullmatch(sentence.text) and at_line_start and has_inline_content:
            dropped.append({"start": sentence.start, "end": sentence.end, "reason": "list_marker"})
            continue
        match = LEADER.match(sentence.text) if at_line_start else None
        start = sentence.start + (match.end() if match else 0)
        for match in re.finditer(r"[^;]+(?:;|$)", text[start : sentence.end]):
            left, right = start + match.start(), start + match.end()
            while left < right and text[left].isspace():
                left += 1
            while right > left and (text[right - 1].isspace() or text[right - 1] == ";"):
                right -= 1
            if right <= left:
                continue
            value = text[left:right]
            if META.fullmatch(value):
                dropped.append(
                    {"start": left, "end": right, "reason": "standalone_meta_or_refusal"}
                )
            else:
                kept.append(TextSpan(left, right, value))
    return kept, dropped
