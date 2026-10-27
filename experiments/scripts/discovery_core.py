"""Small, explicit primitives for the bounded discovery experiments."""

from copy import deepcopy


def content_readout(response: str, row: dict) -> dict:
    """Change response aggregation only; preserve every token score and offset."""
    if len(row["offsets"]) != len(row["token_probabilities"]):
        raise ValueError("unaligned tokens")
    keep = []
    for (start, end), value in zip(row["offsets"], row["token_probabilities"], strict=True):
        if not 0 <= start <= end <= len(response):
            raise ValueError("invalid original offset")
        if any(char.isalnum() for char in response[start:end]):
            keep.append(value)
    out = deepcopy(row)
    out["score"] = max(keep, default=0.0)
    out["content_readout_tokens"] = len(keep)
    return out


def short_error(record) -> bool:
    gold = {
        i
        for a in record.annotations
        for i in range(a.start, a.end)
        if not record.response[i].isspace()
    }
    return (
        bool(gold)
        and len(record.response) >= 500
        and len(gold) <= 0.05 * sum(not char.isspace() for char in record.response)
    )
