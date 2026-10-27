"""Source-level integrity checks; never use response labels or remove examples."""

import hashlib
import unicodedata
from collections import defaultdict
from itertools import combinations
from typing import Any

from grounded_hallucination_lab.data.splits import Partitions


def normalize_source_text(text: str) -> str:
    """Conservative normalization: NFKC, casefold, collapse Unicode whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def isolation_report(parts: Partitions) -> dict[str, Any]:
    """Compare full evidence text once per source, excluding query and response.

    This checks complete context equality, not partial passage overlap, paraphrases
    or shuffled passages. Normalization does not strip punctuation/passage numbers.
    """
    sources = {}
    memberships = {}
    response_ids = {}
    for name, rows in parts.items():
        memberships[name] = {r.source_id for r in rows}
        response_ids[name] = {r.id for r in rows}
        for record in rows:
            if record.source_id in sources and sources[record.source_id] != record.evidence:
                raise ValueError(f"inconsistent evidence for source_id {record.source_id}")
            sources[record.source_id] = record.evidence
    pairs = list(combinations(memberships, 2))
    overlaps = {
        f"{left}__{right}": {
            "source_ids": sorted(memberships[left] & memberships[right]),
            "response_ids": sorted(response_ids[left] & response_ids[right]),
        }
        for left, right in pairs
    }
    report: dict[str, Any] = {
        "pairwise_id_overlap": overlaps,
        "all_ids_disjoint": all(
            not v["source_ids"] and not v["response_ids"] for v in overlaps.values()
        ),
        "scope": "full QA evidence string, unique source IDs; no labels used",
        "normalization": "Unicode NFKC + casefold + whitespace collapse; punctuation retained",
        "action": "report only; no rows deleted or partitions reassigned",
    }
    for mode in ("exact", "normalized"):
        buckets = defaultdict(set)
        for sid, evidence in sources.items():
            text = evidence if mode == "exact" else normalize_source_text(evidence)
            buckets[text].add(sid)
        duplicates = []
        pair_counts = {f"{a}__{b}": 0 for a, b in pairs}
        all_duplicate_groups = 0
        for text, ids in sorted(buckets.items()):
            if len(ids) < 2:
                continue
            all_duplicate_groups += 1
            by_partition = {
                name: sorted(ids & group) for name, group in memberships.items() if ids & group
            }
            if len(by_partition) < 2:
                continue
            for left, right in pairs:
                pair_counts[f"{left}__{right}"] += len(by_partition.get(left, [])) * len(
                    by_partition.get(right, [])
                )
            duplicates.append(
                {
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "source_ids_by_partition": by_partition,
                }
            )
        report[mode] = {
            "all_duplicate_text_groups": all_duplicate_groups,
            "cross_partition_text_groups": len(duplicates),
            "cross_partition_source_pairs": pair_counts,
            "groups": duplicates,
        }
    return report
