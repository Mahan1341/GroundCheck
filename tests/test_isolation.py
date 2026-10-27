from dataclasses import replace

from grounded_hallucination_lab.data.isolation import isolation_report, normalize_source_text
from grounded_hallucination_lab.data.splits import Partitions, grouped_split


def test_duplicate_audit_counts_sources_not_responses(dataset):
    base = dataset.records[0]
    parts = Partitions(
        train=[
            replace(base, id="a1", source_id="a", evidence="Alpha Beta"),
            replace(base, id="a2", source_id="a", evidence="Alpha Beta"),
        ],
        calibration=[replace(base, id="b1", source_id="b", evidence="ＡLPHA\t beta\n")],
        dev=[replace(base, id="c1", source_id="c", evidence="Alpha Beta")],
        test=[replace(base, id="d1", source_id="d", evidence="different", split="test")],
    )
    report = isolation_report(parts)
    assert report["exact"]["cross_partition_text_groups"] == 1
    assert report["exact"]["cross_partition_source_pairs"]["train__dev"] == 1
    assert report["normalized"]["cross_partition_text_groups"] == 1
    assert report["normalized"]["cross_partition_source_pairs"]["train__calibration"] == 1
    assert report["normalized"]["cross_partition_source_pairs"]["calibration__dev"] == 1
    assert sum(len(rows) for _, rows in parts.items()) == 5  # No silent removal.


def test_no_false_duplicates_from_six_responses_per_source(dataset):
    report = isolation_report(grouped_split(dataset.records))
    assert report["all_ids_disjoint"]
    assert report["exact"]["cross_partition_text_groups"] == 0
    assert report["normalized"]["cross_partition_text_groups"] == 0


def test_normalization_retains_punctuation_and_passage_numbers():
    assert normalize_source_text("Ａ b\n\tC ") == "a b c"
    assert normalize_source_text("passage 1: a.") != normalize_source_text("passage 2: a")
