"""Offline safeguards for the external split and sealed confirmation labels."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/discovery_external_data.py"
SPEC = importlib.util.spec_from_file_location("discovery_external_data", SCRIPT)
external = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(external)


def test_locked_values_are_never_json_decoded(monkeypatch):
    original = json.loads

    def guarded(value, *args, **kwargs):
        assert "DO_NOT_DECODE" not in value
        return original(value, *args, **kwargs)

    monkeypatch.setattr(external.json, "loads", guarded)
    value = '[{"nested":{"id":"dev"},"id":"locked","labels":DO_NOT_DECODE},'
    value += '{"id":"dev","response":"brace } and escaped \\\" quote","label":1}]'
    rows = external.selected_objects(value, {"dev"})
    assert [r["id"] for r in rows] == ["dev"]


@pytest.mark.parametrize("text,allowed", [
    ('[{"id":"x"},{"id":"x"}]', {"x"}),
    ('[{"nested":{"id":"x"}}]', {"x"}),
    ('[{"id":"x","id":"y"}]', {"x"}),
    ('[{"id":{"nested":"never decode"}}]', {"x"}),
    ('[{"id":"x"}]', {"missing"}),
    ('{"id":"x"}', {"x"}),
    ('[{"id":"x","broken":[}]', {"x"}),
])
def test_scanner_fails_closed_on_ambiguous_or_missing_ids(text, allowed):
    with pytest.raises(ValueError):
        external.selected_objects(text, allowed)


def test_normalized_duplicates_are_grouped_and_split_is_order_invariant():
    sources = [{"source_id": str(i), "group_id": external.source_group(f"Context {i}")}
               for i in range(10)]
    sources.append({"source_id": "duplicate",
                    "group_id": external.source_group("ＣＯＮＴＥＸＴ 0\n")})
    assert sources[0]["group_id"] == sources[-1]["group_id"]
    first = external.source_partitions(sources, set())
    assert first == external.source_partitions(list(reversed(sources)), set())
    assert first["0"] == first["duplicate"]
    assert Counter(first[str(i)] for i in range(10)) == {
        "development": 4, "locked_confirmation": 6,
    }
    excluded = external.source_partitions(sources, {"duplicate"})
    assert excluded["0"] == excluded["duplicate"] == "excluded_overlap"


@pytest.fixture
def sealed_fixture(tmp_path, monkeypatch):
    base = tmp_path / "external"
    monkeypatch.setattr(external, "ROOT", tmp_path)
    monkeypatch.setattr(external, "BASE", base)
    monkeypatch.setattr(external, "ANNOTATIONS", {"toy": (0, "unused")})
    sources = [{"source_id": name, "group_id": name, "partition": "locked_confirmation",
                "question": "Which object?", "evidence": "A is one; B is two.", "category": "qa"}
               for name in ("eligible", "excluded")]
    external.write_jsonl(base / "source_contexts.jsonl", sources)
    external.write_json(base / "frozen_split.json", {
        "source_contexts_sha256": external.digest((base / "source_contexts.jsonl").read_bytes()),
    })
    split_sha = external.digest((base / "frozen_split.json").read_bytes())
    annotations = ('[{"id":"excluded","response":"sealed","claude2_response_kg":'
                   'DO_NOT_DECODE},{"id":"eligible","response":"A is two.",'
                   '"claude2_response_kg":[{"human_label":"Contradiction"}]}]')
    path = base / "raw/dolly_toy_answers.json"
    external.write_bytes(path, annotations.encode())
    external.write_json(base / "sealed_annotation_manifest.json", {
        "split_sha256": split_sha,
        "files": [{"path": str(path.relative_to(tmp_path)),
                   "sha256": external.digest(path.read_bytes())}],
    })
    external.write_json(base / "frozen_eligibility.json", {
        "eligible_source_ids": {"locked_confirmation": ["eligible"]},
    })
    artifact = tmp_path / "model.fixture"
    external.write_bytes(artifact, b"offline fake model bytes")
    freeze = {
        "stage": "external_confirmation", "selection_complete": True,
        "no_adaptation_after_confirmation": True, "external_split_sha256": split_sha,
        "external_eligibility_sha256": external.digest(
            (base / "frozen_eligibility.json").read_bytes()
        ),
        "metrics": ["macro_f1"],
        "candidates": [{"name": "fixture", "threshold": 0.5,
                        "artifacts": [{"path": "model.fixture",
                                       "sha256": external.digest(artifact.read_bytes())}]}],
    }
    return base, tmp_path / "final_freeze.json", freeze


@pytest.mark.parametrize("change", [
    {"stage": "development"}, {"selection_complete": False},
    {"no_adaptation_after_confirmation": False}, {"external_split_sha256": "wrong"},
    {"external_eligibility_sha256": "wrong"}, {"metrics": []}, {"candidates": []},
])
def test_final_freeze_constraints_fail_before_unlock(sealed_fixture, change):
    base, path, freeze = sealed_fixture
    freeze.update(change)
    external.write_json(path, freeze)
    with pytest.raises(ValueError):
        external.export_partition("locked_confirmation", path, external.digest(path.read_bytes()))
    assert not (base / "confirmation_unlock.json").exists()
    assert not (base / "confirmation_records.jsonl").exists()


@pytest.mark.parametrize("kind", ["missing_hash", "wrong_hash", "changed_model", "bad_threshold"])
def test_final_candidate_hash_and_threshold_constraints(sealed_fixture, kind):
    base, path, freeze = sealed_fixture
    if kind == "changed_model":
        freeze["candidates"][0]["artifacts"][0]["sha256"] = "wrong"
    elif kind == "bad_threshold":
        freeze["candidates"][0]["threshold"] = float("nan")
    external.write_json(path, freeze)
    digest = external.digest(path.read_bytes())
    digest = None if kind == "missing_hash" else "wrong" if kind == "wrong_hash" else digest
    with pytest.raises(ValueError):
        external.export_partition("locked_confirmation", path, digest)
    assert not (base / "confirmation_unlock.json").exists()


def test_masked_confirmation_export_never_decodes_excluded_labels(sealed_fixture, monkeypatch):
    base, path, freeze = sealed_fixture
    external.write_json(path, freeze)
    original = json.loads

    def guarded(value, *args, **kwargs):
        assert "DO_NOT_DECODE" not in value
        return original(value, *args, **kwargs)

    monkeypatch.setattr(external.json, "loads", guarded)
    external.export_partition("locked_confirmation", path, external.digest(path.read_bytes()))
    rows = [json.loads(line) for line in (base / "confirmation_records.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["original_source_id"] == "eligible"
    assert rows[0]["label"] == 1
    assert rows[0]["span_labels_available"] is False


def test_exclusive_write_preserves_existing_artifact(tmp_path):
    path = tmp_path / "immutable.json"
    external.write_json(path, {"frozen": 1})
    with pytest.raises(FileExistsError):
        external.write_json(path, {"frozen": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"frozen": 1}
