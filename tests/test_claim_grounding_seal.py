"""Offline guards: historical versions are narrow; closed artifacts remain opaque."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/scripts/claim_grounding_seal.py"
SPEC = importlib.util.spec_from_file_location("claim_grounding_seal", SCRIPT)
seal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seal)


def source(namespace):
    return (
        "BASE, PRED, REPORT = "
        f"run.BASE / '{namespace}', run.PRED / '{namespace}', run.REPORT / '{namespace}'\n"
        "def prepare():\n    return 12 * (3 - 1)\n"
    )


def test_only_exact_namespace_and_ast_preserving_formatting_allowed():
    historical = source("c2_bounded")
    current = source("c2_ready").replace("12 * (3 - 1)", "(12\n            * (3 - 1))")
    seal.verify_prepare_equivalence(historical, current)
    with pytest.raises(ValueError, match="beyond formatting"):
        seal.verify_prepare_equivalence(historical, current.replace("12", "13"))
    with pytest.raises(ValueError, match="namespace assignment"):
        seal.verify_prepare_equivalence(historical, source("c2_other"))


def test_namespace_literals_elsewhere_are_not_rewritten():
    historical = source("c2_bounded") + "guard = 'c2_bounded'\n"
    current = source("c2_ready") + "guard = 'c2_ready'\n"
    with pytest.raises(ValueError, match="beyond formatting"):
        seal.verify_prepare_equivalence(historical, current)


def test_mapped_closed_data_and_failed_json_are_hash_only(tmp_path):
    closed = tmp_path / "confirmation_labels.jsonl"
    failed = tmp_path / "failed_resource.json"
    closed.write_bytes(b'\xff\x00never deserialize closed values\n')
    failed.write_bytes(b'{"passed":')
    files = {p.name: {"sha256": seal.digest(p)} for p in (closed, failed)}
    assert seal.verify_file_map(tmp_path, files) == 2
    assert seal.count_lines(closed) == 1


def legacy_fixture(root):
    namespace = "old_fixture"
    manifest = root / "models" / namespace / "artifact_manifest.json"
    snapshot = root / "research_snapshots" / f"{namespace}_source.zip"
    document = root / "docs/PROGRESS.md"
    prediction = root / "predictions/confirmation.jsonl"
    for path in (manifest, snapshot, document, prediction):
        path.parent.mkdir(parents=True, exist_ok=True)
    document.write_bytes(b"historical progress")
    prediction.write_bytes(b"\xff\x00sealed predictions are not valid JSON")
    files = {p.relative_to(root).as_posix(): {"sha256": seal.digest(p)}
             for p in (document, prediction)}
    with zipfile.ZipFile(snapshot, "x") as archive:
        archive.write(document, "docs/PROGRESS.md")
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    return namespace, seal.digest(manifest), seal.digest(snapshot), document, prediction


def test_only_four_living_docs_can_use_pinned_historical_zip(tmp_path):
    namespace, manifest_sha, snapshot_sha, document, prediction = legacy_fixture(tmp_path)
    document.write_bytes(b"new progress")
    result = seal.verify_legacy_manifest(tmp_path, namespace, 2, manifest_sha, snapshot_sha)
    assert result["historical_living_documents"] == ["docs/PROGRESS.md"]
    prediction.write_bytes(b"changed closed prediction")
    with pytest.raises(ValueError, match="Immutable historical artifact"):
        seal.verify_legacy_manifest(tmp_path, namespace, 2, manifest_sha, snapshot_sha)


def test_modified_manifest_or_archive_fails_before_historical_exception(tmp_path):
    namespace, manifest_sha, snapshot_sha, _, _ = legacy_fixture(tmp_path)
    path = tmp_path / "models" / namespace / "artifact_manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Hash mismatch"):
        seal.verify_legacy_manifest(tmp_path, namespace, 2, manifest_sha, snapshot_sha)


def test_path_aliases_and_workspace_escape_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="outside workspace"):
        seal.checked_path(tmp_path, "../closed.json")
    with pytest.raises(ValueError, match="Ambiguous paths"):
        seal.verify_file_map(tmp_path, {"docs/file": "a", "docs\\file": "a"})
    with pytest.raises(ValueError, match="unused historical replacement"):
        seal.verify_file_map(tmp_path, {}, replacements={"unexpected": tmp_path / "file"})


def test_exclusive_write_cannot_replace_snapshot_metadata(tmp_path):
    path = tmp_path / "manifest.json"
    seal.write_exclusive(path, {"frozen": True})
    old = path.read_bytes()
    with pytest.raises(FileExistsError):
        seal.write_exclusive(path, {"frozen": False})
    assert path.read_bytes() == old


@pytest.mark.parametrize("hierarchical", [True, False])
def test_source_snapshots_support_exact_hierarchy_and_legacy_flat_layout(tmp_path, hierarchical):
    folder = Path("saved")
    relative = Path("scripts/metrics.py") if hierarchical else Path("metrics.py")
    path = tmp_path / folder / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"expected source")
    files = {"scripts\\metrics.py": seal.digest(path), "src/metrics.py": "wrong hash"}
    assert seal.verify_snapshot_copies(tmp_path, folder, files) == 1
    path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Unverified saved source"):
        seal.verify_snapshot_copies(tmp_path, folder, files)
    path.write_bytes(b"expected source")
    if hierarchical:
        # A same-basename file cannot borrow a different source path's valid hash.
        with pytest.raises(ValueError, match="Unverified saved source"):
            seal.verify_snapshot_copies(tmp_path, folder, {"other/metrics.py": seal.digest(path)})


def test_resume_preserves_raw_crlf_prefix_and_only_decodes_keys(tmp_path):
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    # Deliberately invalid JSON values prove that model values are not decoded.
    prefix = b'{"key": "' + a.encode() + b'", "opaque": \xff}\r\n'
    suffix = b'{"key": "' + b.encode() + b'", "opaque": \xfe}\r\n'
    path = tmp_path / "cache.jsonl"
    path.write_bytes(prefix + suffix)
    freeze = {"cached_pairs": 1, "pending_keys": [b],
              "cache_prefix_sha256": hashlib.sha256(prefix).hexdigest()}
    result = seal.verify_resume_cache(path, freeze, {a, b})
    assert result["complete_unique_keys"] == 2
    assert seal.verify_resume_cache(path, freeze, {a, b}, require_complete=False)[
        "complete_unique_keys"] is None
    path.write_bytes(prefix.replace(b"\r\n", b"\n") + suffix)
    with pytest.raises(ValueError, match="prefix changed"):
        seal.verify_resume_cache(path, freeze, {a, b})
    path.write_bytes(prefix + prefix)
    with pytest.raises(ValueError, match="Duplicate"):
        seal.verify_resume_cache(path, freeze, {a, b})
    path.write_bytes(prefix)
    with pytest.raises(ValueError, match="exactly the frozen pending"):
        seal.verify_resume_cache(path, freeze, {a, b})
    with pytest.raises(ValueError, match="disagree with original prefix"):
        seal.verify_resume_cache(path, freeze, {a, c}, require_complete=False)
