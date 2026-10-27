"""Verify frozen executed versions and seal the follow-up without opening closed labels."""

import argparse
import ast
import hashlib
import zipfile

from scope_analysis import BASE, PRED, REPORT, ROOT, read, rows, write

from grounded_hallucination_lab.research.protocol import digest

SNAPSHOT = ROOT / "research_snapshots/scope_diagnosis_20261026_source.zip"
MANIFEST = BASE / "artifact_manifest.json"
OLD_SNAPSHOT = ROOT / "research_snapshots/discovery_loop_20261026_source.zip"
LIVING = {"docs/PROGRESS.md", "docs/EVALUATION_PROTOCOL.md", "docs/DECISIONS.md",
          "RESEARCH_NOTES.md"}


def verify_old():
    """Only living documents may have a newer version; historical bytes remain verified."""
    assert digest(OLD_SNAPSHOT) == (
        "76ede8539cd9c45de77b5a2b055a3771c4341a1c50ddc7e86ddf94effe895448")
    manifest = read(ROOT / "models/discovery_loop_20261026/artifact_manifest.json")
    historical = []
    with zipfile.ZipFile(OLD_SNAPSHOT) as archive:
        for name, item in manifest["files"].items():
            name = name.replace("\\", "/")
            if digest(ROOT / name) == item["sha256"]:
                continue
            assert name in LIVING, f"Immutable historical artifact changed: {name}"
            assert hashlib.sha256(archive.read(name)).hexdigest() == item["sha256"]
            historical.append(name)
    return {"old_artifacts": len(manifest["files"]), "living_docs_in_old_snapshot": historical,
            "closed_artifacts": "hash-only; no dataset/prediction deserialization"}


def verify_freezes():
    initial = read(BASE / "freeze.json")
    for group in ("input_files", "source"):
        for name, sha in initial[group].items():
            assert digest(ROOT / name) == sha, name
    assert digest(BASE / "pairs.json") == initial["pairs_sha256"]
    assert digest(ROOT / "predictions/discovery_loop_20261026/factcg_pair_cache.jsonl") == (
        initial["old_cache_sha256"])
    for name, sha in read(BASE / "analysis_freeze.json")["files"].items():
        assert digest(ROOT / name) == sha, name
    formatting = read(BASE / "pilot_source_formatting.json")
    for key in ("executed", "current"):
        assert digest(ROOT / formatting[f"{key}_source"]) == formatting[f"{key}_sha256"]
    original = (ROOT / formatting["executed_source"]).read_text(encoding="utf-8")
    current = (ROOT / formatting["current_source"]).read_text(encoding="utf-8")
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(current))
    for name, sha in read(BASE / "pilot_freeze.json")["files"].items():
        path = (BASE / "pilot_source/scope_pilot.py"
                if name == "scripts/scope_pilot.py" else ROOT / name)
        assert digest(path) == sha, name
    cache = rows(PRED / "pair_cache.jsonl")  # New allowlisted cache only.
    assert len(cache) == len({r["key"] for r in cache}) == 7367
    assert {r["key"] for r in cache} == set(read(BASE / "pairs.json"))
    assert all(r["truncated_tokens"] == 0 for r in cache)
    for dataset in ("natural", "r1", "r6"):
        a, b = rows(PRED / f"{dataset}_seg_filtered.jsonl"), rows(PRED / f"pilot_{dataset}.jsonl")
        assert [(x["id"], x["score"]) for x in a] == [(x["id"], x["score"]) for x in b]
    return {"pair_count": len(cache), "max_input_tokens": max(r["input_tokens"] for r in cache),
            "truncated_pairs": 0, "pilot_response_scores_unchanged": True,
            "pilot_executed_source": "original snapshot; current AST identical after formatting"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("seal", "verify"))
    args = parser.parse_args()
    verification = {**verify_old(), **verify_freezes()}
    if args.command == "verify":
        manifest = read(MANIFEST)
        for name, spec in manifest["files"].items():
            assert digest(ROOT / name) == spec["sha256"], name
        print({**verification, "verified_files": len(manifest["files"]),
               "source_snapshot_sha256": digest(SNAPSHOT)})
        return
    assert not MANIFEST.exists() and not SNAPSHOT.exists()
    source = list((ROOT / "src").rglob("*.py"))
    source += list((ROOT / "scripts").glob("*.py"))
    source += list((ROOT / "tests").glob("test_scope*.py"))
    source += list((ROOT / "docs").glob("SCOPE_*.md"))
    source += [ROOT / name for name in LIVING]
    source += [ROOT / name for name in ("AGENTS.md", "PROJECT_SPEC.md", "FIRST_TASK.md",
                                       "pyproject.toml", ".gitignore")]
    source += list((ROOT / "configs").glob("*.txt"))
    with zipfile.ZipFile(SNAPSHOT, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(source)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    write(REPORT / "integrity.json", {**verification,
                                     "source_snapshot_sha256": digest(SNAPSHOT),
                                     "source_files": len(set(source)), "new_training_runs": 0})
    files = set(source + [SNAPSHOT])
    for folder in (BASE, PRED, REPORT):
        files.update(p for p in folder.rglob("*") if p.is_file() and p != MANIFEST
                     and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".part"))
    write(MANIFEST, {"files": {p.relative_to(ROOT).as_posix(): {
        "sha256": digest(p), "bytes": p.stat().st_size} for p in sorted(files)}})
    print({**verification, "artifact_files": len(files),
           "source_snapshot_sha256": digest(SNAPSHOT)})


if __name__ == "__main__":
    main()
