"""Seal and verify the completed packet without rewriting historical artifacts."""

import argparse
import zipfile

from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    read_json,
    read_rows,
    write_json,
)

BASE = ROOT / "models/discovery_loop_20261026"
REPORT = ROOT / "reports/discovery_loop_20261026"
MANIFEST = BASE / "artifact_manifest.json"
SNAPSHOT = ROOT / "research_snapshots/discovery_loop_20261026_source.zip"
DOC_NAMES = [
    "RESEARCH_DISCOVERY_LOOP.md",
    "R11_PLUS_RESEARCH_LOG.md",
    "DISCOVERY_CYCLE2_PREREG.md",
    "DISCOVERY_CYCLE3_PREREG.md",
    "DISCOVERY_CYCLE4_PREREG.md",
    "DISCOVERY_CONFIRMATION_PREREG.md",
    "RESEARCH_LOOP_VERDICT.md",
    "DISCOVERY_REPRODUCE.md",
    "PROGRESS.md",
    "EVALUATION_PROTOCOL.md",
    "DECISIONS.md",
]


def verify_confirmation():
    freeze = read_json(BASE / "external_final_freeze.json")
    assert digest(BASE / "external_final_freeze.json") == (
        "3c31017765ddfcfd7b9aa482e3de9303319dcdb6b112a3cb261410bbc0423158"
    )
    for candidate in freeze["candidates"]:
        for item in candidate["artifacts"]:
            assert digest(ROOT / item["path"]) == item["sha256"], item["path"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["seal", "verify"])
    args = parser.parse_args()
    verify_confirmation()
    if args.command == "verify":
        manifest = read_json(MANIFEST)
        for name, item in manifest["files"].items():
            assert digest(ROOT / name) == item["sha256"], name
        print(
            {
                "verified_files": len(manifest["files"]),
                "snapshot_sha256": digest(SNAPSHOT),
                "confirmation_freeze_intact": True,
            }
        )
        return
    assert not MANIFEST.exists() and not SNAPSHOT.exists()
    source = sorted((ROOT / "src").rglob("*.py"))
    source += sorted((ROOT / "scripts").glob("research*.py"))
    source += sorted((ROOT / "scripts").glob("discovery*.py"))
    source += sorted((ROOT / "tests").glob("test_discovery*.py"))
    source += [ROOT / "docs" / name for name in DOC_NAMES]
    source += [
        ROOT / name
        for name in (
            "AGENTS.md",
            "PROJECT_SPEC.md",
            "FIRST_TASK.md",
            "pyproject.toml",
            ".gitignore",
            "RESEARCH_NOTES.md",
        )
    ]
    source += sorted((ROOT / "configs").glob("research*.txt"))
    source += [ROOT / "configs/checkpoints.json", ROOT / "configs/nltk-data.json"]
    SNAPSHOT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(SNAPSHOT, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(source)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    cache = read_rows(ROOT / "predictions/discovery_loop_20261026/factcg_pair_cache.jsonl")
    assert len({r["key"] for r in cache}) == len(cache)
    assert all(r["truncated_tokens"] == 0 for r in cache)
    final = {
        "snapshot": SNAPSHOT.relative_to(ROOT).as_posix(),
        "snapshot_sha256": digest(SNAPSHOT),
        "source_members": len(set(source)),
        "unique_factcg_pairs": len(cache),
        "truncated_factcg_pairs": 0,
        "max_factcg_input_tokens": max(r["input_tokens"] for r in cache),
        "confirmation_freeze_intact": True,
        "new_fits": 0,
        "unfinished_jobs": 0,
        "gpu_budget": read_json(REPORT / "resource_ledger.json")["total_charged_gpu_minutes"],
    }
    write_json(REPORT / "integrity_final.json", final)
    files = set(source + [SNAPSHOT])
    for folder in (BASE, REPORT, ROOT / "predictions/discovery_loop_20261026"):
        files.update(
            p
            for p in folder.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in (".pyc", ".part")
            and p != MANIFEST
        )
    write_json(
        MANIFEST,
        {
            "files": {
                p.relative_to(ROOT).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                for p in sorted(files)
            },
            "policy": "Existing immutable artifacts preserved; no models/texts in source ZIP; "
            "all weights/data/cache remain ignored. Verify separately before starting next packet.",
        },
    )
    print({**final, "artifact_files": len(files)}, flush=True)


if __name__ == "__main__":
    main()
