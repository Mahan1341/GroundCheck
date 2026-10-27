"""Explicit, optional network entrypoint. Never called by tests or baseline CLI."""

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"


def fetch(destination: Path, revision: str = REVISION) -> dict:
    """Download a fixed Git snapshot, atomically per file, with SHA256 manifest."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase Git commit SHA")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repository": "https://github.com/ParticleMedia/RAGTruth",
        "revision": revision,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "files": {},
    }
    for name in ("response.jsonl", "source_info.jsonl", "README.md", "LICENSE"):
        remote = f"dataset/{name}" if name.endswith(".jsonl") else name
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{revision}/{remote}"
        target = destination / name
        partial = target.with_suffix(target.suffix + ".part")
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    out.write(chunk)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        manifest["files"][name] = {
            "url": url,
            "sha256": digest.hexdigest(),
            "bytes": target.stat().st_size,
        }
    manifest_path = destination / "manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/ragtruth"))
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    print(json.dumps(fetch(args.output_dir, args.revision), indent=2))
