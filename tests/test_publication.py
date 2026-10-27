"""Guard against excluding runtime library modules along with private dataset folders."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_gitignore_keeps_library_data_but_excludes_private_artifacts():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists() or not shutil.which("git"):
        pytest.skip("Git checkout required for publication policy check")

    def ignored(path):
        return subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--no-index", "-q", path],
            check=False,
        ).returncode == 0

    assert not ignored("src/grounded_hallucination_lab/data/ragtruth.py")
    for path in ("data/source_info.jsonl", "models/checkpoint.bin", "predictions/cases.jsonl"):
        assert ignored(path)
    assert not ignored("examples/input.jsonl")
