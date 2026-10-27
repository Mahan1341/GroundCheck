"""Fetch I/O is tested with in-memory HTTP responses, never internet."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_ragtruth",
    Path(__file__).parents[1] / "scripts" / "fetch_ragtruth.py",
)
fetch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_module)


def test_pinned_fetch_hashes_and_manifest(tmp_path, monkeypatch):
    urls = []

    def fake_open(url, timeout):
        urls.append(url)
        return io.BytesIO(b"synthetic download bytes\n")

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", fake_open)
    manifest = fetch_module.fetch(tmp_path)
    assert len(urls) == 4
    assert all(fetch_module.REVISION in url for url in urls)
    assert (tmp_path / "manifest.json").exists()
    for name, info in manifest["files"].items():
        assert info["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.part"))


def test_failed_download_keeps_existing_file(tmp_path, monkeypatch):
    original = tmp_path / "response.jsonl"
    original.write_bytes(b"existing")

    def fail(url, timeout):
        raise OSError("simulated network failure")

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="simulated"):
        fetch_module.fetch(tmp_path)
    assert original.read_bytes() == b"existing"
    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob("*.part"))


def test_mutable_revision_rejected(tmp_path):
    with pytest.raises(ValueError, match="full lowercase Git commit SHA"):
        fetch_module.fetch(tmp_path, "main")
