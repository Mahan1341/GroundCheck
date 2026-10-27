"""Acquire pinned public FactCG artifacts without importing or executing author code.

Weights are downloaded once, directly into the ignored experiment directory.
Resumable HTTP ranges share one .part file; no Hugging Face cache copy is made.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/discovery_loop_20261026"
MODEL_ID = "yaxili96/FactCG-DeBERTa-v3-Large"
MODEL_REVISION = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
AUTHOR_REPO = "derenlei/FactCG"
AUTHOR_COMMIT = "41f185413312ed9b3904ec03963fe899693498e0"
MODEL_FILES = (
    "README.md", "added_tokens.json", "config.json", "special_tokens_map.json",
    "spm.model", "tokenizer.json", "tokenizer_config.json", "model.safetensors",
)
TRAINING_FILES = (
    "data/training_stage1/CG2C_hotpot_qa_rbt_mnli_failed.json",
    "data/training_stage1/CG2C_musique_minhop3_rbt_mnli_failed.json",
    "data/training_stage1/anli_minicheck.json",
    "data/training_stage1/minicheck_c2d.json",
    "data/training_stage2/CG2C_doc.json",
    "data/training_stage2/minicheck_d2c.json",
)
AUTHOR_FILES = (
    "LICENSE", "README.md", "train.sh", "train.py", "benchmark.py",
    "factcg/__init__.py", "factcg/utils.py", "factcg/inference.py",
    "factcg/factcg_score.py", "factcg/grounding_model.py", "factcg/dataloader.py",
)
CHUNK_BYTES = 16 * 1024 * 1024


def json_write(path: Path, value: Any) -> None:
    """Atomically replace acquisition state, never a historical model artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def request(url: str, headers: dict[str, str] | None = None):
    """Open an unauthenticated public URL with a bounded socket timeout."""
    values = {"User-Agent": "GroundCheck-pinned-research-acquisition"}
    values.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=values), timeout=90)


def read_json(url: str) -> dict[str, Any]:
    with request(url) as response:
        return json.load(response)


def hashes(path: Path) -> tuple[str, str]:
    """Compute SHA256 and Git's blob SHA1 without loading a weight file in RAM."""
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha256.update(block)
            git_sha1.update(block)
    return sha256.hexdigest(), git_sha1.hexdigest()


def verify(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on any mismatch against pinned upstream metadata."""
    if path.stat().st_size != metadata["size"]:
        raise ValueError(f"Size mismatch: {path}")
    sha256, blob_sha1 = hashes(path)
    expected = metadata.get("lfs", {}).get("sha256")
    if expected:
        if sha256 != expected:
            raise ValueError(f"Upstream LFS SHA256 mismatch: {path}")
    elif blob_sha1 != metadata.get("blobId", metadata.get("sha")):
        raise ValueError(f"Upstream Git blob SHA1 mismatch: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256, "git_blob_sha1": blob_sha1}


def range_download(url: str, destination: Path, size: int) -> None:
    """Resume four bounded workers into a single local file, without chunk duplicates."""
    partial = destination.with_suffix(destination.suffix + ".part")
    progress = destination.with_suffix(destination.suffix + ".progress.json")
    state: dict[str, Any] = {"url": url, "size": size, "chunk_bytes": CHUNK_BYTES, "done": []}
    if progress.exists():
        state = json.loads(progress.read_text(encoding="utf-8"))
        if (state["url"], state["size"], state["chunk_bytes"]) != (url, size, CHUNK_BYTES):
            raise ValueError("Refusing to resume mismatched acquisition state")
        if not partial.exists() or partial.stat().st_size != size:
            raise ValueError("Partial file does not match saved acquisition state")
    else:
        if partial.exists():
            raise ValueError("Partial file exists without a valid acquisition manifest")
        with partial.open("xb") as stream:
            stream.truncate(size)
        json_write(progress, state)
    completed = set(state["done"])

    def fetch(start: int) -> int:
        end = min(size, start + CHUNK_BYTES) - 1
        for attempt in range(4):
            try:
                # Unique query protects byte ranges from overly aggressive intermediary caches.
                ranged_url = f"{url}?download=true&groundcheck_range={start}"
                with request(ranged_url, {"Range": f"bytes={start}-{end}"}) as response:
                    wanted = f"bytes {start}-{end}/{size}"
                    if response.headers.get("Content-Range") != wanted:
                        raise ValueError(f"Range not honored at byte {start}")
                    content = response.read(end - start + 2)
                if len(content) != end - start + 1:
                    raise ValueError(f"Incomplete range at byte {start}")
                with partial.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                return start
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError("Unreachable")

    offsets = list(range(0, size, CHUNK_BYTES))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, start) for start in offsets if start not in completed]
        for future in concurrent.futures.as_completed(futures):
            completed.add(future.result())
            state["done"] = sorted(completed)
            json_write(progress, state)
            if len(completed) % 8 == 0 or len(completed) == len(offsets):
                print(f"WEIGHTS {len(completed)}/{len(offsets)} ranges", flush=True)
    partial.replace(destination)
    progress.unlink()


def acquire(url: str, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Fetch an absent artifact; an existing artifact must verify and is never overwritten."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if metadata["size"] > 100 * 1024 * 1024:
            range_download(url, destination, metadata["size"])
        else:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with request(url) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            verify(temporary, metadata)
            temporary.replace(destination)
    result = verify(destination, metadata)
    print(f"VERIFIED {destination.relative_to(BASE)} {result['bytes']} bytes", flush=True)
    return {"url": url, "path": str(destination.relative_to(BASE)), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()
    started = time.time()
    model = read_json(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true")
    author = read_json(
        f"https://api.github.com/repos/{AUTHOR_REPO}/git/trees/{AUTHOR_COMMIT}?recursive=1"
    )
    if model["sha"] != MODEL_REVISION or author["sha"] != AUTHOR_COMMIT:
        raise ValueError("Resolved upstream revision differs from pinned revision")
    manifest: dict[str, Any] = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "author_repo": AUTHOR_REPO, "author_commit": AUTHOR_COMMIT,
        "license": model["cardData"].get("license"),
        "custom_remote_code_executed": False, "gpu_used": False,
        "script_sha256": hashes(Path(__file__))[0], "files": {},
    }
    manifest_path = BASE / "factcg/acquisition_manifest.json"
    model_files = {item["rfilename"]: item for item in model["siblings"]}
    author_files = {item["path"]: item for item in author["tree"] if item["type"] == "blob"}
    json_write(BASE / "factcg/provenance/hf_metadata.json", model)
    json_write(BASE / "factcg/provenance/author_tree.json", author)
    for name in TRAINING_FILES + AUTHOR_FILES:
        target = (
            BASE / "factcg_training" / name.removeprefix("data/")
            if name in TRAINING_FILES else BASE / "factcg/provenance/author" / name
        )
        url = f"https://raw.githubusercontent.com/{AUTHOR_REPO}/{AUTHOR_COMMIT}/{name}"
        manifest["files"][str(target.relative_to(BASE))] = acquire(url, target, author_files[name])
        json_write(manifest_path, manifest)
    for name in MODEL_FILES:
        if name == "model.safetensors" and args.skip_weights:
            continue
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
        target = BASE / "factcg" / name
        manifest["files"][str(target.relative_to(BASE))] = acquire(url, target, model_files[name])
        json_write(manifest_path, manifest)
    manifest["complete"] = not args.skip_weights
    manifest["elapsed_seconds"] = time.time() - started
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    json_write(manifest_path, manifest)
    summary = {key: manifest[key] for key in ("complete", "total_bytes", "elapsed_seconds")}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
