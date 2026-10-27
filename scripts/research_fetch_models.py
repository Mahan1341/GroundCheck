"""Download only pinned research artifacts; never executes downloaded custom code."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from research_range_download import download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "keys",
        nargs="+",
        choices=["nli", "minicheck", "hhem", "hhem_foundation", "backbone", "fallback"],
    )
    args = parser.parse_args()
    registry = json.loads((ROOT / "configs/checkpoints.json").read_text())
    for key in args.keys:
        info = registry[key]
        target = ROOT / "models/pretrained" / key
        target.mkdir(parents=True, exist_ok=True)
        manifest = {"repo_id": info["repo_id"], "revision": info["revision"], "files": {}}
        for filename in info["files"]:
            if "/" in filename or not filename.endswith((".json", ".model", ".py", ".md")):
                if (
                    filename
                    != (
                        "model.safetensors"
                        if "model.safetensors" in info["files"]
                        else "pytorch_model.bin"
                    )
                    or key == "hhem_foundation"
                ):
                    continue
            path = target / filename
            url = f"https://huggingface.co/{info['repo_id']}/resolve/{info['revision']}/{filename}"
            if not path.exists():
                print(f"Downloading {key}/{filename}", flush=True)
                if filename.endswith((".safetensors", ".bin")):
                    download(url, path)
                    manifest["files"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
                    continue
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(url, timeout=120) as response:
                            with path.with_suffix(path.suffix + ".part").open("wb") as stream:
                                total = 0
                                while block := response.read(8 * 1024 * 1024):
                                    stream.write(block)
                                    total += len(block)
                                    if total % (128 * 1024 * 1024) == 0:
                                        print(
                                            f"{key}/{filename}: {total // 1048576} MiB", flush=True
                                        )
                        path.with_suffix(path.suffix + ".part").replace(path)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2)
            manifest["files"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        (target / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Complete: {key} {info['revision']}", flush=True)


if __name__ == "__main__":
    main()
