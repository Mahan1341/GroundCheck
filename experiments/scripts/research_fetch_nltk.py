"""Restore precisely hashed Punkt tables, without running package download hooks."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/nltk-data.json").read_text())
destination = root / "models/nltk_data/tokenizers"
archive_path = destination / "punkt_tab.zip"
if archive_path.exists():
    content = archive_path.read_bytes()
else:
    url = (
        f"https://raw.githubusercontent.com/{config['repository']}/"
        f"{config['revision']}/{config['path']}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
if hashlib.sha256(content).hexdigest() != config["sha256"]:
    raise ValueError("Punkt archive differs from the recorded experiment")
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(content)) as archive:
    for name in archive.namelist():
        if not (destination / name).resolve().is_relative_to(destination.resolve()):
            raise ValueError("unsafe archive path")
    archive.extractall(destination)
if not archive_path.exists():
    archive_path.write_bytes(content)
print("Verified and restored pinned Punkt tables", config["sha256"])
