"""Bounded parallel HTTP ranges for large files when the host proxy stalls full transfers."""

import argparse
import concurrent.futures
import hashlib
import re
import time
import urllib.request
from pathlib import Path


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if not match:
            raise ValueError("server did not honor range request")
        total = int(match[1])
        response.read()
    folder = destination.with_suffix(destination.suffix + ".ranges")
    folder.mkdir(exist_ok=True)
    chunk = 8 * 1024 * 1024

    def fetch(start):
        end = min(total, start + chunk) - 1
        path = folder / str(start)
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise ValueError("incorrect byte range")
                    value = response.read()
                if len(value) != end - start + 1:
                    raise ValueError("incomplete chunk")
                path.write_bytes(value)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)

    offsets = list(range(0, total, chunk))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(fetch, offsets), 1):
            if i % 16 == 0:
                print(f"{destination.name}: {i}/{len(offsets)} chunks", flush=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    sha = hashlib.sha256()
    with temporary.open("wb") as stream:
        for start in offsets:
            value = (folder / str(start)).read_bytes()
            sha.update(value)
            stream.write(value)
    temporary.replace(destination)
    # Retain chunks for recovery; caller may remove only this verified workspace directory later.
    print(f"COMPLETE {destination} bytes={total} sha256={sha.hexdigest()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    download(args.url, args.destination)
