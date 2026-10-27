"""Record a separately trained checkpoint; does not claim it reproduces reference quality."""

import argparse
import json
from pathlib import Path

from grounded_hallucination_lab.inference import file_hash, load_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("threshold must be in (0, 1], selected on permitted development sources")
    profile = load_profile()
    profile.update(model_id="groundcheck-retrained-b7", response_threshold=args.threshold,
                   evaluation="User-trained artifact; reference metrics do not apply.")
    profile["files"] = {name: file_hash(args.checkpoint / name) for name in profile["files"]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
