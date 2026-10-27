"""Pinned local POS resource for R8; fail closed and never download during inference."""

import hashlib
import json
from functools import lru_cache

from grounded_hallucination_lab.research.protocol import ROOT


@lru_cache(maxsize=1)
def tagger():
    import nltk
    from nltk.tag import PerceptronTagger

    spec = json.loads((ROOT / "configs/r8-nlp.json").read_text(encoding="utf-8"))
    for name, expected in spec["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("R8 POS resource hash mismatch")
    nltk.data.path.insert(0, str(ROOT / "models/nltk_data"))
    return PerceptronTagger(lang="eng")
