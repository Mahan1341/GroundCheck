"""Compatibility entrypoint for research experiments."""

import sys
from pathlib import Path

_LEGACY_FILE = Path(__file__).resolve()
_IMPL_DIR = _LEGACY_FILE.parents[1] / "experiments" / "scripts"
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))
_IMPL = _IMPL_DIR / "claim_grounding_replay.py"
exec(compile(_IMPL.read_bytes(), str(_LEGACY_FILE), "exec"), globals())
