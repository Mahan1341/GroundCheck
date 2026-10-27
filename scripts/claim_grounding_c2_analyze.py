"""Compatibility entrypoint. Implementation lives in experiments/scripts/claim_grounding_c2_analyze.py."""
from pathlib import Path
import sys

_LEGACY_FILE = Path(__file__).resolve()
_IMPL_DIR = _LEGACY_FILE.parents[1] / "experiments" / "scripts"
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))
_IMPL = _IMPL_DIR / "claim_grounding_c2_analyze.py"
exec(compile(_IMPL.read_bytes(), str(_LEGACY_FILE), "exec"), globals())
