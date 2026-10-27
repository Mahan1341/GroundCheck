"""Compatibility entrypoint. Implementation lives in experiments/scripts/research_r6.py."""
from pathlib import Path
import sys

_LEGACY_FILE = Path(__file__).resolve()
_IMPL_DIR = _LEGACY_FILE.parents[1] / "experiments" / "scripts"
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))
_IMPL = _IMPL_DIR / "research_r6.py"
exec(compile(_IMPL.read_bytes(), str(_LEGACY_FILE), "exec"), globals())
