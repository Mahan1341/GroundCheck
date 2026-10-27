"""Group isolation, inference allowlist, hashes and research artifact I/O."""

import hashlib
import importlib.metadata
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from grounded_hallucination_lab.cli import _provenance
from grounded_hallucination_lab.data.ragtruth import Record, load_ragtruth
from grounded_hallucination_lab.data.splits import Partitions, grouped_split

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "models/phase2b"
PRED = ROOT / "predictions/phase2b"
REPORT = ROOT / "reports/phase2b"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"preserving existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def ordered_sources(records: list[Record], namespace: str) -> list[str]:
    if any(r.split == "test" or not r.labels_available for r in records):
        raise ValueError("official test cannot enter research folds")
    return sorted(
        {r.source_id for r in records},
        key=lambda s: (hashlib.sha256(f"{namespace}:42:{s}".encode()).hexdigest(), s),
    )


def inner_split(train: list[Record]) -> tuple[list[Record], list[Record]]:
    sources = ordered_sources(train, "inner")
    if len(sources) < 2:
        raise ValueError("inner split needs two sources")
    val = set(sources[: min(len(sources) - 1, math.ceil(0.2 * len(sources)))])
    return ([r for r in train if r.source_id not in val], [r for r in train if r.source_id in val])


def cv_split(train: list[Record], fold: int) -> tuple[list[Record], list[Record]]:
    if fold not in (0, 1, 2):
        raise ValueError("three folds only")
    val = set(ordered_sources(train, "cv")[fold::3])
    return ([r for r in train if r.source_id not in val], [r for r in train if r.source_id in val])


def load_parts() -> Partitions:
    directory = ROOT / "data/raw/ragtruth"
    _provenance(directory)  # Check original manifest before reading.
    parts = grouped_split(load_ragtruth(directory).records)
    ref = read_json(ROOT / "tests/fixtures/phase1_reference.json")
    if sorted(r.id for r in parts.dev) != ref["dev"]["response_ids"]:
        raise ValueError("historical dev membership changed")
    return parts


def partition(parts: Partitions, name: str) -> list[Record]:
    if name == "test":
        raise ValueError("test is not authorized")
    fit, val = inner_split(parts.train)
    choices = {
        "inner_train": fit,
        "inner_val": val,
        "train": parts.train,
        "calibration": parts.calibration,
        "dev": parts.dev,
    }
    if name not in choices:
        raise ValueError(name)
    return sorted(choices[name], key=lambda r: r.id)


@dataclass(frozen=True)
class InferenceInput:
    id: str
    source_id: str
    question: str
    evidence: str
    response: str

    @classmethod
    def from_record(cls, record: Record) -> "InferenceInput":
        if record.split == "test" or not record.labels_available:
            raise ValueError("test inference is not authorized")
        return cls(record.id, record.source_id, record.question, record.evidence, record.response)

    def lexical_record(self) -> Record:
        return Record(
            self.id,
            self.source_id,
            "QA",
            "",
            "train",
            "withheld",
            self.response,
            self.question,
            self.evidence,
            (),
            {},
            {},
        )


def provenance() -> dict:
    files = sorted((ROOT / "src").rglob("*.py"))
    files += sorted((ROOT / "scripts").glob("research*.py"))
    files += [
        ROOT / "EXPERIMENT_PLAN.md",
        ROOT / "EXPERIMENT_PLAN_RESUME.md",
        ROOT / "EXPERIMENT_R1_CF.md",
        ROOT / "EXPERIMENT_R1_DIAGNOSTIC.md",
        ROOT / "configs/counterfactual_validation_acceptance.json",
        ROOT / "configs/counterfactual_acceptance.json",
        ROOT / "configs/checkpoints.json",
        ROOT / "configs/research-requirements.txt",
        ROOT / "configs/research-lock.txt",
        ROOT / "configs/nltk-data.json",
    ]
    packages = {}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "tokenizers",
        "sentencepiece",
        "huggingface-hub",
        "nltk",
        "safetensors",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "code": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "packages": packages,
        "dataset": _provenance(ROOT / "data/raw/ragtruth")["sha256"],
    }


def now() -> str:
    return datetime.now(UTC).isoformat()


def preserve_code() -> dict:
    """Content-addressed source snapshot; never include data, weights or predictions."""
    state = provenance()
    key = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    target = OUT / "code_snapshots" / f"{key}.zip"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as archive:
            for name, expected in state["code"].items():
                content = (ROOT / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("source changed while snapshotting")
                archive.writestr(name, content)
            archive.writestr("provenance.json", json.dumps(state, indent=2))
    return state
