"""Hash-only historical verification and exclusive sealing of the claim-grounding packet.

No model imports, prediction deserialization, or closed-label access. The sole source
version substitution is the explicitly preserved, never GPU-executed C2 preparation.
"""

import argparse
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "claim_grounding_20261026"
BASE = Path("models") / NAMESPACE
PRED = Path("predictions") / NAMESPACE
REPORT = Path("reports") / NAMESPACE
MANIFEST = BASE / "artifact_manifest.json"
SNAPSHOT = Path("research_snapshots") / f"{NAMESPACE}_source.zip"
LIVING = frozenset({"docs/PROGRESS.md", "docs/EVALUATION_PROTOCOL.md",
                    "docs/DECISIONS.md", "RESEARCH_NOTES.md"})
LEGACY = (
    ("discovery_loop_20261026", 291,
     "2c410bc5c514b11d4210e9ec7b05ffad309f8064080a866616019803a2302437",
     "76ede8539cd9c45de77b5a2b055a3771c4341a1c50ddc7e86ddf94effe895448"),
    ("scope_diagnosis_20261026", 177,
     "ca382545ddcf71ad41f363a692188ed79ee07a85bbc93317c0b9fa23e636bc4b",
     "1a930bdeaee5c2c1d162c3eedbe517fb9f34f3a0b45b11a2f162d2cc2993449b"),
)
PREPARE_SOURCE = "scripts/claim_grounding_c2_bounded.py"
HISTORICAL_PREPARE = BASE / "c2_bounded/executed_prepare_source.py"
HISTORICAL_PREPARE_SHA = "274cf85157a04d01bc127a997139d37c517e670f548792c2f87df3d4b369f1ab"
FAILED_RESOURCE = REPORT / "c1_profile_resources.json"
FAILED_RESOURCE_SHA = "09287f31610373284010e08692edef0b60dfd5d333faab3404e841c9d0493e47"
DATASETS = ("natural", "r1", "r6", "external", "factorial")


def digest(path: Path) -> str:
    """Read opaque bytes only, including closed or incomplete JSON artifacts."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(root: Path, name: str | Path) -> Path:
    """Manifest paths may not escape the workspace, even via symlinks."""
    path = root / str(name).replace("\\", "/")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path outside workspace: {name}")
    return path


def read_manifest(path: Path) -> dict:
    """Only callers' explicitly named manifests/configuration are deserialized."""
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")


def verify_file_map(root: Path, files: dict, *, replacements: dict | None = None) -> int:
    """Do not decode mapped files: a map may contain closed labels or broken JSON."""
    replacements = replacements or {}
    normalized = {str(k).replace("\\", "/"): v for k, v in files.items()}
    if len(normalized) != len(files) or not set(replacements) <= normalized.keys():
        raise ValueError("Ambiguous paths or an unused historical replacement")
    for name, spec in normalized.items():
        path = replacements.get(name, checked_path(root, name))
        expected = spec["sha256"] if isinstance(spec, dict) else spec
        require_hash(path, expected)
    return len(normalized)


def verify_legacy_manifest(root: Path, namespace: str, count: int,
                           manifest_sha: str, snapshot_sha: str) -> dict:
    manifest_path = root / "models" / namespace / "artifact_manifest.json"
    snapshot_path = root / "research_snapshots" / f"{namespace}_source.zip"
    require_hash(manifest_path, manifest_sha)
    require_hash(snapshot_path, snapshot_sha)
    files = read_manifest(manifest_path)["files"]
    if len(files) != count:
        raise ValueError(f"Unexpected legacy artifact count: {namespace}")
    historical = []
    with zipfile.ZipFile(snapshot_path) as archive:
        for raw_name, spec in files.items():
            name = raw_name.replace("\\", "/")
            current = checked_path(root, name)
            if digest(current) == spec["sha256"]:
                continue
            if name not in LIVING:
                raise ValueError(f"Immutable historical artifact changed: {name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != spec["sha256"]:
                raise ValueError(f"Invalid historical living-document copy: {name}")
            historical.append(name)
    return {"files": count, "historical_living_documents": sorted(historical),
            "closed_files": "opaque SHA256 only; no labels/predictions/metrics decoded"}


def namespace_ast(source: str, namespace: str) -> str:
    """Normalize only the three literals in the exact known namespace assignment."""
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Tuple)
                       and [getattr(x, "id", None) for x in t.elts] == ["BASE", "PRED", "REPORT"]
                       for t in node.targets)]
    expected = ast.parse(
        "BASE, PRED, REPORT = " + ", ".join(
            f"run.{name} / {namespace!r}" for name in ("BASE", "PRED", "REPORT"))
    ).body[0]
    if len(matches) != 1 or ast.dump(matches[0]) != ast.dump(expected):
        raise ValueError("Unexpected C2 namespace assignment")
    for value in matches[0].value.elts:
        value.right.value = "__exact_namespace_only__"
    return ast.dump(tree)


def verify_prepare_equivalence(historical: str, current: str) -> None:
    if namespace_ast(historical, "c2_bounded") != namespace_ast(current, "c2_ready"):
        raise ValueError("C2 preparation changed beyond formatting and namespace")


def verify_snapshot_copies(root: Path, folder: Path, files: dict) -> int:
    """Saved source copies must match a specifically frozen source, never arbitrary bytes."""
    count = 0
    normalized = {name.replace("\\", "/"): sha for name, sha in files.items()}
    if len(normalized) != len(files):
        raise ValueError("Ambiguous paths in source snapshot manifest")
    for path in sorted((root / folder).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root / folder)
        if len(relative.parts) > 1:
            # Hierarchical snapshots preserve exact repo paths, including duplicate basenames.
            key = relative.as_posix()
            candidates = [normalized[key]] if key in normalized else []
        else:
            # Early snapshots used a flat layout; keep their existing content-based check.
            candidates = [sha for name, sha in normalized.items() if Path(name).name == path.name]
        if not candidates or digest(path) not in candidates:
            raise ValueError(f"Unverified saved source: {path}")
        count += 1
    if not count:
        raise ValueError(f"Empty source snapshot: {folder}")
    return count


def count_lines(path: Path) -> int:
    """Prediction cardinality only; no decoding of values, labels, or scores."""
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def verify_resume_cache(path: Path, freeze: dict, all_keys: set[str], *,
                        require_complete: bool = True) -> dict:
    """Validate raw prefix bytes and keys only; never decode cached model scores."""
    cached = freeze["cached_pairs"]
    pending = freeze["pending_keys"]
    if cached < 0 or pending != sorted(set(pending)) or cached + len(pending) != len(all_keys):
        raise ValueError("Invalid resume membership")
    prefix_hash, seen = hashlib.sha256(), set()
    key_pattern = re.compile(rb'\s*\{\s*"key"\s*:\s*"([a-f0-9]{64})"\s*,')
    appended = []
    with path.open("rb") as stream:
        index = 0
        while require_complete or index < cached:
            line = stream.readline()
            if not line:
                break
            match = key_pattern.match(line)
            if not match or not line.endswith(b"\n"):
                raise ValueError("Malformed cache key or incomplete cache row")
            key = match[1].decode("ascii")
            if key in seen or key not in all_keys:
                raise ValueError("Duplicate or unexpected cache key")
            seen.add(key)
            if index < cached:
                prefix_hash.update(line)
            else:
                appended.append(key)
            index += 1
    if index < cached or prefix_hash.hexdigest() != freeze["cache_prefix_sha256"]:
        raise ValueError("Pre-resume cache prefix changed")
    if require_complete:
        if seen != all_keys or appended != pending:
            raise ValueError("Resume did not append exactly the frozen pending keys")
    elif sorted(all_keys - seen) != pending:
        raise ValueError("Frozen pending keys disagree with original prefix")
    return {"preserved_prefix_rows": cached, "frozen_pending_keys": len(pending),
            "complete_unique_keys": len(seen) if require_complete else None,
            "model_values": "not deserialized; key metadata and raw prefix bytes only"}


def verify_resume(root: Path, *, require_complete: bool) -> dict:
    path = root / BASE / "c2_ready/c2_resume1_freeze.json"
    if not path.exists() and not require_complete:
        return {"status": "not yet started"}
    freeze = read_manifest(path)
    require_hash(root / "scripts/claim_grounding_resume.py", freeze["source_sha256"])
    require_hash(root / BASE / "c2_ready/c2_freeze.json", freeze["original_freeze_sha256"])
    require_hash(root / REPORT / "c2_ready/c2_run_resources.json", freeze["failed_report_sha256"])
    # The frozen pair dictionary uses two-space-indented SHA keys. Do not decode its texts.
    pair_bytes = (root / BASE / "c2_ready/c2_pairs.json").read_bytes()
    all_keys = {m.decode("ascii") for m in re.findall(
        rb'^  "([a-f0-9]{64})": \{\r?$', pair_bytes, re.MULTILINE)}
    ready = read_manifest(root / BASE / "c2_ready/c2_freeze.json")
    if len(all_keys) != ready["pairs"]:
        raise ValueError("Unexpected pair-file schema or key count")
    return verify_resume_cache(root / PRED / "c2_ready/c2_pair_cache.jsonl", freeze, all_keys,
                               require_complete=require_complete)


def verify_freezes(root: Path, *, require_complete: bool = True) -> dict:
    frozen = {}
    simple = [BASE / "c1_freeze.json", BASE / "c1_bounded/c1_freeze.json",
              BASE / "c2_freeze.json", BASE / "c2_ready/c2_freeze.json"]
    analysis = [BASE / "c1_bounded/c1_analysis_freeze.json",
                BASE / "c2_ready/c2_analysis_freeze.json"]
    for relative in simple + analysis:
        if not (root / relative).exists() and not require_complete and relative in analysis:
            continue
        data = read_manifest(root / relative)
        verify_file_map(root, data["files"])
        frozen[relative.as_posix()] = data
    historical = root / HISTORICAL_PREPARE
    require_hash(historical, HISTORICAL_PREPARE_SHA)
    initial = read_manifest(root / BASE / "c2_bounded/c2_freeze.json")
    verify_file_map(root, initial["files"], replacements={PREPARE_SOURCE: historical})
    verify_prepare_equivalence(historical.read_text(encoding="utf-8"),
                               (root / PREPARE_SOURCE).read_text(encoding="utf-8"))
    ready = frozen[(BASE / "c2_ready/c2_freeze.json").as_posix()]
    for key in ("counts", "pairs", "reused"):
        if initial[key] != ready[key]:
            raise ValueError(f"C2 prepared membership differs: {key}")
    if set(ready["counts"]) != set(DATASETS):
        raise ValueError("Unexpected C2 datasets")
    for filename in ["c2_pairs.json"] + [f"c2_{d}_layout.jsonl" for d in DATASETS]:
        require_hash(root / BASE / "c2_ready" / filename,
                     digest(root / BASE / "c2_bounded" / filename))
    external = read_manifest(root / BASE / "external/samsum_manifest.json")
    b7 = read_manifest(root / BASE / "external_B7_freeze.json")
    verify_file_map(root, external["files_sha256"])
    verify_file_map(root, b7["files_sha256"])
    launch = read_manifest(root / BASE / "external_B7_launch.json")
    require_hash(root / BASE / "external_B7_freeze.json", launch["source_freeze_sha256"])
    require_hash(checked_path(root, launch["selected_c2_plan_path"]),
                 launch["selected_c2_plan_sha256"])
    require_hash(root / FAILED_RESOURCE, FAILED_RESOURCE_SHA)
    copies = {}
    snapshot_maps = [(BASE / "c1_source", frozen[(BASE / "c1_freeze.json").as_posix()]),
                     (BASE / "c2_source", frozen[(BASE / "c2_freeze.json").as_posix()]),
                     (BASE / "external/source_snapshot", {"files": external["files_sha256"]}),
                     (BASE / "external_B7_source", {"files": b7["files_sha256"]})]
    for name in analysis:
        if name.as_posix() in frozen:
            snapshot_maps.append((name.parent / name.name.replace("_freeze.json", "_source"),
                                  frozen[name.as_posix()]))
    for folder, data in snapshot_maps:
        copies[folder.as_posix()] = verify_snapshot_copies(root, folder, data["files"])
    predictions = {}
    if require_complete:
        counts = {PRED / "external_B7.jsonl": 295,
                  PRED / "c1_bounded/c1_gradients.jsonl":
                  frozen[(BASE / "c1_bounded/c1_freeze.json").as_posix()]["unique_pairs"],
                  PRED / "c2_ready/c2_pair_cache.jsonl": ready["pairs"]}
        for dataset, n in ready["counts"].items():
            for arm in ("original", "joint_window", "delta"):
                counts[PRED / "c2_ready" / f"c2_{dataset}_{arm}.jsonl"] = n
        for path, expected in counts.items():
            actual = count_lines(root / path)
            if actual != expected:
                raise ValueError(f"Incomplete prediction file: {path}: {actual}/{expected}")
            predictions[path.as_posix()] = actual
    return {"current_freezes": len(frozen) + 3, "source_snapshot_files": copies,
            "C2_resume": verify_resume(root, require_complete=require_complete),
            "historical_C2_prepare": HISTORICAL_PREPARE.as_posix(),
            "C2_only_namespace_and_formatting_changed": True,
            "C2_prepared_pairs_and_layouts_identical": True,
            "failed_C1_resource": "preserved original opaque bytes; never JSON decoded",
            "prediction_counts_only": predictions}


def source_files(root: Path) -> set[Path]:
    files = set((root / "src").rglob("*.py"))
    files.update((root / "scripts").glob("*.py"))
    files.update((root / "tests").rglob("*.py"))
    files.update((root / "docs").glob("CLAIM_GROUNDING*.md"))
    files.update((root / "configs").glob("*.txt"))
    files.update(root / name for name in LIVING)
    files.update(root / name for name in ("AGENTS.md", "PROJECT_SPEC.md", "FIRST_TASK.md",
                                         "pyproject.toml", ".gitignore"))
    files.update((root / BASE).rglob("*.py"))
    return {path for path in files if "__pycache__" not in path.parts}


def write_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "seal", "verify"))
    args = parser.parse_args()
    verification = {"legacy": {spec[0]: verify_legacy_manifest(ROOT, *spec) for spec in LEGACY},
                    "current": verify_freezes(ROOT, require_complete=args.command != "audit")}
    if args.command == "audit":
        print(json.dumps(verification, indent=2))
        return
    if args.command == "verify":
        manifest = read_manifest(ROOT / MANIFEST)
        count = verify_file_map(ROOT, manifest["files"])
        print(json.dumps({"verified_files": count, "source_snapshot_sha256":
                          digest(ROOT / SNAPSHOT), "legacy_verified": [s[1] for s in LEGACY]}))
        return
    destinations = [ROOT / MANIFEST, ROOT / SNAPSHOT, ROOT / REPORT / "integrity.json"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Seal is exclusive; never overwrite a prior seal or snapshot")
    sources = source_files(ROOT)
    with zipfile.ZipFile(ROOT / SNAPSHOT, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            archive.write(path, path.relative_to(ROOT).as_posix())
    write_exclusive(ROOT / REPORT / "integrity.json", {
        **verification, "source_snapshot_sha256": digest(ROOT / SNAPSHOT),
        "source_files": len(sources), "new_training_runs": 0,
        "data_policy": "All data/weights remain outside source ZIP; artifact SHA256 only"})
    files = sources | {ROOT / SNAPSHOT}
    for folder in (BASE, PRED, REPORT):
        files.update(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and path != ROOT / MANIFEST
                     and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".part"))
    write_exclusive(ROOT / MANIFEST, {"files": {
        path.relative_to(ROOT).as_posix(): {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(files)}})
    print(json.dumps({"artifact_files": len(files), "source_files": len(sources),
                      "source_snapshot_sha256": digest(ROOT / SNAPSHOT)}))


if __name__ == "__main__":
    main()
