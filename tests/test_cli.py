import json
import subprocess
import sys
from pathlib import Path

from grounded_hallucination_lab.cli import main

FIXTURE = Path(__file__).parent / "fixtures"


def test_module_end_to_end(tmp_path):
    output = tmp_path / "baseline.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grounded_hallucination_lab.cli",
            "baseline",
            "--data-dir",
            str(FIXTURE),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["dataset"]["kind"] == "synthetic_fixture"
    assert report["evaluation_partition"] == "dev"
    assert report["metrics"]["n"] == 4
    assert report["test_reserved"]["responses"] == 4
    assert report["environment"]["source_sha256"]
    assert len(report["dev_predictions"]) == 4
    assert report["baseline"]["train_size"] == 6
    assert report["partitions"]["calibration"]["n_before_quality_filter"] == 2


def test_processed_roundtrip_and_reproducible_metrics(tmp_path):
    output = tmp_path / "a.json"
    processed = tmp_path / "qa.jsonl"
    argv = [
        "baseline",
        "--data-dir",
        str(FIXTURE),
        "--output",
        str(output),
        "--processed-output",
        str(processed),
    ]
    assert main(argv) == 0
    first = json.loads(output.read_text())
    records = [json.loads(line) for line in processed.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8
    assert {r["split"] for r in records} == {"train", "calibration"}
    assert all(r["official_split"] == "train" and "raw_source" not in r for r in records)
    assert records[0]["evidence"].startswith("passage 1:")
    assert main(argv) == 0
    second = json.loads(output.read_text())
    for key in ("metrics", "partitions", "dev_predictions", "baseline"):
        assert first[key] == second[key]


def test_missing_files_fails_cleanly(tmp_path, capsys):
    assert (
        main(["audit", "--data-dir", str(tmp_path / "missing"), "--output", str(tmp_path / "out")])
        == 2
    )
    assert "source_info.jsonl" in capsys.readouterr().err


def test_manifest_tampering_rejected(mutable_data, tmp_path, capsys):
    _, save = mutable_data
    directory = save()
    (directory / "manifest.json").write_text(json.dumps({"files": {"response.jsonl": {}}}))
    assert (
        main(
            [
                "audit",
                "--data-dir",
                str(directory),
                "--output",
                str(tmp_path.parent / "manifest-failure.json"),
            ]
        )
        == 2
    )
    assert "SHA256 mismatch" in capsys.readouterr().err


def test_config_and_quality_policy(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('seed = 5\nquality = "good-only"\nimplicit_true = "exclude"\n')
    out = tmp_path / "audit.json"
    assert (
        main(["audit", "--data-dir", str(FIXTURE), "--output", str(out), "--config", str(config)])
        == 0
    )
    report = json.loads(out.read_text())
    assert report["official_train_summary"]["excluded_quality_count"] == 2
    assert report["config"]["seed"] == 5
