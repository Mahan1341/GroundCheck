import json
from pathlib import Path

import pytest

from grounded_hallucination_lab.cli import main

FIXTURE = Path(__file__).parent / "fixtures"


def _args(tmp_path, command):
    return [
        command,
        "--data-dir",
        str(FIXTURE),
        "--output",
        str(tmp_path / f"{command}.json"),
        "--artifacts-dir",
        str(tmp_path / "models"),
        "--predictions-dir",
        str(tmp_path / "predictions"),
    ]


def test_calibrate_then_evaluate_fixture(tmp_path):
    assert main(_args(tmp_path, "lexical-calibrate")) == 0
    frozen_path = tmp_path / "models" / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    assert not frozen["dev_evaluated"] and not frozen["test_evaluated"]
    assert frozen["selection"]["selection_partition"] == "calibration"
    assert set(frozen["fit"]["source_ids"]) == set(frozen["membership"]["train"]["source_ids"])
    before = frozen_path.read_bytes()
    assert main(_args(tmp_path, "lexical-evaluate")) == 0
    report = json.loads((tmp_path / "lexical-evaluate.json").read_text())
    assert all(report["freeze_checks"].values())
    assert report["metrics"]["lexical"]["n"] == 4
    assert report["bootstrap"]["resamples"] == 1000
    assert report["threshold"] == frozen["selection"]["threshold"]
    assert not report["test_evaluated"]
    assert before == frozen_path.read_bytes()
    # The original calibration cannot be silently overwritten after dev.
    assert main(_args(tmp_path, "lexical-calibrate")) == 2


@pytest.mark.parametrize("field", ["code", "model", "config", "membership"])
def test_freeze_rejects_mutations_before_dev(tmp_path, field):
    assert main(_args(tmp_path, "lexical-calibrate")) == 0
    path = tmp_path / "models" / "frozen.json"
    frozen = json.loads(path.read_text())
    if field == "code":
        frozen["environment"]["source_sha256"] = {}
    elif field == "model":
        frozen["model_sha256"] = "wrong"
    elif field == "config":
        frozen["lexical_config"]["top_k"] = 99
    else:
        frozen["membership"]["dev"]["response_ids"] = []
    path.write_text(json.dumps(frozen))
    assert main(_args(tmp_path, "lexical-evaluate")) == 2
    assert not (tmp_path / "predictions" / "dev.jsonl").exists()


def test_lexical_rejects_changed_split(tmp_path):
    assert main([*_args(tmp_path, "lexical-calibrate"), "--seed", "99"]) == 2
