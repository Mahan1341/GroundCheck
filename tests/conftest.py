import json
from pathlib import Path

import pytest

from grounded_hallucination_lab.data.ragtruth import load_ragtruth

FIXTURE = Path(__file__).parent / "fixtures"


@pytest.fixture
def dataset():
    return load_ragtruth(FIXTURE)


@pytest.fixture
def mutable_data(tmp_path):
    rows = {
        name: [
            json.loads(line) for line in (FIXTURE / name).read_text(encoding="utf-8").splitlines()
        ]
        for name in ("source_info.jsonl", "response.jsonl")
    }

    def save():
        for name, content in rows.items():
            (tmp_path / name).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in content),
                encoding="utf-8",
            )
        return tmp_path

    return rows, save
