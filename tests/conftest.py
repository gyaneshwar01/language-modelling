import pickle
from pathlib import Path

import pytest

SNAPSHOTS_PATH = Path(__file__).resolve().parent / "_snapshots"


class Snapshot:
    """Compares test results against data pickled in tests/_snapshots."""

    def __init__(self, test_name: str):
        self.path = SNAPSHOTS_PATH / f"{test_name}.pkl"

    def assert_match(self, actual: dict):
        with open(self.path, "rb") as f:
            expected = pickle.load(f)

        for key in actual:
            assert key in expected, f"Key '{key}' not found in {self.path.name}"
            assert actual[key] == expected[key], f"Data for key '{key}' does not match {self.path.name}"


@pytest.fixture
def snapshot(request):
    return Snapshot(request.node.name)
