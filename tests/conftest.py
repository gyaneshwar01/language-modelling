import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from .common import FIXTURES_PATH

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


class NumpySnapshot:
    """Compares a tensor or array against the one saved in tests/_snapshots/<test_name>.npz."""

    def __init__(self, test_name: str):
        self.path = SNAPSHOTS_PATH / f"{test_name}.npz"

    def assert_match(self, actual: Tensor | np.ndarray, rtol: float = 1e-4, atol: float = 1e-2):
        if isinstance(actual, Tensor):
            actual = actual.detach().cpu().numpy()
        expected = np.load(self.path)["array"]
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=f"Mismatch with {self.path.name}")


@pytest.fixture
def snapshot(request):
    return Snapshot(request.node.name)


@pytest.fixture
def numpy_snapshot(request):
    return NumpySnapshot(request.node.name)


@pytest.fixture
def ts_state_dict():
    """Weights and config of a small reference transformer."""
    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    with open(FIXTURES_PATH / "ts_tests" / "model_config.json") as f:
        config = json.load(f)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict, config


# Model sizes the snapshots were generated with


@pytest.fixture
def vocab_size():
    return 10_000


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def n_queries():
    return 12


@pytest.fixture
def n_heads():
    return 4


@pytest.fixture
def d_head():
    return 16


@pytest.fixture
def d_model(n_heads, d_head):
    return n_heads * d_head


@pytest.fixture
def d_ff():
    return 128


@pytest.fixture
def theta():
    return 10000.0


@pytest.fixture
def in_embeddings(batch_size, n_queries, d_model):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def in_indices(batch_size, n_queries):
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))


@pytest.fixture
def pos_ids(n_queries):
    return torch.arange(0, n_queries)
