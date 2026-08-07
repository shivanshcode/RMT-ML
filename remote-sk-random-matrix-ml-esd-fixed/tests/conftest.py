import numpy as np
import pytest

SEED = 1234


@pytest.fixture
def seed():
    return SEED


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)
