import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tests.synthetic_models import tiny_llama, tiny_causal_lm        # noqa: E402

from rmt import activations as A                                 # noqa: E402

pytestmark = pytest.mark.torch


def test_feature_layer_forward_is_identity_to_linear():
    lin = nn.Linear(8, 5, bias=True)
    fl = A.FeatureLayer(lin)
    x = torch.randn(3, 7, 8)
    import torch.nn.functional as F
    assert torch.allclose(fl(x), F.linear(x, lin.weight, lin.bias), atol=1e-6)


def test_feature_layer_is_centered_covariance():
    torch.manual_seed(0)
    lin = nn.Linear(6, 4, bias=False)
    fl = A.FeatureLayer(lin)
    X = torch.randn(200, 6)
    # mean pass
    fl.mode = "mean"
    fl(X)
    # FM pass
    fl.mode = "FM"
    fl(X)
    Xn = X.numpy().astype(np.float64)
    expected = np.cov(Xn, rowvar=False, bias=True)               # centered, /N
    assert np.allclose(fl.cov_, expected, atol=1e-6), np.abs(fl.cov_ - expected).max()


def test_separate_fm_counter_not_shared():
    # one mean pass + two FM passes; a SHARED counter would give (2/3)*batch_cov.
    torch.manual_seed(1)
    lin = nn.Linear(5, 3, bias=False)
    fl = A.FeatureLayer(lin)
    X = torch.randn(150, 5)
    fl.mode = "mean"; fl(X)
    fl.mode = "FM"; fl(X); fl(X)             # identical batch twice
    Xn = X.numpy().astype(np.float64)
    expected = np.cov(Xn, rowvar=False, bias=True)
    assert np.allclose(fl.cov_, expected, atol=1e-6)
    assert fl.computation == 150
    assert fl.fm_computation == 300


def test_set_feature_mode_toggles():
    m = tiny_llama(n_layers=1, d=16)
    A.replace_with_feature_layers(m, layer_indices=[0], device="cpu")
    A.set_feature_mode(m, "FM")
    fls = [mod for mod in m.modules() if isinstance(mod, A.FeatureLayer)]
    assert fls and all(f.mode == "FM" for f in fls)
    A.set_feature_mode(m, "mean")
    assert all(f.mode == "mean" for f in fls)


def test_replace_and_collect_roundtrip():
    m = tiny_causal_lm(n_layers=2, d=16)
    wrapped = A.replace_with_feature_layers(m, layer_indices=[0], device="cpu")
    assert wrapped and all(".0." in w for w in wrapped)
    assert not any("embed" in w or "lm_head" in w for w in wrapped)
    x = torch.randint(0, 50, (1, 12))
    m(input_ids=x)                          # populate (mean mode default)
    fm = A.collect_feature_matrices(m)
    assert set(fm) == set(wrapped)
    for name, d in fm.items():
        assert set(d) == {"FM", "mean", "mean_count", "fm_count"}
        assert d["mean_count"] > 0
        assert d["FM"].shape[0] == d["FM"].shape[1]       # square d_in
        assert d["FM"].shape[0] == m.get_submodule(name).kernel_dim
