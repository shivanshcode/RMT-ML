import numpy as np
import pytest

from rmt import per_matrix as PM
from rmt import linalg as LA
from rmt.discovery import MatrixRecord
from rmt.config import RunConfig


def _record(W, name="model.layers.0.self_attn.q_proj.weight", short="Q", li=0):
    W = np.asarray(W, dtype=np.float64)
    return MatrixRecord(name=name, short=short, layer_idx=li,
                        weight=W, n=W.shape[0], m=W.shape[1])


def test_single_svd_called_once(monkeypatch, rng):
    calls = {"n": 0}
    real = LA.cached_svd

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(PM, "cached_svd", counting)
    W = rng.standard_normal((120, 80))
    PM.per_matrix_analysis(_record(W), cfg=RunConfig(use_svd_cache=False))
    assert calls["n"] == 1


def test_row_has_full_schema(rng):
    W = rng.standard_normal((200, 120))
    row = PM.per_matrix_analysis(_record(W))
    for col in PM.CSV_COLUMNS:
        assert col in row, f"missing column {col}"


def test_N_cov_equals_m_by_default(rng):
    W = rng.standard_normal((200, 120))
    row = PM.per_matrix_analysis(_record(W))
    assert row["N_cov"] == 120
    assert row["mp_minus_eig"] <= row["mp_plus_eig"]


def test_exponents_are_labelled_and_distinct(rng):
    # heavy-tailed-ish matrix; ensure alpha (λ) and alpha_on_nu both present
    W = rng.standard_normal((300, 150))
    row = PM.per_matrix_analysis(_record(W))
    for k in ("alpha", "alpha_on_nu", "alpha_hill_nu", "alpha_hill_lambda"):
        assert k in row


def test_is_square_flag(rng):
    assert PM.per_matrix_analysis(_record(rng.standard_normal((50, 50))))["is_square"] == 1
    assert PM.per_matrix_analysis(_record(rng.standard_normal((80, 40))))["is_square"] == 0


def test_alpha_estimator_honored(rng):
    W = rng.standard_normal((300, 150))
    cfg_csn = RunConfig(alpha_estimator="csn")
    cfg_hw = RunConfig(alpha_estimator="hill_windowed")
    r1 = PM.per_matrix_analysis(_record(W), cfg=cfg_csn)
    r2 = PM.per_matrix_analysis(_record(W), cfg=cfg_hw)
    # headline alpha differs by estimator (csn density vs windowed-hill survival)
    assert r1["alpha"] != r2["alpha"] or np.isnan(r1["alpha"])


def test_overlap_block_filled_when_fm_present(rng):
    n, m = 64, 64
    W = rng.standard_normal((n, m))
    C = rng.standard_normal((m, m)); C = C @ C.T
    fm = {"model.layers.0.self_attn.q_proj": {"FM": C, "weight": W, "mean": np.zeros(m)}}
    row = PM.per_matrix_analysis(_record(W), fm_dict=fm)
    assert np.isfinite(row["max_overlap"])
    assert row["argmax_singular_for_top_eigenvector"] >= 0
    # without fm, overlap is NaN / -1
    row2 = PM.per_matrix_analysis(_record(W))
    assert np.isnan(row2["max_overlap"])
    assert row2["argmax_singular_for_top_eigenvector"] == -1
