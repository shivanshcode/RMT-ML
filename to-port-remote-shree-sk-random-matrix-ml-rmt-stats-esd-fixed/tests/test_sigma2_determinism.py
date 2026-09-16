"""Regression tests for priority item 6 — the Sigma^2 stopping rule stops early
(review §4.4).

Thamm's rule tracks the spread of the last ``min_iters`` *running* variances.
A cumulative mean moves by O(1/k) whether or not it has converged, so that
spread falls below ``tol`` on its own and the loop halts at a data-dependent
time.  The consequence is +/-1.4% of pure seed noise on a fixed spectrum, and a
Monte-Carlo error that differs from matrix to matrix and cannot be quoted.

The fix is a fixed window count.  These tests pin both the defect and the fix.
"""
import numpy as np
import pytest

from rmt import spacing as SP
from rmt import ensembles as E
from rmt.config import RunConfig


@pytest.fixture(scope="module")
def xi():
    """One fixed unfolded spectrum. Everything below varies only the seed."""
    rng = np.random.default_rng(4242)
    s = np.sort(np.linalg.svd(E.wishart_factor(900, 900, rng=rng),
                              compute_uv=False))
    lv = SP.bulk_levels(s, mode="center", center_frac=0.7)
    return np.sort(SP.unfold(lv, method="cheb"))


def _seed_spread(xi, L, **kw):
    v = np.array([SP.sigma2(None, L, unfolded=xi, rng=sd, **kw)
                  for sd in range(6)], dtype=np.float64)
    return float((v.max() - v.min()) / v.mean()), v


# --- the defect -------------------------------------------------------------#
@pytest.mark.parametrize("L", [5, 20])
def test_adaptive_rule_leaves_seed_noise(xi, L):
    """Reproduces §4.4: the legacy path's answer depends on the seed at the
    percent level, which is larger than several effects one might want to
    claim."""
    rel, _ = _seed_spread(xi, L, n_windows=None)
    assert rel > 3e-3, rel


def test_adaptive_rule_stops_early(xi):
    """Direct evidence that the stopping rule is the cause: forcing many more
    windows changes the answer by more than the tolerance it claims to enforce.
    """
    loose = SP.sigma2(None, 20, unfolded=xi, rng=0, n_windows=None)
    tight = SP.sigma2(None, 20, unfolded=xi, rng=0, n_windows=None,
                      tol=1e-6, min_iters=20000, max_iters=400000)
    assert abs(loose - tight) > 1e-3


# --- the fix ----------------------------------------------------------------#
def test_fixed_window_count_is_the_default():
    assert SP.SIGMA2_N_WINDOWS == 200_000
    assert RunConfig().sigma2_n_windows == 200_000


def test_fixed_estimator_is_deterministic(xi):
    """Same spectrum, same seed, same L -> bit-identical. No stopping time."""
    a = SP.sigma2(None, 10, unfolded=xi, rng=0)
    b = SP.sigma2(None, 10, unfolded=xi, rng=0)
    assert a == b


@pytest.mark.parametrize("L", [5, 20])
def test_fixed_estimator_cuts_seed_noise(xi, L):
    """The headline of §4.4: ~1.4% of seed scatter reduced by an order of
    magnitude, and now bounded by a number we can report."""
    rel_adaptive, _ = _seed_spread(xi, L, n_windows=None)
    rel_fixed, vals = _seed_spread(xi, L)
    assert rel_fixed < rel_adaptive / 2.0, (rel_fixed, rel_adaptive)
    assert rel_fixed < 1e-2
    # and the residual scatter is consistent with the *predicted* MC error
    predicted = SP.sigma2_mc_error(float(vals.mean()))
    assert vals.std(ddof=1) < 4.0 * predicted


def test_mc_error_matches_the_observed_scatter(xi):
    """sigma2_mc_error is a real error bar, not decoration: the spread over
    seeds must agree with sigma^2*sqrt(2/N) to within a small factor."""
    v = np.array([SP.sigma2(None, 10, unfolded=xi, rng=sd, n_windows=20_000)
                  for sd in range(12)])
    predicted = SP.sigma2_mc_error(float(v.mean()), 20_000)
    observed = float(v.std(ddof=1))
    assert 0.3 < observed / predicted < 3.0, (observed, predicted)


def test_mc_error_shrinks_as_one_over_sqrt_n():
    a = SP.sigma2_mc_error(1.0, 10_000)
    b = SP.sigma2_mc_error(1.0, 40_000)
    assert a / b == pytest.approx(2.0, rel=1e-9)
    assert SP.sigma2_mc_error(1.0, 200_000) < 0.005      # <0.5% at the default


def test_fixed_estimator_still_agrees_with_theory(xi):
    """Determinism is worthless if it is deterministically wrong."""
    for L in (5, 10, 20):
        got = SP.sigma2(None, L, unfolded=xi, rng=0)
        th = SP.sigma2_goe_theory(L)
        assert abs(got - th) / th < 0.20, (L, got, th)


def test_adaptive_path_is_still_reachable(xi):
    """Old results must remain reproducible."""
    v = SP.sigma2(None, 10, unfolded=xi, rng=0, n_windows=None)
    assert np.isfinite(v)
    assert v != SP.sigma2(None, 10, unfolded=xi, rng=0)


def test_residual_scatter_is_window_limited_not_spectrum_limited(xi):
    """Worth knowing where the remaining noise lives.  If it were set by the
    finite spectrum (only ~30 independent windows of width 20 fit in 630
    levels) then raising the window count would not help and the error bar
    would be a lie.  It does help, and it tracks 1/sqrt(N)."""
    def sd(nw):
        v = np.array([SP.sigma2(None, 20, unfolded=xi, rng=sd_, n_windows=nw)
                      for sd_ in range(6)])
        return float(v.std(ddof=1))
    lo, hi = sd(20_000), sd(1_000_000)
    assert hi < lo / 2.0, (lo, hi)


def test_more_windows_converges_toward_the_large_sample_answer(xi):
    ref = SP.sigma2(None, 10, unfolded=xi, rng=7, n_windows=400_000)
    near = abs(SP.sigma2(None, 10, unfolded=xi, rng=0, n_windows=200_000) - ref)
    far = abs(SP.sigma2(None, 10, unfolded=xi, rng=0, n_windows=2_000) - ref)
    assert near < far


# --- wiring -----------------------------------------------------------------#
def test_per_matrix_emits_the_window_count_and_error_bar():
    from rmt.per_matrix import per_matrix_analysis, CSV_COLUMNS

    class Rec:
        name = "t.weight"; short = "t"; layer_idx = 0
        n = 300; m = 300
        weight = E.wishart_factor(300, 300, rng=np.random.default_rng(5))

    cfg = RunConfig(do_powerlaw=False, do_overlap=False, do_ipr=False,
                    do_porter_thomas=False, do_brody=False,
                    sigma2_L=[5], delta3_L=[5])
    row = per_matrix_analysis(Rec(), cfg=cfg)
    assert row["sigma2_n_windows"] == 200_000
    assert np.isfinite(row["sigma2_L5_mcerr"])
    assert row["sigma2_L5_mcerr"] < 0.01 * abs(row["sigma2_L5"]) + 1e-9
    for col in ("sigma2_n_windows", "sigma2_L5_mcerr"):
        assert col in CSV_COLUMNS


def test_cli_exposes_the_window_count():
    from rmt.cli import build_parser
    a = build_parser().parse_args(["--sigma2_n_windows", "50000"])
    assert a.sigma2_n_windows == 50000
