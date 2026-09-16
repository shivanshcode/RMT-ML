"""Executable calibration for ``spacing.SIGMA2_RELIABLE_LMAX`` (review §4.5).

The review's complaint was that the dict was annotated "calibrated numerically"
with no pointer to the calibration, so nobody could check it or reproduce it.
This module *is* the calibration.  If you change those constants, this test is
what has to pass afterwards.

The claim being pinned: a local Gaussian kernel cannot preserve count
fluctuations on scales beyond its own window, so Sigma^2 from the 'gauss'
branch degrades an order of magnitude earlier in L than the global 'cheb' fit.
"""
import numpy as np
import pytest

from rmt import spacing as SP
from rmt import ensembles as E

N = 1200
REPS = 3
RTOL = 0.05          # the "within ~5% of theory" the constants claim


@pytest.fixture(scope="module")
def spectra():
    rng = np.random.default_rng(808)
    return [np.sort(np.linalg.svd(E.wishart_factor(N, N, rng=rng),
                                  compute_uv=False)) for _ in range(REPS)]


def _rel_err(spectra, L, method, win=15):
    vals = []
    for s in spectra:
        xi = np.sort(SP.unfold(s, method=method, win_size=win))
        vals.append(SP.sigma2(None, L, unfolded=xi, method=method, rng=0,
                              win_size=win))
    th = SP.sigma2_goe_theory(L)
    return abs(float(np.mean(vals)) - th) / th


def test_sigma2_reliable_lmax_is_calibrated(spectra):
    """Each unfolding must be within 5% of Mehta at its own LMAX."""
    for method in ("cheb", "gauss"):
        lmax = SP.sigma2_reliable_lmax(method, 15)
        err = _rel_err(spectra, lmax, method)
        assert err < RTOL, (method, lmax, err)


def test_the_old_gauss_constant_of_15_was_wrong(spectra):
    """Pins the correction. The shipped value used to be 15.0, annotated
    "within ~5% of theory"; it is 17% low there, and 11% low at L=10, which
    per_matrix was retaining and reporting."""
    assert SP.SIGMA2_RELIABLE_LMAX["gauss"] == 5.0
    assert _rel_err(spectra, 15, "gauss") > 0.10
    assert _rel_err(spectra, 10, "gauss") > 0.05


def test_gauss_fails_beyond_its_lmax(spectra):
    """...and the constant is not merely conservative: well past its LMAX the
    local kernel is materially wrong, which is why it is capped."""
    lmax = SP.sigma2_reliable_lmax("gauss", 15)
    err = _rel_err(spectra, 4 * lmax, "gauss")
    assert err > 2 * RTOL, err


def test_local_kernel_reach_scales_with_its_window(spectra):
    """The justification for making LMAX a function of win_size rather than a
    constant: a wider kernel really does reach further."""
    assert _rel_err(spectra, 10, "gauss", win=30) < _rel_err(spectra, 10, "gauss", win=15)
    assert SP.sigma2_reliable_lmax("gauss", 30) > SP.sigma2_reliable_lmax("gauss", 15)


def test_the_window_scaling_is_capped(spectra):
    """...but only so far: at win=60 the kernel over-smooths and degrades at
    every L, so the rule must not extrapolate."""
    assert SP.sigma2_reliable_lmax("gauss", 60) == SP.sigma2_reliable_lmax(
        "gauss", SP.GAUSS_WIN_CALIBRATED_MAX)
    assert _rel_err(spectra, 10, "gauss", win=60) > RTOL


def test_cheb_lmax_is_higher_than_gauss_lmax():
    assert SP.sigma2_reliable_lmax("cheb") > 3 * SP.sigma2_reliable_lmax("gauss", 15)
    # 'auto' resolves to whichever branch ran; its entry must not be laxer
    assert SP.SIGMA2_RELIABLE_LMAX["auto"] == SP.SIGMA2_RELIABLE_LMAX["cheb"]


def test_cheb_beats_gauss_at_long_range(spectra):
    """The ordering that justifies preferring the global fit."""
    L = 50
    assert _rel_err(spectra, L, "cheb") < _rel_err(spectra, L, "gauss")


def test_warning_fires_past_the_calibrated_range(spectra):
    xi = np.sort(SP.unfold(spectra[0], method="gauss"))
    with pytest.warns(RuntimeWarning, match="calibrated range"):
        SP.sigma2(None, 30, unfolded=xi, method="gauss", rng=0, win_size=15)
