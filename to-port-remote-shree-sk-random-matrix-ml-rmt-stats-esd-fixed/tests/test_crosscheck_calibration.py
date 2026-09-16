"""Executable calibration for ``spacing.AUTO_CROSSCHECK_TOL`` (review §4.5).

The review's complaint: "the constant 1.5 is unjustified in-repo".  It is
justified by a *gap* — the cross-check ratio is tightly clustered near 1 when
the global basis can represent the density, and several-fold larger when it
cannot — and 1.5 sits inside that gap.  This module measures both sides of it.

If either side moves, the constant has to move with it, which is exactly the
property the review asked for.
"""
import numpy as np
import pytest

from rmt import spacing as SP
from rmt import ensembles as E

REPS = 5


def _ratio(levels):
    local = SP.GaussBroadening(15).unfold_spectrum(levels)
    return SP.unfolding_crosscheck(SP._unfold_cheb(levels), local)


@pytest.fixture(scope="module")
def ratios():
    """Cross-check ratios in the two regimes.

    good: singular values of a square Wishart, trimmed to the central 70% — a
          density a degree-7 Chebyshev fit represents easily.
    bad:  the SAME matrices in the lambda = nu^2/N domain, untrimmed, where the
          hard edge produces a density the basis cannot follow.
    """
    rng = np.random.default_rng(2718)
    good, bad = [], []
    for _ in range(REPS):
        s = np.sort(np.linalg.svd(E.wishart_factor(500, 500, rng=rng),
                                  compute_uv=False))
        good.append(_ratio(SP.bulk_levels(s, mode="center", center_frac=0.7)))
        bad.append(_ratio(s ** 2 / 500.0))
    return np.array(good), np.array(bad)


def test_good_regime_sits_near_one(ratios):
    good, _ = ratios
    assert good.max() < SP.AUTO_CROSSCHECK_TOL, good
    assert abs(good.mean() - 1.0) < 0.25, good.mean()


def test_bad_regime_is_far_above_the_tolerance(ratios):
    _, bad = ratios
    assert bad.min() > SP.AUTO_CROSSCHECK_TOL, bad
    assert bad.mean() > 2.0, bad.mean()


def test_the_tolerance_lies_inside_the_gap(ratios):
    """The actual justification for 1.5: it separates the two populations with
    margin on both sides, and no observation lands near it."""
    good, bad = ratios
    assert good.max() < SP.AUTO_CROSSCHECK_TOL < bad.min()
    # margin: the threshold is not scraping either population
    assert SP.AUTO_CROSSCHECK_TOL > 1.15 * good.max()
    assert SP.AUTO_CROSSCHECK_TOL < 0.85 * bad.min()


def test_the_branch_choice_follows_the_ratio(ratios):
    """The ratio is not decorative — it is what unfold_auto actually switches
    on, and the switch is what produced the NaN pattern in review §3/F3."""
    rng = np.random.default_rng(99)
    s = np.sort(np.linalg.svd(E.wishart_factor(500, 500, rng=rng),
                              compute_uv=False))
    assert SP.unfold_auto(SP.bulk_levels(s, mode="center",
                                         center_frac=0.7))[1] == "cheb"
    # WAS: == "gauss".  Flipped deliberately.  Falling to the local kernel on a
    # square-lambda spectrum is the pathology, not the cure -- it is what cost
    # ~11 % on Sigma^2(10) through the window-reach bias.  The transform search
    # now removes the lambda^(-1/2) hard-edge singularity exactly (x -> sqrt x),
    # so the global fit is recovered and the crosscheck drops 4.45 -> ~1.0.
    assert SP.unfold_auto(s ** 2 / 500.0)[1] == "cheb"
    assert SP.unfold_transform_auto(s ** 2 / 500.0)[2] == "sqrt"
    assert SP.unfold_transform_auto(s ** 2 / 500.0)[3] < SP.AUTO_CROSSCHECK_TOL


def test_crosscheck_docstring_records_the_justification():
    doc = SP.unfolding_crosscheck.__doc__
    assert "1.5" in doc
    assert "test_crosscheck_calibration" in doc
