"""Regression tests for priority item 7 — the documentation defects
(review §4.2, §4.3, §4.5).

These are "minor" only until someone believes one of them.  Each claim that was
wrong in a docstring is turned here into an assertion, so the docstring cannot
drift back out of agreement with the code.
"""
import numpy as np
import pytest

from rmt import spacing as SP
from rmt import ensembles as E


@pytest.fixture
def rng():
    return np.random.default_rng(90210)


# --------------------------------------------------------------------------- #
# §4.2 — GaussBroadening silently discards levels                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w,expected", [(5, 1014), (15, 994), (30, 964)])
def test_replicate_does_not_retain_every_level(rng, w, expected):
    """The old docstring said 'every input level is retained'. It is not:
    ``unfold_spectrum`` pads by w and then slices [2w:-2w], so n -> n - 2w.
    These are the exact numbers from the review's table."""
    lv = np.sort(rng.uniform(0, 1, 1024))
    out = SP.GaussBroadening(win_size=w, method="replicate").unfold_spectrum(lv)
    assert out.size == expected == 1024 - 2 * w


@pytest.mark.parametrize("w", [5, 15])
def test_drop_loses_four_windows(rng, w):
    lv = np.sort(rng.uniform(0, 1, 1024))
    out = SP.GaussBroadening(win_size=w, method="drop").unfold_spectrum(lv)
    assert out.size == 1024 - 4 * w


@pytest.mark.parametrize("method,w", [("replicate", 15), ("drop", 15),
                                      ("replicate", 7), ("drop", 7)])
def test_n_out_predicts_the_level_count(rng, method, w):
    """The helper must let a caller budget for the loss in advance."""
    gb = SP.GaussBroadening(win_size=w, method=method)
    lv = np.sort(rng.uniform(0, 1, 800))
    assert gb.n_out(800) == gb.unfold_spectrum(lv).size


def test_the_discarded_levels_are_the_extreme_ones(rng):
    """Why it matters: the dropped levels are exactly where MP-edge deviations
    live, so unfold_n_levels must be reported, not assumed equal to input."""
    lv = np.sort(rng.uniform(0, 1, 600))
    xi = SP.GaussBroadening(win_size=15).unfold_spectrum(lv)
    # the unfolded staircase spans the interior only: it starts well above 0
    # and ends well below n
    assert xi[0] > 5.0
    assert xi[-1] < lv.size - 5.0


# --------------------------------------------------------------------------- #
# §4.3 — GinUE constants used for a real matrix                                #
# --------------------------------------------------------------------------- #
def test_reference_table_carries_error_bars():
    """The review's GinOE/GinUE <cos> gap is not reproducible, so the table has
    to carry the standard errors that show why -- otherwise the next reader
    re-derives the same wrong conclusion from the same point estimates."""
    for k in ("ginue", "ginoe"):
        assert SP.CSR_REFERENCE[k]["cos_sem"] > 0
    ue = SP.CSR_REFERENCE["ginue"]
    oe = SP.CSR_REFERENCE["ginoe"]
    gap = abs(ue["cos_mean"] - oe["cos_mean"])
    joint = np.hypot(ue["cos_sem"], oe["cos_sem"])
    # the two ensembles' <cos> agree to well within one joint standard error
    assert gap < joint, (gap, joint)


def test_the_literature_bulk_value_is_kept_separate():
    """The measured constants include the spectral edge and so sit ~0.03 above
    the literature bulk value; conflating the two is the §4.1 error again."""
    lit = SP.CSR_REFERENCE["ginue_bulk_literature"]["cos_mean"]
    meas = SP.CSR_REFERENCE["ginue"]["cos_mean"]
    assert lit == pytest.approx(-0.240)
    assert abs(meas - lit) > 0.02


def test_auto_picks_ginoe_for_a_real_matrix(rng):
    out = SP.complex_spacing_ratio(E.ginoe(300, rng), ensemble="auto")
    assert out["ensemble"] == "ginoe"
    assert out["ref_cos_mean"] == SP.CSR_REFERENCE["ginoe"]["cos_mean"]


def test_auto_picks_ginue_for_a_complex_matrix(rng):
    out = SP.complex_spacing_ratio(E.ginue(300, rng), ensemble="auto")
    assert out["ensemble"] == "ginue"
    assert out["ref_cos_mean"] == SP.CSR_REFERENCE["ginue"]["cos_mean"]


def test_real_ginibre_has_exactly_real_eigenvalues(rng):
    """The structural reason GinOE differs: an O(sqrt(N)) subset of the
    spectrum is exactly real, and those points sit on a line."""
    out = SP.complex_spacing_ratio(E.ginoe(400, rng), ensemble="auto")
    assert out["frac_real"] > 0.01
    # ... and the complex ensemble has none
    assert SP.complex_spacing_ratio(E.ginue(400, rng))["frac_real"] == 0.0


def test_frac_real_is_the_real_discriminant_not_cos(rng):
    """Records BOTH halves of the §4.3 finding.

    True: GinOE carries an O(sqrt(N)) subset of exactly real eigenvalues and
    GinUE carries none -- a clean, large, structural separation.
    Not true: that this shifts <cos arg z> by 25%.  A 4% subset cannot move a
    mean that far, and measured over replicas the two ensembles' <cos> overlap.
    """
    def measure(gen, reps=6):
        outs = [SP.complex_spacing_ratio(gen(300, rng)) for _ in range(reps)]
        return (np.array([o["frac_real"] for o in outs]),
                np.array([o["cos_mean"] for o in outs]))

    f_oe, c_oe = measure(E.ginoe)
    f_ue, c_ue = measure(E.ginue)

    # frac_real separates them completely -- no overlap at all
    assert f_ue.max() == 0.0
    assert f_oe.min() > 0.01

    # <cos> does not: the difference in means is small next to the scatter
    sem = np.hypot(c_oe.std(ddof=1), c_ue.std(ddof=1)) / np.sqrt(len(c_oe))
    assert abs(c_oe.mean() - c_ue.mean()) < 3.0 * sem


def test_stripping_real_eigenvalues_restores_the_ginue_reference(rng):
    out = SP.complex_spacing_ratio(E.ginoe(400, rng), strip_real=True)
    assert "real_stripped" in out["ensemble"]
    assert out["ref_cos_mean"] == SP.CSR_REFERENCE["ginue"]["cos_mean"]
    assert abs(out["cos_mean"] - SP.CSR_REFERENCE["ginue"]["cos_mean"]) < 0.06


# --------------------------------------------------------------------------- #
# §4.5 — negative spacings were silently dropped                               #
# --------------------------------------------------------------------------- #
def test_clean_spacings_tolerates_a_few_dips():
    s = np.full(5000, 1.0)
    s[0] = -0.1                       # 1 bad in 5000 = 2e-4 < 1e-3
    out, frac, ok = SP.clean_spacings(s)
    assert ok
    assert out.size == 4999
    assert frac == pytest.approx(2e-4)


def test_clean_spacings_rejects_a_broken_unfolding():
    s = np.full(1000, 1.0)
    s[:20] = -0.1                     # 2% -- far beyond tolerance
    _out, frac, ok = SP.clean_spacings(s)
    assert not ok
    assert frac == pytest.approx(0.02)


def test_ks_nans_the_row_rather_than_truncating_the_sample():
    """Dropping negatives changes n, which changes the KS critical value, so
    the reported p-value would refer to a test that was never run."""
    xi = np.arange(600.0)
    xi[100], xi[101] = xi[101], xi[100]        # inject non-monotonicity
    xi[200], xi[201] = xi[201], xi[200]
    xi[300], xi[301] = xi[301], xi[300]
    broken = xi.copy()
    # 3 dips in 599 spacings ~ 0.5% > 0.1% tolerance
    out = SP.nn_spacing_ks(None, unfolded=broken)
    assert np.isnan(out["nn_KS_GOE"])
    assert np.isnan(out["nn_KS_GOE_p"])
    assert out["nn_frac_nonpositive"] > SP.UNFOLD_NONMONO_TOL


def test_brody_nans_on_a_broken_unfolding():
    xi = np.arange(600.0)
    for i in (100, 200, 300):
        xi[i], xi[i + 1] = xi[i + 1], xi[i]
    out = SP.brody_beta(None, unfolded=xi, n_bootstrap=20)
    assert np.isnan(out["brody_beta"])


def test_good_unfolding_reports_the_sample_size(rng):
    """The counts must be surfaced so a reader can see what n the KS used."""
    ev = np.linalg.eigvalsh(E.goe(400, rng))
    out = SP.nn_spacing_ks(ev)
    assert out["nn_frac_nonpositive"] == 0.0
    assert out["nn_n_spacings"] > 300
    assert np.isfinite(out["nn_KS_GOE"])


def test_uncalibrated_pvalue_is_documented_as_such():
    """§3/F2: the docstring must warn, since the column is still emitted."""
    doc = SP.nn_spacing_ks.__doc__
    assert "not calibrated" in doc
    assert "ks_pvalue_mc" in doc
