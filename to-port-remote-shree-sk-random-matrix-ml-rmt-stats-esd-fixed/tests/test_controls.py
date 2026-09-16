"""Regression tests for priority item 8 — the destructive controls (review §8).

The review calls these "the single thing that turns 'we measured something' into
'we measured something randomness does not explain'".  The tests below check
that each control preserves what it claims to preserve and destroys what it
claims to destroy, and that the decision ladder in :func:`rmt.controls.interpret`
reaches the right verdict on matrices whose right answer is known by
construction.
"""
import numpy as np
import pytest

from rmt import controls as CT
from rmt import spacing as SP
from rmt import nulls as NU
from rmt import ensembles as E


@pytest.fixture
def rng():
    return np.random.default_rng(31337)


@pytest.fixture
def heteroscedastic(rng):
    """An iid Gaussian matrix with a strong row/column variance profile --
    a caricature of a trained weight matrix."""
    n, m = 240, 240
    W = rng.standard_normal((n, m))
    row_scale = np.exp(rng.normal(0, 0.8, size=(n, 1)))
    col_scale = np.exp(rng.normal(0, 0.8, size=(1, m)))
    return W * row_scale * col_scale


# --- what each control preserves / destroys --------------------------------- #
def test_entry_shuffle_preserves_the_entry_multiset_exactly(rng):
    W = rng.standard_normal((60, 80))
    S = CT.entry_shuffle(W, rng)
    assert S.shape == W.shape
    assert np.array_equal(np.sort(S.ravel()), np.sort(W.ravel()))


def test_entry_shuffle_destroys_the_variance_profile(heteroscedastic):
    before = CT.variance_profile(heteroscedastic)
    after = CT.variance_profile(CT.entry_shuffle(heteroscedastic, 0))
    assert before["row_sd_cv"] > 0.3
    assert after["row_sd_cv"] < before["row_sd_cv"] / 2.0
    # ... while leaving every entry-level moment untouched
    assert after["global_sd"] == pytest.approx(before["global_sd"], rel=1e-12)
    assert after["excess_kurtosis"] == pytest.approx(before["excess_kurtosis"],
                                                     rel=1e-9)


@pytest.mark.parametrize("fn", [CT.row_shuffle, CT.col_shuffle,
                                CT.row_col_shuffle])
def test_shuffles_keep_the_entry_multiset(heteroscedastic, fn):
    S = fn(heteroscedastic, 0)
    assert np.array_equal(np.sort(S.ravel()),
                          np.sort(heteroscedastic.ravel()))
    assert not np.array_equal(S, heteroscedastic)


def test_row_shuffle_preserves_row_scale_exactly(heteroscedastic):
    """This is the control that matters: it must keep the heteroscedasticity an
    iid null lacks, or it is just a slower entry shuffle."""
    S = CT.row_shuffle(heteroscedastic, 0)
    assert np.allclose(S.std(axis=1), heteroscedastic.std(axis=1))
    assert CT.variance_profile(S)["row_sd_cv"] == pytest.approx(
        CT.variance_profile(heteroscedastic)["row_sd_cv"], rel=1e-9)


def test_col_shuffle_preserves_col_scale_exactly(heteroscedastic):
    S = CT.col_shuffle(heteroscedastic, 0)
    assert np.allclose(S.std(axis=0), heteroscedastic.std(axis=0))


def test_composed_shuffle_does_not_preserve_the_profile(heteroscedastic):
    """Pins the correction to §8.2: composing the two passes destroys what each
    preserves, landing at the entry-shuffle value. Anyone tempted to use the
    composition as a variance-matched null fails here."""
    before = CT.variance_profile(heteroscedastic)["row_sd_cv"]
    composed = CT.variance_profile(CT.row_col_shuffle(heteroscedastic, 0))["row_sd_cv"]
    entry = CT.variance_profile(CT.entry_shuffle(heteroscedastic, 0))["row_sd_cv"]
    assert composed < 0.6 * before
    assert abs(composed - entry) < 0.3 * entry


def test_gaussian_match_matches_only_sigma(rng):
    W = rng.standard_normal((100, 100)) * 3.0
    G = CT.gaussian_match(W, rng)
    assert G.shape == W.shape
    assert abs(G.std() / W.std() - 1.0) < 0.05
    assert abs(CT.variance_profile(G)["excess_kurtosis"]) < 0.3


@pytest.mark.parametrize("scheme,expect", [("normal", 0.02),
                                           ("xavier", np.sqrt(2 / 400.0)),
                                           ("kaiming", np.sqrt(2 / 200.0))])
def test_init_control_uses_the_requested_law(scheme, expect):
    W = CT.init_control((200, 200), scheme=scheme, rng=0)
    assert W.shape == (200, 200)
    assert abs(W.std() / expect - 1.0) < 0.06


def test_init_control_rejects_an_unknown_scheme():
    with pytest.raises(ValueError):
        CT.init_control((10, 10), scheme="nope")


# --- variance-profile diagnostic -------------------------------------------- #
def test_variance_profile_flags_heteroscedasticity(heteroscedastic, rng):
    het = CT.variance_profile(heteroscedastic)
    iid = CT.variance_profile(rng.standard_normal((240, 240)))
    # an iid matrix sits at the analytic expectation; the trained caricature
    # is an order of magnitude above it
    assert iid["row_sd_cv"] < 3.0 * iid["row_sd_cv_iid_expected"]
    assert het["row_sd_cv"] > 10.0 * het["row_sd_cv_iid_expected"]


def test_variance_profile_detects_heavy_tails(rng):
    gauss = CT.variance_profile(rng.standard_normal((200, 200)))
    heavy = CT.variance_profile(rng.standard_t(3, size=(200, 200)))
    assert abs(gauss["excess_kurtosis"]) < 0.3
    assert heavy["excess_kurtosis"] > 1.0


# --- the suite and the decision ladder -------------------------------------- #
@pytest.fixture(scope="module")
def band():
    return NU.null_band((200, 200), n_reps=12, seed=0, brody=False,
                        bulk_frac=SP.BULK_CENTER_FRAC,
                        delta3_L=(10,), sigma2_L=(10,))


def test_control_suite_runs_every_control_and_z_scores_it(band, rng):
    W = E.wishart_factor(200, 200, rng=rng)
    suite = CT.control_suite(W, band=band, brody=False,
                             delta3_L=(10,), sigma2_L=(10,))
    assert set(suite) == set(CT.CONTROLS)
    for name, row in suite.items():
        assert np.isfinite(row["delta3_L10"]), name
        assert "z_delta3_L10" in row, name


def test_a_random_matrix_is_consistent_with_random(band, rng):
    """The null case must come back clean, or every verdict below is noise."""
    W = E.wishart_factor(200, 200, rng=rng)
    suite = CT.control_suite(W, band=band, brody=False,
                             delta3_L=(10,), sigma2_L=(10,))
    assert abs(suite["real"]["z_delta3_L10"]) < 3.0
    assert CT.interpret(suite) == "consistent-with-random"


def test_block_structure_is_caught_as_structure_beyond_controls(band):
    """Two independent blocks: real spectrum deviates hard, and BOTH shuffles
    destroy the structure, so the deviation is attributable to the structure
    rather than to the entries or the variance profile."""
    g = np.random.default_rng(5)
    n = 200
    W = np.zeros((n, n))
    h = n // 2
    W[:h, :h] = g.standard_normal((h, h))
    W[h:, h:] = g.standard_normal((h, h))
    suite = CT.control_suite(W, band=band, brody=False, rng=1,
                             delta3_L=(10,), sigma2_L=(10,))
    assert abs(suite["real"]["z_delta3_L10"]) > 3.0
    assert CT.interpret(suite) == "structure-beyond-controls"


def test_interpret_returns_undetermined_without_a_band(rng):
    W = E.wishart_factor(200, 200, rng=rng)
    suite = CT.control_suite(W, brody=False, delta3_L=(10,), sigma2_L=(10,))
    assert CT.interpret(suite) == "undetermined"


def test_interpret_blames_the_entry_distribution_when_the_shuffle_also_deviates():
    """Synthetic ladder check: if entry_shuffle is out of band too, the verdict
    must NOT be 'learned structure' -- this is the misattribution §8.1 exists
    to prevent."""
    suite = {"real": {"z_delta3_L10": 6.0},
             "entry_shuffle": {"z_delta3_L10": 5.5},
             "row_col_shuffle": {"z_delta3_L10": 5.0}}
    assert CT.interpret(suite) == "entry-distribution"


def test_interpret_blames_the_variance_profile_when_only_that_survives():
    suite = {"real": {"z_delta3_L10": 6.0},
             "entry_shuffle": {"z_delta3_L10": 0.4},
             "row_col_shuffle": {"z_delta3_L10": 5.2}}
    assert CT.interpret(suite) == "variance-profile"


def test_band_bulk_fraction_mismatch_is_refused(band, rng):
    """A band built on a different fraction of the spectrum makes every z-score
    meaningless, so it must fail loudly rather than quietly."""
    W = E.wishart_factor(200, 200, rng=rng)
    with pytest.raises(ValueError, match="bulk_frac"):
        CT.control_suite(W, band=band, bulk_center_frac=0.5, brody=False,
                         delta3_L=(10,), sigma2_L=(10,))


def test_null_band_records_its_bulk_fraction(band):
    assert band["bulk_frac"] == pytest.approx(SP.BULK_CENTER_FRAC)
