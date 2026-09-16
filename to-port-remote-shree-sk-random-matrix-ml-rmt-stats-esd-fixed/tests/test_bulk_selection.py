"""Regression tests for priority item 5 — the MP bulk cut is a no-op on square
matrices, so the bulk statistics were being computed with the spectral edges
still in the sample (review §4.1).

Each test pins one link in that argument:

  1. mp_bounds really does return nu_- = 0 for n = m  (the defect exists);
  2. the edges really are a different ensemble  (<r> differs edge vs centre);
  3. bulk_levels(mode='center') really removes them at any aspect ratio;
  4. the pipeline actually uses it, and the flag reaches per_matrix;
  5. the conclusion is stable across the 0.6/0.7/0.8 sweep.
"""
import numpy as np
import pytest

from rmt import spacing as SP
from rmt import mp as MP
from rmt import ensembles as E
from rmt.config import RunConfig


@pytest.fixture
def rng():
    return np.random.default_rng(20240501)


# --- 1. the defect ---------------------------------------------------------- #
def test_mp_cut_is_a_noop_on_square_matrices():
    """nu_- = sigma(sqrt(max) - sqrt(min)) = 0 when n == m, so nothing is cut."""
    lo, hi = MP.mp_bounds(4096, 4096, 1.0)
    assert lo == 0.0
    assert hi == pytest.approx(128.0)
    # ... whereas a rectangular shape does get a real lower edge
    lo_r, hi_r = MP.mp_bounds(1024, 4096, 1.0)
    assert lo_r == pytest.approx(32.0)
    assert hi_r == pytest.approx(96.0)


def test_mp_cut_keeps_the_hard_edge_in_the_sample(rng):
    W = E.wishart_factor(600, 600, rng=rng)
    s = np.sort(np.linalg.svd(W, compute_uv=False))
    sig = MP.estimate_sigma_gd_median(s=np.sort(s)[::-1], n=600, m=600)
    lo, hi = MP.mp_bounds(600, 600, sig)
    kept = SP.bulk_levels(s, mode="mp", lo=lo, hi=hi)
    # the smallest singular values -- the Bessel hard edge -- survive the trim
    assert kept.min() < 0.05 * s.max()
    assert kept.size > 0.95 * s.size


def test_lambda_domain_edge_breaks_the_global_unfolding(rng):
    """The measurable consequence of leaving the edges in.

    NOTE ON THE EVIDENCE.  The review motivates this cut with <r> = 0.5165 at
    the hard edge against 0.5490 mid-spectrum on a square Gaussian.  That
    comparison does not reproduce: measured at N = 1000 over 20 replicas the
    smallest decile gives <r> = 0.5305 +/- 0.009 against 0.5301 +/- 0.003 for
    the central 70%.  It should not be expected to -- <r> is a *ratio* of
    consecutive spacings and is therefore insensitive to smooth density
    variation at first order, which is the whole reason it needs no unfolding.
    That number looks like seed noise and this test does not depend on it.

    The real, reproducible damage is to the UNFOLDING.  In the lambda = nu^2/N
    domain the square-matrix hard edge produces a density a degree-7 Chebyshev
    basis cannot represent, the cross-check ratio blows up, and every matrix is
    forced onto the local-kernel branch -- which is precisely the branch that
    makes per_matrix write sigma2_L20 = NaN (review §3/F3).  Trimming by rank
    removes the edge and restores the global fit.
    """
    N = 500
    def probe(frac):
        ratios, branches, txs = [], [], []
        for _ in range(3):
            s = np.sort(np.linalg.svd(E.wishart_factor(N, N, rng=rng),
                                      compute_uv=False))
            lam = s ** 2 / N
            lv = SP.bulk_levels(lam, mode="center", center_frac=frac)
            local = SP.GaussBroadening(15).unfold_spectrum(lv)
            ratios.append(SP.unfolding_crosscheck(SP._unfold_cheb(lv), local))
            branches.append(SP.unfold_auto(lv)[1])
            txs.append(SP.unfold_transform_auto(lv)[2])
        return float(np.mean(ratios)), branches, txs

    r_full, b_full, t_full = probe(1.0)
    r_cut, b_cut, t_cut = probe(0.7)

    # untrimmed: the global fit fails the cross-check and every matrix falls
    # back to the local kernel
    assert r_full > SP.AUTO_CROSSCHECK_TOL, r_full
    # UPDATED.  Was: falls back to "gauss".  That fallback was the defect, not
    # the cure -- it put the matrix on the local kernel, whose window-reach bias
    # costs ~11 % on Sigma^2(10).  unfold_transform_auto now removes the
    # lambda^(-1/2) hard-edge singularity exactly via x -> sqrt(x), so the global
    # fit is recovered.  What the test still pins is the ORIGINAL diagnosis: in
    # the identity coordinate the global fit really does leave a huge modulation.
    assert set(b_full) == {"cheb"}, b_full
    assert set(t_full) == {"sqrt"}, t_full
    # trimmed: the fit is representable again and the global branch is used
    assert r_cut < SP.AUTO_CROSSCHECK_TOL, r_cut
    assert set(b_cut) == {"cheb"}, b_cut
    assert r_cut < 0.5 * r_full


def test_r_statistic_is_insensitive_to_the_edge(rng):
    """Pins the negative result above, so nobody re-derives the cut from <r>."""
    edge, mid = [], []
    for _ in range(8):
        s = np.sort(np.linalg.svd(E.wishart_factor(600, 600, rng=rng),
                                  compute_uv=False))
        edge.append(SP.r_statistic(s[:s.size // 10]))
        mid.append(SP.r_statistic(s[int(.15 * s.size):int(.85 * s.size)]))
    # both are GOE to within their own scatter; the claimed 0.03 gap is absent
    assert abs(np.mean(edge) - np.mean(mid)) < 0.02
    assert abs(np.mean(mid) - 0.5307) < 0.02


# --- 2. the fix ------------------------------------------------------------- #
@pytest.mark.parametrize("frac", [0.6, 0.7, 0.8])
def test_center_cut_keeps_the_requested_fraction(frac):
    x = np.arange(1000.0)
    kept = SP.bulk_levels(x, mode="center", center_frac=frac)
    assert abs(kept.size / x.size - frac) < 0.01
    # symmetric about the centre
    assert abs((kept[0] - x[0]) - (x[-1] - kept[-1])) <= 1.0


@pytest.mark.parametrize("shape", [(600, 600), (300, 1200)])
def test_center_cut_removes_both_edges_at_any_aspect_ratio(rng, shape):
    s = np.sort(np.linalg.svd(E.wishart_factor(*shape, rng=rng),
                              compute_uv=False))
    kept = SP.bulk_levels(s, mode="center", center_frac=0.7)
    assert kept.min() > s.min()
    assert kept.max() < s.max()


def test_center_cut_does_not_bias_a_null_spectrum(rng):
    """The cut must be free on a matrix that really is random: Delta_3 stays on
    the GOE law.  (A cut that 'improved' agreement would be selecting levels to
    fit the answer, which is worse than the defect it fixes.)"""
    d3 = []
    for _ in range(4):
        s = np.sort(np.linalg.svd(E.wishart_factor(700, 700, rng=rng),
                                  compute_uv=False))
        lv = SP.bulk_levels(s, mode="center", center_frac=0.7)
        xi, _ = SP.unfold_auto(lv)
        d3.append(SP.delta3(None, 10, unfolded=np.sort(xi)))
    assert abs(np.mean(d3) - SP.delta3_goe_theory(10)) / SP.delta3_goe_theory(10) < 0.15


def test_none_and_frac_one_are_identity():
    x = np.linspace(0, 1, 501)
    assert np.array_equal(SP.bulk_levels(x, mode="none"), x)
    assert np.array_equal(SP.bulk_levels(x, mode="center", center_frac=1.0), x)


def test_mp_mode_refuses_an_overaggressive_trim():
    """A trim that would keep <25% means sigma is wrong, not that the spectrum
    is all outliers -- the input must come back unchanged."""
    x = np.linspace(1.0, 10.0, 400)
    out = SP.bulk_levels(x, mode="mp", lo=9.9, hi=10.0)
    assert out.size == x.size


def test_bad_arguments_raise():
    x = np.linspace(0, 1, 100)
    with pytest.raises(ValueError):
        SP.bulk_levels(x, mode="center", center_frac=0.0)
    with pytest.raises(ValueError):
        SP.bulk_levels(x, mode="mp")
    with pytest.raises(ValueError):
        SP.bulk_levels(x, mode="nonsense")


# --- 3. stability sweep ----------------------------------------------------- #
def test_bulk_stability_reports_a_small_spread_on_a_null_matrix(rng):
    s = np.sort(np.linalg.svd(E.wishart_factor(800, 800, rng=rng),
                              compute_uv=False))
    st = SP.bulk_stability(s, SP.r_statistic)
    assert set(SP.BULK_CENTER_FRAC_SWEEP).issubset(st)
    # <r> on a random matrix must not depend on where the bulk window is drawn
    assert st["rel_spread"] < 0.03, st


# --- 4. the pipeline actually uses it --------------------------------------- #
def test_runconfig_defaults_to_the_centred_cut():
    cfg = RunConfig()
    assert cfg.spacing_bulk_mode == "center"
    assert cfg.spacing_bulk_center_frac == 0.7
    assert list(cfg.spacing_bulk_frac_sweep) == [0.6, 0.7, 0.8]


def test_cli_exposes_the_flag():
    from rmt.cli import build_parser
    a = build_parser().parse_args(["--spacing_bulk_mode", "mp",
                                   "--spacing_bulk_center_frac", "0.6"])
    assert a.spacing_bulk_mode == "mp"
    assert a.spacing_bulk_center_frac == 0.6


def test_per_matrix_honours_the_bulk_mode(rng):
    from rmt.per_matrix import per_matrix_analysis

    class Rec:
        name = "test.weight"; short = "test"; layer_idx = 0
        n = 300; m = 300
        weight = E.wishart_factor(300, 300, rng=np.random.default_rng(3))

    cfg = RunConfig(do_powerlaw=False, do_overlap=False, do_ipr=False,
                    do_porter_thomas=False, do_brody=False,
                    spacing_bulk_mode="center", spacing_bulk_center_frac=0.7,
                    sigma2_L=[5], delta3_L=[5])
    row = per_matrix_analysis(Rec(), cfg=cfg)
    assert row["bulk_mode"] == "center"
    assert row["bulk_center_frac"] == 0.7
    assert abs(row["n_levels_bulk"] / 300 - 0.7) < 0.02

    cfg_mp = RunConfig(**{**cfg.to_dict(), "spacing_bulk_mode": "mp"})
    row_mp = per_matrix_analysis(Rec(), cfg=cfg_mp)
    assert row_mp["bulk_mode"] == "mp"
    # the defect, reproduced through the real pipeline: on a square matrix the
    # MP mode keeps essentially every level
    assert row_mp["n_levels_bulk"] > 0.95 * 300
    assert row_mp["n_levels_bulk"] > row["n_levels_bulk"]
