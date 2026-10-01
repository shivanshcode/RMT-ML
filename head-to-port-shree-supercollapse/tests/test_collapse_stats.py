"""Tests for collapse_stats.py -- see docs/5_TESTS.md.

    python -m pytest tests/test_collapse_stats.py -v

No GPU, no jax, no network. T1/T13/T14 skip cleanly if the repo's notebook or
reference logs are absent.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import collapse_stats as cs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO, "figures", "collapse.ipynb")
# The authors' published MLP logs. Keep this file read-only and NEVER let a
# sweep write over it -- `logs/` is untracked in practice, so git cannot
# restore it and there is no warning when it is clobbered.
REF_PKL = next(
    (p for p in (os.path.join(REPO, "logs", "mlp_reference.pkl"),
                 os.path.join(REPO, "logs", "mlp.pkl"))
     if os.path.exists(p)),
    os.path.join(REPO, "logs", "mlp_reference.pkl"),
)


# ---------------------------------------------------------------------------
# Synthetic ladder builders
# ---------------------------------------------------------------------------

def make_run(p, seed, L0=0.002, mu=0.34, nu=0.29, gamma=None, a_t=1.0, a_p=1.0,
             n=400, x_min=1e-3, noise=None, rng=None):
    """One run in make_logs.py schema, following paper Eq. 5.

        t*(p) = p**gamma           (gamma = nu/mu is compute-optimal, Eq. 6)
        L(t)  = L0 + a_t t^-mu + a_p p^-nu

    ``noise`` is an optional callable (rng, x) -> psi, applied multiplicatively
    to the reducible part: L -> L0 + (L - L0) * (1 + psi).
    """
    gamma = nu / mu if gamma is None else gamma
    rng = np.random.default_rng(1000 * int(seed) + int(np.log(p) * 100)) \
        if rng is None else rng

    t_star = float(p) ** gamma
    x = np.geomspace(x_min, 1.0, n)
    t = x * t_star
    reducible = a_t * t ** (-mu) + a_p * float(p) ** (-nu)
    if noise is not None:
        reducible = reducible * (1.0 + noise(rng, x))
    loss = L0 + reducible

    hist = pd.DataFrame(
        {"step": np.arange(n), "compute": 6.0 * t * p, "test_loss": loss}
    )
    return dict(
        history=hist,
        num_params=int(p),
        D=int(round(np.sqrt(p))),
        seed=int(seed),
        opt_L=float(loss[-1]),
        opt_C=float(hist["compute"].iloc[-1]),
    )


def make_ladder(ps=(1e6, 2e6, 4e6, 8e6, 16e6, 32e6), seeds=(0, 1, 2), **kw):
    return pd.DataFrame(
        [make_run(int(p), s, **kw) for p in ps for s in seeds]
    )


def iid_noise(scale=2e-3):
    """Independent fluctuation at every eval point -> Delta ~ sqrt(2) sigma."""
    def f(rng, x):
        return rng.normal(0.0, scale, size=x.shape)
    return f


def walk_noise(scale=2e-3, decay=True):
    """Random walk whose increments scale with a decaying learning rate.

    This is the paper's section 3.3 picture: optimization noise per step scales
    with the instantaneous LR, so psi(t) - psi(t*) shrinks as t -> t* while
    psi(t) itself does not. Should produce Delta < sigma.
    """
    def f(rng, x):
        lr = (1.0 - x) if decay else np.ones_like(x)
        inc = rng.normal(0.0, scale, size=x.shape) * np.sqrt(np.maximum(lr, 0))
        return np.cumsum(inc)
    return f


def pipeline(runs, L0, n_grid=201, x_min=0.05, ddof=1):
    grid = cs.make_grid(n_grid, x_min=x_min, x_max=1.0)
    curves = cs.resample(runs, L0, grid)
    return curves, cs.collapse_deviation(curves, ddof=ddof), \
        cs.noise_floor(curves, ddof=ddof)


# ---------------------------------------------------------------------------
# T1 -- vendored fit matches the notebook
# ---------------------------------------------------------------------------

def _normalize(src):
    return [ln.rstrip() for ln in src.splitlines() if ln.strip()]


@pytest.mark.skipif(not os.path.exists(NOTEBOOK), reason="collapse.ipynb absent")
def test_vendored_fit_matches_notebook():
    with open(NOTEBOOK) as f:
        nb = json.load(f)
    cell = None
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "def fit_power_law_constant" in src:
            cell = src
            break
    assert cell is not None, "no fit_power_law_constant cell in the notebook"

    start = cell.index("def fit_power_law_constant")
    notebook_fn = _normalize(cell[start:])
    vendored_fn = _normalize(inspect.getsource(cs.fit_power_law_constant))
    assert vendored_fn == notebook_fn, (
        "collapse_stats.fit_power_law_constant has drifted from "
        "figures/collapse.ipynb. Re-vendor it verbatim."
    )


# ---------------------------------------------------------------------------
# T2 -- the fit recovers a known L0
# ---------------------------------------------------------------------------

def test_fit_recovers_L0():
    L0_true, a, b = 0.002, 0.05, 0.16
    C = np.geomspace(1.0, 1000.0, 8)
    L = a * C ** (-b) + L0_true
    res = cs.fit_power_law(C, L, num_inits=4)
    assert abs(res["L0"] - L0_true) / L0_true < 0.05
    assert res["r2"] > 0.999


# ---------------------------------------------------------------------------
# T3 / T4 -- exact power laws collapse iff the data exponent is optimal
# ---------------------------------------------------------------------------

def _max_width_spread(runs, L0):
    _, dev, _ = pipeline(runs, L0, x_min=0.05)
    return np.nanmax(dev["delta_width"])


def test_exact_power_law_collapses():
    L0 = 0.002
    runs = make_ladder(seeds=(0,), L0=L0)  # noiseless
    assert _max_width_spread(runs, L0) < 1e-6


def test_wrong_data_exponent_breaks_collapse():
    L0, mu, nu = 0.002, 0.34, 0.29
    good = _max_width_spread(make_ladder(seeds=(0,), L0=L0, mu=mu, nu=nu), L0)
    bad = _max_width_spread(
        make_ladder(seeds=(0,), L0=L0, mu=mu, nu=nu, gamma=1.15 * nu / mu), L0
    )
    assert bad > 100 * max(good, 1e-12)
    assert bad > 1e-3


# ---------------------------------------------------------------------------
# T5 -- Delta(1) == 0
# ---------------------------------------------------------------------------

def test_delta_is_zero_at_x_equals_one():
    L0 = 0.002
    runs = make_ladder(L0=L0, noise=walk_noise())
    curves, dev, _ = pipeline(runs, L0)
    assert curves.grid[-1] == 1.0
    assert np.all(curves.ell[:, -1] == 1.0)
    assert dev["delta"][-1] == 0.0
    assert dev["delta_pooled"][-1] == 0.0


# ---------------------------------------------------------------------------
# T6 -- sigma is computed on the reducible loss, not the normalized curve
# ---------------------------------------------------------------------------

def test_sigma_uses_reducible_not_normalized_loss():
    L0 = 0.002
    runs = make_ladder(L0=L0, noise=walk_noise())
    _, dev_a, sig_a = pipeline(runs, L0)

    # Perturb each run's opt_L only. Unphysical, but it changes ell and must
    # leave sigma -- a statistic of the raw reducible loss -- untouched.
    rng = np.random.default_rng(0)
    bumped = runs.copy()
    bumped["opt_L"] = runs["opt_L"] * (1.0 + rng.uniform(0.01, 0.05, len(runs)))
    _, dev_b, sig_b = pipeline(bumped, L0)

    for p in sig_a["per_p"]:
        np.testing.assert_array_equal(sig_a["per_p"][p], sig_b["per_p"][p])
    assert not np.allclose(
        dev_a["delta"][:-1], dev_b["delta"][:-1], equal_nan=True
    )


# ---------------------------------------------------------------------------
# T7 -- the control variate beats the noise floor
# ---------------------------------------------------------------------------

def test_control_variate_beats_noise_floor():
    L0 = 0.002
    runs = make_ladder(L0=L0, seeds=(0, 1, 2, 3, 4), noise=walk_noise())
    curves, dev, sig = pipeline(runs, L0, x_min=0.05)

    late = (curves.grid > 0.9) & (curves.grid < 1.0)
    assert np.all(dev["delta"][late] < sig["mean"][late])

    onset = cs.supercollapse_onset(curves.grid, dev["delta"], sig["mean"])
    assert onset > 0.2

    # Independent noise gives no time-correlation to cancel: Var(psi(t)-psi(t*))
    # = 2 Var(psi), so Delta sits ABOVE sigma.
    runs_iid = make_ladder(L0=L0, seeds=(0, 1, 2, 3, 4), noise=iid_noise())
    curves_i, dev_i, sig_i = pipeline(runs_iid, L0, x_min=0.05)
    mid = (curves_i.grid > 0.2) & (curves_i.grid < 0.9)
    frac_above = np.mean(dev_i["delta"][mid] > sig_i["mean"][mid])
    assert frac_above > 0.8


# ---------------------------------------------------------------------------
# T8 -- normalizing by the MEAN final loss reduces Delta-tilde to sigma
# ---------------------------------------------------------------------------

def test_normalizing_by_mean_final_loss_reduces_delta_to_sigma():
    """Paper section 3.3, stated as an identity."""
    L0 = 0.002
    runs = make_ladder(L0=L0, seeds=(0, 1, 2, 3, 4), noise=walk_noise())
    curves, _, sig = pipeline(runs, L0, x_min=0.05)

    # Rebuild ell dividing by the seed-MEAN final reducible loss.
    mean_curves = cs.Curves(
        grid=curves.grid, ell=curves.ell.copy(), lred=curves.lred,
        p=curves.p, seed=curves.seed, D=curves.D, L0=curves.L0,
    )
    for p in np.unique(curves.p):
        m = curves.p == p
        mean_final = np.nanmean(curves.lred[m][:, -1])
        mean_curves.ell[m] = curves.lred[m] / mean_final

    tilde = cs.per_model_collapse_deviation(mean_curves)
    for p, s in sig["per_p"].items():
        ok = np.isfinite(s) & np.isfinite(tilde[p])
        np.testing.assert_allclose(tilde[p][ok], s[ok], rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# T9 -- variance decomposition identity
# ---------------------------------------------------------------------------

def test_variance_decomposition_identity():
    L0 = 0.002
    runs = make_ladder(L0=L0, noise=walk_noise())  # balanced: 6 widths x 3 seeds
    _, dev, _ = pipeline(runs, L0, ddof=0)

    lhs = dev["delta_decomposed"] ** 2
    rhs = dev["delta_seed"] ** 2 + dev["delta_width"] ** 2
    ok = np.isfinite(lhs) & np.isfinite(rhs)
    np.testing.assert_allclose(lhs[ok], rhs[ok], rtol=1e-10, atol=1e-14)

    ok = np.isfinite(dev["delta_decomposed"]) & np.isfinite(dev["delta_pooled"])
    np.testing.assert_allclose(
        dev["delta_decomposed"][ok], dev["delta_pooled"][ok],
        rtol=1e-10, atol=1e-14,
    )


# ---------------------------------------------------------------------------
# T10 -- resampling is exact at knots and does not extrapolate
# ---------------------------------------------------------------------------

def test_resample_is_exact_at_knots():
    L0 = 0.002
    runs = make_ladder(ps=(4e6,), seeds=(0,), L0=L0, n=200, x_min=1e-2)
    h = runs["history"].iloc[0]
    x = h["compute"].to_numpy() / runs["opt_C"].iloc[0]

    knots = x[np.array([20, 60, 100, 150, 199])]
    curves = cs.resample(runs, L0, knots)
    expected = (h["test_loss"].to_numpy() - L0)[np.array([20, 60, 100, 150, 199])]
    np.testing.assert_allclose(curves.lred[0], expected, rtol=1e-12)

    # Below the run's first x -> nan, never a clamped extrapolation.
    below = cs.resample(runs, L0, np.array([x[0] * 0.1, x[10], 1.0]))
    assert np.isnan(below.lred[0, 0])
    assert np.isfinite(below.lred[0, 1])


# ---------------------------------------------------------------------------
# T11 -- loader schema and filters
# ---------------------------------------------------------------------------

def test_loader_filters_by_width():
    runs = make_ladder(ps=(1e4, 1e6, 4e6, 16e6), seeds=(0, 1, 2))
    out = cs.load_runs(runs, min_D=200, verbose=False)
    assert set(out["num_params"]) == {1_000_000, 4_000_000, 16_000_000}


def test_loader_drops_thin_widths(capsys):
    rows = [make_run(int(p), s) for p in (1e6, 4e6, 16e6) for s in (0, 1, 2)]
    rows.append(make_run(64_000_000, 0))  # one seed only
    out = cs.load_runs(pd.DataFrame(rows), verbose=False)
    assert 64_000_000 not in set(out["num_params"])
    assert "dropping" in capsys.readouterr().out


def test_loader_reconstructs_opt_L_and_opt_C():
    runs = make_ladder(ps=(1e6, 4e6, 16e6)).drop(columns=["opt_L", "opt_C"])
    out = cs.load_runs(runs, verbose=False)
    h = out["history"].iloc[0]
    assert out["opt_L"].iloc[0] == pytest.approx(h["test_loss"].iloc[-1])
    assert out["opt_C"].iloc[0] == pytest.approx(h["compute"].iloc[-1])


def test_loader_accepts_P_only_and_rejects_mismatch():
    runs = make_ladder(ps=(1e6, 4e6, 16e6))
    p_only = runs.rename(columns={"num_params": "P"})
    assert len(cs.load_runs(p_only, verbose=False)) == len(runs)

    bad = runs.copy()
    bad["P"] = bad["num_params"] * 2
    with pytest.raises(ValueError, match="disagree"):
        cs.load_runs(bad, verbose=False)


def test_resample_rejects_opt_C_mismatch():
    runs = make_ladder(ps=(1e6, 4e6, 16e6), seeds=(0,))
    runs = runs.copy()
    runs.loc[0, "opt_C"] = runs.loc[0, "opt_C"] * 2.0
    with pytest.raises(ValueError, match="not 1"):
        cs.resample(runs, 0.002, cs.make_grid(50, x_min=0.05))


# ---------------------------------------------------------------------------
# T12 -- an L0 above the minimum loss is refused
# ---------------------------------------------------------------------------

def test_L0_above_min_loss_raises():
    runs = cs.load_runs(make_ladder(ps=(1e6, 4e6, 16e6)), verbose=False)
    with pytest.raises(ValueError, match="min"):
        cs.fit_L0(runs, L0=2.0 * runs["opt_L"].min())


# ---------------------------------------------------------------------------
# T13 / T14 -- against the authors' reference logs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(REF_PKL), reason="logs/mlp.pkl absent")
def test_end_to_end_on_reference_logs():
    res = cs.analyze(REF_PKL, min_D=384, n_grid=201, verbose=False)

    # The authors' ladder: 8 widths x 5 seeds = 40 runs after the 16 exact
    # duplicate rows in the published pickle are dropped.
    assert res["runs"]["D"].nunique() == 8
    assert res["runs"]["seed"].nunique() == 5
    assert len(res["runs"]) == 40

    L0 = res["fit"]["L0"]
    assert 0 < L0 < res["runs"]["opt_L"].min()
    assert res["fit"]["r2"] > 0.999

    grid = res["grid"]
    i = int(np.argmin(np.abs(grid - 0.95)))
    assert res["delta"]["delta"][i] < res["sigma"]["mean"][i], (
        "Delta is not below the noise floor at x=0.95 on the authors' own "
        "MLP logs -- supercollapse should be present there."
    )
    # Measured on the authors' 5-seed pickle: onset 0.26, frac_below 0.32,
    # L0 = 0.00120, R2 = 0.99989. The strict onset stops at the LAST
    # violation, so an isolated spike can halve it; frac_below is the
    # spike-insensitive companion. Thresholds sit a little under the measured
    # values so a grid-resolution change does not flip the test.
    assert res["onset"] >= 0.2
    assert res["frac_below"] > 0.25


@pytest.mark.skipif(not os.path.exists(REF_PKL), reason="logs/mlp.pkl absent")
def test_cli_smoke(tmp_path):
    cli = os.path.join(REPO, "plot_collapse_deviation.py")
    assert os.path.exists(cli)
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run(
        [sys.executable, cli, REF_PKL, "--out-dir", str(tmp_path),
         "--name", "smoke", "--min-D", "384", "--grid", "101", "--no-show"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    for suffix in ("_deviation.png", "_ratio.png", "_deviation.csv",
                   "_summary.json"):
        assert (tmp_path / f"smoke{suffix}").exists(), suffix

    csv = pd.read_csv(tmp_path / "smoke_deviation.csv")
    assert len(csv) == 101
    assert "sigma_D384" in csv.columns
    assert {"x", "delta", "sigma_mean", "ratio"} <= set(csv.columns)

    summary = json.loads((tmp_path / "smoke_summary.json").read_text())
    ref = cs.analyze(REF_PKL, min_D=384, n_grid=101, verbose=False)
    assert summary["delta_onset"] == pytest.approx(ref["onset"])
    assert summary["frac_below_floor"] == pytest.approx(ref["frac_below"])


# ---------------------------------------------------------------------------
# T15 / T16 -- duplicate handling
# ---------------------------------------------------------------------------

def test_identical_duplicates_are_dropped(capsys):
    """The authors' published pickle carries exact duplicate rows.

    save_logs.py iterates a comma-separated tag list and extends its run list
    per tag, so a run carrying two tags is ingested twice. The copies are
    identical down to the history, so dropping one cannot change a statistic.
    """
    runs = make_ladder(ps=(1e6, 4e6, 16e6), seeds=(0, 1, 2))
    doubled = pd.concat([runs, runs[runs["seed"] > 0]], ignore_index=True)
    out = cs.load_runs(doubled, verbose=True)

    assert len(out) == len(runs)
    assert not out.duplicated(subset=["num_params", "seed"]).any()
    assert "exact duplicate" in capsys.readouterr().out

    # and the statistics are unchanged by the round trip
    a = pipeline(cs.load_runs(runs, verbose=False), 0.002)[1]["delta"]
    b = pipeline(out, 0.002)[1]["delta"]
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_conflicting_duplicates_still_raise():
    """Two DIFFERENT runs sharing a label must never be silently merged.

    This is what a glob collision between tags 'mlp' and 'mlp_no_mup'
    produces, and picking between them changes the answer.
    """
    runs = make_ladder(ps=(1e6, 4e6, 16e6), seeds=(0, 1, 2))
    other = runs[runs["seed"] > 0].copy()
    other["history"] = other["history"].map(
        lambda h: h.assign(test_loss=h["test_loss"] * 1.05)
    )
    mixed = pd.concat([runs, other], ignore_index=True)
    with pytest.raises(ValueError, match="NOT identical"):
        cs.load_runs(mixed, verbose=False)
