#!/usr/bin/env python
"""Figure 1c / 3c: collapse deviation vs per-model noise floor.

Plots the two statistics of arXiv 2507.02119 section 2.4 that
figures/collapse.ipynb does not compute:

    Delta(x)   -- relative spread of the SELF-NORMALIZED loss curves, pooled
                  over model size and seed                        (Eq. 2)
    sigma(x,p) -- relative spread of the RAW REDUCIBLE loss across seeds, at
                  fixed model size                                (Eq. 3)

Supercollapse is Delta(x) < sigma(x,p) over a substantial late fraction of
training. It is the claim that the collapse across model sizes is tighter than
our ability to predict any single model's loss.

Usage
-----
    # one sweep
    python plot_collapse_deviation.py logs/mlp.pkl --min-D 384

    # the actual deliverable: muP vs the no-muP ablation
    python plot_collapse_deviation.py logs/mlp.pkl logs/mlp_no_mup.pkl \
        --labels muP no-muP --name mlp_mup_vs_nomup --min-D 384

Exit codes: 0 ok, 2 data problem (no runs / too few seeds / bad L0), 1 other.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

import collapse_stats as cs

X_SAMPLES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_style():
    """Match figures/collapse.ipynb so these sit beside the existing figures.

    seaborn is optional -- matplotlib defaults are a fine fallback and this
    script should never be the reason an analysis env needs another package.
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(
            style="whitegrid",
            rc={
                "axes.spines.left": True, "axes.spines.bottom": True,
                "axes.spines.right": True, "axes.spines.top": True,
                "axes.edgecolor": "black", "axes.linewidth": 1.0,
            },
        )
        sns.set_context(
            "paper", font_scale=1.8,
            rc={"lines.linewidth": 2, "axes.grid": True, "grid.alpha": 0.3},
        )
        cmap = sns.color_palette("plasma", as_cmap=True)
    except ImportError:
        print("[sc] seaborn not installed; using matplotlib defaults")
        plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3,
                             "font.size": 13})
        cmap = plt.get_cmap("plasma")
    return plt, cmap


def _width_colors(widths, cmap):
    """Plasma by log(p), the notebook's convention (cell 6)."""
    w = np.asarray(widths, dtype=float)
    if len(w) == 1:
        return {int(w[0]): cmap(0.5)}
    t = 0.9 * (np.log(w) - np.log(w.min())) / (np.log(w.max()) - np.log(w.min()))
    return {int(p): cmap(ti) for p, ti in zip(w, t)}


def _smooth(y, window):
    if not window or window < 2:
        return y
    return (
        pd.Series(y)
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy()
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_deviation(res, label, out_base, plt, cmap, args):
    """Figure 1c / 3c."""
    grid = res["grid"]
    delta = _smooth(res["delta"]["delta"], args.smooth)
    sigma = res["sigma"]

    curves = res["curves"]
    colors = _width_colors(curves.widths, cmap)

    ncols = 2 if args.per_model else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8.5 * ncols, 6), dpi=args.dpi,
                             squeeze=False)
    ax = axes[0][0]

    for p, s in sigma["per_p"].items():
        ax.plot(grid, _smooth(s, args.smooth), ls="--", lw=1.0, alpha=0.75,
                color=colors[int(p)], zorder=5)
    # one proxy handle so the legend reads like the paper's
    ax.plot([], [], ls="--", lw=1.5, color="darkorange",
            label=r"Noise floor $\sigma$")
    ax.plot(grid, delta, lw=2.6, color="tab:blue", zorder=10,
            label=r"Collapse deviation $\Delta$")

    onset = res["onset"]
    if np.isfinite(onset) and 0 < onset < 1:
        ax.axvspan(1 - onset, 1.0, color="tab:blue", alpha=0.07, zorder=0)
        ax.axvline(1 - onset, color="tab:blue", alpha=0.35, lw=1.2, ls=":")

    ax.set_yscale("log")
    ax.set_xlim(grid[0], 1.0)
    ax.set_xlabel("Normalized Compute")
    ax.set_ylabel("Rel. Variation")
    ax.set_title(f"{label}   ($\\delta$ = {onset:.2f})" if np.isfinite(onset)
                 else label)
    ax.legend(loc="lower left", fontsize=14)
    _sane_ylim(ax, [delta] + list(sigma["per_p"].values()))

    if args.per_model:
        ax2 = axes[0][1]
        tilde = cs.per_model_collapse_deviation(curves, ddof=args.ddof)
        for p in sigma["per_p"]:
            ax2.plot(grid, _smooth(sigma["per_p"][p], args.smooth), ls="--",
                     lw=1.0, alpha=0.75, color=colors[int(p)])
            ax2.plot(grid, _smooth(tilde[p], args.smooth), ls="-", lw=1.4,
                     alpha=0.9, color=colors[int(p)])
        ax2.plot([], [], ls="--", color="k", label=r"$\sigma(x,p)$")
        ax2.plot([], [], ls="-", color="k", label=r"$\tilde\Delta(x,p)$")
        ax2.set_yscale("log")
        ax2.set_xlim(grid[0], 1.0)
        ax2.set_xlabel("Normalized Compute")
        ax2.set_ylabel("Rel. Variation")
        ax2.set_title("Per-model: control variate isolated")
        ax2.legend(loc="lower left", fontsize=14)
        _sane_ylim(ax2, list(tilde.values()) + list(sigma["per_p"].values()))

    fig.tight_layout()
    _save(fig, out_base + "_deviation", plt, args)


def plot_ratio(res, label, out_base, plt, args):
    """Delta / sigma-bar. Below 1 is supercollapse."""
    grid = res["grid"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = _smooth(res["delta"]["delta"], args.smooth) / \
            _smooth(res["sigma"]["mean"], args.smooth)

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=args.dpi)
    ax.plot(grid, ratio, lw=2.4, color="tab:blue")
    ax.axhline(1.0, color="k", ls="--", lw=1.4)
    ax.text(0.02, 1.08, "above: ordinary collapse", fontsize=12,
            transform=ax.get_yaxis_transform(), va="bottom")
    ax.text(0.02, 0.92, "below: supercollapse", fontsize=12,
            transform=ax.get_yaxis_transform(), va="top")
    ax.set_yscale("log")
    ax.set_xlim(grid[0], 1.0)
    ax.set_xlabel("Normalized Compute")
    ax.set_ylabel(r"$\Delta\,/\,\bar\sigma$")
    ax.set_title(label)
    _sane_ylim(ax, [ratio])
    fig.tight_layout()
    _save(fig, out_base + "_ratio", plt, args)


def plot_compare(results, labels, out_base, plt, args):
    """Two or more sweeps overlaid -- the muP vs no-muP deliverable."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), dpi=args.dpi)
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    series = []

    for res, label, c in zip(results, labels, palette):
        grid = res["grid"]
        d = _smooth(res["delta"]["delta"], args.smooth)
        s = _smooth(res["sigma"]["mean"], args.smooth)
        axes[0].plot(grid, d, lw=2.4, color=c, label=rf"$\Delta$  {label}")
        axes[0].plot(grid, s, lw=1.4, ls="--", color=c, alpha=0.8,
                     label=rf"$\bar\sigma$  {label}")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = d / s
        axes[1].plot(grid, r, lw=2.4, color=c, label=label)
        series += [d, s, r]

    axes[0].set_ylabel("Rel. Variation")
    axes[0].set_title("Collapse deviation vs noise floor")
    axes[1].axhline(1.0, color="k", ls="--", lw=1.4)
    axes[1].set_ylabel(r"$\Delta\,/\,\bar\sigma$")
    axes[1].set_title("Below 1 = supercollapse")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlim(results[0]["grid"][0], 1.0)
        ax.set_xlabel("Normalized Compute")
        ax.legend(loc="lower left", fontsize=13)
    _sane_ylim(axes[0], series[:-1])
    _sane_ylim(axes[1], [series[-1]])
    fig.tight_layout()
    _save(fig, out_base + "_compare", plt, args)


def _sane_ylim(ax, arrays, lo=1e-6, hi=1e2):
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return
    ax.set_ylim(max(lo, vals.min() * 0.5), min(hi, vals.max() * 2.0))


def _save(fig, base, plt, args):
    for ext in ("png", "pdf"):
        path = f"{base}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
        print(f"[sc] wrote {path}")
    if args.no_show:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_csv(res, path):
    grid = res["grid"]
    dev, sigma, curves = res["delta"], res["sigma"], res["curves"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = dev["delta"] / sigma["mean"]
    out = {
        "x": grid,
        "delta": dev["delta"],
        "delta_seed": dev["delta_seed"],
        "delta_width": dev["delta_width"],
        "delta_pooled": dev["delta_pooled"],
        "sigma_mean": sigma["mean"],
        "sigma_median": sigma["median"],
        "ratio": ratio,
    }
    for p, s in sigma["per_p"].items():
        out[f"sigma_D{curves.D_of(p)}"] = s
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"[sc] wrote {path}")


def summary_dict(res, label):
    grid = res["grid"]
    dev, sigma = res["delta"], res["sigma"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = dev["delta"] / sigma["mean"]

    samples = {}
    for x in X_SAMPLES:
        i = int(np.argmin(np.abs(grid - x)))
        samples[f"{x:g}"] = {
            "x": float(grid[i]),
            "delta": _f(dev["delta"][i]),
            "sigma_mean": _f(sigma["mean"][i]),
            "ratio": _f(ratio[i]),
        }
    runs = res["runs"]
    return {
        "label": label,
        "L0": float(res["fit"]["L0"]),
        "a": _f(res["fit"]["a"]),
        "b": _f(res["fit"]["b"]),
        "r2": _f(res["fit"]["r2"]),
        "L0_source": res["fit"]["source"],
        "n_runs": int(len(runs)),
        "widths": sorted(int(d) for d in runs["D"].unique()),
        "seeds": sorted(int(s) for s in runs["seed"].unique()),
        "delta_onset": _f(res["onset"]),
        "frac_below_floor": _f(res["frac_below"]),
        "samples": samples,
    }


def _f(v):
    v = float(v)
    return None if not np.isfinite(v) else v


def print_table(summary):
    print(f"\n[sc] {summary['label']}: L0 = {summary['L0']:.6g} "
          f"({summary['L0_source']}), R2 = {summary['r2']}")
    print(f"[sc]   {len(summary['widths'])} widths x "
          f"{len(summary['seeds'])} seeds = {summary['n_runs']} runs")
    print(f"\n{'x':>6} {'Delta':>12} {'sigma_bar':>12} {'ratio':>10}")
    for k, s in summary["samples"].items():
        f = lambda v: "     nan" if v is None else f"{v:8.3e}"
        r = "   nan" if s["ratio"] is None else f"{s['ratio']:6.2f}"
        print(f"{k:>6} {f(s['delta']):>12} {f(s['sigma_mean']):>12} {r:>10}")
    d = summary["delta_onset"]
    if d is None:
        print("\n[sc] supercollapse onset: undetermined")
    else:
        print(f"\n[sc] supercollapse onset delta = {d:.2f}  "
              f"(Delta < sigma_bar for x > {1 - d:.2f})")
    fb = summary["frac_below_floor"]
    if fb is not None:
        print(f"[sc] Delta < sigma_bar at {fb:.0%} of all x "
              "(spike-insensitive companion to delta)")


def print_verdict(summaries):
    """The muP-vs-no-muP call, using 2_HANDOFF.md section 8's acceptance rule."""
    key = "0.9"
    ratios = [(s["label"], s["samples"][key]["ratio"]) for s in summaries]
    if any(r is None for _, r in ratios):
        print("[sc] verdict: ratio undefined at x=0.9; cannot compare")
        return
    (la, ra), (lb, rb) = ratios[0], ratios[1]
    gap = max(ra, rb) / min(ra, rb)
    print(f"\n[sc] {la} vs {lb}: ratio at x=0.9 -- {ra:.2f} vs {rb:.2f} "
          f"({gap:.1f}x gap)")
    if gap >= 1.5:
        better = la if ra < rb else lb
        print(f"[sc] verdict: real effect. {better} collapses markedly tighter.")
    elif gap >= 1.2:
        print("[sc] verdict: weak. Suggestive but not conclusive.")
    else:
        print("[sc] verdict: AMBIGUOUS (<1.2x). Add seeds 3-4 "
              "(2_HANDOFF.md section 10) before drawing a conclusion.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Figure 1c/3c: collapse deviation vs noise floor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pkls", nargs="+", help="logs/*.pkl from make_logs.py")
    p.add_argument("--labels", nargs="*", default=None,
                   help="one label per pickle (default: file stems)")
    p.add_argument("--out-dir", default="figures_out")
    p.add_argument("--name", default=None,
                   help="output basename (default: stem of the first pickle)")
    p.add_argument("--min-D", type=float, default=None,
                   help="drop widths below this (384 for the MLP ladder)")
    p.add_argument("--max-D", type=float, default=None)
    p.add_argument("--min-seeds", type=int, default=2)
    p.add_argument("--grid", type=int, default=200, dest="n_grid")
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--spacing", choices=("linear", "log"), default="linear")
    p.add_argument("--L0", type=float, default=None,
                   help="override the fitted irreducible loss")
    p.add_argument("--ddof", type=int, default=1)
    p.add_argument("--estimator", choices=("decomposed", "pooled"),
                   default="decomposed")
    p.add_argument("--per-model", action="store_true",
                   help="extra panel: per-model Delta-tilde(x,p) vs sigma(x,p)")
    p.add_argument("--smooth", type=int, default=0,
                   help="rolling-median window for DISPLAY only; CSV is raw")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no-show", action="store_true", default=True)
    p.add_argument("--show", dest="no_show", action="store_false")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.no_show:
        os.environ.setdefault("MPLBACKEND", "Agg")
    plt, cmap = setup_style()

    labels = args.labels or [
        os.path.splitext(os.path.basename(p))[0] for p in args.pkls
    ]
    if len(labels) != len(args.pkls):
        print(f"[sc] ERROR: {len(labels)} labels for {len(args.pkls)} pickles",
              file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.pkls[0]))[0]

    results, summaries = [], []
    for pkl, label in zip(args.pkls, labels):
        if not os.path.exists(pkl):
            print(f"[sc] ERROR: {pkl} not found", file=sys.stderr)
            return 2
        print(f"\n[sc] === {label}  ({pkl}) ===")
        res = cs.analyze(
            pkl, min_D=args.min_D, max_D=args.max_D, L0=args.L0,
            n_grid=args.n_grid, x_min=args.x_min, spacing=args.spacing,
            ddof=args.ddof, estimator=args.estimator,
            min_seeds=args.min_seeds, verbose=False,
        )
        results.append(res)

        base = os.path.join(args.out_dir, f"{name}_{label}"
                            if len(args.pkls) > 1 else name)
        summary = summary_dict(res, label)
        summaries.append(summary)
        print_table(summary)

        plot_deviation(res, label, base, plt, cmap, args)
        plot_ratio(res, label, base, plt, args)
        write_csv(res, base + "_deviation.csv")
        with open(base + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[sc] wrote {base}_summary.json")

    if len(results) >= 2:
        plot_compare(results, labels, os.path.join(args.out_dir, name),
                     plt, args)
        print_verdict(summaries)

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, TypeError, FileNotFoundError) as e:
        print(f"[sc] DATA ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
