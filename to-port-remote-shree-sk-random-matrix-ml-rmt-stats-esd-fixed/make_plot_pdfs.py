#!/usr/bin/env python3
"""make_plot_pdfs.py — stack the RMT per-matrix plots into contact-sheet PDFs.

Reads the output tree written by run_rmt_supercollapse.slurm:

    RMT_Out_D1024/
        mlp_D1024_N5_seed0_frac0p05/
            esd/model_layers_0_mlp_fc1_weight.png
            hill/…  porter_thomas/…  porter_thomas_decile/…
            spacing/…  rigidity/…

and writes one PDF per plot type:

    RMT_Out_D1024/pdfs/D1024_esd.pdf
    RMT_Out_D1024/pdfs/D1024_hill.pdf
    …

Each PAGE is one (seed, layer, fc) combination, with the training fractions
laid out left-to-right, top-to-bottom. So page 1 of the ESD pdf is seed 0,
layer 0, fc1 across every fraction; page 2 is seed 0, layer 0, fc2; and so on
through all seeds and layers.

USAGE
    cd /home/shivansh/remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed
    python make_plot_pdfs.py                          # defaults to D=1024
    python make_plot_pdfs.py --D 384
    python make_plot_pdfs.py --root RMT_Out_D1024 --tag mlp --seeds 0 1 2
    python make_plot_pdfs.py --fracs 0.05 0.10 0.20 0.50 1.00    # a subset
    python make_plot_pdfs.py --kinds esd hill                    # a subset
    python make_plot_pdfs.py --nomup                             # ablation tree

NOTES
  * The grid is 3 columns wide by default and grows downward to fit however
    many fractions there are — 11 fractions gives 4 rows (12 cells, one blank).
    A 3x3 grid cannot hold 10 or 11 fractions, which is why it is not fixed.
  * frac 0.00 is excluded by default: fc2 is zero-initialised, so no fc2 plot
    exists at frac 0, and the fc1 plot there is the untrained init. Pass
    --include-zero to keep it.
  * A missing PNG leaves a labelled blank cell rather than shifting every
    later image into the wrong slot.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# plot sub-directory -> (pdf stem, human title)
KINDS = {
    "esd":                  ("esd", "ESD / Marchenko-Pastur"),
    "hill":                 ("hill", "Hill estimator"),
    "porter_thomas":        ("porter_thomas", "Porter-Thomas"),
    "porter_thomas_decile": ("porter_thomas_decile", "Porter-Thomas by decile"),
    "spacing":              ("spacing", "Nearest-neighbour spacing"),
    "rigidity":             ("rigidity", "Spectral rigidity"),
}

ROLE = {"fc1": "U", "fc2": "D"}
_FRAC_RE = re.compile(r"_frac(\d+)p(\d+)$")


def frac_tag(f: float) -> str:
    return f"{f:.2f}".replace(".", "p")


def run_dir(root: str, tag: str, D: int, N: int, seed: int, frac: float) -> str:
    return os.path.join(root, f"{tag}_D{D}_N{N}_seed{seed}_frac{frac_tag(frac)}")


def discover_fracs(root: str, tag: str, D: int, N: int, seed: int) -> list:
    """Training fractions actually present on disk for this seed."""
    out = []
    prefix = f"{tag}_D{D}_N{N}_seed{seed}"
    for name in os.listdir(root):
        if not name.startswith(prefix + "_frac"):
            continue
        if not os.path.isdir(os.path.join(root, name)):
            continue
        m = _FRAC_RE.search(name)
        if m:
            out.append(float(f"{m.group(1)}.{m.group(2)}"))
    return sorted(set(out))


def discover_layers(root: str, tag: str, D: int, N: int, seed: int,
                    fracs: list) -> list:
    """Layer indices that actually have plots, from any fraction/kind."""
    pat = re.compile(r"model_layers_(\d+)_mlp_fc[12]_weight\.png$")
    found = set()
    for f in fracs:
        d = run_dir(root, tag, D, N, seed, f)
        for kind in KINDS:
            sub = os.path.join(d, kind)
            if not os.path.isdir(sub):
                continue
            for fn in os.listdir(sub):
                m = pat.match(fn)
                if m:
                    found.add(int(m.group(1)))
        if found:
            break
    return sorted(found)


def draw_page(pdf, paths_and_labels, suptitle, cols, cell_w, cell_h):
    """One contact-sheet page: images in a cols-wide grid, each labelled."""
    n = len(paths_and_labels)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * cell_w, rows * cell_h + 0.5))
    axes = [axes] if rows * cols == 1 else list(axes.ravel())

    for ax in axes:
        ax.set_axis_off()

    for ax, (path, label) in zip(axes, paths_and_labels):
        if path and os.path.exists(path):
            try:
                ax.imshow(mpimg.imread(path))
            except Exception as e:                      # unreadable / truncated
                ax.text(0.5, 0.5, f"unreadable\n{e.__class__.__name__}",
                        ha="center", va="center", fontsize=7, color="crimson",
                        transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    fontsize=9, color="0.6", style="italic",
                    transform=ax.transAxes)
            ax.set_facecolor("0.97")
            ax.patch.set_visible(True)
        ax.set_title(label, fontsize=10, pad=4)

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    # rect leaves room for the suptitle; tight_layout keeps cells from touching
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.4, w_pad=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Stack RMT plots into per-plot-type PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--D", type=int, default=1024, help="model width")
    p.add_argument("--N", type=int, default=5, help="number of MLP blocks")
    p.add_argument("--root", default=None,
                   help="results dir (default: RMT_Out_D<D>, or "
                        "nomup_RMT_Out_D<D> with --nomup)")
    p.add_argument("--out", default=None,
                   help="pdf output dir (default: <root>/pdfs)")
    p.add_argument("--tag", default=None,
                   help="filename tag (default: mlp, or mlp_no_mup with --nomup)")
    p.add_argument("--nomup", action="store_true",
                   help="use the ablation tree and tag")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fracs", type=float, nargs="+", default=None,
                   help="fractions to show (default: all found on disk)")
    p.add_argument("--include-zero", action="store_true",
                   help="keep frac 0.00 (excluded by default)")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="layer indices (default: all found)")
    p.add_argument("--fcs", nargs="+", default=["fc1", "fc2"],
                   choices=["fc1", "fc2"])
    p.add_argument("--kinds", nargs="+", default=list(KINDS),
                   choices=list(KINDS))
    p.add_argument("--cols", type=int, default=3, help="images per row")
    p.add_argument("--cell-w", type=float, default=5.0,
                   help="cell width in inches")
    p.add_argument("--cell-h", type=float, default=3.6,
                   help="cell height in inches")
    a = p.parse_args(argv)

    tag = a.tag or ("mlp_no_mup" if a.nomup else "mlp")
    root = a.root or (f"nomup_RMT_Out_D{a.D}" if a.nomup else f"RMT_Out_D{a.D}")
    root = os.path.abspath(root)
    out_dir = os.path.abspath(a.out or os.path.join(root, "pdfs"))

    if not os.path.isdir(root):
        print(f"error: results dir not found: {root}", file=sys.stderr)
        return 2
    os.makedirs(out_dir, exist_ok=True)

    # --- resolve fractions and layers per seed ----------------------------- #
    fracs_by_seed = {}
    for s in a.seeds:
        fr = a.fracs if a.fracs is not None else discover_fracs(
            root, tag, a.D, a.N, s)
        if not a.include_zero:
            fr = [f for f in fr if round(f, 2) != 0.0]
        fracs_by_seed[s] = sorted(fr)

    seeds = [s for s in a.seeds if fracs_by_seed[s]]
    if not seeds:
        print(f"error: no run directories matching "
              f"{tag}_D{a.D}_N{a.N}_seed*_frac* under {root}", file=sys.stderr)
        return 2
    for s in a.seeds:
        if not fracs_by_seed[s]:
            print(f"warning: seed {s} has no fraction directories; skipping")

    layers = a.layers
    if layers is None:
        layers = discover_layers(root, tag, a.D, a.N, seeds[0],
                                 fracs_by_seed[seeds[0]])
    if not layers:
        print("error: no layer plots found (looked for "
              "model_layers_<i>_mlp_fc<n>_weight.png)", file=sys.stderr)
        return 2

    print(f"root   : {root}")
    print(f"seeds  : {seeds}")
    print(f"layers : {layers}   fcs: {a.fcs}")
    print(f"fracs  : {fracs_by_seed[seeds[0]]}")
    print(f"grid   : {a.cols} cols -> "
          f"{(len(fracs_by_seed[seeds[0]]) + a.cols - 1)//a.cols} rows/page")
    print(f"out    : {out_dir}\n")

    # --- one pdf per plot kind --------------------------------------------- #
    for kind in a.kinds:
        stem, nice = KINDS[kind]
        pdf_path = os.path.join(out_dir, f"D{a.D}_{stem}"
                                + ("_no_mup" if a.nomup else "") + ".pdf")
        pages = missing = 0
        with PdfPages(pdf_path) as pdf:
            for seed in seeds:
                for layer in layers:
                    for fc in a.fcs:
                        cells = []
                        for f in fracs_by_seed[seed]:
                            img = os.path.join(
                                run_dir(root, tag, a.D, a.N, seed, f), kind,
                                f"model_layers_{layer}_mlp_{fc}_weight.png")
                            if not os.path.exists(img):
                                missing += 1
                                img = None
                            cells.append((img, f"frac {f:.2f}"))
                        title = (f"{nice}  —  D={a.D}  seed {seed}  "
                                 f"layer {layer}  {fc} (\"{ROLE[fc]}\")"
                                 + ("  [no-muP]" if a.nomup else ""))
                        draw_page(pdf, cells, title, a.cols, a.cell_w, a.cell_h)
                        pages += 1
            d = pdf.infodict()
            d["Title"] = f"{nice} — D={a.D}" + (" (no-muP)" if a.nomup else "")
            d["Subject"] = (f"seeds={seeds} layers={layers} fcs={a.fcs} "
                            f"fracs={fracs_by_seed[seeds[0]]}")
        size_mb = os.path.getsize(pdf_path) / 1e6
        note = f"   ({missing} missing panels)" if missing else ""
        print(f"{pdf_path}   {pages} pages, {size_mb:.1f} MB{note}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
