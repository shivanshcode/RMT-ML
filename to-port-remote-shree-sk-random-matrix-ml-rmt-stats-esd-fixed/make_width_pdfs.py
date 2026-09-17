#!/usr/bin/env python3
"""make_width_pdfs.py — contact sheets across MODEL WIDTH at a fixed fraction.

Companion to make_plot_pdfs.py. That one holds the width fixed and tiles the
training fractions; this one holds the fraction fixed (default 1.00, the fully
trained checkpoint) and tiles the widths, which is the comparison supercollapse
is actually about.

Reads the per-width trees:

    RMT_Out_D384/mlp_D384_N5_seed0_frac1p00/esd/model_layers_0_mlp_fc1_weight.png
    RMT_Out_D512/mlp_D512_N5_seed0_frac1p00/esd/…
    …
    RMT_Out_D2048/…

and writes one PDF per plot type:

    pdfs_by_width/width_esd.pdf
    pdfs_by_width/width_hill.pdf
    …

Each PAGE is one (seed, layer, fc) combination, with the eight widths laid out
in a FIXED 3 x 3 grid (8 panels used, 1 blank). Page 1 of the ESD pdf is
seed 0, layer 0, fc1 across D = 384 … 2048; page 2 is seed 0, layer 0, fc2.
30 pages per pdf: 3 seeds x 5 layers x 2 fc.

USAGE
    cd /home/shivansh/remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed
    python make_width_pdfs.py                      # muP, frac 1.00, seeds 0 1 2
    python make_width_pdfs.py --nomup              # the ablation trees
    python make_width_pdfs.py --frac 0.50          # a different fraction
    python make_width_pdfs.py --kinds esd hill     # a subset of plot types
    python make_width_pdfs.py --widths 384 1024 2048

NOTES
  * The grid never auto-resizes. More widths than rows*cols spill onto a second
    page rather than shrinking the panels.
  * A width whose PNG is absent (run not finished, or a layer index that width
    does not have) leaves a labelled blank cell, so the remaining panels stay
    in their correct slots and pages stay comparable across the pdf.
  * Panels are ordered by ascending width, left to right, top to bottom.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

KINDS = {
    "esd":                  ("esd", "ESD / Marchenko-Pastur"),
    "hill":                 ("hill", "Hill estimator"),
    "porter_thomas":        ("porter_thomas", "Porter-Thomas"),
    "porter_thomas_decile": ("porter_thomas_decile", "Porter-Thomas by decile"),
    "spacing":              ("spacing", "Nearest-neighbour spacing"),
    "rigidity":             ("rigidity", "Spectral rigidity"),
}

ROLE = {"fc1": "U", "fc2": "D"}
WIDTHS = [384, 512, 645, 812, 1024, 1290, 1625, 2048]


def frac_tag(f: float) -> str:
    return f"{f:.2f}".replace(".", "p")


def png_path(base: str, D: int, tag: str, N: int, seed: int, frac: float,
             kind: str, layer: int, fc: str, nomup: bool) -> str:
    root = os.path.join(base, f"{'nomup_' if nomup else ''}RMT_Out_D{D}")
    run = f"{tag}_D{D}_N{N}_seed{seed}_frac{frac_tag(frac)}"
    return os.path.join(root, run, kind,
                        f"model_layers_{layer}_mlp_{fc}_weight.png")


def draw_page(pdf, cells, suptitle, cols, rows, cell_w, cell_h):
    """One page: a FIXED rows x cols grid. Unused cells are left blank so the
    geometry is identical on every page of the pdf."""
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * cell_w, rows * cell_h + 0.5))
    axes = [axes] if rows * cols == 1 else list(axes.ravel())
    for ax in axes:
        ax.set_axis_off()

    for ax, (path, label) in zip(axes, cells):
        if path and os.path.exists(path):
            try:
                ax.imshow(mpimg.imread(path))
            except Exception as e:
                ax.text(0.5, 0.5, f"unreadable\n{e.__class__.__name__}",
                        ha="center", va="center", fontsize=7, color="crimson",
                        transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    fontsize=9, color="0.6", style="italic",
                    transform=ax.transAxes)
            ax.set_facecolor("0.97")
            ax.patch.set_visible(True)
        ax.set_title(label, fontsize=11, pad=4)

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.4, w_pad=0.8)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Contact sheets across model width at a fixed fraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base", default=".",
                   help="dir containing the RMT_Out_D* / nomup_RMT_Out_D* trees")
    p.add_argument("--out", default=None,
                   help="pdf output dir (default: <base>/pdfs_by_width)")
    p.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    p.add_argument("--frac", type=float, default=1.00,
                   help="training fraction to show")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--fcs", nargs="+", default=["fc1", "fc2"],
                   choices=["fc1", "fc2"])
    p.add_argument("--kinds", nargs="+", default=list(KINDS),
                   choices=list(KINDS))
    p.add_argument("--N", type=int, default=5)
    p.add_argument("--tag", default=None,
                   help="filename tag (default: mlp, or mlp_no_mup with --nomup)")
    p.add_argument("--nomup", action="store_true",
                   help="use the nomup_RMT_Out_D* trees and the ablation tag")
    p.add_argument("--cols", type=int, default=3, help="panels per row")
    p.add_argument("--rows", type=int, default=3, help="panel rows per page")
    p.add_argument("--cell-w", type=float, default=5.0)
    p.add_argument("--cell-h", type=float, default=3.6)
    a = p.parse_args(argv)

    base = os.path.abspath(a.base)
    tag = a.tag or ("mlp_no_mup" if a.nomup else "mlp")
    out_dir = os.path.abspath(
        a.out or os.path.join(base, "pdfs_by_width"
                              + ("_no_mup" if a.nomup else "")))

    # --- check the trees exist before building anything -------------------- #
    widths, absent = [], []
    for D in sorted(a.widths):
        root = os.path.join(base, f"{'nomup_' if a.nomup else ''}RMT_Out_D{D}")
        run = os.path.join(root, f"{tag}_D{D}_N{a.N}_seed{a.seeds[0]}"
                                 f"_frac{frac_tag(a.frac)}")
        (widths if os.path.isdir(run) else absent).append(D)
    if absent:
        print(f"warning: no frac{frac_tag(a.frac)} dir for width(s) "
              f"{absent} — they will render as blank panels")
        widths = sorted(a.widths)
    if not widths:
        print(f"error: found no run directories under {base}", file=sys.stderr)
        return 2
    os.makedirs(out_dir, exist_ok=True)

    per_page = a.cols * a.rows
    n_pages_per_combo = (len(widths) + per_page - 1) // per_page
    print(f"base   : {base}")
    print(f"widths : {widths}")
    print(f"frac   : {a.frac:.2f}   seeds: {a.seeds}   layers: {a.layers}   "
          f"fcs: {a.fcs}")
    print(f"grid   : fixed {a.rows} x {a.cols} = {per_page} panels/page"
          + (f"  ({len(widths)} widths -> {n_pages_per_combo} pages per "
             f"seed/layer/fc)" if n_pages_per_combo > 1 else ""))
    print(f"out    : {out_dir}\n")

    for kind in a.kinds:
        stem, nice = KINDS[kind]
        name = (f"width_{stem}_frac{frac_tag(a.frac)}"
                + ("_no_mup" if a.nomup else "") + ".pdf")
        pdf_path = os.path.join(out_dir, name)
        pages = missing = 0
        with PdfPages(pdf_path) as pdf:
            for seed in a.seeds:
                for layer in a.layers:
                    for fc in a.fcs:
                        cells = []
                        for D in widths:
                            img = png_path(base, D, tag, a.N, seed, a.frac,
                                           kind, layer, fc, a.nomup)
                            if not os.path.exists(img):
                                missing += 1
                                img = None
                            cells.append((img, f"D = {D}"))
                        chunks = [cells[i:i + per_page]
                                  for i in range(0, len(cells), per_page)] or [[]]
                        for k, chunk in enumerate(chunks, 1):
                            title = (f"{nice}  —  frac {a.frac:.2f}  "
                                     f"seed {seed}  layer {layer}  "
                                     f"{fc} (\"{ROLE[fc]}\")"
                                     + ("  [no-muP]" if a.nomup else "")
                                     + (f"   page {k}/{len(chunks)}"
                                        if len(chunks) > 1 else ""))
                            draw_page(pdf, chunk, title, a.cols, a.rows,
                                      a.cell_w, a.cell_h)
                            pages += 1
            d = pdf.infodict()
            d["Title"] = (f"{nice} across width — frac {a.frac:.2f}"
                          + (" (no-muP)" if a.nomup else ""))
            d["Subject"] = (f"widths={widths} seeds={a.seeds} "
                            f"layers={a.layers} fcs={a.fcs}")
        note = f"   ({missing} missing panels)" if missing else ""
        print(f"{pdf_path}   {pages} pages, "
              f"{os.path.getsize(pdf_path)/1e6:.1f} MB{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
