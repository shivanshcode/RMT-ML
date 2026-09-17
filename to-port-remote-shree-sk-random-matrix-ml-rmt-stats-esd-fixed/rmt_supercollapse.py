"""rmt_supercollapse.py — feed supercollapse MLP checkpoints into the RMT-ML suite.

Drop this next to your RMT package (same directory as the ``rmt/`` folder).

The training side writes one ``.npz`` per checkpoint, with keys already named
``model.layers.{i}.mlp.fc{1,2}.weight`` and weights already in torch
``(out, in)`` order. This module wraps them in a plain ``nn.Module`` so that
``rmt.discovery.discover_weight_matrices`` finds them via the generic spec:
``fc1 -> "U"``, ``fc2 -> "D"``, layer index from ``\\.layers?\\.(\\d+)\\.``.

Verified against the RMT suite: ``get_model_spec`` resolves to ``generic``,
discovery returns 2N records with the right ``short``/``layer_idx``, and
``analyze_one_model`` emits the 133-column CSV plus the esd / hill /
porter_thomas / porter_thomas_decile / spacing / rigidity plot trees.

--------------------------------------------------------------------------
WHAT CHANGED vs. the first version (all of it forced by the actual suite)
--------------------------------------------------------------------------
1. ZERO MATRICES ARE DROPPED.  ``fc2`` is zero-initialised, so at ``frac0p00``
   every "D" matrix is exactly 0.  ``per_matrix_analysis`` then dies inside
   ``rmt.mp.mp_median`` -> ``scipy.optimize.brentq``: "f(a) and f(b) must have
   different signs" (sigma = 0 makes the MP support degenerate).  The pipeline
   catches it and silently drops the row, so you get a CSV that is missing half
   its matrices with no explanation.  ``load_npz_model`` now skips all-zero
   matrices up front and records them in ``model.rmt_dropped``.

2. STABLE RANK IS TRACKED PER ROLE, NOT AVERAGED.  ``rmt.pipeline.analyze_
   checkpoints`` averages every record in a layer, i.e. mean(fc1, fc2).  At
   frac 0 that is mean(fc1, 0) = half the fc1 value — the exact artifact the
   handoff warns about.  ``track_scalars`` replaces it with a tidy long-format
   CSV keyed by (frac, layer, short), so U and D never mix.

3. DELTA MODELS ARE BUILT IN MEMORY, in float64.  The old version wrote a
   temp ``.npz`` into the checkpoint directory (fails on a read-only or full
   scratch mount) and did the subtraction in the stored fp32.

4. META PARSING IS ROBUST to dict / str / json / 0-d object array.

5. A DRIVER + CLI.  ``analyze_width`` sweeps seeds x fractions, gives each
   (seed, frac) its own output directory so the plot filenames never collide,
   and concatenates every per-matrix CSV into one tidy table with D / seed /
   frac / step columns prepended — which is what you actually plot from.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
Command line (the D=384 sweep, all seeds, all fractions):

    python rmt_supercollapse.py --ckpt-dir ~/supercollapse/ckpts \\
        --D 384 --seeds 0 1 2 --out ./RMT_Out_D384

Same thing on the update matrices W(t) - W(0):

    python rmt_supercollapse.py --ckpt-dir ~/supercollapse/ckpts \\
        --D 384 --seeds 0 1 2 --out ./RMT_Out_D384_delta --delta

Library, single checkpoint:

    from rmt_supercollapse import load_npz_model
    from rmt.pipeline import analyze_one_model
    m = load_npz_model("ckpts/mlp_D384_N5_seed0_mupTrue_frac1p00.npz")
    analyze_one_model(m, "mlp_D384_seed0_frac1.00", "./RMT_Out",
                      do_overlap=False, do_perplexity=False)

NOTE: activation-covariance (``do_overlap``) and perplexity blocks need a
runnable forward pass and a tokenizer.  These checkpoints are weights only, so
both stay False everywhere in this module.
"""
from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "load_npz_model", "load_delta_model", "list_checkpoints",
    "checkpoint_loader_for", "track_scalars", "analyze_run", "analyze_width",
    "frac_tag", "read_meta",
]

_KEY_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.(fc[12])\.weight$")
_FRAC_RE = re.compile(r"_frac(\d+)p(\d+)\.npz$")


# --------------------------------------------------------------------------- #
# model construction                                                          #
# --------------------------------------------------------------------------- #
class _Cfg:
    """Minimal config so rmt.discovery.get_model_spec() resolves to _GENERIC.

    'supercollapse_mlp' contains none of the registry keys (llama / gpt2 /
    qwen / bert / gpt_neox), so the generic spec is returned, which is the one
    whose MATRIX_PATTERNS map fc1 -> U and fc2 -> D.
    """

    def __init__(self, meta: Optional[dict] = None):
        self.model_type = "supercollapse_mlp"
        self.architectures = ["SupercollapseMLP"]
        self.num_attention_heads = None      # no fused QKV anywhere
        self.hidden_size = (meta or {}).get("D")
        self.num_hidden_layers = (meta or {}).get("N")


def read_meta(z) -> dict:
    """``__meta__`` -> dict, whatever np.savez turned it into."""
    if "__meta__" not in getattr(z, "files", []):
        return {}
    raw = z["__meta__"]
    try:
        raw = raw.item()
    except (AttributeError, ValueError):
        pass
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        for parser in (ast.literal_eval, json.loads):
            try:
                out = parser(raw)
                if isinstance(out, dict):
                    return out
            except Exception:
                continue
    return {}


def _arrays_from_npz(z) -> Dict[int, Dict[str, np.ndarray]]:
    layers: Dict[int, Dict[str, np.ndarray]] = {}
    for k in z.files:
        if k == "__meta__":
            continue
        m = _KEY_RE.match(k)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = z[k]
    return layers


def _build_model(layers: Dict[int, Dict[str, np.ndarray]], meta: dict,
                 *, drop_zero: bool = True, source: str = "weight") -> nn.Module:
    """Wrap {layer: {fc1: W, fc2: W}} in a module whose named_modules() match.

    Weights are held in float64.  ``rmt.discovery._weight_2d`` upcasts to
    float64 anyway; holding them there means a delta W(t)-W(0) is not rounded
    back down to fp32 on the way in.
    """
    model = nn.Module()
    model.config = _Cfg(meta)
    model.model = nn.Module()
    blocks = nn.ModuleList()
    dropped: List[str] = []

    for i in sorted(layers):
        blk = nn.Module()
        mlp = nn.Module()
        for which in ("fc1", "fc2"):
            if which not in layers[i]:
                continue
            W = np.ascontiguousarray(layers[i][which], dtype=np.float64)
            if W.ndim != 2:
                dropped.append(f"model.layers.{i}.mlp.{which}.weight[ndim={W.ndim}]")
                continue
            if drop_zero and not np.any(W):
                # fc2 is zero-initialised (mlp.py: nnx.update(fc2.kernel, zeros)).
                # sigma = 0 -> MP support degenerates -> brentq raises inside
                # rmt.mp.mp_median and the pipeline drops the row anyway, but
                # silently.  Drop it here, loudly, instead.
                dropped.append(f"model.layers.{i}.mlp.{which}.weight")
                continue
            lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
            lin.weight = nn.Parameter(torch.from_numpy(W), requires_grad=False)
            setattr(mlp, which, lin)
        blk.mlp = mlp
        blocks.append(blk)

    model.model.layers = blocks
    model.rmt_meta = dict(meta)
    model.rmt_dropped = dropped
    model.rmt_source = source
    model.eval()
    return model


def load_npz_model(path: str, *, drop_zero: bool = True) -> nn.Module:
    """Rebuild an nn.Module whose named_modules() match the RMT patterns."""
    with np.load(path, allow_pickle=True) as z:
        layers = _arrays_from_npz(z)
        meta = read_meta(z)
        if not layers:
            raise ValueError(
                f"{path}: no keys matched model.layers.N.mlp.fcX.weight")
        layers = {i: {k: np.asarray(v) for k, v in d.items()}
                  for i, d in layers.items()}
    return _build_model(layers, meta, drop_zero=drop_zero, source="weight")


# --------------------------------------------------------------------------- #
# checkpoint enumeration                                                      #
# --------------------------------------------------------------------------- #
def frac_tag(frac: float) -> str:
    """0.05 -> '0p05', 1.0 -> '1p00' (the training side's filename convention)."""
    return f"{frac:.2f}".replace(".", "p")


def list_checkpoints(ckpt_dir: str, run_prefix: str) -> List[Tuple[float, str]]:
    """[(frac, path), ...] sorted by training fraction."""
    out: List[Tuple[float, str]] = []
    for p in glob.glob(os.path.join(ckpt_dir, f"{run_prefix}*_frac*.npz")):
        base = os.path.basename(p)
        if base.startswith("."):
            continue
        m = _FRAC_RE.search(base)
        if m:
            out.append((float(f"{m.group(1)}.{m.group(2)}"), p))
    return sorted(out)


def checkpoint_loader_for(ckpt_dir: str, run_prefix: str, *, drop_zero: bool = True
                          ) -> Tuple[Callable[[float], nn.Module], List[float]]:
    """Return (loader, fracs) ready for rmt.pipeline.analyze_checkpoints.

    Prefer :func:`track_scalars` over ``analyze_checkpoints`` — the latter
    averages fc1 and fc2 within a layer, which is meaningless here.
    """
    items = list_checkpoints(ckpt_dir, run_prefix)
    if not items:
        raise FileNotFoundError(f"no checkpoints for '{run_prefix}' in {ckpt_dir}")
    table = {f: p for f, p in items}

    def loader(frac: float) -> nn.Module:
        key = frac if frac in table else min(table, key=lambda f: abs(f - frac))
        return load_npz_model(table[key], drop_zero=drop_zero)

    return loader, [f for f, _ in items]


def load_delta_model(ckpt_dir: str, run_prefix: str, frac: float,
                     *, drop_zero: bool = True) -> nn.Module:
    """Model whose weights are ``W(frac) - W(0)`` — the update matrix.

    Under muP the update, not the raw weight, carries the width-scaling
    prediction; its spectrum is usually where the structure is.  Built in
    memory at float64 — nothing is written to ``ckpt_dir``.

    For fc2 the delta is identical to the raw weight (W_init = 0), so delta
    analysis is only distinct for fc1.
    """
    table = dict(list_checkpoints(ckpt_dir, run_prefix))
    if 0.0 not in table:
        raise FileNotFoundError(f"{run_prefix}: no frac0p00 checkpoint to subtract")
    nearest = frac if frac in table else min(table, key=lambda f: abs(f - frac))

    with np.load(table[0.0], allow_pickle=True) as z0, \
            np.load(table[nearest], allow_pickle=True) as z1:
        a0 = _arrays_from_npz(z0)
        a1 = _arrays_from_npz(z1)
        meta = read_meta(z1)
        layers: Dict[int, Dict[str, np.ndarray]] = {}
        for i, d in a1.items():
            for which, W in d.items():
                if i in a0 and which in a0[i]:
                    layers.setdefault(i, {})[which] = (
                        np.asarray(W, dtype=np.float64)
                        - np.asarray(a0[i][which], dtype=np.float64))
    meta = dict(meta)
    meta["delta_from_frac"] = 0.0
    return _build_model(layers, meta, drop_zero=drop_zero, source="delta")


# --------------------------------------------------------------------------- #
# per-role scalar tracking (replaces analyze_checkpoints)                      #
# --------------------------------------------------------------------------- #
_SCALAR_FIELDS = [
    "frac", "step", "layer", "short", "matrix", "n", "m",
    "frob_norm", "spectral_norm", "stable_rank", "spectral_entropy",
    "sigma_med", "mp_plus", "n_right_outliers", "mp_softrank", "bulk_mass_frac",
]


def track_scalars(ckpt_dir: str, run_prefix: str, *, probe_layers=None,
                  fracs: Optional[Sequence[float]] = None,
                  delta: bool = False, out_csv: Optional[str] = None) -> List[dict]:
    """Cheap spectral scalars per (fraction, layer, role) — long/tidy format.

    One SVD per matrix, no plots.  This is the honest replacement for
    ``rmt.pipeline.analyze_checkpoints``: it never averages fc1 with fc2, and
    it reports zero matrices as zeros rather than dragging a layer mean down.
    """
    from rmt import mp as MP
    from rmt.scalars import (stable_rank, spectral_entropy, mp_softrank,
                             bulk_mass_frac)

    items = list_checkpoints(ckpt_dir, run_prefix)
    if not items:
        raise FileNotFoundError(f"no checkpoints for '{run_prefix}' in {ckpt_dir}")
    if fracs is not None:
        keep = {round(f, 2) for f in fracs}
        items = [(f, p) for f, p in items if round(f, 2) in keep]

    want = set(probe_layers) if probe_layers is not None else None
    rows: List[dict] = []

    base = dict(list_checkpoints(ckpt_dir, run_prefix))
    zero_ref = None
    if delta:
        if 0.0 not in base:
            raise FileNotFoundError(f"{run_prefix}: no frac0p00 to subtract")
        with np.load(base[0.0], allow_pickle=True) as z0:
            zero_ref = {i: {k: np.asarray(v, dtype=np.float64)
                            for k, v in d.items()}
                        for i, d in _arrays_from_npz(z0).items()}

    for frac, path in items:
        with np.load(path, allow_pickle=True) as z:
            layers = _arrays_from_npz(z)
            meta = read_meta(z)
        for i in sorted(layers):
            if want is not None and i not in want:
                continue
            for which, short in (("fc1", "U"), ("fc2", "D")):
                if which not in layers[i]:
                    continue
                W = np.asarray(layers[i][which], dtype=np.float64)
                if delta:
                    W = W - zero_ref[i][which]
                n, m = W.shape
                row = {f: float("nan") for f in _SCALAR_FIELDS}
                row.update({
                    "frac": frac, "step": meta.get("step", ""), "layer": i,
                    "short": short,
                    "matrix": f"model.layers.{i}.mlp.{which}.weight",
                    "n": n, "m": m,
                    "frob_norm": float(np.linalg.norm(W)),
                })
                if np.any(W):
                    s = np.sort(np.linalg.svd(W, compute_uv=False))[::-1]
                    sig = MP.estimate_sigma_gd_median(s=s, n=n, m=m)
                    _, mp_plus = MP.mp_bounds(n, m, sig)
                    row.update({
                        "spectral_norm": float(s[0]),
                        "stable_rank": float(stable_rank(s=s)),
                        "spectral_entropy": float(spectral_entropy(s)),
                        "sigma_med": float(sig),
                        "mp_plus": float(mp_plus),
                        "n_right_outliers": int(np.sum(s > mp_plus)),
                        "mp_softrank": float(mp_softrank(s, mp_plus)),
                        "bulk_mass_frac": float(bulk_mass_frac(s, mp_plus)),
                    })
                else:
                    # exact zero (fc2 at frac 0) — report it, don't average it
                    row.update({"spectral_norm": 0.0, "stable_rank": 0.0})
                rows.append(row)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_SCALAR_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- #
# drivers                                                                     #
# --------------------------------------------------------------------------- #
_PIPELINE_FLAGS = dict(
    do_overlap=False,        # needs a forward pass — weights only here
    do_perplexity=False,     # needs a tokenizer — weights only here
    do_qkv_heatmap=False,    # no attention in this MLP
    do_ipr=True,
    do_powerlaw=True,        # -> hill/ plots + alpha columns
    do_spacing=True,         # -> spacing/ + rigidity/ plots
    do_porter_thomas=True,   # -> porter_thomas/ + porter_thomas_decile/ plots
    do_ww=False,
    offline=True,
)


def analyze_run(ckpt_dir: str, run_prefix: str, output_dir: str, *,
                fracs: Optional[Sequence[float]] = None, delta: bool = False,
                layers: Optional[Sequence[int]] = None,
                extra_flags: Optional[dict] = None,
                verbose: bool = True) -> List[dict]:
    """Full RMT analysis of one (D, seed) run across training fractions.

    Each fraction gets its own output directory, so the per-matrix plot
    filenames (which are derived from the matrix name alone) never collide.
    Returns every per-matrix row, with D / seed / frac / step prepended.
    """
    from rmt.pipeline import analyze_one_model

    items = list_checkpoints(ckpt_dir, run_prefix)
    if not items:
        raise FileNotFoundError(f"no checkpoints for '{run_prefix}' in {ckpt_dir}")
    if fracs is not None:
        keep = {round(f, 2) for f in fracs}
        items = [(f, p) for f, p in items if round(f, 2) in keep]

    flags = dict(_PIPELINE_FLAGS)
    if layers:
        flags["layers"] = list(layers)
    if extra_flags:
        flags.update(extra_flags)

    all_rows: List[dict] = []
    for frac, path in items:
        if delta and frac == 0.0:
            continue                       # W(0) - W(0) = 0, nothing to analyse
        tag = f"{run_prefix}_frac{frac_tag(frac)}" + ("_delta" if delta else "")
        out = os.path.join(output_dir, tag)
        t0 = time.time()
        model = (load_delta_model(ckpt_dir, run_prefix, frac) if delta
                 else load_npz_model(path))
        if verbose and model.rmt_dropped:
            print(f"  [{tag}] dropped all-zero: "
                  f"{', '.join(model.rmt_dropped)}", flush=True)
        _, rows = analyze_one_model(model, tag, out, **flags)
        meta = model.rmt_meta
        for r in rows:
            r.update({
                "run": run_prefix, "source": "delta" if delta else "weight",
                "frac": frac, "step": meta.get("step", ""),
                "D": meta.get("D", ""), "N": meta.get("N", ""),
                "seed": meta.get("seed", ""), "mup": meta.get("mup", ""),
                "num_params": meta.get("num_params", ""),
            })
        all_rows.extend(rows)
        del model
        if verbose:
            print(f"  [{tag}] {len(rows)} matrices in "
                  f"{time.time() - t0:.1f}s -> {out}", flush=True)
    return all_rows


_PREPEND = ["run", "source", "D", "N", "seed", "mup", "frac", "step", "num_params"]


def _write_combined(rows: List[dict], path: str) -> str:
    from rmt.per_matrix import CSV_COLUMNS
    cols = _PREPEND + [c for c in CSV_COLUMNS if c not in _PREPEND]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return path


def analyze_width(ckpt_dir: str, D: int, seeds: Sequence[int], output_dir: str, *,
                  tag: str = "mlp", N: int = 5,
                  fracs: Optional[Sequence[float]] = None, delta: bool = False,
                  layers: Optional[Sequence[int]] = None,
                  scalars: bool = True, extra_flags: Optional[dict] = None,
                  verbose: bool = True) -> Dict[str, str]:
    """Sweep one width across seeds and fractions; write the combined tables.

    Returns {"metrics": path, "scalars": path or ""}.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_rows: List[dict] = []
    scalar_rows: List[dict] = []
    suffix = "_delta" if delta else ""

    for seed in seeds:
        prefix = f"{tag}_D{D}_N{N}_seed{seed}"
        if verbose:
            print(f"[{prefix}] starting", flush=True)
        all_rows.extend(analyze_run(ckpt_dir, prefix, output_dir, fracs=fracs,
                                    delta=delta, layers=layers,
                                    extra_flags=extra_flags, verbose=verbose))
        if scalars:
            sr = track_scalars(ckpt_dir, prefix, fracs=fracs, delta=delta,
                               probe_layers=layers)
            for r in sr:
                r.update({"run": prefix, "D": D, "N": N, "seed": seed,
                          "source": "delta" if delta else "weight"})
            scalar_rows.extend(sr)

    metrics_path = _write_combined(
        all_rows, os.path.join(output_dir, f"{tag}_D{D}{suffix}_all_matrix_metrics.csv"))
    scalars_path = ""
    if scalar_rows:
        scalars_path = os.path.join(
            output_dir, f"{tag}_D{D}{suffix}_scalars_by_role.csv")
        cols = ["run", "source", "D", "N", "seed"] + _SCALAR_FIELDS
        with open(scalars_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(scalar_rows)
    if verbose:
        print(f"\nwrote {metrics_path} ({len(all_rows)} rows)")
        if scalars_path:
            print(f"wrote {scalars_path} ({len(scalar_rows)} rows)")
    return {"metrics": metrics_path, "scalars": scalars_path}


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="RMT analysis of supercollapse MLP checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="directory holding the .npz files")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--D", type=int, required=True, help="model width")
    p.add_argument("--N", type=int, default=5, help="number of MLP blocks")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tag", default="mlp",
                   help="filename tag ('mlp' or 'mlp_no_mup')")
    p.add_argument("--fracs", type=float, nargs="+", default=None,
                   help="training fractions to analyse (default: all found)")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="restrict to these layer indices (default: all)")
    p.add_argument("--delta", action="store_true",
                   help="analyse W(t) - W(0) instead of W(t)")
    p.add_argument("--no-scalars", action="store_true",
                   help="skip the per-role scalar table")
    p.add_argument("--scalars-only", action="store_true",
                   help="only the cheap scalar sweep; no plots, no 133-col CSV")
    p.add_argument("--complex-spacing", action="store_true",
                   help="also run the GinOE complex spacing ratio (square W only)")
    p.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto",
                   help="SVD backend. 'auto' goes to the GPU only when CUDA is "
                        "visible AND max(n,m) >= --gpu-svd-min-dim (default 1024), "
                        "so D<1024 runs numpy and D>=1024 runs cuSOLVER. Pin "
                        "'numpy' to keep every width on one code path.")
    p.add_argument("--gpu-svd-min-dim", type=int, default=1024,
                   help="dimension at which the 'auto' backend switches to GPU")
    p.add_argument("--quick", action="store_true",
                   help="fewer Porter-Thomas draws and Sigma^2 windows (~3x faster)")
    p.add_argument("--list", action="store_true",
                   help="list the checkpoints that would be analysed, then exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = _parse_args(argv)

    if a.list:
        for seed in a.seeds:
            prefix = f"{a.tag}_D{a.D}_N{a.N}_seed{seed}"
            items = list_checkpoints(a.ckpt_dir, prefix)
            print(f"{prefix}: {len(items)} checkpoints")
            for f, p in items:
                print(f"   frac {f:.2f}  {os.path.basename(p)}")
        return 0

    if a.scalars_only:
        os.makedirs(a.out, exist_ok=True)
        rows: List[dict] = []
        for seed in a.seeds:
            prefix = f"{a.tag}_D{a.D}_N{a.N}_seed{seed}"
            sr = track_scalars(a.ckpt_dir, prefix, fracs=a.fracs,
                               delta=a.delta, probe_layers=a.layers)
            for r in sr:
                r.update({"run": prefix, "D": a.D, "N": a.N, "seed": seed,
                          "source": "delta" if a.delta else "weight"})
            rows.extend(sr)
            print(f"[{prefix}] {len(sr)} rows", flush=True)
        suffix = "_delta" if a.delta else ""
        path = os.path.join(a.out, f"{a.tag}_D{a.D}{suffix}_scalars_by_role.csv")
        cols = ["run", "source", "D", "N", "seed"] + _SCALAR_FIELDS
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")
        return 0

    extra = {"backend": a.backend, "gpu_svd_min_dim": a.gpu_svd_min_dim}
    if a.complex_spacing:
        extra["do_complex_spacing"] = True
    if a.quick:
        extra.update(pt_n_samples=1000, sigma2_n_windows=20_000,
                     brody_bootstrap=50)

    analyze_width(a.ckpt_dir, a.D, a.seeds, a.out, tag=a.tag, N=a.N,
                  fracs=a.fracs, delta=a.delta, layers=a.layers,
                  scalars=not a.no_scalars, extra_flags=extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
