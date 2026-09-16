#!/usr/bin/env python
"""export_weights_npy.py — dump selected weight matrices to .npy.

The §8 destructive controls need the **actual matrix**, not the CSV: an entry
shuffle or a row shuffle cannot be reconstructed from summary statistics.  This
script is the bridge between the pipeline and
``rmt_null_calibration.py controls``.

Usage
-----
    # every q_proj / k_proj in layers 0, 15, 31
    python export_weights_npy.py \\
        --model_path /path/to/local/Llama-3.1-8B \\
        --patterns q_proj k_proj \\
        --layers 0 15 31 \\
        --out weights_npy/

    # then, per matrix
    python rmt_null_calibration.py controls --npy weights_npy/<name>.npy \\
        --reps 30 --bulk-frac 0.7 --out cal/

PRECISION.  Weights are cast to float64 on the way out, from whatever the
checkpoint holds.  If the checkpoint is bf16, that cast does NOT recover the
fp32 matrix -- bf16 rounding moves singular values by ~6% of a mean spacing
(review §2.4), so a bf16 checkpoint and its fp32 cast are different matrices.
Load the fp32 weights if you have them, and keep --dtype fp32 in the pipeline
so the controls and the pipeline see the same object.

DEFINE "THE WEIGHT MATRIX" (review §8.6).  This script exports W as stored.  If
your claim is about the operator the network applies, that is
``W @ diag(rmsnorm_gain)``, which has different singular values.  Either is
defensible; state which, and be consistent between the pipeline, the bands and
the controls.  ``--rmsnorm_gain`` applies the gain if you want that convention.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser("export_weights_npy")
    p.add_argument("--model_path", required=True,
                   help="local HF model directory (offline)")
    p.add_argument("--patterns", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"],
                   help="substrings matched against parameter names")
    p.add_argument("--layers", nargs="*", type=int, default=[],
                   help="layer indices to keep; empty = all")
    p.add_argument("--out", default="weights_npy")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--rmsnorm_gain", action="store_true",
                   help="export W @ diag(gain) instead of W (review §8.6)")
    p.add_argument("--max_matrices", type=int, default=0,
                   help="stop after N matrices (0 = no limit)")
    a = p.parse_args()

    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("this script needs torch + transformers (the rest of the suite "
              "does not)", file=sys.stderr)
        return 1

    dt = {"fp32": torch.float32, "bf16": torch.bfloat16,
          "fp16": torch.float16}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=dt, local_files_only=True)

    os.makedirs(a.out, exist_ok=True)
    gains = {}
    if a.rmsnorm_gain:
        for name, prm in model.named_parameters():
            if "norm" in name and prm.ndim == 1:
                gains[name] = prm.detach().float().cpu().numpy()

    n_done = 0
    for name, prm in model.named_parameters():
        if prm.ndim != 2:
            continue
        if not any(pat in name for pat in a.patterns):
            continue
        if a.layers:
            m = re.search(r"layers\.(\d+)\.", name)
            if not m or int(m.group(1)) not in a.layers:
                continue

        W = prm.detach().float().cpu().numpy().astype(np.float64)
        if a.rmsnorm_gain:
            layer = re.search(r"layers\.(\d+)\.", name)
            g = None
            if layer:
                pref = f"model.layers.{layer.group(1)}."
                for gname, gv in gains.items():
                    if gname.startswith(pref) and gv.size == W.shape[1]:
                        g = gv
                        break
            if g is None:
                print(f"  !! no matching rmsnorm gain for {name}; exporting W "
                      f"unscaled -- do NOT mix conventions in one run")
            else:
                W = W * g[None, :]

        safe = name.replace(".", "_").replace("/", "_")
        path = os.path.join(a.out, f"{safe}.npy")
        np.save(path, W)
        print(f"  wrote {path}  shape={W.shape}  dtype=float64")
        n_done += 1
        if a.max_matrices and n_done >= a.max_matrices:
            break

    if n_done == 0:
        print("no matrices matched --patterns/--layers", file=sys.stderr)
        return 1
    print(f"\n{n_done} matrices written to {a.out}/")
    print("next: python rmt_null_calibration.py controls --npy "
          f"{a.out}/<name>.npy --reps 30 --bulk-frac 0.7 --out cal/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
