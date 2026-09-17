"""Offline replacement for save_logs.py.

Builds logs/<name>.pkl from the CSV/JSON files written by the local logger in
train.py, in exactly the format figures/collapse.ipynb expects. No wandb, no
network.

Usage:
    python make_logs.py --tag mlp --out logs/mlp.pkl
    python make_logs.py --tag mlp_no_mup --out logs/mlp_no_mup.pkl
"""
import argparse
import glob
import json
import os
import pickle

import pandas as pd

HIST_COLS = ["step", "compute", "test_loss", "lr", "tau", "Buvar"]


def build(log_dir: str, tag: str) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(log_dir, f"{tag}_*.csv"))):
        meta_path = csv_path[:-4] + ".json"
        if not os.path.exists(meta_path):
            print(f"  skip (no meta): {csv_path}")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        hist = pd.read_csv(csv_path)
        if len(hist) < 2:
            print(f"  skip (empty/incomplete): {csv_path}")
            continue

        # compute is logged in raw FLOPs; the notebooks expect petaflops
        hist["compute"] = hist["compute"] / 1e15
        hist = hist[[c for c in HIST_COLS if c in hist.columns]]
        hist = hist.sort_values("step").ffill().dropna(subset=["test_loss"])

        rows.append(
            dict(
                history=hist.reset_index(drop=True),
                num_params=int(meta["num_params"]),
                N=int(meta["N"]),
                D=int(meta["D"]),
                V=int(meta.get("V", 8)),
                L=int(meta.get("L", 1)),
                B=int(meta["B"]),
                lr=float(meta["lr"]),
                seed=int(meta["seed"]),
                schedule=meta.get("schedule", "linear"),
                decay_frac=float(meta.get("decay_frac", 1.0)),
                mup=str(meta.get("mup", True)),
                test_loss=float(hist["test_loss"].iloc[-1]),
                opt_L=float(hist["test_loss"].iloc[-1]),
                opt_C=float(hist["compute"].iloc[-1]),
            )
        )
        print(f"  + D={rows[-1]['D']:>5} seed={rows[-1]['seed']} "
              f"mup={rows[-1]['mup']} evals={len(hist):>4} "
              f"final_loss={rows[-1]['opt_L']:.5f}")

    if not rows:
        raise SystemExit(f"No runs found for tag '{tag}' in {log_dir}/")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="wandb_tag used for the runs")
    p.add_argument("--log-dir", default="local_logs")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out = a.out or f"logs/{a.tag}.pkl"
    print(f"Collecting tag='{a.tag}' from {a.log_dir}/ ...")
    df = build(a.log_dir, a.tag)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(df, f)
    print(f"\nWrote {out}: {len(df)} runs, "
          f"{df['D'].nunique()} widths, {df['seed'].nunique()} seeds")
