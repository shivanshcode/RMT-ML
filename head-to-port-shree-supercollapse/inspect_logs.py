#!/usr/bin/env python
"""Inspect a logs/*.pkl and explain any duplicate (D, seed) rows.

    python inspect_logs.py logs/mlp.pkl
    python inspect_logs.py logs/*.pkl

Answers the question load_runs cannot: WHY are there duplicates? Two very
different causes need very different fixes.

  IDENTICAL duplicates -- same final loss, same compute, same row count.
      The same run was ingested twice. Harmless data, harmless to drop.
      Usual cause: a glob collision (one tag being a prefix of another), or a
      pickle concatenated with itself.

  DISTINCT duplicates -- same (D, seed) but different losses.
      Two genuinely different training runs are wearing the same label. This
      is the dangerous one: the two are not interchangeable and dropping the
      wrong one silently changes the answer. Usual cause: two sweeps with
      different hyperparameters collected under one tag (e.g. muP and the
      no-muP ablation, which per 2_HANDOFF.md section 5 both write mupTrue in
      their filenames and are only distinguished by the tag prefix).

With --split, distinct duplicates are separated into groups by their loss
signature and written to <stem>_group{k}.pkl so each can be inspected on its
own. Nothing is ever overwritten.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd


def describe(df: pd.DataFrame, path: str) -> pd.DataFrame:
    print(f"\n=== {path} ===")
    print(f"rows: {len(df)}")
    print(f"columns: {sorted(df.columns)}")

    key = "num_params" if "num_params" in df.columns else "P"
    rows = []
    for i, r in df.iterrows():
        h = r["history"]
        rows.append({
            "row": i,
            "D": int(r["D"]),
            "seed": int(r["seed"]),
            "p": int(r[key]),
            "evals": len(h),
            "final_loss": float(h["test_loss"].iloc[-1]),
            "final_C": float(h["compute"].iloc[-1]),
            "lr": r.get("lr", np.nan),
            "sched": r.get("schedule", "?"),
            "mup": r.get("mup", "?"),
        })
    t = pd.DataFrame(rows).sort_values(["D", "seed", "row"]).reset_index(drop=True)

    print(f"widths: {sorted(t['D'].unique())}")
    print(f"seeds:  {sorted(t['seed'].unique())}")
    for col in ("lr", "sched", "mup"):
        vals = t[col].unique()
        if len(vals) > 1:
            print(f"!! MIXED {col}: {list(vals)}  <- more than one sweep in here")
    return t


def report_duplicates(t: pd.DataFrame) -> pd.DataFrame:
    dup = t.duplicated(subset=["D", "seed"], keep=False)
    if not dup.any():
        print("\nno duplicate (D, seed) rows -- this pickle is clean")
        return t.iloc[0:0]

    d = t[dup].copy()
    print(f"\n{len(d)} duplicated rows over "
          f"{d.groupby(['D','seed']).ngroups} (D, seed) pairs")

    identical, distinct = [], []
    for (D, s), g in d.groupby(["D", "seed"]):
        same = np.allclose(g["final_loss"], g["final_loss"].iloc[0], rtol=1e-12) \
            and np.allclose(g["final_C"], g["final_C"].iloc[0], rtol=1e-12)
        (identical if same else distinct).append((D, s, g))

    if identical:
        print(f"\n  {len(identical)} pair(s) are IDENTICAL copies -- safe to "
              f"de-duplicate (keep any one)")
    if distinct:
        print(f"\n  {len(distinct)} pair(s) are DISTINCT runs sharing a label. "
              f"These are NOT interchangeable:")
        print(f"\n  {'D':>6} {'seed':>5} {'evals':>7} {'final_loss':>13} "
              f"{'final_C':>13}")
        for D, s, g in distinct:
            for _, r in g.iterrows():
                print(f"  {D:>6} {s:>5} {r['evals']:>7} "
                      f"{r['final_loss']:>13.6f} {r['final_C']:>13.4f}")
            print()
    return d


def split_groups(df: pd.DataFrame, t: pd.DataFrame, path: str) -> None:
    """Separate a mixed pickle into self-consistent groups.

    Rows are grouped by their rank within each (D, seed): the k-th copy of
    every pair goes to group k. That recovers two interleaved sweeps cleanly
    when each contributed exactly one row per (D, seed), which is what a glob
    collision produces.
    """
    t = t.copy()
    t["rank"] = t.groupby(["D", "seed"]).cumcount()
    stem = os.path.splitext(path)[0]
    for k, g in t.groupby("rank"):
        out = df.loc[g["row"].to_numpy()]
        dest = f"{stem}_group{k}.pkl"
        if os.path.exists(dest):
            print(f"  refusing to overwrite {dest}")
            continue
        with open(dest, "wb") as f:
            pickle.dump(out.reset_index(drop=True), f)
        losses = g["final_loss"]
        print(f"  wrote {dest}: {len(out)} rows, "
              f"{g['D'].nunique()} widths, final loss "
              f"{losses.min():.6f}..{losses.max():.6f}")
    print("\n  Inspect both groups, decide which is which (the muP sweep should "
          "\n  have the LOWER loss at large D; at D=384 the two sweeps are "
          "\n  matched and should agree), then rename accordingly.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkls", nargs="+")
    ap.add_argument("--split", action="store_true",
                    help="write <stem>_group{k}.pkl for a mixed pickle")
    ap.add_argument("--full", action="store_true", help="print every row")
    a = ap.parse_args(argv)

    bad = 0
    for path in a.pkls:
        if not os.path.exists(path):
            print(f"!! {path} not found")
            bad = 1
            continue
        df = pd.read_pickle(path)
        t = describe(df, path)
        if a.full:
            print(t.to_string(index=False))
        d = report_duplicates(t)
        if len(d):
            bad = 1
            if a.split:
                split_groups(df, t, path)
    return bad


if __name__ == "__main__":
    sys.exit(main())
