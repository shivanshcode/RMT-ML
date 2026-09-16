# File tree — where each file goes

Target repo: `nandinikatyayan1111/RMT-ML`, folder
`remote-sk-random-matrix-ml-rmt-stats-esd-fixed/`.

**Read this first.** `main` is the **pre-fix** code. Verified against a fresh
clone: `main` has no `rmt/nulls.py`, no `rmt/controls.py`, no
`rmt_null_calibration.py`, and its `rmt/spacing.py` is 26 KB against 44 KB in
`files(3).zip`. So `files(3).zip` has never been merged. **Apply it first, then
this delivery on top.**

---

## Legend

| mark | meaning |
|---|---|
| **NEW** | file does not exist in the repo at all |
| **REPLACE** | overwrite the file that `files(3).zip` puts there |
| **REPLACE (repo)** | overwrite a file that exists on `main` today |
| *(f3)* | comes from `files(3).zip`, unchanged by me — listed so the tree is complete |

---

## Tree

```
remote-sk-random-matrix-ml-rmt-stats-esd-fixed/
│
├── benchmark.py                          (repo, unchanged — keep it)
├── benchmark3.py                         ← NEW          three-way harness
├── benchmark_results.csv                 (delivered, keep as the baseline)
├── benchmark_results_3way.csv            ← NEW          output, 70 rows, 6 seeds
├── rmt_null_calibration.py               (f3)  ← ALSO NEEDS O8 PATCH, see below
├── export_weights_npy.py                 (f3)
├── run_rmt.slurm                         (repo, superseded — keep for reference)
├── run_rmt_calibrated.slurm              (f3, superseded)
├── run_rmt_calibrated_v2.slurm           ← NEW          use this one
│
├── design.md                             ← NEW          all design rationale
├── resolved-issues.md                    ← NEW          what is fixed + how to verify
├── outstanding-issues.md                 ← NEW          what is still open
├── README_APPLY.md                       ← NEW          install + verification
├── BENCHMARK_TABLE.md                    ← NEW          full results table
│
├── diffs/
│   ├── rmt_spacing.py.diff               ← NEW   (362 lines)
│   ├── rmt_nulls.py.diff                 ← NEW   ( 52 lines)
│   ├── rmt_controls.py.diff              ← NEW   ( 58 lines)
│   ├── test_spacing.py.diff              ← NEW   ( 20 lines)
│   ├── test_bulk_selection.py.diff       ← NEW   ( 39 lines)
│   └── test_crosscheck_calibration.py.diff ← NEW ( 18 lines)
│
├── rmt/
│   ├── reference.py                      ← NEW          exact finite-L laws
│   ├── spacing.py                        ← REPLACE      (56 KB)
│   ├── nulls.py                          ← REPLACE      (+ critical_value)
│   ├── controls.py                       ← REPLACE      (+ variance_profile_gaussian)
│   ├── per_matrix.py                     (f3)
│   ├── config.py                         (f3)  ← see O3 if you change the default
│   ├── cli.py                            (f3)  ← add 'tx' to --unfold_method choices
│   ├── ensembles.py                      (f3)
│   ├── linalg.py                         (repo, unchanged — verified precision-safe)
│   ├── mp.py                             (repo, unchanged — see O2)
│   ├── porter_thomas.py                  (repo, unchanged — see design.md D14)
│   ├── scalars.py                        (repo, unchanged)
│   ├── selftest.py                       (repo, unchanged)
│   ├── pipeline.py  discovery.py  decile.py  activations.py
│   ├── mp_fit.py  tail.py  overlap.py  perplexity.py
│   ├── model_io.py  svd_cache.py  __init__.py  __main__.py
│   ├── baselines/                        (repo, unchanged)
│   └── plots/                            (repo, unchanged)
│
└── tests/
    ├── conftest.py                       (repo, unchanged — provides the rng fixture)
    ├── test_reference_and_unfolding.py   ← NEW          23 tests
    ├── test_spacing.py                   ← REPLACE (repo)  1 assertion inverted
    ├── test_bulk_selection.py            ← REPLACE (f3)    1 assertion inverted
    ├── test_crosscheck_calibration.py    ← REPLACE (f3)    1 assertion inverted
    ├── test_nulls.py                     (f3)
    ├── test_controls.py                  (f3)
    ├── test_sigma2_determinism.py        (f3)
    ├── test_sigma2_lmax_calibration.py   (f3)
    ├── test_docs_and_references.py       (f3)
    └── test_*.py                         (repo, unchanged: mp, scalars, tail,
                                           porter_thomas, per_matrix, decile,
                                           discovery, ensembles, overlap,
                                           pipeline_smoke, activations, svd_cache)
```

---

## Copy commands

```bash
git clone https://github.com/nandinikatyayan1111/RMT-ML.git
cd RMT-ML/remote-sk-random-matrix-ml-rmt-stats-esd-fixed

# --- step 1: files(3).zip (main does NOT contain it) -----------------------
cp files_3/spacing.py files_3/per_matrix.py files_3/config.py \
   files_3/cli.py files_3/ensembles.py files_3/controls.py files_3/nulls.py  rmt/
cp files_3/test_*.py                                                          tests/
cp files_3/rmt_null_calibration.py files_3/export_weights_npy.py \
   files_3/run_rmt_calibrated.slurm                                           .

# --- step 2: this delivery on top ----------------------------------------
cp rmt_fix_v2/rmt/*.py     rmt/
cp rmt_fix_v2/tests/*.py   tests/
cp rmt_fix_v2/benchmark3.py rmt_fix_v2/run_rmt_calibrated_v2.slurm \
   rmt_fix_v2/benchmark_results_3way.csv                                      .
cp rmt_fix_v2/*.md                                                            .
mkdir -p diffs && cp rmt_fix_v2/diffs/*.diff                                  diffs/

# --- step 3: gates --------------------------------------------------------
python -m rmt --selftest                    # 11/11 PASS
python -m pytest tests/ -q -m "not torch"   # 263 passed, 4 skipped
OLD_REPO=../remote-sk-random-matrix-ml-esd-fixed NEW_REPO=. FIX_REPO=. \
  python benchmark3.py 6
```

---

## Two hand-edits not shipped as files

**1. `rmt/cli.py`** — add `tx` to the `--unfold_method` choices so the new
default is reachable from the command line:

```python
p.add_argument("--unfold_method", default="auto",
               choices=["tx", "auto", "cheb", "gauss", "poly"])
```

`auto` already delegates to the transform search, so this is only needed if you
want to name it explicitly (as `run_rmt_calibrated_v2.slurm` does).

**2. `rmt_null_calibration.py`, `score` stage, ~line 469** — replace the
anti-conservative normal quantile (outstanding-issues.md **O8**):

```python
# before
abs(scipy.stats.norm.ppf(0.025 / (len(rows) * len(stats))))
# after
from rmt import nulls as NU
NU.critical_value(n_reps, len(rows) * len(stats))   # n_reps from null_bands.csv
```

The v2 SLURM script computes the correct threshold inline as a stopgap, so the
run is safe either way — but the printed value from the unpatched calibrator is
wrong and should not be quoted.

---

## SLURM: what differs from your repo's script

`run_rmt_calibrated_v2.slurm` supersedes both `run_rmt.slurm` (repo) and
`run_rmt_calibrated.slurm` (files(3).zip). Six substantive differences:

| # | change | reason |
|---|---|---|
| 1 | `--unfold_method tx` (was `cheb`) | design.md D4 |
| 2 | `--spacing_bulk_mode auto` (was `center` 0.7) | design.md D8, O3 |
| 3 | GATE 0 expects 263/4 (was 240/4) | 23 new tests |
| 4 | GATE 0b runs `benchmark3.py` | estimators re-verified against exact laws before touching weights |
| 5 | Stage 2 bands use the same method and bulk mode as stage 6, `--reps 40` | a band under different level selection is not a null for the measurement |
| 6 | Stage 7 computes the correct t-quantile inline | O8 |

Also: stage 5b now sweeps `1.0` as well as 0.6/0.7/0.8, because without the
uncut baseline an `EDGE-SENSITIVE` flag cannot be distinguished from a
level-count effect — which is exactly what went wrong on `sigma2_L20`.

If you prefer to keep the delivered `center`/0.7 behaviour, set `BULK_MODE=center`
at the top of the script **and rebuild the bands the same way**. The `score`
stage refuses a `bulk_frac` mismatch, which is correct behaviour — do not work
around it.
