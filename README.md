# RMT-ML

RMT attribution and learning-mode research code with two independent workflows:

- `compute_optimal_analysis/` — Spectral-Chinchilla training/allocation experiments.
- `remote-sk-random-matrix-ml-esd-fixed/` — offline analysis of existing model snapshots.

Both projects intentionally contain a top-level Python package named `rmt`, but their APIs are incompatible. Both runners now atomically claim fresh output directories; their numerical and intervention contracts remain independent. Run commands and tests in **separate processes from the respective project root**; do not put both roots on one `PYTHONPATH` or import both in one interpreter. Each production entry point validates/isolates its own project. See each directory's README for its CLI, output, precision, and asset contracts, and its `bug_report.md` for live per-finding remediation status.
