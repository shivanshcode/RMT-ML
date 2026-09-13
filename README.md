# RMT-ML

Random Matrix Theory (RMT) uses matrix spectra to examine complex systems. This repository contains two independent RMT workflows:

- `compute_optimal_analysis/` contains Spectral-Chinchilla experiments for training and compute allocation.
- `remote-sk-random-matrix-ml-esd-fixed/` contains RMT tools for offline analysis of existing model snapshots.

Both projects have a top-level Python package named `rmt`, but their APIs are incompatible. Both runners use atomic claims for new output directories. Their numerical contracts and intervention contracts are independent.

Operate commands and tests in separate processes from the applicable project root. Do not put both roots on one `PYTHONPATH`. Do not import both packages in one interpreter. These actions can load the incorrect `rmt` package.

Each production entry point isolates its project. Read the applicable `README.md` for the CLI, output, precision, and asset contracts. Read `bug_report.md` and `bug_reportv2.md` for the repair status of each audit finding.
