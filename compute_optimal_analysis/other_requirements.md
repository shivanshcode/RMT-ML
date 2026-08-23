# Offline HPC prerequisites

This repository is designed so compute-node jobs perform no network access. Asset acquisition and wheel staging are separate, explicit operations performed once on an internet-connected staging machine with the same operating-system, Python, and accelerator ABI as the cluster.

## 1. Fixed platform contract

| Component | Required value |
|---|---|
| Operating system | 64-bit Linux on the staging machine and compute nodes |
| Python | CPython 3.10.x; 3.10.14 is the recorded production target |
| Shell | Bash 4.4 or newer for arrays and strict-mode batch execution |
| Accelerator | NVIDIA A100 or H100; CPU execution remains supported |
| CUDA runtime | CUDA 12.1-compatible runtime for the selected `torch==2.4.1` wheel |
| NVIDIA driver | A cluster-supported driver compatible with CUDA 12.1 |
| System ABI | glibc and `libstdc++` compatible with every staged manylinux wheel |
| Scheduler | SLURM with `sbatch`, `srun`, and a configured GPU generic resource |
| Host memory | At least 64 GiB for the default batch request |
| Local storage | At least 25 GiB plus space for checkpoints and requested token limits |

All declared Python dependencies are exactly pinned in `requirements.txt`. The connected staging machine must build a complete wheelhouse for the target platform, including resolver-selected transitive wheels; compute nodes install only from that directory. Preserve the staging resolver report or package inventory with the wheelhouse so the full transitive environment can be audited.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip download --dest wheelhouse --requirement requirements.txt
python -m pip install --no-index --find-links wheelhouse --requirement requirements.txt
python -m pip check
python -m pip freeze --all > wheelhouse/resolved-environment.txt
sha256sum wheelhouse/*.whl > wheelhouse/SHA256SUMS
```

For CUDA, stage the `torch==2.4.1` wheel built for CUDA 12.1 and its matching NVIDIA dependency wheels. Do not mix a CPU-only wheelhouse with a GPU job. Preserve the complete wheelhouse rather than copying only the top-level packages, because the installer must resolve transitive dependencies without an index.

## 2. Mandatory offline environment

The SLURM harness exports these values before invoking Python:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_HOME="./cache/torch"
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="./cache/huggingface/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=".:${PYTHONPATH:-}"
```

Jobs must not set proxy variables or point any cache variable outside the staged project tree. `run_experiments.py` accepts only a local NPY/NPZ token array and contains no download fallback.

## 3. Required directory layout

```text
cache/
├── huggingface/
│   ├── datasets/
│   └── hub/
└── torch/
data/
├── asset_manifest.json
├── raw/
│   └── wikitext-103-raw-v1/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── tokenized/
│   ├── wikitext-103-raw-v1_gpt2.npy
│   └── synthetic_zipf.npy
└── tokenizers/
    ├── gpt2/
    │   ├── merges.txt
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── vocab.json
    └── synthetic/
        └── vocabulary.json
logs/
results/
wheelhouse/
```

Create `logs/` before submitting the first job because SLURM opens output files before the batch script starts:

```bash
mkdir -p logs results wheelhouse
```

## 4. Asset manifest

The reproducible default corpus is Hugging Face dataset `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`. The tokenizer snapshot is `openai-community/gpt2`. On the connected host, the asset script resolves both moving repository names to immutable commit SHAs, downloads by those revisions, copies tokenizer configuration locally, exports each raw split as JSON Lines, writes the training token stream as an integer NPY file, and records the revisions plus SHA-256 checksums in `data/asset_manifest.json`.

No pretrained checkpoint is required: Spectral-Chinchilla constructs and trains its causal Transformer from the local configuration. If an operator adds a pretrained initialization, its complete checkpoint, tokenizer, configuration, license, source revision, byte size, and SHA-256 checksum must be placed under `data/checkpoints/<model-name>/` and added to `data/asset_manifest.json` before cluster transfer.

The deterministic synthetic fallback requires no network and creates both a Zipf-distributed token stream and an integer-identity vocabulary. It is appropriate for software calibration, not for language-model claims.

## 5. Connected-node prefetch

Run this only where internet access is explicitly permitted:

```bash
source .venv/bin/activate
python scripts/download_assets.py --assets all --allow-network
```

To cap staging size while validating the pipeline:

```bash
python scripts/download_assets.py --assets all --allow-network --max-wikitext-tokens 10000000
```

To generate only the fully local synthetic corpus:

```bash
python scripts/download_assets.py --assets synthetic --synthetic-tokens 1000000
```

Copy the repository, `wheelhouse/`, `cache/`, and `data/` trees to the cluster without dereferencing or omitting files. Verify the copied `data/asset_manifest.json` checksums with the cluster's standard checksum utility before submission.

The repository's local verifier performs the same check without network access:

```bash
python scripts/download_assets.py --verify-only
```

## 6. Air-gap installation and validation

On the cluster login node:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse --requirement requirements.txt
python -m pip check
test -f data/asset_manifest.json
test -f data/tokenized/wikitext-103-raw-v1_gpt2.npy
python scripts/download_assets.py --verify-only
mkdir -p logs results
python -m pytest -q
bash -n run_hpc.slurm
sbatch run_hpc.slurm
```

The operator performs these commands manually. Batch jobs inherit strict offline variables and should fail immediately on missing local assets rather than attempt remote recovery.

## 7. Storage and reproducibility notes

- A full WikiText-103 token stream is hundreds of megabytes; raw exports and package caches require additional space.
- Checkpoints can dominate storage. `--save-checkpoints` is disabled by default.
- `data/asset_manifest.json`, `requirements.txt`, the SLURM job ID, `run_config.json`, and `spectral_method_config.json` together identify a run.
- The default covariance path accumulates float32 matrices on the active accelerator and transfers only final reduced matrices to host memory.
- FARMS and Lanczos remain host-side NumPy/SciPy algorithms. Accelerator results cross that boundary only as explicit host arrays.
