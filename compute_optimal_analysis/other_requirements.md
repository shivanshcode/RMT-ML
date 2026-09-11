# Offline HPC prerequisites

Compute-node jobs do not use a network. Operators acquire assets, reconcile the environment, and stage wheels. Preserve the working sibling ESD environment.

## 1. Known cluster facts and open tests

| Component | Known value | Required action |
|---|---|---|
| Interpreter | `/home/shivansh/.conda/envs/rmt_ml_env/bin/python` | The successful ESD job recorded it. `run_hpc.slurm` uses it by default. |
| Conda environment | `rmt_ml_env` | Do not replace it with `.venv` or a guessed environment named `rmt`. |
| Scheduler partition | `gpulong` | The known ESD launcher and this launcher use it. |
| Resources | One GPU and 64 GiB of host RAM | Measure this training load separately. |
| Accelerator target | A100 | The ESD launcher states this value. No saved device and driver inventory supports it. |
| Python | This project requires version 3.10 or later. | Bytecode indicates 3.10. Record the active patch version. |
| CUDA and modules | The ESD script contained unversioned `module load cuda` with hidden errors. | This does not prove that `cuda/12.1` or another CUDA module exists or is necessary. Record the environment first. |
| ESD packages | No freeze or Conda export is committed. | Record them. Do not infer versions from unpinned requirements. |
| Scheduler policy | Account, QoS, CPU limit, and time limit are unknown. | Ask the site for these values. |

`run_hpc.slurm` calls the recorded interpreter. It does not use `module purge`, a Python module, or a guessed Conda base. Set `RMT_PYTHON` only for a prefix that passed its tests. Set `RMT_CUDA_MODULE` only after inventory and a GPU test show that it is necessary.

The default request is 16 CPUs for 24 hours. The sibling ESD script requested four days. This does not show that this workload needs or permits four days. Measure one track before resource changes. The runner uses one process and one GPU. It has no DDP or FSDP path.

## 2. Record the environment before package changes

Use the known interpreter on the cluster. Do GPU commands in an allocated GPU job. No GPU on a login node does not show an incompatibility.

```bash
PY=/home/shivansh/.conda/envs/rmt_ml_env/bin/python
mkdir -p environment_inventory
"$PY" -V > environment_inventory/python.txt 2>&1
"$PY" -m pip freeze --all > environment_inventory/pip-freeze.txt
"$PY" -m pip check > environment_inventory/pip-check.txt
module -t list > environment_inventory/modules.txt 2>&1
nvidia-smi > environment_inventory/nvidia-smi.txt
(gcc --version; g++ --version) > environment_inventory/compiler.txt 2>&1

"$PY" - <<'PY' > environment_inventory/runtime.json
import json
import platform
import sys
from importlib import metadata
import torch

names = (
    "numpy", "scipy", "torch", "matplotlib", "pytest", "datasets",
    "transformers", "tokenizers", "huggingface-hub", "safetensors",
    "pyarrow", "triton",
)
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = "not-installed"
record = {
    "executable": sys.executable,
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    record.update({
        "device": device.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "vram_bytes": device.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    })
print(json.dumps(record, indent=2))
PY

conda list -p /home/shivansh/.conda/envs/rmt_ml_env --explicit \
    > environment_inventory/conda-explicit.txt
```

Keep the inventory with the execution records. A successful `pip check` examines declared package metadata only. It is not a test of numerical APIs, compilation, CUDA SVD, or scientific behavior.

## 3. Dependency rules

`requirements.txt` is the original direct-pin contract for a standalone environment:

| Package | Standalone pin |
|---|---:|
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| Torch | 2.4.1 |
| Matplotlib | 3.9.2 |
| pytest | 8.3.3 |
| datasets | 3.0.1 |
| transformers | 4.45.2 |
| tokenizers | 0.20.1 |
| huggingface-hub | 0.25.2 |
| safetensors | 0.4.5 |
| pyarrow | 17.0.0 |

Do not install these pins into `rmt_ml_env` without analysis. The sibling ESD Delta3 path uses `np.trapezoid` or NumPy 1.x `np.trapz`. Thus, NumPy 1.26 alone does not show an incompatibility.

Preserve and record the full ESD stack. Before changes, do tests of required APIs, CUDA, the toolchain, and numerical behavior. If the live stack supports this project, use it without changes.

If it does not support the project, make a separate approved environment. Select it with `RMT_PYTHON`. Make `requirements-cluster.txt` from the environment that passed its tests and get a review.

Compute nodes do not require staging-only Hugging Face packages after full token and manifest staging. Site policy can require one common environment.

### Optional standalone wheelhouse

This procedure is separate from reuse of `rmt_ml_env`. Use a connected Linux host that matches the target system. Match Python ABI, architecture, glibc, libstdc++, and the selected Torch and CUDA build. Do not use the Windows interpreter from this checkout.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip download --only-binary=:all: --dest wheelhouse \
    --requirement requirements.txt
sha256sum wheelhouse/*.whl > wheelhouse/SHA256SUMS
python -m pip install --no-index --find-links wheelhouse \
    --requirement requirements.txt
python -m pip check
python -m pip freeze --all > wheelhouse/resolved-environment.txt
```

For a cluster clone, use the reviewed `requirements-cluster.txt`. Include each transitive wheel selected by the resolver. Include the applicable official Torch build, runtime dependencies, and compatible Triton when compilation is active.

A CUDA module cannot change a CPU Torch wheel into a CUDA build. Do not resolve pip or Conda packages remotely in a compute job.

## 4. Offline variables and writable caches

The launcher sets these values:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/cache/huggingface/hub"
export PYTHONPATH="$PROJECT_ROOT"
```

`PYTHONPATH` contains only this source root. This prevents import of the incompatible sibling package that also uses the name `rmt`. Operate the two projects in separate interpreter processes.

The launcher puts Inductor, Triton, and temporary files under `${SLURM_TMPDIR}` when available. Otherwise, it uses `cache/runtime/$SLURM_JOB_ID`. Operators can set `RMT_RUNTIME_CACHE` to approved local storage for the job.

Before compilation, make sure that file permissions and compiled artifact loading operate correctly. Also make sure that the Torch and driver versions support the allocator configuration.

## 5. Assets and directory structure

This project trains a new `CausalTransformer`. It does not load the Pythia snapshot from the sibling project. `--dataset-path` does not accept a raw WikiText text file.

Stage the GPT-2 tokenizer and WikiText-103 integer stream:

```bash
# Connected staging host only
python scripts/download_assets.py --assets all --allow-network

# Deployed offline project
/home/shivansh/.conda/envs/rmt_ml_env/bin/python \
    scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only
```

The default stream is `data/tokenized/wikitext-103-raw-v1_gpt2.npy`. Its vocabulary size is 50257. Copy `data/asset_manifest.json` and each listed file. The local examination covers the full manifest.

Do not mix tokenizer identities. For a custom corpus, record identity, vocabulary, split rules, sizes, and SHA-256 checksums. Synthetic data supports software calibration only. Do not use it for language-model claims.

Use this structure:

```text
cache/
├── huggingface/
├── runtime/
└── torch/
data/
├── asset_manifest.json
├── raw/
├── tokenized/
│   └── wikitext-103-raw-v1_gpt2.npy
└── tokenizers/
logs/
results/
```

Before `sbatch`, make the submission directories:

```bash
mkdir -p logs results cache/torch cache/huggingface
```

You must make `logs/` before submission. SLURM opens `#SBATCH --output` and `--error` files before the batch shell starts.

Production writes to a new `results/jobs/$SLURM_JOB_ID` root. A custom `OUTPUT_ROOT` can be absent or intentionally pre-created and empty. The launcher atomically claims it with `.job-owner`. Each track claims its child directory with `.run-owner.json`. The launcher rejects a nonempty or owned location.

## 6. Accelerator and numerical tests

ESD success with FP32 inference and SVD is not a test of this BF16 training workflow. It does not examine compilation, covariance, or an explicit CUDA SVD driver.

In a short GPU allocation, record these items and do these tests:

- Record the GPU model, VRAM, capability, driver, and Torch CUDA build. Record `torch.cuda.is_bf16_supported()`.
- Do one forward, backward, and optimizer update at the selected precision.
- Do evaluation and activation covariance at the selected device and dtype.
- Apply `torch.linalg.svd(..., driver="gesvdj")` to representative shapes.
- Apply each requested lesion tranche and record host-memory peaks.
- Do tests of the compiler, toolchain, and writable Inductor and Triton caches.

First use `--no-compile-model` or `COMPILE_MODEL=0`. Do a separate test of compilation. CPU SVD, CPU covariance, and another precision are explicit choices. They are not automatic substitutes. Record and calibrate each choice. Match BLAS and OpenMP threads and data workers to allocated CPUs.

## 7. Preflight and submission

From the deployed `compute_optimal_analysis` directory, enter:

```bash
PROJECT_ROOT="$PWD"
PY=/home/shivansh/.conda/envs/rmt_ml_env/bin/python
mkdir -p logs results cache/torch cache/huggingface
export PYTHONPATH="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" - <<'PY'
from pathlib import Path
import rmt
root = Path.cwd().resolve()
actual = Path(rmt.__file__).resolve()
print("rmt source:", actual)
assert actual == root / "rmt" / "__init__.py", actual
from rmt.factory import RMTMethodConfig
from models.transformer import CausalTransformer
from pipelines.trainer import LanguageModelTrainer
print("compute project imports resolved correctly")
PY

"$PY" -m pip check
"$PY" scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only
"$PY" -m pytest -q
bash -n run_hpc.slurm
if LC_ALL=C grep -q $'\r' run_hpc.slurm; then
    echo "Convert run_hpc.slurm to LF before submission" >&2
    exit 2
fi
```

If the allocated-GPU tests succeed, submit one track first:

```bash
sbatch --chdir="$PWD" \
    --export=ALL,PROJECT_ROOT="$PWD",TRACK=golden \
    run_hpc.slurm
```

The launcher examines project files, local `rmt`, Python, CUDA, BF16, the dataset, and the asset manifest. It uses strict shell mode and stops after an error. Ask the site about account, QoS, CPUs, and time. Do not request many GPUs because the code has no parallel path.

## 8. Reproducibility records

`.gitattributes` requires LF for `*.slurm`. Examine the deployed copy too. Keep the environment inventory, asset manifest, source revision, SLURM ID, logs, and runtime record. Also keep the resolved method and execution manifests.

A full WikiText-103 tree and checkpoints can use much storage. Measure the quota before production. A manifest-only execution is not a test of training, CUDA, covariance, SVD, lesions, or compilation.

Operate tests from this project directory only. Collection of both projects in one Python process can cause an `rmt` namespace collision.
