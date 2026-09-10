# Offline HPC prerequisites

Compute-node jobs perform no network access. Asset acquisition, environment reconciliation, and any wheel staging are explicit operator tasks. The working sibling ESD environment must be preserved.

## 1. Verified cluster facts and unresolved checks

| Component | Established value | Status/action |
|---|---|---|
| Interpreter | `/home/shivansh/.conda/envs/rmt_ml_env/bin/python` | Recorded by the successful ESD job; default in `run_hpc.slurm` |
| Conda environment | `rmt_ml_env` | Do not replace it with `.venv` or a guessed environment named `rmt` |
| Scheduler partition | `gpulong` | Declared in the known ESD launcher and this launcher |
| Resources | One GPU and 64 GiB host RAM | Shared declaration; measure this training workload independently |
| Accelerator target | A100 | Stated by the ESD launcher, not a saved device/driver inventory |
| Python | At least 3.10 required by this project | Bytecode suggests 3.10, but the live patch version must be recorded |
| CUDA/module setup | Unversioned `module load cuda` appeared in ESD, with errors suppressed | Not proof that `cuda/12.1` exists or is needed; inventory first |
| ESD package versions | No committed freeze/Conda export | Must be inventoried; do not infer versions from unpinned requirements |
| Scheduler policy | Account, QoS, allowed CPU count and wall time are not recorded | Confirm with the site |

`run_hpc.slurm` calls the verified interpreter directly. It deliberately does not run `module purge`, load a Python module, or activate a guessed Conda base. Set `RMT_PYTHON` for a validated alternate prefix. Set `RMT_CUDA_MODULE` only when inventory and a GPU smoke test show that a module is required.

The default header requests 16 CPUs and 24 hours. The sibling ESD launcher demonstrates that a four-day request was used, not that this workload needs or is allowed that limit. Measure one track before changing resources. The runner is single-process/single-GPU and has no DDP/FSDP path.

## 2. Inventory before changing dependencies

Run inventory commands on the cluster with the known interpreter. Run GPU commands in an allocated GPU job; the absence of a GPU on a login node is not a compatibility failure.

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

Archive the inventory with run records. A successful `pip check` only verifies declared package metadata; it does not validate numerical APIs, compilation, CUDA SVD, or scientific behavior.

## 3. Dependency policy

`requirements.txt` is the original standalone direct-pin contract:

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

Do **not** install these pins into `rmt_ml_env` blindly. The sibling ESD Delta3 path now falls back from `np.trapezoid` to NumPy 1.x's `np.trapz`, so NumPy 1.26 is not by itself a demonstrated incompatibility. Preserve and inventory the complete ESD stack, then validate all required APIs, CUDA/toolchain behavior, and numerical tests before changing it.

If the live stack supplies the APIs this project needs, use it unchanged. If not, clone/create a separate approved environment, select it with `RMT_PYTHON`, and produce a reviewed `requirements-cluster.txt` from that validated environment. A different pin by itself is not evidence that source or environment changes are needed. Staging-only Hugging Face packages need not be installed on compute nodes once a complete integer token array and manifest exist, unless site policy requires one uniform environment.

### Optional standalone/wheelhouse workflow

The standalone workflow remains available, but it is separate from reuse of `rmt_ml_env`. Build wheels on a connected **Linux** host matching the target Python ABI, architecture, glibc/libstdc++, and selected Torch/CUDA build—not with this checkout's Windows interpreter.

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

For a cluster-specific clone, substitute the reviewed `requirements-cluster.txt`. Include all resolver-selected transitive wheels, the appropriate official Torch build and runtime dependencies, and compatible Triton when compilation is enabled. A CUDA module cannot turn a CPU Torch wheel into a CUDA build. Never perform remote pip/Conda resolution in a compute job.

## 4. Offline variables and writable caches

The launcher exports:

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

`PYTHONPATH` intentionally contains only this source root, preventing the sibling ESD package—also named `rmt`—from being imported in this process. Run the two projects in separate interpreter processes.

Inductor, Triton, and temporary files are placed under `${SLURM_TMPDIR}` when available, otherwise under `cache/runtime/$SLURM_JOB_ID`. Operators may set `RMT_RUNTIME_CACHE` to approved job-local scratch. Verify filesystem permissions and compiled-artifact loading before enabling compilation. The allocator setting must also be checked against the installed Torch/driver.

## 5. Required assets and directory layout

The project trains a new `CausalTransformer`; it does not load the sibling project's Pythia snapshot and does not accept a raw WikiText text file for `--dataset-path`. Stage this project's GPT-2 tokenizer and WikiText-103 integer stream explicitly:

```bash
# Connected staging host only
python scripts/download_assets.py --assets all --allow-network

# Deployed offline project
/home/shivansh/.conda/envs/rmt_ml_env/bin/python \
    scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only
```

The default token stream is `data/tokenized/wikitext-103-raw-v1_gpt2.npy`, with vocabulary size 50257. Copy `data/asset_manifest.json` and every manifest-listed file; the verifier checks the complete manifest. Do not mix tokenizer identities. Custom corpora must record identity, vocabulary, split policy, sizes, and SHA-256 checksums. Synthetic data is suitable only for software calibration, not language-model claims.

Required layout:

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

Create submission-time directories before `sbatch`:

```bash
mkdir -p logs results cache/torch cache/huggingface
```

This is mandatory for `logs/`, because SLURM opens `#SBATCH --output` and `--error` before the batch shell can execute. Production outputs use a fresh `results/jobs/$SLURM_JOB_ID` root. A custom `OUTPUT_ROOT` may be absent or intentionally pre-created and empty; the SLURM launcher atomically claims it with `.job-owner`, and each track runner claims its child directory with `.run-owner.json`. Nonempty or already-owned targets are rejected.

## 6. Accelerator and numerical validation

ESD success in FP32 inference/SVD does not validate this project's BF16 training, compilation, covariance path, or explicit CUDA SVD driver. In a short GPU allocation, verify:

- the allocated model, VRAM, compute capability, driver, Torch CUDA build, and `torch.cuda.is_bf16_supported()`;
- a forward/backward optimizer update in the requested precision;
- evaluation and activation covariance on the requested device/dtype;
- `torch.linalg.svd(..., driver="gesvdj")` on representative shapes;
- all requested lesion tranches and host-memory peaks;
- compiler/toolchain and writable Inductor/Triton caches.

Start calibration with `--no-compile-model` or submit with `COMPILE_MODEL=0`, then validate compilation separately. CPU SVD/covariance and another precision are explicit scientific/runtime choices, not silent fallbacks; record and recalibrate them. Match BLAS/OpenMP thread counts and data workers to the allocated CPUs.

## 7. Preflight and submission

From the deployed `compute_optimal_analysis` directory:

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

After the allocated-GPU smoke tests pass, submit one track first:

```bash
sbatch --chdir="$PWD" \
    --export=ALL,PROJECT_ROOT="$PWD",TRACK=golden \
    run_hpc.slurm
```

The launcher verifies its project files, exact `rmt` source, interpreter/version, CUDA/BF16 access, selected dataset, and complete asset manifest. It uses strict mode and propagates failures. Confirm account/QoS, CPU count, and wall time with the site; do not request multiple GPUs expecting unsupported parallelism.

## 8. Reproducibility notes

- `.gitattributes` forces LF for `*.slurm`; still check the deployed copy explicitly.
- Preserve the environment inventory, asset manifest, source revision, SLURM job ID/logs, runtime environment record, and resolved method/run manifests.
- A full WikiText-103 asset tree and checkpoints may require substantial storage; measure quota usage before production.
- One successful manifest-only run does not test training, CUDA, covariance, SVD, lesions, or compilation.
- Run test suites from this project directory only; collecting both sibling projects in one Python process risks `rmt` namespace collisions.
