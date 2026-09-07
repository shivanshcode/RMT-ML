# Changes needed to run compute_optimal_analysis in the existing ESD HPC environment

## 1. What the repository actually establishes

This is a migration checklist, **not changes already applied**. No cluster access, installation, downloads, or job submissions were performed during review. The working ESD environment should be preserved.

| Item | Evidence from the sibling ESD directory | Implication for this project |
|---|---|---|
| Actual interpreter used | `RMT_All_Models_376402.log:1`: `/home/shivansh/.conda/envs/rmt_ml_env/bin/python` | Replace the mandatory project `.venv` with this verified Conda interpreter/prefix, or a separately validated clone. |
| Environment name | `run_rmt.slurm:43`: `conda activate rmt_ml_env` | This is not `.venv`, and not the script's erroneous fallback environment named `rmt`. |
| Scheduler declaration | `run_rmt.slurm:5-8`: `gpulong`, one GPU, 64G RAM, four days | Add the known GPU partition. One GPU/64G already match. Wall time and CPU counts require workload/site validation. |
| CUDA module declaration | `run_rmt.slurm:35`: `module load cuda`, errors suppressed | There is **no evidence** that `cuda/12.1` or `python/3.10` are installed module names, or even that the unversioned CUDA load succeeded. |
| Accelerator target | ESD launcher comments describe an offline A100 run | A100 is the stated target, not a saved `nvidia-smi`/VRAM/driver inventory. Verify the actual allocation. |
| Offline settings | ESD exports all three HF offline switches, disables tokenizer parallelism, sets expandable CUDA segments | Keep these controls; this project already contains them. |
| Model/data actually requested | Local `./models/pythia-160m`, layers 0/5/10, FP32, local WikiText-2 text path | These are ESD application arguments, not assets or flags to copy into the from-scratch training runner. |
| Actual text used | `.error:2`: requested WikiText file missing; fallback corpus used | The successful ESD job does **not** prove that WikiText assets exist. Stage this project's training data explicitly. |
| Runtime package versions | ESD requirements are unpinned; no Conda export/pip freeze is committed | Exact replacement pins cannot be inferred. Inventory the live environment before deciding what to install/change. |
| Python version evidence | Committed bytecode names contain `cpython-310`; logs omit a version string | Python 3.10 is suggested, not established as the live interpreter's exact version. Verify it. This project's declared target is CPython 3.10. |
| Local script line endings | Git index LF, Windows working-tree CRLF for both launchers | Transfer LF scripts; do not assume a Windows copy is submission-ready. |

**Important incompatibility:** this project's `requirements.txt` pins `numpy==1.26.4`, but ESD's default Delta3 path calls `np.trapezoid`, which requires NumPy 2.0+. Blindly installing this requirements file into `rmt_ml_env` can break the already working ESD code. The saved 18-row run is consistent with that API being available, but is not an exact package inventory.

## 2. Required launcher/path changes

All file paths in this section refer to `compute_optimal_analysis/`.

| ID | File/location | Change to make |
|---|---|---|
| ENV-01 | `run_hpc.slurm:20-25` | Remove the mandatory `.venv/bin/activate` existence check, activation, and `.venv/bin/python` assignment. Select `/home/shivansh/.conda/envs/rmt_ml_env/bin/python` explicitly, with an optional operator override. Validate executable existence and print `sys.executable`/version. If activation is needed for Conda libraries, source the actual Conda profile and activate the **verified prefix**. Do not reconstruct the path as `$CONDA_BASE/envs/rmt`. |
| ENV-02 | `run_hpc.slurm:14-15` | Replace the hard-coded `module purge; module load cuda/12.1 python/3.10` policy with the module configuration actually needed by the existing Conda environment. Avoid a conflicting Python module. Do not purge modules that the validated environment/toolchain depends upon. A matching CUDA module may be unnecessary for a runtime-bundled Torch wheel; compiler requirements for `torch.compile` are separate. Verify rather than blindly copying the old script's error suppression. |
| ENV-03 | SLURM header | Add `#SBATCH --partition=gpulong` for the demonstrated GPU queue. Keep one node, one task/process, and one GPU. Confirm any required account/QoS/CPU directives with the site; none are recorded in the ESD launcher. |
| ENV-04 | `run_hpc.slurm:17-18`; submission commands | Resolve **this directory**, not the ESD directory or the repository parent. Accept a validated absolute `PROJECT_ROOT` or require submission from this directory. Check `run_experiments.py`, `pipelines/cli_config.py`, and the dataset before launching. Do not derive the project from `$0` inside SLURM, where the script may live in a spool directory. |
| ENV-05 | Log/output paths and deployment procedure | Create `logs/`, results, and writable caches **before** `sbatch`. SLURM opens log paths before the shell can execute `mkdir`. Use unique per-job output directories or reject a pre-existing output directory to avoid overwriting another run's manifests/metrics. |
| ENV-06 | `run_hpc.slurm:35`; Python invocation | Put this project's source root first in import resolution, remove the sibling ESD source root from PYTHONPATH for this process, and verify `rmt.__file__`. Keep `python run_experiments.py`, not ESD's `python -m rmt`. Run the two projects in separate interpreter processes; both packages are named `rmt` and cannot be safely co-imported under that name. |
| ENV-07 | Script transfer; optionally root `.gitattributes` | Normalize `run_hpc.slurm` to LF on deployment. Add a repository rule such as `*.slurm text eol=lf` if cross-platform checkout is routine. A grammar-only `bash -n` check does not catch this deployment problem. |
| ENV-08 | `README.md`, `other_requirements.md`, relevant examples in `differences.md` | Document the actual Conda-prefix workflow and partition instead of making `.venv`, versioned modules, CUDA 12.1, and Python 3.10.14 sound like verified cluster facts. Retain the old wheelhouse workflow as a separate, optional environment, not as instructions to overwrite `rmt_ml_env`. |

### Minimal replacement for the project/environment block

Use this as a proposed edit to the existing launcher, **not as a second complete SLURM script**. Keep the existing strict mode, offline exports, asset preflight, `COMMON_ARGS`, and four track functions. Add the partition in the header separately.

```bash
set -euo pipefail

# Set PROJECT_ROOT at submission, or submit from compute_optimal_analysis.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the project directory}}"
cd "$PROJECT_ROOT"
[[ -f run_experiments.py && -f pipelines/cli_config.py ]] || {
    echo "PROJECT_ROOT is not compute_optimal_analysis: $PROJECT_ROOT" >&2
    exit 2
}

# This exact path is recorded in the successful ESD job log.
PYTHON="${RMT_PYTHON:-/home/shivansh/.conda/envs/rmt_ml_env/bin/python}"
[[ -x "$PYTHON" ]] || {
    echo "Missing cluster interpreter: $PYTHON" >&2
    exit 2
}
"$PYTHON" -c 'import sys; print(sys.executable); print(sys.version)'

# Add only the CUDA/compiler module setup validated for this interpreter.
# Do not retain the unconditional cuda/12.1 + python/3.10 module load.

# Safe minimal import path: do not append the sibling ESD project.
export PYTHONPATH="$PROJECT_ROOT"
```

Do not copy the ESD script's lack of strict mode or its final unconditional completion message after an unchecked Python command. The current compute launcher already propagates failures with `set -euo pipefail`; preserve that behavior.

## 3. Inventory and reconcile Python dependencies before changing any pins

### ENV-09 — Capture the live environment, do not guess its versions

Run the following **on the cluster** with the known interpreter. GPU checks must be performed in an allocated GPU job, not interpreted as failures merely because a login node has no GPU.

```bash
PY=/home/shivansh/.conda/envs/rmt_ml_env/bin/python
mkdir -p environment_inventory
"$PY" -V
"$PY" -m pip freeze --all > environment_inventory/pip-freeze.txt
"$PY" -m pip check > environment_inventory/pip-check.txt
module -t list > environment_inventory/modules.txt 2>&1
nvidia-smi > environment_inventory/nvidia-smi.txt
"$PY" - <<'PY'
import json
import platform
import sys
from importlib import metadata
import torch
names = (
    'numpy', 'scipy', 'torch', 'matplotlib', 'pytest', 'datasets',
    'transformers', 'tokenizers', 'huggingface-hub', 'safetensors',
    'pyarrow', 'triton'
)
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = 'not-installed'
info = {
    'executable': sys.executable,
    'python': sys.version,
    'platform': platform.platform(),
    'packages': packages,
    'torch_cuda_build': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
}
if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    info.update({
        'device': device.name,
        'capability': list(torch.cuda.get_device_capability(0)),
        'vram_bytes': device.total_memory,
        'bf16_supported': torch.cuda.is_bf16_supported(),
    })
print(json.dumps(info, indent=2))
PY
```

If Conda is available, also archive its package/channel/build inventory:

```bash
conda list -p /home/shivansh/.conda/envs/rmt_ml_env --explicit \
    > environment_inventory/conda-explicit.txt
```

Record OS/architecture, driver, compiler (`gcc`/`g++`), available modules, and any site-specific library-path setup. No exact CUDA version or Python patch version can be supplied honestly from the committed job log alone.

### ENV-10 — Use a reviewed cluster lock or a separate clone, not an in-place downgrade

Current direct pins:

| Package | `requirements.txt` | Needed by |
|---|---|---|
| numpy | 1.26.4 | Arrays, datasets, numerical analysis |
| scipy | 1.13.1 | Spectral algorithms, integration, optimization |
| torch | 2.4.1 | Training, CUDA SVD, covariance, lesions |
| matplotlib | 3.9.2 | Runner imports it even for manifest-only mode |
| pytest | 8.3.3 | Tests/preflight |
| datasets | 3.0.1 | Connected WikiText asset staging |
| transformers | 4.45.2 | Connected/local tokenizer staging |
| tokenizers | 0.20.1 | Staging tokenizer dependency |
| huggingface-hub | 0.25.2 | Staging revision resolution/downloads |
| safetensors | 0.4.5 | Staging/HF dependency |
| pyarrow | 17.0.0 | Dataset staging dependency |

Required actions:

- [ ] Verify Python is at least 3.10 for this source/API contract. If the exact pinned environment is chosen instead, stage for the documented CPython 3.10 target; the local review interpreter, Python 3.13.2, is **not** evidence that these old wheels install there.
- [ ] Compare live versions and required APIs with the table. **A differing pin alone does not mean a code change is needed.** Validate the existing stack first.
- [ ] Preserve the ESD stack. In particular, do not force NumPy 1.26.4, Torch 2.4.1, or Transformers 4.45.2 into it without a separate compatibility plan.
- [ ] Add a reviewed `requirements-cluster.txt` or equivalent lock/inventory matching the validated existing stack, and point the cluster documentation/preflight to it. Preserve `requirements.txt` as the original standalone environment contract if both workflows remain supported.
- [ ] If incompatible dependencies really are necessary, clone/create a **separate environment** and select it through `RMT_PYTHON`; document that this preserves the same cluster/hardware but is no longer the identical ESD package environment.
- [ ] Install only genuinely missing packages into an approved environment. The training runner needs NumPy/SciPy/Torch/Matplotlib; the Hugging Face/dataset packages are chiefly needed on the staging host once local integer arrays exist. `accelerate`, `weightwatcher`, `powerlaw`, `pandas`, and `seaborn` from ESD are not new requirements of this custom training runner.
- [ ] Validate the NumPy/SciPy numerical APIs, Torch AMP/SVD/compile APIs, plotting calls, and import namespace against the actual versions. A successful `pip check` verifies declared dependencies, not scientific behavior or undeclared API minimums.

### ENV-11 — If installation is necessary, stage an ABI-matching offline wheelhouse

- Use a connected **Linux** staging host matching the cluster architecture, Python ABI, glibc/libstdc++ compatibility, and selected CUDA/Torch build. Do not download Windows wheels with this checkout's Windows Python and copy them to Linux.
- Download all transitive dependencies for the reviewed lock, including the appropriate Torch CUDA runtime dependencies and compatible Triton when compilation is enabled. Choose the official Torch wheel index/build based on the inventory, not an assumed CUDA 12.1 module.
- Keep a resolver report, complete version/build inventory, and SHA-256 sums. Review/remove local editable source paths from a portable dependency lock; archive them separately rather than expecting `pip download` to reproduce a cluster-local editable install.
- Install only from the wheelhouse on offline nodes. Do not run remote pip/Conda resolution in a compute job.

Example workflow after producing a reviewed, portable `requirements-cluster.txt`:

```bash
# Connected matching Linux staging environment only:
python -m pip download --only-binary=:all: --dest wheelhouse \
    --requirement requirements-cluster.txt
sha256sum wheelhouse/*.whl > wheelhouse/SHA256SUMS

# Approved target environment only; not an instruction to overwrite rmt_ml_env:
"$PY" -m pip install --no-index --find-links wheelhouse \
    --requirement requirements-cluster.txt
"$PY" -m pip check
```

Supply the correct Torch index/build when staging if the reviewed lock requires it. A CUDA module does not turn a CPU-only Torch wheel into a CUDA build, and Torch's bundled runtime does not replace the kernel driver.

## 4. Required application assets and offline configuration

### ENV-12 — Stage the new training corpus; do not reuse ESD paths blindly

`run_experiments.py` trains a new `CausalTransformer`. It does **not** load `models/pythia-160m` and does not accept a raw WikiText text file as `--dataset-path`.

- [ ] On a connected staging node, stage the project's intended GPT-2 tokenizer and WikiText-103 training tokens using `scripts/download_assets.py --assets all --allow-network` under the verified staging environment.
- [ ] Copy `data/asset_manifest.json`, the selected integer NPY/NPZ stream, and all other **manifest-listed** files into the new project. The existing verifier checks the whole manifest, so copying just the one NPY while leaving listed raw/tokenizer files behind fails preflight.
- [ ] Set `DATASET_PATH` to `data/tokenized/wikitext-103-raw-v1_gpt2.npy` or a deliberately staged alternative. Keep `--vocab-size` consistent with the chosen tokenizer and verify every ID is in range.
- [ ] Do not mix Pythia tokenizer IDs with a claimed GPT-2 asset contract. A new custom corpus/tokenizer must have explicit identity, vocabulary, checksums, and split policy.
- [ ] Run `scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only` offline after transfer.
- [ ] Fix/extend verification to require a matching manifest record for the actual selected dataset and verify its digest (bug report CO-12). The present script can verify an unrelated asset tree.
- [ ] Preserve existing source-revision metadata when adding assets; currently partial staging overwrites manifest provenance (CO-13). Use isolated staging roots until fixed.

The saved ESD log explicitly reports that `./wikitext-2-raw/wiki.test.raw` was missing. Treat the needed data as unstaged until verified. Do not copy the ESD `models/` snapshots over this project's `models/` directory, which contains Python source.

Synthetic data may be generated entirely offline for a calibration run, but it is not WikiText or a substitute for a language-model result. Generate it in a fresh/isolated asset root so it cannot overwrite an existing production manifest.

### ENV-13 — Keep offline exports and create writable runtime caches

These existing values should remain enabled:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Keep `HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `TORCH_HOME` under this project's writable staged tree. Do not point to ESD's outputs or assume the two projects share compatible assets.

With compilation enabled, additionally configure writable, quota-appropriate temporary/compile caches, preferably in site-approved job-local scratch:

```bash
RUNTIME_CACHE="${SLURM_TMPDIR:-$PROJECT_ROOT/cache/runtime}"
mkdir -p "$RUNTIME_CACHE/inductor" "$RUNTIME_CACHE/triton" "$RUNTIME_CACHE/tmp"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_CACHE/inductor"
export TRITON_CACHE_DIR="$RUNTIME_CACHE/triton"
export TMPDIR="$RUNTIME_CACHE/tmp"
```

Ensure the filesystem permits the compiler's required operations and loading compiled artifacts. `TORCH_HOME` alone does not configure Inductor/Triton caches. Verify the allocator setting is supported by the installed Torch/driver; change it only if actual runtime validation requires that.

## 5. Hardware/resource changes that are conditional, not proven requirements

### ENV-14 — Validate new capabilities not exercised by the ESD run

| New path | Why ESD success is insufficient | Action |
|---|---|---|
| BF16 AMP training | ESD explicitly loads/analyzes in FP32; it does not train this model | Verify `torch.cuda.is_bf16_supported()` on the allocation. A100 normally supports BF16. If unsupported, explicitly select a validated precision, update manifests, and re-calibrate; do not claim an identical numerical experiment. |
| `torch.compile`/Inductor/Triton | ESD runs inference/SVD, not this compiled training loop | Verify compatible Torch/Triton, a usable C/C++ toolchain where required, CUDA driver support, and writable caches. Use `--no-compile-model` for initial smoke calibration; enable compilation only after a real forward/backward compile test succeeds. No automatic compile fallback is implemented. |
| CUDA `gesvdj` | ESD uses Torch SVD without an explicit driver and analyzes in float64 | Check the selected driver and analysis dtype on the installed stack. SVD precision is a scientific change (CO-08), not merely an environment setting. `--svd-backend cpu` is an explicit slower fallback, not evidence of GPU compatibility. |
| GPU covariance | This project uses its own hook/float32 moment path | Calibrate the covariance dtype/device and memory independently. `--covariance-device cpu` is an explicit fallback when justified. |
| CPU linear algebra + data workers | Current compute header requests 16 CPUs; ESD does not declare that count | Keep 16 only if allowed/available. Match `OMP_NUM_THREADS`, BLAS thread settings, worker counts, and prefetch settings to the allocation; avoid oversubscription. |
| Training memory | Optimizer state, gradients, logits, activations, compile workspace, and lesion snapshots are new costs | Start small, measure GPU and host peaks, then choose batch/context/caps. One GPU/64G host RAM already match the declared ESD resources but are not a proof every configuration fits. |
| Wall time | Default compute `TRACK=all` runs four training/analysis tracks sequentially; ESD analyzes an existing checkpoint | Consider the demonstrated `4-00:00:00` limit if allowed and necessary, but do not blindly assume four days suffices. Start with one track and measured throughput. Fix CO-04/06 before extrapolating timing. |
| Long-run recoverability | This runner currently persists metrics/checkpoints late | Implement incremental metrics and periodic/recoverable checkpoints (CO-05) before long allocations. A longer SLURM time limit alone does not solve lost results. |

Do **not** request multiple GPUs expecting speedup: the runner is single-process/single-device and implements no DDP/FSDP. Keep existing scientific track flags rather than translating the ESD underscore-style CLI arguments literally.

## 6. Validation and submission sequence

### ENV-15 — Preflight with the selected interpreter and isolated imports

From the **deployed compute_optimal_analysis directory**:

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
print('rmt source:', actual)
assert actual == root / 'rmt' / '__init__.py', actual
from rmt.factory import RMTMethodConfig
from models.transformer import CausalTransformer
from pipelines.trainer import LanguageModelTrainer
print('compute project imports resolved correctly')
PY

"$PY" -m pip check
"$PY" scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only
bash -n run_hpc.slurm
"$PY" run_experiments.py --device cpu --output-dir results/manifest_preflight
```

Check line endings explicitly on Linux as part of deployment, for example:

```bash
if LC_ALL=C grep -q $'\r' run_hpc.slurm; then
    echo 'Convert run_hpc.slurm to LF before submission' >&2
    exit 2
fi
```

Then:

1. Resolve **CO-36** in `bug_report.md`: the current test suite contains a contradictory overlap expectation. Do not call it an environment failure or silently skip it.
2. Run `"$PY" -m pytest -q` **from this project directory only** after that correction; do not collect both projects in one process. Archive results and the dependency inventory.
3. In a short GPU allocation, run a tiny **custom-mode** experiment on an isolated, verified synthetic asset tree, with one cell, a small token budget, small vocabulary, few batches, `--no-compile-model`, and deliberate diagnostic choices. Preset modes force some features on, and disabled overlap currently still captures covariance (CO-10/11); fix those before relying on flags as resource controls.
4. Exercise CUDA forward/backward, evaluation, covariance, SVD, and all requested lesion tranches. Then separately validate the compile path. A manifest-only run does not test any of these capabilities.
5. Confirm actual per-cell tokens/parameters/FLOPs, initialization reproducibility, output persistence, and spectrum-domain consistency. Environment compatibility does not resolve CO-01/02/03/07 or the other scientific bugs.
6. Submit one production track first, from the correct directory, using the adapted launcher and per-job outputs. Example submission after the proposed edits:

```bash
cd /absolute/deployed/path/to/compute_optimal_analysis
mkdir -p logs results
sbatch --chdir="$PWD" \
    --export=ALL,PROJECT_ROOT="$PWD",TRACK=golden \
    run_hpc.slurm
```

Replace the example deployment path with the actual location; the new project location is **not** established by the ESD log. Validate that `gpulong` and the requested wall time/CPU resources are permitted for your account.

## 7. What does not need to be copied or changed merely for environment reuse

- The three HF offline flags, tokenizer parallelism setting, headless Matplotlib backend, and one-GPU execution design are already present.
- The one-GPU/64G request already matches the ESD script's declared resources; more memory/time is a measured workload decision.
- No pretrained Pythia checkpoint, `AutoModelForCausalLM` loader, ESD `--models`/`--layers` list, or `python -m rmt --selftest` command belongs in the compute runner. Use this project's tests and tiny execution preflight instead.
- Do not import ESD's `rmt` implementation to make this project's imports succeed. They are different applications sharing a name, not interchangeable versions.
- Do not downgrade working cluster packages solely to match the unverified standalone pins. The exact changes to version constraints must follow the live inventory and compatibility checks above.

**Bottom line:** the immediately identifiable migration changes are the Conda interpreter/prefix, module policy, GPU partition, validated project/import paths, LF/log/cache deployment, and newly staged token assets. Package-version/toolchain changes remain conditional on the actual cluster inventory. Separately, the source bugs identified in `bug_report.md` need repair before a successful launch can be treated as a trustworthy scientific run.
