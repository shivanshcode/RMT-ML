# Running supercollapse (MLP task), fully offline

Everything below was tested end to end: env resolution, a complete training run,
CSV logging, and the pickle the figure notebook consumes.

---

## 1. Environment (needs internet — do this once)

```bash
srun --gres=gpu:a100:1 --mem=64G --time=2:00:00 --pty bash   # get a free GPU node
conda create -n supercollapse python=3.11 -y
conda activate supercollapse

echo $LD_LIBRARY_PATH        # if it contains /usr/local/cuda -> unset LD_LIBRARY_PATH

pip install "jax[cuda12]==0.6.2" "flax==0.10.6"
pip install optax hydra-core omegaconf wandb tqdm chex \
            numpy pandas matplotlib seaborn scipy jupyter
```

**These versions are verified, not guessed.** `jax==0.6.2` + `flax==0.10.6`
(optax resolves to 0.2.8). The repo was written ~July 2025; your jax 0.10.2 is
from June 2026 and is a year too new. Both of these must hold:

```bash
python -c "import jax; print(jax.devices())"                # -> [CudaDevice(id=0)]
python -c "from flax.nnx.training.optimizer import _opt_state_variables_to_state; print('ok')"
```

Freeze immediately once green — you cannot rediscover this offline:

```bash
pip freeze > ~/supercollapse-env.txt
pip download -d ~/wheel-cache -r ~/supercollapse-env.txt
```

## 2. Apply the three repo fixes

The repo does not run as shipped. Copy in these files (all included):

| File | Why |
|---|---|
| `configs/local.yaml` | **new** — `main.py` hardcodes `config_name='local'` but that file was never committed. This is the `MissingConfigException` you hit. |
| `train.py` | **patched** — adds a CSV logger so runs work with no network. 3 small insertions; see `train.patch`. |
| `make_logs.py` | **new** — offline replacement for `save_logs.py`, which calls `wandb.Api()` and needs the network. |
| `run_mlp_sweep.sh` | **new** — sweep driver with resume support. |

## 3. Smoke test (2 minutes)

```bash
export WANDB_MODE=offline XLA_PYTHON_CLIENT_PREALLOCATE=false
python main.py -cn local model.D=128 model.N=2 opt.B=256 \
    scale=null exponent=null T=100_000 T_eval=30_000 num_evals=10
```

Expect a decreasing loss and `Local log written: local_logs/...csv`.
The `fourier` dataset is synthetic — no download needed.

## 4. Run the sweep

```bash
./run_mlp_sweep.sh            # muP, 8 widths x 3 seeds
python make_logs.py --tag mlp --out logs/mlp.pkl
```

Then the ablation, where collapse should visibly degrade:

```bash
./run_mlp_sweep.sh no_mup
python make_logs.py --tag mlp_no_mup --out logs/mlp_no_mup.pkl
```

The script skips runs whose logs already exist, so it is safe to re-run after a
job dies.

## 5. Figures

`figures/collapse.ipynb`, cell 4 — point it at your pickle:

```python
ds_name = 'mlp'          # loads ../logs/mlp.pkl
```

Cell 5 computes `opt_L` / `opt_C` from the last row of each history and
normalises loss and compute to 1 at the end of training. That normalisation is
the collapse. Verified working against the pickle `make_logs.py` produces.

---

## Things worth knowing before you commit GPU hours

**Use at least 3 seeds.** `experiments/mlp.yaml` sets `seed: 0` only, which gives
you *collapse* but not *supercollapse*. The claim is that the spread across
widths falls **below the seed-to-seed noise floor** — with one seed there is no
noise floor to compare against. The authors' own `logs/mlp.pkl` has 5 seeds
across 8 widths (56 runs). `run_mlp_sweep.sh` uses 3 as a compromise.

**`configs/mlp.yaml` alone does not reproduce the paper.** It has `lr: 0.4`,
`N: 3`, `schedule: const`, `readout_lr_mult: 0.1`. The sweep overrides every one
of those to `lr: 0.001`, `N: 5`, `schedule: linear`, `readout_lr_mult: 1`.
`configs/local.yaml` bakes the correct values in.

**`scale`/`exponent` override `T`.** In `train.py` (lines 97–105) the horizon is
derived as `C = 1e15*(num_params/scale)**exponent`, `T = C/(6*num_params)`. The
static `T: 1_000_000_000` in `configs/mlp.yaml` is ignored whenever both are set.
Don't hardcode `T` in `local.yaml` — that would break compute-optimal scaling,
which is exactly what supercollapse depends on.

**Cost.** Per seed, derived from the fitted horizon:

| D | params | tokens | steps |
|---|---|---|---|
| 384 | 1.5M | 1.4e8 | 33k |
| 512 | 2.6M | 2.5e8 | 61k |
| 645 | 4.2M | 4.0e8 | 98k |
| 812 | 6.6M | 6.5e8 | 158k |
| 1024 | 10.5M | 1.0e9 | 256k |
| 1290 | 16.7M | 1.7e9 | 414k |
| 1625 | 26.4M | 2.7e9 | 670k |
| 2048 | 42.0M | 4.4e9 | 1.08M |

≈2.8M steps per seed, ≈8.3M for three. D=2048 dominates. Budget a few days on one
A100; run seeds in parallel across GPUs if you have them. Verified: these
params/compute numbers match the authors' `logs/mlp.pkl` exactly (D=384 →
1,478,016 params, opt_C = 1.2099 PF).

**Skip `mlp_fit.yaml`.** That sweep fits the horizon and is the most expensive
step. The `scale`/`exponent` above are the authors' fitted values.

**For Slurm**, put activation inside the script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:a100:1 --mem=64G --time=24:00:00
source ~/miniconda3/etc/profile.d/conda.sh
conda activate supercollapse
cd ~/supercollapse
./run_mlp_sweep.sh
```

---

## Next: weight-matrix analysis

For your actual research question, the hook goes in the eval block of `train.py`
(around line 243, where `nnx.split(model, nnx.Param)` already gives you the
params pytree). Log per-layer SVD spectra, Frobenius/spectral norms, stable rank,
and the same for `W - W_init` — not full checkpoints. `model.match_mup_at_D` and
`model.init_std_mult` are your initialization axes. Ask when you're ready and
I'll write it against this patched file.
