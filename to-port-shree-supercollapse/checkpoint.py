"""checkpoint.py — save MLP weight matrices in a form the RMT-ML suite reads.

Design notes
------------
* Written as ``.npz`` (numpy), NOT torch ``.pt``, so the JAX training env never
  needs torch installed. The RMT side rebuilds an ``nn.Module`` from the npz.

* Key names are chosen so that ``rmt.discovery`` classifies them without any
  change to your suite:

      model.layers.{i}.mlp.fc1.weight   -> short "U"   (MATRIX_PATTERNS["U"] has "fc1")
      model.layers.{i}.mlp.fc2.weight   -> short "D"   (MATRIX_PATTERNS["D"] has "fc2")

  and ``extract_layer_index`` picks up ``i`` via the generic regex
  ``\\.layers?\\.(\\d+)\\.``.

* TRANSPOSED ON SAVE. Flax ``nnx.Linear`` stores the kernel as
  ``(in_features, out_features)``; torch ``nn.Linear.weight`` is
  ``(out_features, in_features)``, and ``rmt.discovery._weight_2d`` assumes the
  torch convention (it only transposes for HF ``Conv1D``). These are square for
  fc1/fc2 so the SVD spectrum is unchanged either way, but row/column semantics
  matter for ``--N_cov_mode cols`` and for the overlap block, so we store the
  torch convention.

* embed (V x D) and readout (D x 1) are deliberately not saved: ``discovery``
  skips anything matching "embed", and a rank-1 readout carries no RMT signal.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# 0.0 is included on purpose: you need W_init to compute W - W_init, which is
# where the muP width-scaling predictions actually live.
DEFAULT_FRACS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def checkpoint_steps(num_train_steps: int, fracs=DEFAULT_FRACS) -> dict:
    """Map {step_index: fraction}. Deduplicated, so tiny runs still work."""
    out = {}
    last = num_train_steps - 1
    for f in fracs:
        s = int(round(f * last))
        out.setdefault(s, f)
    return out


def save_checkpoint(model, out_dir: str, run_id: str, frac: float, step: int,
                    meta: dict | None = None) -> str:
    """Write one checkpoint. Returns the path written."""
    os.makedirs(out_dir, exist_ok=True)
    state = nnx.state(model, nnx.Param)

    arrays = {}
    flat = jax.tree_util.tree_flatten_with_path(state)[0]
    for path, leaf in flat:
        parts = [str(k) for k in path]
        joined = ".".join(parts)
        # paths look like: ['blocks'].[0].['fc1'].['kernel']..value
        if "blocks" not in joined:
            continue                      # skip embed / readout
        layer = None
        which = None
        for p in parts:
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits and layer is None and "[" in p and "'" not in p:
                layer = int(digits)
            if "fc1" in p:
                which = "fc1"
            elif "fc2" in p:
                which = "fc2"
        if layer is None or which is None:
            continue
        W = np.asarray(jax.device_get(leaf), dtype=np.float32)
        if W.ndim != 2:
            continue
        # Flax (in, out) -> torch (out, in)
        arrays[f"model.layers.{layer}.mlp.{which}.weight"] = np.ascontiguousarray(W.T)

    if not arrays:
        raise RuntimeError("no fc1/fc2 kernels found — parameter layout changed?")

    tag = f"frac{frac:.2f}".replace(".", "p")
    path = os.path.join(out_dir, f"{run_id}_{tag}.npz")
    payload = dict(arrays)
    payload["__meta__"] = np.array(
        repr(dict(meta or {}, frac=float(frac), step=int(step),
                  n_matrices=len(arrays))), dtype=object)
    np.savez_compressed(path, **payload)
    return path
