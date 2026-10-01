"""checkpoint_chess.py — save every 2-D weight matrix of the TRANSFORMER.

Used for the chess / CIFAR-5M transformer runs. The MLP ladder keeps the
original ``checkpoint.py`` untouched, so the MLP checkpoints and every RMT
result already derived from them are unchanged. ``train.py`` picks between the
two on ``cfg.arch``.

WHY THIS FILE EXISTS
--------------------
``checkpoint.py``'s save loop keeps a parameter only when its path contains
"fc1" or "fc2". On the MLP that is every weight matrix. On the transformer it
silently discards all four attention matrices per block --- 12 of 18, and the
bulk of the parameters --- leaving an RMT analysis of a transformer that never
looks at attention.

Design notes
------------
* Written as ``.npz`` (numpy), NOT torch ``.pt``, so the JAX training env never
  needs torch installed. The RMT side rebuilds an ``nn.Module`` from the npz.

* SAVES EVERYTHING 2-D. Every 2-D Param is written and the RMT side decides
  what to analyse: 18 block matrices for a 3-block transformer, plus the two
  embeddings and the readout.

* Key names are chosen so ``rmt.discovery.classify`` resolves them under the
  generic spec without touching the suite. ``classify`` tries
  ``[QKV, Q, K, V, G, U, D, O]`` in order, first substring match wins:

      model.layers.{i}.attn.query_proj.weight   -> "Q"   (via "query")
      model.layers.{i}.attn.key_proj.weight     -> "K"   (via "key")
      model.layers.{i}.attn.value_proj.weight   -> "V"   (via "value")
      model.layers.{i}.attn.output_proj.weight  -> "O"   (via "proj")
      model.layers.{i}.mlp.fc1.weight           -> "U"   (via "fc1")
      model.layers.{i}.mlp.fc2.weight           -> "D"   (via "fc2")

  and ``extract_layer_index`` picks up ``i`` via the generic regex
  ``\\.layers?\\.(\\d+)\\.``.

  Q/K/V are checked before O, so ``query_proj`` is not swallowed by O's very
  broad "proj" pattern. Order is load-bearing; do not rename these.

* Embeddings and the readout are saved under names ``rmt.discovery``
  deliberately skips (``_SKIP_SUBSTRINGS`` contains "embed" and "lm_head", and
  "readout" matches no pattern so it classifies to None). They are in the file
  so nothing is lost; the RMT loader exposes ``--include-embed`` to rename them
  into analysable roles when you want them.

* TRANSPOSED ON SAVE — kernels only. Flax ``nnx.Linear`` stores the kernel as
  ``(in, out)``; torch ``nn.Linear.weight`` is ``(out, in)``, which is what
  ``rmt.discovery._weight_2d`` assumes. ``nnx.Embed`` stores ``embedding`` as
  ``(num_embeddings, features)``, which already matches torch
  ``nn.Embedding.weight`` — so embeddings are NOT transposed. The distinction
  is made on the leaf name ("kernel" vs "embedding"), not on the module.

  For square matrices the SVD spectrum is unchanged either way, but row/column
  semantics matter for ``--N_cov_mode cols`` and for the overlap block, and the
  transformer's embed (V x D) and readout (D x V) are not square.
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

# Flax submodule name -> (RMT parent, RMT leaf). Anything not listed is kept
# under its own name, so a new layer type is saved rather than silently lost.
_BLOCK_LEAF = {
    "query_proj":  ("attn", "query_proj"),
    "key_proj":    ("attn", "key_proj"),
    "value_proj":  ("attn", "value_proj"),
    "output_proj": ("attn", "output_proj"),
    "fc1":         ("mlp",  "fc1"),
    "fc2":         ("mlp",  "fc2"),
}

# Top-level params. These names are all skipped or unclassified by
# rmt.discovery on purpose — see the module docstring.
_TOP_LEVEL = {
    "embed":     "model.embed_tokens.weight",
    "pos_embed": "model.pos_embed.weight",
    "readout":   "lm_head.weight",
}


def checkpoint_steps(num_train_steps: int, fracs=DEFAULT_FRACS) -> dict:
    """Map {step_index: fraction}. Deduplicated, so tiny runs still work."""
    out = {}
    last = num_train_steps - 1
    for f in fracs:
        s = int(round(f * last))
        out.setdefault(s, f)
    return out


def _path_parts(path) -> list:
    """jax tree path -> list of plain strings.

    Dict keys arrive as ``['blocks']``, list indices as ``[0]``, and the final
    element is flax's ``VariableState.value`` accessor, which ``str()``s to
    ``.value``. Strip the decoration and DROP anything that normalises to
    nothing, so a path renders as

        ['blocks', '0', 'attn', 'key_proj', 'kernel']

    and ``parts[-1]`` is the real leaf. Without the drop the trailing
    ``.value`` became an empty final element, ``parts[-1]`` was ``''``, and
    every parameter was skipped as "not a kernel" --- the whole checkpoint came
    out empty.
    """
    parts = []
    for k in path:
        s = str(k).strip().strip("[]").strip("'\"").lstrip(".")
        if s.endswith(".value"):
            s = s[: -len(".value")]
        if s in ("", "value"):
            continue
        parts.append(s)
    return parts


def _classify_path(parts: list):
    """(rmt_key, is_kernel) for one parameter path, or (None, _) to skip."""
    leaf = parts[-1]                      # 'kernel' | 'embedding' | 'scale' ...
    is_kernel = leaf == "kernel"
    if leaf not in ("kernel", "embedding"):
        return None, is_kernel            # RMSNorm scale etc.

    if "blocks" in parts:
        bi = parts.index("blocks")
        # the element right after 'blocks' is the block index
        if bi + 1 >= len(parts) or not parts[bi + 1].isdigit():
            return None, is_kernel
        layer = int(parts[bi + 1])
        inner = parts[bi + 2:-1]          # e.g. ['attn', 'query_proj']
        if not inner:
            return None, is_kernel
        name = inner[-1]
        parent, leafname = _BLOCK_LEAF.get(name, (".".join(inner[:-1]) or "misc", name))
        return f"model.layers.{layer}.{parent}.{leafname}.weight", is_kernel

    # top level: embed / pos_embed / readout
    for key, mapped in _TOP_LEVEL.items():
        if key in parts:
            return mapped, is_kernel
    return None, is_kernel


def save_checkpoint(model, out_dir: str, run_id: str, frac: float, step: int,
                    meta: dict | None = None) -> str:
    """Write one checkpoint. Returns the path written."""
    os.makedirs(out_dir, exist_ok=True)
    state = nnx.state(model, nnx.Param)

    arrays = {}
    skipped = []
    flat = jax.tree_util.tree_flatten_with_path(state)[0]
    for path, leaf in flat:
        parts = _path_parts(path)
        key, is_kernel = _classify_path(parts)
        if key is None:
            skipped.append(".".join(parts))
            continue
        W = np.asarray(jax.device_get(leaf), dtype=np.float32)
        if W.ndim != 2:
            skipped.append(".".join(parts) + f"[ndim={W.ndim}]")
            continue
        # Flax Linear kernel is (in, out) -> torch (out, in).
        # Flax Embed embedding is (V, D), which already matches torch.
        arrays[key] = np.ascontiguousarray(W.T if is_kernel else W)

    if not arrays:
        raw = [".".join(_path_parts(p)) for p, _ in flat[:10]]
        raise RuntimeError(
            "no 2-D parameters found — parameter layout changed?\n"
            f"  normalised paths: {raw}\n"
            f"  skipped: {skipped[:10]}")

    tag = f"frac{frac:.2f}".replace(".", "p")
    path = os.path.join(out_dir, f"{run_id}_{tag}.npz")
    payload = dict(arrays)
    payload["__meta__"] = np.array(
        repr(dict(meta or {}, frac=float(frac), step=int(step),
                  n_matrices=len(arrays), skipped=skipped)), dtype=object)
    np.savez_compressed(path, **payload)
    return path
