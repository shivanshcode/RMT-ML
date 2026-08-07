"""rmt.svd_cache — persist per-matrix SVD factors to disk (npz).

Filenames are sanitized so module names like ``model.layers.0.self_attn.q_proj``
or fused tags ``...query_key_value.weight[Q]`` become safe file stems.
"""
from __future__ import annotations

import os
import re
import numpy as np

_SANITIZE = re.compile(r"""[ /.\[\]'"]+""")


def _safe_stem(name: str) -> str:
    stem = _SANITIZE.sub("_", name).strip("_")
    return stem or "matrix"


def save_svd(cache_dir, name, U, s, Vh) -> str:
    """Save (U, s, Vh) to ``cache_dir/<sanitized name>.npz``; return the path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _safe_stem(name) + ".npz")
    np.savez_compressed(path, U=np.asarray(U), s=np.asarray(s), Vh=np.asarray(Vh),
                        name=np.array(name))
    return path


def load_svd(cache_dir, name):
    """Load (U, s, Vh); return None if absent."""
    path = os.path.join(cache_dir, _safe_stem(name) + ".npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as d:
        return d["U"], d["s"], d["Vh"]
