"""rmt.svd_cache — persist per-matrix SVD factors to disk (npz).

Filenames are sanitized so module names like ``model.layers.0.self_attn.q_proj``
or fused tags ``...query_key_value.weight[Q]`` become safe file stems.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import numpy as np

_SANITIZE = re.compile(r"""[ /.\[\]'"]+""")


def _safe_stem(name: str) -> str:
    stem = _SANITIZE.sub("_", name).strip("_") or "matrix"
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:16]
    return f"{stem}_{digest}"


def weight_digest(weight) -> str:
    array = np.ascontiguousarray(np.asarray(weight))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_svd(cache_dir, name, U, s, Vh, *, digest=None) -> str:
    """Save (U, s, Vh) to ``cache_dir/<sanitized name>.npz``; return the path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _safe_stem(name) + ".npz")
    fd, temporary = tempfile.mkstemp(prefix=".svd-", suffix=".npz", dir=cache_dir)
    os.close(fd)
    try:
        np.savez_compressed(temporary, U=np.asarray(U), s=np.asarray(s), Vh=np.asarray(Vh),
                            name=np.array(name), digest=np.array(digest or ""))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_svd(cache_dir, name, *, digest=None):
    """Load (U, s, Vh); return None if absent."""
    path = os.path.join(cache_dir, _safe_stem(name) + ".npz")
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as d:
        if "name" not in d or str(d["name"].item()) != str(name):
            return None
        if digest is not None and ("digest" not in d or str(d["digest"].item()) != digest):
            return None
        U, s, Vh = d["U"], d["s"], d["Vh"]
        if U.ndim != 2 or s.ndim != 1 or Vh.ndim != 2:
            return None
        if U.shape[1] != s.size or Vh.shape[0] != s.size:
            return None
        return U, s, Vh
