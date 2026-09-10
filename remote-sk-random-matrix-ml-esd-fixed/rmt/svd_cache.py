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


def save_svd(cache_dir, name, U, s, Vh, *, digest=None,
             backend="unknown", factorization_dtype=None, degraded=False) -> str:
    """Save (U, s, Vh) to ``cache_dir/<sanitized name>.npz``; return the path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _safe_stem(name) + ".npz")
    fd, temporary = tempfile.mkstemp(prefix=".svd-", suffix=".npz", dir=cache_dir)
    os.close(fd)
    try:
        inferred_dtype = factorization_dtype or str(np.asarray(s).dtype)
        np.savez_compressed(
            temporary, U=np.asarray(U), s=np.asarray(s), Vh=np.asarray(Vh),
            name=np.array(name), digest=np.array(digest or ""),
            backend=np.array(str(backend)),
            factorization_dtype=np.array(str(inferred_dtype)),
            degraded=np.array(bool(degraded)),
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_svd(cache_dir, name, *, digest=None, required_dtype=None,
             allow_degraded=False, return_metadata=False, expected_shape=None):
    """Load factors only when identity and requested precision contract match.

    ``return_metadata=True`` appends the persisted backend, factorization dtype,
    and degraded marker so aggregation cannot erase precision provenance.
    """
    path = os.path.join(cache_dir, _safe_stem(name) + ".npz")
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if "name" not in d or str(d["name"].item()) != str(name):
                return None
            if digest is not None and ("digest" not in d or str(d["digest"].item()) != digest):
                return None
            cached_dtype = (str(d["factorization_dtype"].item())
                            if "factorization_dtype" in d else None)
            degraded = bool(d["degraded"].item()) if "degraded" in d else False
            if required_dtype is not None and cached_dtype != str(required_dtype):
                return None
            if degraded and not allow_degraded:
                return None
            U, s, Vh = d["U"], d["s"], d["Vh"]
            factor_dtypes = (U.dtype, s.dtype, Vh.dtype)
            if (not all(np.issubdtype(dtype, np.floating) for dtype in factor_dtypes)
                    or len(set(factor_dtypes)) != 1):
                return None
            actual_dtype = str(s.dtype)
            if cached_dtype is not None and cached_dtype != actual_dtype:
                return None
            if required_dtype is not None and actual_dtype != str(required_dtype):
                return None
            if U.ndim != 2 or s.ndim != 1 or Vh.ndim != 2:
                return None
            if (U.shape[1] != s.size or Vh.shape[0] != s.size
                    or s.size > min(U.shape[0], Vh.shape[1])):
                return None
            if expected_shape is not None:
                expected_n, expected_m = map(int, expected_shape)
                expected_k = min(expected_n, expected_m)
                if (U.shape != (expected_n, expected_k)
                        or s.shape != (expected_k,)
                        or Vh.shape != (expected_k, expected_m)):
                    return None
            if (not np.all(np.isfinite(U)) or not np.all(np.isfinite(s))
                    or not np.all(np.isfinite(Vh)) or np.any(s < 0.0)):
                return None
            # Every vector-associated diagnostic assumes LAPACK's descending
            # singular-value convention.  A consistently permuted cache can
            # reconstruct the weight while violating that semantic invariant.
            if s.size > 1 and np.any(s[:-1] < s[1:]):
                return None
            if return_metadata:
                backend = str(d["backend"].item()) if "backend" in d else "unknown-cache"
                return U, s, Vh, {
                    "backend": backend,
                    "factorization_dtype": cached_dtype or str(s.dtype),
                    "degraded": degraded,
                }
            return U, s, Vh
    except Exception:
        # Cache files are an optional optimization.  Truncated ZIPs, malformed
        # scalar metadata, and missing members are cache misses; callers can
        # recompute from the live trusted weight and atomically replace them.
        return None
