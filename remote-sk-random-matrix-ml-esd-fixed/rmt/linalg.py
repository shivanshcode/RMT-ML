"""rmt.linalg — SVDResult dataclass and the single cached SVD entry point.

The ``SVDResult`` dataclass is *pure* (no torch). ``cached_svd`` dispatches to a
GPU (torch+cuda) or CPU (numpy) backend but always returns numpy arrays, so the
scientific core downstream never sees torch.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SVDResult:
    """One SVD factorization, threaded through all per-matrix analysis.

    Fields
    ------
    U  : (n, k) left singular vectors
    s  : (k,)   singular values, **descending**, k = min(n, m)
    Vh : (k, m) right singular vectors (rows are v_k, each length m = in_features)
    n, m : matrix shape
    """
    U: np.ndarray
    s: np.ndarray
    Vh: np.ndarray
    n: int
    m: int
    backend: str = "unknown"
    factorization_dtype: str = "float64"
    degraded: bool = False

    @property
    def gamma(self) -> float:
        """Aspect ratio min(n, m) / max(n, m) in (0, 1]."""
        return min(self.n, self.m) / max(self.n, self.m)


def singular_basis_status(result: SVDResult) -> str:
    """Qualify diagnostics that depend on individual singular vectors."""

    values = np.asarray(result.s, dtype=np.float64)
    scale = float(np.max(values)) if values.size else 0.0
    epsilon = np.finfo(np.dtype(result.factorization_dtype)).eps
    tolerance = epsilon * max(result.n, result.m) * scale
    if scale == 0.0 or np.any(values <= tolerance):
        return "unavailable: nonidentifiable null singular subspace"
    if values.size > 1 and np.any(np.abs(np.diff(values)) <= tolerance):
        return "unavailable: unresolved repeated singular-value subspace"
    return "available"


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def cached_svd(weight, full_matrices: bool = False, *, backend: str = "auto",
               gpu_min_dim: int = 1024) -> SVDResult:
    """Compute one SVD; dispatch to a GPU or CPU backend.

    Parameters
    ----------
    weight : np.ndarray | torch.Tensor   2-D matrix (n, m)
    full_matrices : bool
    backend : {"auto", "numpy", "torch"}
        "torch" runs ``torch.linalg.svd`` on cuda (drives the A100); "numpy" runs
        ``np.linalg.svd``; "auto" picks torch+cuda when available and
        ``max(n, m) >= gpu_min_dim``, else numpy.
    gpu_min_dim : int
        Threaded from ``cfg.gpu_svd_min_dim``; the "auto" backend only goes to the
        GPU when ``max(n, m)`` reaches this dimension.

    Returns
    -------
    SVDResult with **numpy float64** arrays (U, s, Vh) regardless of backend.

    Precision (REPORT §0): the torch path runs the SVD in **float64** and returns
    float64 U/s/Vh. A float32 SVD floors the smallest singular values at
    ~1e-7·σ_max — exactly the regime Paper 3 studies — so float32 is only used as
    a logged last-resort OOM fallback.
    """
    # Accept torch tensors too (cast to numpy float64 for the numpy path).
    W = _as_float_2d(weight)
    n, m = int(W.shape[0]), int(W.shape[1])
    min_dim_for_gpu = int(gpu_min_dim)

    use_torch = False
    if backend == "torch":
        use_torch = True
    elif backend == "auto":
        use_torch = _gpu_available() and max(n, m) >= min_dim_for_gpu
    elif backend == "numpy":
        use_torch = False
    else:
        raise ValueError(f"unknown backend {backend!r}")

    actual_backend = "torch" if use_torch else "numpy"
    factorization_dtype = "float64"
    degraded = False
    if use_torch:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.no_grad():
                # float64 SVD — preserves the small singular values (REPORT §0).
                t = torch.as_tensor(W, dtype=torch.float64, device=dev)
                U, s, Vh = torch.linalg.svd(t, full_matrices=full_matrices)
                U = U.detach().to("cpu").numpy().astype(np.float64)
                s = s.detach().to("cpu").numpy().astype(np.float64)
                Vh = Vh.detach().to("cpu").numpy().astype(np.float64)
        except Exception as exc:  # pragma: no cover - GPU/OOM last resort
            # Last-resort: free cache and retry on CPU in numpy float64; only if
            # that also fails do we ever touch float32 (never silently).
            try:
                import torch
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                U, s, Vh = np.linalg.svd(W, full_matrices=full_matrices)
                s = s.astype(np.float64)
                actual_backend = "numpy-fallback"
            except Exception:
                from .config import get_logger
                get_logger("rmt.linalg").warning(
                    "float64 SVD failed (%s); falling back to float32", exc)
                U, s, Vh = np.linalg.svd(W.astype(np.float32),
                                         full_matrices=full_matrices)
                s = s.astype(np.float64)
                actual_backend = "numpy-fallback"
                factorization_dtype = "float32"
                degraded = True
    else:
        U, s, Vh = np.linalg.svd(W, full_matrices=full_matrices)
        s = s.astype(np.float64)

    return SVDResult(U=np.asarray(U, dtype=np.float64),
                     s=np.asarray(s, dtype=np.float64),
                     Vh=np.asarray(Vh, dtype=np.float64), n=n, m=m,
                     backend=actual_backend,
                     factorization_dtype=factorization_dtype,
                     degraded=degraded)


def _as_float_2d(weight) -> np.ndarray:
    """Cast a weight (numpy or torch) up to float64 2-D numpy for stable RMT math.

    REPORT §0: never round-trip through float32 here — a torch float64 weight must
    survive as float64, or the smallest singular values become float32 noise.
    """
    if hasattr(weight, "detach"):  # torch.Tensor
        import torch
        if bool(weight.is_complex()):
            raise TypeError("complex weights are not supported by the real RMT SVD API")
        weight = weight.detach().to("cpu").to(torch.float64).numpy()
    W = np.asarray(weight)
    if np.iscomplexobj(W):
        raise TypeError("complex weights are not supported by the real RMT SVD API")
    if W.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {W.shape}")
    return W.astype(np.float64, copy=False)
