"""rmt — Random Matrix Theory analysis of LLM weight matrices.

Importing :mod:`rmt` only pulls in the pure numpy/scipy scientific core; the
torch/HF model-I/O modules (discovery, activations, decile, perplexity,
pipeline) are imported lazily so the pure test groups never require torch.
"""
__version__ = "1.0.0"

from . import config, ensembles, mp, tail, scalars, spacing, overlap, linalg  # noqa: F401
from .linalg import SVDResult, cached_svd  # noqa: F401
from .config import RunConfig, TOL, OfflineGuard, get_logger  # noqa: F401

__all__ = [
    "config", "ensembles", "mp", "tail", "scalars", "spacing", "overlap", "linalg",
    "SVDResult", "cached_svd", "RunConfig", "TOL", "OfflineGuard", "get_logger",
    "__version__",
]
