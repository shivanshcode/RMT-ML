"""rmt.baselines.weightwatcher — optional WeightWatcher comparison.

Best-effort: if the local `weightwatcher` package is unavailable, returns None
without raising, so a missing baseline never breaks the main analysis.
"""
from __future__ import annotations


def run_weightwatcher(model, *, normalize=False, glorot_fix=False):
    try:
        import weightwatcher as ww  # local, optional
    except Exception:
        return None
    try:
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze(normalize=normalize, glorot_fix=glorot_fix)
        summary = watcher.get_summary(details)
        return {"summary": summary,
                "details": details.to_dict() if hasattr(details, "to_dict") else None}
    except Exception:
        return None
