"""rmt.baselines.weightwatcher — optional WeightWatcher comparison.

Best-effort: dependency and execution failures return structured status records,
so a missing baseline never breaks the main analysis or disappears silently.
"""
from __future__ import annotations


def run_weightwatcher(model, *, normalize=False, glorot_fix=False):
    try:
        import weightwatcher as ww  # local, optional
    except Exception as error:
        return {"status": "unavailable", "reason": repr(error)}
    try:
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze(normalize=normalize, glorot_fix=glorot_fix)
        summary = watcher.get_summary(details)
        return {"status": "complete", "summary": summary,
                "details": details.to_dict() if hasattr(details, "to_dict") else None}
    except Exception as error:
        return {"status": "failed", "reason": repr(error)}
