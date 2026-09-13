import json
import os
from pathlib import Path

import pytest

from rmt.frontier_sweep import (
    SnapshotManager,
    _select_models,
    layer_batches,
    load_catalog,
    representative_layers,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_catalog_contains_supported_frontier_models():
    catalog = load_catalog(PROJECT_ROOT / "frontier_models.json")
    by_id = {model["id"]: model for model in catalog["models"]}
    assert "qwen3-4b-instruct-2507" in by_id
    assert "qwen3-8b-exploratory" in by_id
    assert by_id["qwen3-8b-exploratory"]["dtype"] == "bf16"
    blocked = " ".join(item["repo_id"] for item in catalog["blocked_models"])
    assert "Qwen/Qwen3.8-27B" in blocked


def test_default_selection_omits_gated_and_optional_models():
    catalog = load_catalog(PROJECT_ROOT / "frontier_models.json")
    selected = _select_models(catalog, None, include_optional=False, include_gated=False)
    assert selected
    assert all(model["enabled_by_default"] for model in selected)
    assert all(model["access"] == "public" for model in selected)


def test_representative_layers_and_small_batches():
    assert representative_layers(36, 3) == [0, 18, 35]
    assert representative_layers(1, 3) == [0]
    assert representative_layers(8, 1) == [4]
    assert layer_batches([3, 2, 2, 1, 0], 2) == [[0, 1], [2, 3]]
    with pytest.raises(ValueError):
        layer_batches([0], 0)


def test_snapshots_keep_old_versions_and_prune(tmp_path):
    source = tmp_path / "results"
    snapshots = tmp_path / "snapshots"
    source.mkdir()
    result = source / "result.json"
    stable = source / "stable.csv"
    result.write_text('{"value": 1}\n', encoding="utf-8")
    stable.write_text("a,b\n1,2\n", encoding="utf-8")
    manager = SnapshotManager(source, snapshots, interval_seconds=3600, keep=2)

    first = manager.snapshot("first")
    old_stat = result.stat()
    result.write_text('{"value": 2}\n', encoding="utf-8")
    os.utime(result, ns=(old_stat.st_atime_ns + 1_000_000,
                         old_stat.st_mtime_ns + 1_000_000))
    second = manager.snapshot("second")

    assert json.loads((first / "result.json").read_text())["value"] == 1
    assert json.loads((second / "result.json").read_text())["value"] == 2
    manager.snapshot("third")
    retained = sorted(path for path in snapshots.iterdir() if path.name.startswith("snapshot-"))
    assert len(retained) == 2
    assert not first.exists()


def test_snapshot_root_cannot_be_inside_results(tmp_path):
    source = tmp_path / "results"
    with pytest.raises(ValueError, match="must not be inside"):
        SnapshotManager(source, source / "snapshots")
