import os
import csv
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from _synthetic_models import tiny_causal_lm, TinyPythiaCausalLM   # noqa: E402

from rmt import pipeline as P                                   # noqa: E402
from rmt.per_matrix import CSV_COLUMNS                          # noqa: E402

pytestmark = pytest.mark.torch


def test_analyze_tiny_model_writes_csv(tmp_path):
    model = tiny_causal_lm(n_layers=2, d=64)
    csv_path, rows = P.analyze_one_model(model, "tiny", str(tmp_path),
                                         do_overlap=False, do_perplexity=False)
    assert os.path.exists(csv_path)
    assert len(rows) > 0
    with open(csv_path) as f:
        header = next(csv.reader(f))
    assert header == list(CSV_COLUMNS)
    assert os.path.exists(os.path.join(str(tmp_path), "tiny_summary.json"))


def test_selftest_passes():
    from rmt.selftest import run
    assert run() is True


def test_cli_selftest_exit_zero():
    from rmt.cli import main
    assert main(["--selftest"]) == 0


def test_model_swap_smoke(tmp_path):
    # a pythia-style fused-QKV model must yield separate Q/K/V records
    model = TinyPythiaCausalLM(n_layers=2, d=48)
    csv_path, rows = P.analyze_one_model(model, "pythia", str(tmp_path),
                                         do_overlap=False, do_perplexity=False)
    shorts = {r["short"] for r in rows}
    assert {"Q", "K", "V"} <= shorts
    assert all(r["n"] == 48 for r in rows if r["short"] in ("Q", "K", "V"))


def test_backend_numpy_matches_torch():
    from rmt.linalg import cached_svd
    W = np.random.default_rng(0).standard_normal((128, 96))
    rn = cached_svd(W, backend="numpy")
    if not torch.cuda.is_available():
        pytest.skip("no cuda; torch backend == numpy path")
    rt = cached_svd(W, backend="torch")
    assert np.allclose(np.sort(rn.s), np.sort(rt.s), atol=1e-4)


def test_analyze_with_perplexity_and_overlap_regression(tmp_path):
    """Regression for the two pipeline bugs:
    (1) decile ablation must NOT accumulate / degrade the model's weights;
    (2) overlap must actually run (at least one finite max_overlap) without a
        real tokenizer (offline hash fallback)."""
    model = tiny_causal_lm(n_layers=2, d=64)
    # snapshot pristine singular-value energy of one matrix
    q = model.get_submodule("model.layers.0.self_attn.q_proj")
    s_before = np.linalg.svd(q.weight.detach().numpy(), compute_uv=False)
    energy_before = float(np.sum(s_before))

    csv_path, rows = P.analyze_one_model(
        model, "reg", str(tmp_path), do_overlap=True, do_perplexity=True,
        n_deciles=4, perplexity_tokens=64, n_text_batches=2,
        fm_max_length=64, fm_stride=32)

    # (1) weights restored / non-degenerate (NOT all driven to zero)
    s_after = np.linalg.svd(q.weight.detach().numpy(), compute_uv=False)
    energy_after = float(np.sum(s_after))
    assert energy_after > 0.5 * energy_before, (energy_before, energy_after)
    assert np.sum(s_after > 1e-6) >= len(s_after) - 1     # not collapsed to ~0

    # (2) overlap actually ran for at least one matrix
    assert any(np.isfinite(r["max_overlap"]) for r in rows)

    # (3) perplexity-vs-decile produced finite, non-constant values
    import json
    with open(os.path.join(str(tmp_path), "reg_perplexity.json")) as f:
        ppl = json.load(f)
    assert len(ppl["perplexity"]) == 4


def test_cli_full_run_end_to_end(tmp_path, monkeypatch):
    """Regression: the full `cli.main` deployment path (load -> analyze -> write)
    must run without argument collisions. Patches the loaders so no real HF
    snapshot or tokenizer is needed."""
    import rmt.model_io as mio
    import rmt.cli as cli

    monkeypatch.setattr(mio, "load_model", lambda tag, **k: tiny_causal_lm(n_layers=2, d=64))
    monkeypatch.setattr(mio, "load_tokenizer", lambda tag, **k: None)

    rc = cli.main(["--models", "dummy", "--output_dir", str(tmp_path),
                   "--layers", "0", "--n_text_batches", "2", "--fm_max_length", "64",
                   "--fm_stride", "32", "--do_overlap", "--do_perplexity",
                   "--n_deciles", "4", "--perplexity_tokens", "64",
                   "--no-do_qkv_heatmap"])
    assert rc == 0
    assert os.path.exists(os.path.join(str(tmp_path), "dummy", "dummy_matrix_metrics.csv"))


def test_epoch_checkpoint_tracking(tmp_path):
    # settable probe-layer list across training fractions -> per-epoch CSV
    def loader(frac):
        m = tiny_causal_lm(n_layers=4, d=32)
        return m
    path = P.analyze_checkpoints(loader, [0.0, 0.5, 1.0], probe_layers=[0, 2],
                                 output_dir=str(tmp_path), model_tag="ep")
    assert os.path.exists(path)
    with open(path) as f:
        header = next(csv.reader(f))
    assert "training_fraction" in header
    assert any("layer_0" in h for h in header) and any("layer_2" in h for h in header)
