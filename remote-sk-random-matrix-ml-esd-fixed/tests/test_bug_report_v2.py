import json
from types import SimpleNamespace

import numpy as np
import pytest

from rmt.config import RunConfig
from rmt.linalg import SVDResult
from rmt.mp import mp_cdf, mp_pdf, small_sv_deviation
from rmt.mp_fit import modified_mp
from rmt.overlap import overlap_analysis
from rmt.per_matrix import per_matrix_analysis
from rmt.pipeline import analyze_checkpoints, analyze_one_model
from rmt.spacing import delta3, sigma2
from rmt import tail


def test_overlap_uses_actual_fp32_factorization_precision() -> None:
    singular = np.asarray([5.000001, 5.0, 4.0, 3.0])
    svd = SVDResult(
        U=np.eye(4), s=singular, Vh=np.eye(4), n=4, m=4,
        factorization_dtype="float32", degraded=True,
    )
    covariance = np.diag([4.0, 3.0, 2.0, 1.0])
    result = overlap_analysis(np.diag(singular), covariance, svd=svd)
    assert result["available"] is False
    assert "repeated" in result["status"]


def test_duplicate_checkpoint_probes_fail_before_loading(tmp_path) -> None:
    called = False

    def loader(frac):
        nonlocal called
        called = True
        raise AssertionError(frac)

    with pytest.raises(ValueError, match="duplicates"):
        analyze_checkpoints(loader, [0.0], [0, 0], tmp_path, "run")
    assert called is False
    assert list(tmp_path.iterdir()) == []


def test_library_model_tags_cannot_escape_output_directory(tmp_path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"safe")
    with pytest.raises(ValueError, match="safe filename"):
        analyze_one_model(object(), "../sentinel", tmp_path / "output")
    with pytest.raises(ValueError, match="safe filename"):
        analyze_checkpoints(lambda _: object(), [0.0], [0], tmp_path / "output", "/tmp/x")
    assert sentinel.read_bytes() == b"safe"
    assert not (tmp_path / "output").exists()


def test_direct_matrix_analysis_rejects_complex_weight() -> None:
    record = SimpleNamespace(
        weight=1j * np.eye(4), n=4, m=4, name="complex.weight",
        short="Q", layer_idx=0, source_dtype="complex128",
    )
    with pytest.raises(TypeError, match="complex"):
        per_matrix_analysis(record, cfg=RunConfig(n_deciles=4))


def test_mp_and_modified_density_are_stable_under_rescaling() -> None:
    n = m = 4
    base_pdf = float(mp_pdf(1.0, n, m, 0.5))
    base_cdf = float(mp_cdf(1.0, n, m, 0.5))
    for scale in (1e-100, 1e100):
        observed = float(mp_pdf(scale, n, m, 0.5 * scale))
        assert np.isfinite(observed)
        assert observed * scale == pytest.approx(base_pdf)
        assert mp_cdf(scale, n, m, 0.5 * scale) == pytest.approx(base_cdf)
        modified = float(modified_mp(np.asarray(scale), 2.0, 2.0 * scale, 0.5 * scale))
        expected = float(modified_mp(np.asarray(1.0), 2.0, 2.0, 0.5)) * scale
        assert np.isfinite(modified)
        assert modified == pytest.approx(expected)
    spectrum = np.asarray([0.1, 0.5, 1.0, 2.0])
    base = small_sv_deviation(spectrum, 4, 16, 0.5)
    scaled = small_sv_deviation(spectrum * 1e150, 4, 16, 0.5e150)
    assert scaled["frac_mass_below_minus"] == pytest.approx(base["frac_mass_below_minus"])


def test_hill_plateau_search_considers_overlapping_bands(monkeypatch) -> None:
    values = np.asarray([1.0, 1.0, 1.19, 1.4, 1.4, 1.4, 1.4])
    monkeypatch.setattr(
        tail, "hill_estimator_windowed",
        lambda sample, window: (np.arange(1, values.size + 1), values.copy()),
    )
    result = tail.hill_plateau(np.arange(20.0) + 1.0, window=5, flat_tol=0.20)
    assert result["hill_plateau_width"] == 5
    assert result["hill_is_powerlaw"] is True


def test_low_rank_roundoff_spectrum_makes_mp_unavailable(monkeypatch) -> None:
    from rmt import per_matrix as module

    singular = np.asarray([3.0, 2.0, 1.0, 1e-15, 1e-16, 0.0, 0.0, 0.0])
    fake = SVDResult(
        U=np.eye(8), s=singular, Vh=np.eye(8), n=8, m=8,
        backend="test", factorization_dtype="float64",
    )
    monkeypatch.setattr(module, "cached_svd", lambda *args, **kwargs: fake)
    record = SimpleNamespace(
        weight=np.diag(singular), n=8, m=8, name="low_rank.weight",
        short="Q", layer_idx=0, source_dtype="float64",
    )
    cfg = RunConfig(
        do_powerlaw=False, do_spacing=False, do_ipr=False,
        do_porter_thomas=False, do_overlap=False, n_deciles=4,
    )
    row = per_matrix_analysis(record, cfg=cfg)
    assert row["mp_available"] == 0
    assert row["mp_resolvable_count"] == 3
    assert np.isnan(row["sigma_med"])


@pytest.mark.parametrize("length", [-1.0, 0.0, np.nan, np.inf])
def test_interval_statistics_reject_invalid_lengths(length) -> None:
    levels = np.linspace(0.0, 20.0, 100)
    with pytest.raises(ValueError, match="finite and positive"):
        sigma2(levels, length)
    with pytest.raises(ValueError, match="finite and positive"):
        delta3(levels, length)


def test_strict_analysis_rejects_partly_missing_layers(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    del torch
    from tests.synthetic_models import tiny_causal_lm

    model = tiny_causal_lm(n_layers=1, d=16)
    with pytest.raises(ValueError, match="requested layers"):
        analyze_one_model(
            model, "strict", tmp_path, layers=[0, 99], strict=True,
            do_overlap=False, do_perplexity=False,
        )


def test_checkpoint_coverage_changes_are_recorded(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    del torch
    from tests.synthetic_models import tiny_causal_lm

    def loader(frac):
        model = tiny_causal_lm(n_layers=1, d=8)
        if frac > 0.0:
            del model.model.layers[0].self_attn.q_proj
        return model

    analyze_checkpoints(loader, [0.0, 1.0], [0], tmp_path, "coverage", strict=False)
    status = json.loads((tmp_path / "coverage_checkpoint_status.json").read_text())
    assert status["status"] == "partial"
    assert status["coverage_changes"]


def test_complex_decile_preflight_preserves_all_parameters() -> None:
    torch = pytest.importorskip("torch")
    from tests.synthetic_models import tiny_llama
    from rmt.decile import set_layer_svd_decile
    from rmt.discovery import discover_weight_metadata

    model = tiny_llama(n_layers=1, d=4).to(torch.complex64)
    records = discover_weight_metadata(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(TypeError, match="complex"):
        set_layer_svd_decile(model, records, 1, n_deciles=2)
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
