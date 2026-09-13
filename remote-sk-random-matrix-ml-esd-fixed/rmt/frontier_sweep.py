"""Crash-tolerant, one-model-at-a-time frontier-model RMT sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

CATALOG_NAME = "frontier_models.json"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_TEST_SHA256 = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"
WIKITEXT_TEST_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
    f"{WIKITEXT_REVISION}/wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
SUPPORTED_MODEL_TYPES = {
    "qwen2", "qwen3", "llama", "mistral", "granite", "olmo", "olmo2",
    "smollm3", "gemma", "gemma2", "gemma3_text", "gpt2", "gpt_neox",
}
_ACTIVE_PROCESS: subprocess.Popen | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported frontier model catalog schema")
    seen = set()
    for model in catalog.get("models", []):
        required = {
            "id", "repo_id", "revision", "model_type", "layers", "download_gb",
            "dtype", "access", "enabled_by_default",
        }
        missing = sorted(required - model.keys())
        if missing:
            raise ValueError(f"catalog model lacks fields {missing}: {model!r}")
        if model["id"] in seen:
            raise ValueError(f"duplicate catalog model id: {model['id']}")
        seen.add(model["id"])
        if model["model_type"] not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"catalog model {model['id']} has unsupported type {model['model_type']}"
            )
        if int(model["layers"]) < 1 or float(model["download_gb"]) <= 0:
            raise ValueError(f"catalog model has invalid resource data: {model['id']}")
        if model["dtype"] not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"catalog model has invalid dtype: {model['id']}")
    return catalog


def representative_layers(layer_count: int, count: int = 3) -> list[int]:
    layer_count, count = int(layer_count), int(count)
    if layer_count < 1 or count < 1:
        raise ValueError("layer_count and count must be positive")
    if count >= layer_count:
        return list(range(layer_count))
    if count == 1:
        return [layer_count // 2]
    return sorted({round(index * (layer_count - 1) / (count - 1)) for index in range(count)})


def layer_batches(layers: Iterable[int], batch_size: int) -> list[list[int]]:
    values = sorted({int(layer) for layer in layers})
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    return [values[index:index + int(batch_size)] for index in range(0, len(values), int(batch_size))]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _same_file_version(source: Path, previous: Path) -> bool:
    try:
        left, right = source.stat(), previous.stat()
        return left.st_size == right.st_size and left.st_mtime_ns == right.st_mtime_ns
    except FileNotFoundError:
        return False


def _copy_snapshot_tree(source: Path, target: Path, previous: Path | None) -> None:
    for root, directories, filenames in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        target_dir = target / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            (target_dir / directory).mkdir(exist_ok=True)
        for filename in filenames:
            source_file = root_path / filename
            target_file = target_dir / filename
            previous_file = None if previous is None else previous / relative / filename
            if source_file.is_symlink():
                target_file.symlink_to(os.readlink(source_file))
            elif previous_file is not None and _same_file_version(source_file, previous_file):
                try:
                    os.link(previous_file, target_file)
                except OSError:
                    shutil.copy2(previous_file, target_file)
            else:
                shutil.copy2(source_file, target_file)


class SnapshotManager:
    """Copy a results tree on a timer and hard-link unchanged snapshot files."""

    def __init__(self, source: Path, root: Path, interval_seconds: int = 3600,
                 keep: int = 24):
        self.source = source.resolve()
        self.root = root.resolve()
        self.interval_seconds = int(interval_seconds)
        self.keep = int(keep)
        if self.interval_seconds < 1 or self.keep < 1:
            raise ValueError("snapshot interval and retention must be positive")
        if _is_within(self.root, self.source):
            raise ValueError("snapshot root must not be inside the results root")
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _snapshots(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            path for path in self.root.iterdir()
            if path.is_dir() and path.name.startswith("snapshot-")
        )

    def snapshot(self, reason: str) -> Path:
        with self._lock:
            self.source.mkdir(parents=True, exist_ok=True)
            self.root.mkdir(parents=True, exist_ok=True)
            previous_items = self._snapshots()
            previous = previous_items[-1] if previous_items else None
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            final = self.root / f"snapshot-{stamp}"
            temporary = self.root / f".snapshot-{stamp}-{os.getpid()}.tmp"
            temporary.mkdir()
            try:
                _copy_snapshot_tree(self.source, temporary, previous)
                _atomic_json(temporary / ".snapshot.json", {
                    "created_at_utc": _utc_now(),
                    "reason": str(reason),
                    "source": str(self.source),
                    "previous": None if previous is None else previous.name,
                })
                os.replace(temporary, final)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
            items = self._snapshots()
            for old in items[:-self.keep]:
                shutil.rmtree(old)
            return final

    def start(self) -> None:
        if self._thread is not None:
            return

        def worker() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self.snapshot("hourly")
                except Exception as error:  # noqa: BLE001 - keep the timer alive
                    print(f"snapshot failed: {error}", file=sys.stderr, flush=True)

        self._thread = threading.Thread(target=worker, name="rmt-snapshots", daemon=True)
        self._thread.start()

    def stop(self, final: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1, min(self.interval_seconds, 10)))
            self._thread = None
        if final:
            self.snapshot("shutdown")


def _select_models(catalog: dict, requested: list[str] | None,
                   include_optional: bool, include_gated: bool) -> list[dict]:
    models = list(catalog.get("models", []))
    by_id = {model["id"]: model for model in models}
    if requested:
        missing = sorted(set(requested) - by_id.keys())
        if missing:
            raise ValueError(f"unknown catalog model ids: {missing}")
        selected = [by_id[name] for name in requested]
    else:
        selected = [
            model for model in models
            if bool(model["enabled_by_default"]) or include_optional
        ]
    if not include_gated:
        selected = [model for model in selected if model["access"] != "gated"]
    return selected


def _stage_wikitext(text_path: Path) -> None:
    if text_path.is_file() and text_path.stat().st_size > 0:
        return
    print(f"downloading WikiText-2 test data to {text_path}", flush=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = text_path.with_suffix(text_path.suffix + ".parquet.tmp")
    with urlopen(WIKITEXT_TEST_URL, timeout=120) as response, parquet_path.open("wb") as output:
        shutil.copyfileobj(response, output)
    if _sha256(parquet_path) != WIKITEXT_TEST_SHA256:
        parquet_path.unlink(missing_ok=True)
        raise RuntimeError("the WikiText-2 test download failed its SHA-256 test")
    from pyarrow import parquet
    table = parquet.read_table(parquet_path, columns=["text"])
    rows = table.column("text").to_pylist()
    temporary = text_path.with_suffix(text_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(str(row))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, text_path)
    parquet_path.unlink(missing_ok=True)
    _atomic_json(text_path.parent / "source.json", {
        "dataset": "Salesforce/wikitext",
        "configuration": "wikitext-2-raw-v1",
        "split": "test",
        "revision": WIKITEXT_REVISION,
        "parquet_sha256": WIKITEXT_TEST_SHA256,
        "created_at_utc": _utc_now(),
    })


def _download_marker(model_dir: Path) -> Path:
    return model_dir / ".rmt-download.json"


def _download_model(model: dict, model_dir: Path, reserve_gb: float) -> None:
    marker = _download_marker(model_dir)
    if marker.is_file():
        with marker.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("revision") == model["revision"]:
            return
        raise RuntimeError(f"model directory has another revision: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(model_dir.parent).free / 1e9
    required_gb = float(model["download_gb"]) * 1.10 + float(reserve_gb)
    if free_gb < required_gb:
        raise RuntimeError(
            f"{model['id']} needs about {required_gb:.1f} GB free before download, "
            f"but only {free_gb:.1f} GB is free"
        )
    print(
        f"downloading {model['repo_id']} at {model['revision']} "
        f"({model['download_gb']} GB)", flush=True
    )
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model["repo_id"],
        revision=model["revision"],
        local_dir=str(model_dir),
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=[
            "*.json", "*.safetensors", "*.bin", "*.model", "*.tiktoken",
            "tokenizer*", "vocab.*", "merges.txt", "special_tokens_map.json",
            "added_tokens.json",
        ],
        ignore_patterns=["*.gguf", "*.onnx", "onnx/*", "original/*"],
    )
    configs = list(model_dir.glob("config.json"))
    weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not configs or not weights:
        raise RuntimeError(f"downloaded model is incomplete: {model_dir}")
    with configs[0].open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    actual_type = config.get("model_type")
    if actual_type != model["model_type"]:
        raise RuntimeError(
            f"catalog expected model_type={model['model_type']}, got {actual_type}"
        )
    _atomic_json(marker, {
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "model_type": actual_type,
        "download_completed_at_utc": _utc_now(),
    })


def _task_name(stage: str, layers: list[int]) -> str:
    joined = "-".join(f"{layer:03d}" for layer in layers)
    return f"{stage}-layers-{joined}"


def _complete_marker(task_root: Path) -> Path:
    return task_root / "complete.json"


def _task_contract(model: dict, stage: str, layers: list[int],
                   args: argparse.Namespace) -> dict:
    contract = {
        "pipeline_version": 1,
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "dtype": model["dtype"],
        "stage": stage,
        "layers": list(layers),
        "gpu_svd_min_dim": args.gpu_svd_min_dim,
    }
    if stage == "weights":
        contract.update({
            "activation_batches": args.activation_batches,
            "activation_length": args.activation_length,
            "activation_stride": args.activation_stride,
        })
    else:
        contract.update({
            "deciles": args.deciles,
            "perplexity_tokens": args.perplexity_tokens,
            "perplexity_stride": args.perplexity_stride,
            "decile_scope": "analyzed",
        })
    return contract


def _task_is_complete(task_root: Path, contract: dict) -> bool:
    marker = _complete_marker(task_root)
    try:
        with marker.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        return record.get("contract") == contract
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _next_attempt(task_root: Path) -> Path:
    existing = [
        path for path in task_root.glob("attempt-*")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    ]
    number = max([int(path.name.split("-")[-1]) for path in existing], default=0) + 1
    return task_root / f"attempt-{number:03d}"


def _status_files(path: Path) -> list[Path]:
    return sorted(path.glob("**/*_run_status.json"))


def _build_command(model_dir: Path, attempt: Path, layers: list[int], stage: str,
                   model: dict, args: argparse.Namespace, text_path: Path) -> list[str]:
    command = [
        sys.executable, "-m", "rmt",
        "--models", str(model_dir),
        "--output_dir", str(attempt),
        "--layers", *[str(layer) for layer in layers],
        "--dtype", str(model["dtype"]),
        "--backend", "auto",
        "--gpu_svd_min_dim", str(args.gpu_svd_min_dim),
        "--sigma_estimator", "gd_median",
        "--N_cov_mode", "cols",
        "--text_path", str(text_path),
        "--no-use_svd_cache",
        "--strict",
        "--offline",
    ]
    if stage == "weights":
        command.extend([
            "--alpha_estimator", "all",
            "--hill_window", "20",
            "--n_text_batches", str(args.activation_batches),
            "--fm_max_length", str(args.activation_length),
            "--fm_stride", str(min(args.activation_stride, args.activation_length - 1)),
            "--do_overlap", "--do_qkv_heatmap", "--do_spacing", "--do_ipr",
            "--do_powerlaw", "--no-do_perplexity", "--no-do_complex_spacing",
            "--no-do_porter_thomas",
        ])
    elif stage == "decile":
        command.extend([
            "--no-do_overlap", "--no-do_qkv_heatmap", "--no-do_spacing",
            "--no-do_ipr", "--no-do_powerlaw", "--do_perplexity",
            "--decile_scope", "analyzed",
            "--n_deciles", str(args.deciles),
            "--perplexity_tokens", str(args.perplexity_tokens),
            "--ppl_stride", str(args.perplexity_stride),
            "--no-do_complex_spacing", "--no-do_porter_thomas",
        ])
    else:
        raise ValueError(f"unknown task stage: {stage}")
    return command


def _run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    global _ACTIVE_PROCESS
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        _ACTIVE_PROCESS = process
        try:
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return process.wait()
        finally:
            _ACTIVE_PROCESS = None


def _run_task(project_root: Path, results_root: Path, model_dir: Path, model: dict,
              stage: str, layers: list[int], args: argparse.Namespace,
              text_path: Path, snapshots: SnapshotManager) -> bool:
    task_root = results_root / model["id"] / "tasks" / _task_name(stage, layers)
    contract = _task_contract(model, stage, layers, args)
    if _task_is_complete(task_root, contract):
        print(f"skipping completed task {model['id']} {stage} {layers}", flush=True)
        return True
    attempt = _next_attempt(task_root)
    attempt.mkdir(parents=True)
    command = _build_command(model_dir, attempt, layers, stage, model, args, text_path)
    _atomic_json(attempt / "launch.json", {
        "model": model,
        "stage": stage,
        "layers": layers,
        "command": command,
        "started_at_utc": _utc_now(),
        "thinking_mode": "not_applicable_no_generation",
    })
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(project_root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(args.cpu_threads),
        "MKL_NUM_THREADS": str(args.cpu_threads),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
    })
    if args.gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    if args.temp_root is not None:
        env["TMPDIR"] = str(args.temp_root.resolve())
    return_code = _run_logged(command, attempt / "process.log", env)
    statuses = _status_files(attempt)
    complete = False
    status_values = []
    for status_path in statuses:
        try:
            with status_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            status_values.append({"path": str(status_path), "status": value.get("status")})
            complete = complete or value.get("status") == "complete"
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            status_values.append({"path": str(status_path), "status": "unreadable"})
    record = {
        "model_id": model["id"],
        "stage": stage,
        "layers": layers,
        "contract": contract,
        "return_code": return_code,
        "statuses": status_values,
        "finished_at_utc": _utc_now(),
    }
    _atomic_json(attempt / "finish.json", record)
    if return_code == 0 and complete:
        _atomic_json(_complete_marker(task_root), record)
        snapshots.snapshot(f"completed-{model['id']}-{stage}")
        return True
    snapshots.snapshot(f"failed-{model['id']}-{stage}")
    return False


def _task_plan(model: dict, args: argparse.Namespace) -> list[tuple[str, list[int]]]:
    all_layers = list(range(int(model["layers"])))
    representative = representative_layers(int(model["layers"]), args.representative_layers)
    plan: list[tuple[str, list[int]]] = []
    if args.weight_layer_strategy != "none":
        selected = all_layers if args.weight_layer_strategy == "all" else representative
        plan.extend(("weights", batch) for batch in layer_batches(selected, args.layers_per_task))
    if args.decile_layer_strategy != "none":
        selected = all_layers if args.decile_layer_strategy == "all" else representative
        plan.extend(("decile", [layer]) for layer in selected)
    return plan


def _model_complete(results_root: Path, model: dict) -> Path:
    return results_root / model["id"] / "model-complete.json"


def _run_model(project_root: Path, results_root: Path, models_root: Path, model: dict,
               args: argparse.Namespace, text_path: Path,
               snapshots: SnapshotManager) -> bool:
    completion = _model_complete(results_root, model)
    model_dir = models_root / model["id"]
    remove_after = False
    plan = _task_plan(model, args)
    if not plan:
        raise ValueError("the sweep plan contains no tasks")
    if all(_task_is_complete(
        results_root / model["id"] / "tasks" / _task_name(stage, layers),
        _task_contract(model, stage, layers, args),
    ) for stage, layers in plan):
        print(f"skipping completed model {model['id']}", flush=True)
        if not completion.is_file():
            _atomic_json(completion, {
                "model": model,
                "completed_at_utc": _utc_now(),
                "task_count": len(plan),
                "thinking_mode": "not_applicable_no_generation",
            })
        if not args.keep_models and model_dir.exists():
            shutil.rmtree(model_dir)
        return True
    try:
        _download_model(model, model_dir, args.reserve_free_gb)
        for stage, layers in plan:
            if not _run_task(
                project_root, results_root, model_dir, model, stage, layers,
                args, text_path, snapshots,
            ):
                raise RuntimeError(f"task failed: {model['id']} {stage} {layers}")
        _atomic_json(completion, {
            "model": model,
            "completed_at_utc": _utc_now(),
            "task_count": len(plan),
            "task_contracts": [
                _task_contract(model, stage, layers, args)
                for stage, layers in plan
            ],
            "thinking_mode": "not_applicable_no_generation",
        })
        (results_root / model["id"] / "last-failure.json").unlink(missing_ok=True)
        snapshots.snapshot(f"completed-model-{model['id']}")
        remove_after = True
        return True
    except Exception as error:  # noqa: BLE001 - record one model and continue
        _atomic_json(results_root / model["id"] / "last-failure.json", {
            "model": model,
            "failed_at_utc": _utc_now(),
            "error": repr(error),
        })
        snapshots.snapshot(f"failed-model-{model['id']}")
        print(f"model failed: {model['id']}: {error}", file=sys.stderr, flush=True)
        remove_after = True
        return False
    finally:
        if remove_after and not args.keep_models and model_dir.exists():
            print(f"deleting staged model {model_dir}", flush=True)
            shutil.rmtree(model_dir)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=project_root / CATALOG_NAME)
    parser.add_argument("--results-root", type=Path,
                        default=Path(os.environ.get("RMT_RESULTS_ROOT", project_root / "RMT_Frontier_Results")))
    parser.add_argument("--snapshot-root", type=Path,
                        default=Path(os.environ.get("RMT_SNAPSHOT_ROOT", project_root / "RMT_Result_Snapshots")))
    parser.add_argument("--models-root", type=Path,
                        default=Path(os.environ.get("RMT_MODELS_ROOT", project_root / "models" / "frontier")))
    parser.add_argument("--text-path", type=Path,
                        default=project_root / "wikitext-2-raw" / "wiki.test.raw")
    parser.add_argument("--temp-root", type=Path,
                        default=(Path(os.environ["RMT_TMPDIR"])
                                 if os.environ.get("RMT_TMPDIR") else None))
    parser.add_argument("--models", nargs="*", default=None,
                        help="catalog model ids; default selects enabled public models")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--weight-layer-strategy", choices=("all", "representative", "none"),
                        default="all")
    parser.add_argument("--decile-layer-strategy", choices=("all", "representative", "none"),
                        default="representative")
    parser.add_argument("--representative-layers", type=int, default=3)
    parser.add_argument("--layers-per-task", type=int, choices=(1, 2), default=2)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--perplexity-tokens", type=int, default=2048)
    parser.add_argument("--perplexity-stride", type=int, default=512)
    parser.add_argument("--activation-batches", type=int, default=2)
    parser.add_argument("--activation-length", type=int, default=1024)
    parser.add_argument("--activation-stride", type=int, default=512)
    parser.add_argument("--gpu-svd-min-dim", type=int, default=1024)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--snapshot-interval", type=int, default=3600)
    parser.add_argument("--snapshot-retention", type=int, default=24)
    parser.add_argument("--reserve-free-gb", type=float, default=8.0)
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _handle_signal(signum, _frame) -> None:
    process = _ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    raise KeyboardInterrupt(f"received signal {signum}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    catalog = load_catalog(args.catalog.resolve())
    selected = _select_models(
        catalog, args.models, args.include_optional, args.include_gated
    )
    if args.list_models or args.dry_run:
        for model in selected:
            layers = _task_plan(model, args)
            print(
                f"{model['id']}: {model['repo_id']} type={model['model_type']} "
                f"dtype={model['dtype']} download={model['download_gb']}GB tasks={len(layers)}"
            )
        if args.list_models or args.dry_run:
            return 0
    if not selected:
        raise SystemExit("no models were selected")
    if args.include_gated and not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is required for gated models")
    for value, name in (
        (args.representative_layers, "representative-layers"),
        (args.deciles, "deciles"),
        (args.perplexity_tokens, "perplexity-tokens"),
        (args.snapshot_interval, "snapshot-interval"),
    ):
        if int(value) < 1:
            raise SystemExit(f"{name} must be positive")
    results_root = args.results_root.resolve()
    if args.temp_root is not None:
        args.temp_root = args.temp_root.resolve()
        args.temp_root.mkdir(parents=True, exist_ok=True)
    snapshots = SnapshotManager(
        results_root, args.snapshot_root.resolve(),
        interval_seconds=args.snapshot_interval,
        keep=args.snapshot_retention,
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)
    results_root.mkdir(parents=True, exist_ok=True)
    snapshots.start()
    failures = []
    try:
        _stage_wikitext(args.text_path.resolve())
        for model in selected:
            if not _run_model(
                project_root, results_root, args.models_root.resolve(), model,
                args, args.text_path.resolve(), snapshots,
            ):
                failures.append(model["id"])
    finally:
        snapshots.stop(final=True)
    if failures:
        print(f"failed models: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
