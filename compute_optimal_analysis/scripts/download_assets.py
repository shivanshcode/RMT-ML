"""Prefetch and serialize all assets required by air-gapped experiment jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
TOKENIZER_ID = "openai-community/gpt2"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _configure_cache(root: Path) -> dict[str, Path]:
    cache = root / "cache"
    paths = {
        "hf_home": cache / "huggingface",
        "hf_datasets": cache / "huggingface" / "datasets",
        "hf_hub": cache / "huggingface" / "hub",
        "torch_home": cache / "torch",
        "raw": root / "data" / "raw",
        "tokenized": root / "data" / "tokenized",
        "tokenizers": root / "data" / "tokenizers",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(paths["hf_home"].resolve())
    os.environ["HF_DATASETS_CACHE"] = str(paths["hf_datasets"].resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(paths["hf_hub"].resolve())
    os.environ["TORCH_HOME"] = str(paths["torch_home"].resolve())
    return paths


def _require_connected_mode(allow_network: bool) -> None:
    if not allow_network:
        raise RuntimeError(
            "remote asset acquisition requires --allow-network on a connected staging node"
        )


def _resolve_hub_revisions(allow_network: bool) -> dict[str, str]:
    """Resolve moving repository names to immutable source commit identifiers."""

    _require_connected_mode(allow_network)
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(variable, None)
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError("install the pinned staging requirements before prefetching") from error
    api = HfApi()
    dataset_revision = str(api.dataset_info(DATASET_ID).sha)
    tokenizer_revision = str(api.model_info(TOKENIZER_ID).sha)
    if not dataset_revision or not tokenizer_revision:
        raise RuntimeError("the asset service did not return immutable source revisions")
    return {
        "dataset_revision": dataset_revision,
        "tokenizer_revision": tokenizer_revision,
    }


def _download_tokenizer(
    paths: dict[str, Path],
    allow_network: bool,
    *,
    revision: str,
) -> Path:
    _require_connected_mode(allow_network)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install the pinned staging requirements before prefetching") from error
    destination = paths["tokenizers"] / "gpt2"
    snapshot_download(
        repo_id=TOKENIZER_ID,
        repo_type="model",
        revision=revision,
        local_dir=destination,
        local_dir_use_symlinks=False,
        allow_patterns=(
            "config.json",
            "added_tokens.json",
            "LICENSE*",
            "README.md",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "special_tokens_map.json",
        ),
    )
    return destination


def _write_raw_splits(dataset: Any, destination: Path) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name, split in dataset.items():
        output = destination / f"{split_name}.jsonl"
        count = 0
        with output.open("w", encoding="utf-8") as handle:
            for record in split:
                handle.write(json.dumps({"text": str(record["text"])}, ensure_ascii=False) + "\n")
                count += 1
        counts[str(split_name)] = count
    return counts


def _token_batches(
    texts: Iterable[str],
    tokenizer: Any,
    *,
    batch_size: int,
) -> Iterable[np.ndarray]:
    pending: list[str] = []
    eos = int(tokenizer.eos_token_id)
    for text in texts:
        pending.append(str(text))
        if len(pending) < batch_size:
            continue
        encoded = tokenizer(pending, add_special_tokens=False)["input_ids"]
        for tokens in encoded:
            yield np.asarray([*tokens, eos], dtype=np.int32)
        pending.clear()
    if pending:
        encoded = tokenizer(pending, add_special_tokens=False)["input_ids"]
        for tokens in encoded:
            yield np.asarray([*tokens, eos], dtype=np.int32)


def _serialize_token_stream(
    batches: Iterable[np.ndarray],
    output: Path,
    *,
    max_tokens: int | None,
) -> int:
    temporary = output.with_suffix(".tokens.bin")
    count = 0
    with temporary.open("wb") as handle:
        for tokens in batches:
            selected = tokens
            if max_tokens is not None:
                remaining = max_tokens - count
                if remaining <= 0:
                    break
                selected = tokens[:remaining]
            selected.tofile(handle)
            count += int(selected.size)
    if count < 2:
        temporary.unlink(missing_ok=True)
        raise ValueError("tokenization produced fewer than two tokens")
    source = np.memmap(temporary, mode="r", dtype=np.int32, shape=(count,))
    destination = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.int32,
        shape=(count,),
    )
    chunk = 4_000_000
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        destination[start:stop] = source[start:stop]
    destination.flush()
    del destination
    del source
    temporary.unlink()
    return count


def _download_wikitext(
    paths: dict[str, Path],
    tokenizer_path: Path,
    *,
    allow_network: bool,
    dataset_revision: str,
    tokenizer_revision: str,
    max_tokens: int | None,
) -> dict[str, Any]:
    _require_connected_mode(allow_network)
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install the pinned staging requirements before prefetching") from error
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        revision=dataset_revision,
        cache_dir=str(paths["hf_datasets"]),
    )
    raw_counts = _write_raw_splits(
        dataset,
        paths["raw"] / DATASET_CONFIG,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    output = paths["tokenized"] / "wikitext-103-raw-v1_gpt2.npy"
    token_count = _serialize_token_stream(
        _token_batches(dataset["train"]["text"], tokenizer, batch_size=256),
        output,
        max_tokens=max_tokens,
    )
    return {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": dataset_revision,
        "raw_records": raw_counts,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": tokenizer_revision,
        "tokenized_path": str(output),
        "token_count": token_count,
    }


def _generate_synthetic_assets(
    paths: dict[str, Path],
    *,
    token_count: int,
    vocab_size: int,
    seed: int,
) -> dict[str, Any]:
    if token_count < 2 or vocab_size < 8:
        raise ValueError("synthetic token count and vocabulary size are too small")
    generator = np.random.default_rng(seed)
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probabilities = 1.0 / np.power(ranks, 1.07)
    probabilities /= np.sum(probabilities)
    tokens = generator.choice(vocab_size, size=token_count, p=probabilities).astype(np.int32)
    output = paths["tokenized"] / "synthetic_zipf.npy"
    np.save(output, tokens, allow_pickle=False)
    tokenizer_directory = paths["tokenizers"] / "synthetic"
    tokenizer_directory.mkdir(parents=True, exist_ok=True)
    vocabulary = {f"token_{index}": index for index in range(vocab_size)}
    _write_json(
        tokenizer_directory / "vocabulary.json",
        {
            "type": "integer_identity",
            "vocab_size": vocab_size,
            "tokens": vocabulary,
        },
    )
    return {
        "tokenized_path": str(output),
        "token_count": token_count,
        "vocab_size": vocab_size,
        "seed": seed,
    }


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    manifest_path = (root / "data" / "asset_manifest.json").resolve()
    for base in (root / "data",):
        if not base.exists():
            continue
        for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
            if path.resolve() == manifest_path:
                continue
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return files


def _verify_manifest(root: Path) -> int:
    """Verify every staged file without consulting a remote service."""

    manifest_path = root / "data" / "asset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"asset manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("asset manifest contains no file records")
    verified = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("asset manifest contains a malformed file record")
        relative = Path(str(record.get("path", "")))
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            raise ValueError("asset manifest contains a path outside the project root")
        if not target.is_file():
            raise FileNotFoundError(f"staged asset is missing: {relative}")
        if target.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"staged asset size mismatch: {relative}")
        if _sha256(target) != str(record.get("sha256", "")):
            raise ValueError(f"staged asset checksum mismatch: {relative}")
        verified += 1
    return verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--assets",
        choices=("all", "wikitext103", "synthetic"),
        default="all",
    )
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-wikitext-tokens", type=int, default=0)
    parser.add_argument("--synthetic-tokens", type=int, default=1_000_000)
    parser.add_argument("--synthetic-vocab-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_wikitext_tokens < 0:
        raise ValueError("max-wikitext-tokens must be nonnegative")
    root = args.root.resolve()
    if args.verify_only:
        verified = _verify_manifest(root)
        print(f"verified {verified} staged asset files")
        return 0
    if args.assets in {"all", "wikitext103"}:
        _require_connected_mode(args.allow_network)
    paths = _configure_cache(root)
    assets: dict[str, Any] = {}
    revisions: dict[str, str] = {}
    if args.assets in {"all", "synthetic"}:
        assets["synthetic"] = _generate_synthetic_assets(
            paths,
            token_count=args.synthetic_tokens,
            vocab_size=args.synthetic_vocab_size,
            seed=args.seed,
        )
    if args.assets in {"all", "wikitext103"}:
        revisions = _resolve_hub_revisions(args.allow_network)
        tokenizer_path = _download_tokenizer(
            paths,
            args.allow_network,
            revision=revisions["tokenizer_revision"],
        )
        assets["wikitext103"] = _download_wikitext(
            paths,
            tokenizer_path,
            allow_network=args.allow_network,
            dataset_revision=revisions["dataset_revision"],
            tokenizer_revision=revisions["tokenizer_revision"],
            max_tokens=(
                None if args.max_wikitext_tokens == 0 else args.max_wikitext_tokens
            ),
        )
    manifest = {
        "schema_version": 2,
        "assets": assets,
        "source_revisions": revisions,
        "files": _file_manifest(root),
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": "./cache/huggingface",
            "TORCH_HOME": "./cache/torch",
        },
    }
    _write_json(root / "data" / "asset_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
