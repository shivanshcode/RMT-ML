"""Local token datasets and deterministic synthetic language data."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class CharTokenizer:
    """Minimal reproducible character tokenizer for local-corpus experiments."""

    special_tokens = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, alphabet: Sequence[str]) -> None:
        symbols = sorted(set(str(character) for character in alphabet))
        if any(len(character) != 1 for character in symbols):
            raise ValueError("alphabet entries must be individual characters")
        self.id_to_token = list(self.special_tokens) + symbols
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("text must be nonempty")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return 0

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(character, 3) for character in text]
        if add_bos:
            ids.insert(0, 1)
        if add_eos:
            ids.append(2)
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        output: list[str] = []
        for token_id in token_ids:
            index = int(token_id)
            token = self.id_to_token[index] if 0 <= index < len(self.id_to_token) else "<unk>"
            if skip_special_tokens and token in self.special_tokens:
                continue
            output.append(token)
        return "".join(output)


class TokenSequenceDataset(Dataset[dict[str, Tensor]]):
    """Create overlapping fixed-length causal-LM windows from a token stream."""

    def __init__(
        self,
        tokens: Sequence[int] | np.ndarray | Tensor,
        sequence_length: int,
        *,
        stride: int | None = None,
    ) -> None:
        source = tokens.detach().cpu().numpy() if isinstance(tokens, Tensor) else np.asarray(tokens)
        if source.ndim != 1 or not np.issubdtype(source.dtype, np.integer):
            raise ValueError("tokens must be a one-dimensional integer sequence")
        tensor = torch.as_tensor(source.astype(np.int64, copy=False), dtype=torch.long).clone()
        sequence_length = int(sequence_length)
        stride = sequence_length if stride is None else int(stride)
        if sequence_length < 2 or stride < 1:
            raise ValueError("sequence_length must exceed one and stride must be positive")
        if tensor.numel() < sequence_length:
            raise ValueError("token stream is shorter than one sequence")
        if torch.any(tensor < 0):
            raise ValueError("token ids must be nonnegative")
        self.tokens = tensor
        self.sequence_length = sequence_length
        self.stride = stride
        self.starts = list(range(0, tensor.numel() - sequence_length + 1, stride))
        if self.starts[-1] != tensor.numel() - sequence_length:
            self.starts.append(tensor.numel() - sequence_length)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        start = self.starts[int(index)]
        segment = self.tokens[start : start + self.sequence_length].clone()
        return {
            "input_ids": segment,
            "labels": segment.clone(),
            "attention_mask": torch.ones_like(segment),
        }


def load_token_array(path: str | Path) -> np.ndarray:
    """Load a local one-dimensional integer token array from NPY or NPZ."""

    source = Path(path)
    if source.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("token arrays must use .npy or .npz")
    loaded = np.load(source, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "tokens" not in loaded.files:
                raise ValueError("NPZ token archives must contain an explicit 'tokens' array")
            array = np.asarray(loaded["tokens"])
        finally:
            loaded.close()
    else:
        array = np.asarray(loaded)
    if array.ndim != 1:
        raise ValueError("token array must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("token array must use an integer dtype")
    if array.size < 2 or np.any(array < 0):
        raise ValueError("token array must contain at least two nonnegative ids")
    array = array.astype(np.int64, copy=False)
    return array


def split_tokens(
    tokens: Sequence[int] | np.ndarray,
    *,
    train_fraction: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one token stream contiguously to avoid train/validation leakage."""

    source = np.asarray(tokens)
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.integer):
        raise ValueError("tokens must be a one-dimensional integer array")
    array = source.astype(np.int64, copy=False)
    fraction = float(train_fraction)
    if array.size < 4 or not 0.0 < fraction < 1.0:
        raise ValueError("tokens and train_fraction do not support a nonempty split")
    boundary = min(array.size - 2, max(2, int(round(fraction * array.size))))
    return array[:boundary].copy(), array[boundary:].copy()


def build_synthetic_corpus(
    length: int,
    vocab_size: int,
    *,
    seed: int = 0,
    noise_probability: float = 0.05,
) -> np.ndarray:
    """Generate a learnable periodic/Markov token stream for smoke experiments."""

    length, vocab_size = int(length), int(vocab_size)
    noise_probability = float(noise_probability)
    if length < 2 or vocab_size < 8 or not 0.0 <= noise_probability <= 1.0:
        raise ValueError("synthetic-corpus configuration is invalid")
    generator = np.random.default_rng(seed)
    period = max(4, min(vocab_size - 1, int(round(np.sqrt(vocab_size) * 2))))
    positions = np.arange(length, dtype=np.int64)
    base = 1 + ((positions + (positions // period) ** 2) % (vocab_size - 1))
    noisy = generator.random(length) < noise_probability
    base[noisy] = generator.integers(1, vocab_size, size=int(np.count_nonzero(noisy)))
    return base


__all__ = [
    "CharTokenizer",
    "TokenSequenceDataset",
    "build_synthetic_corpus",
    "load_token_array",
    "split_tokens",
]

