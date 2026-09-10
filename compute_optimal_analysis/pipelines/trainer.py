"""AdamW causal-LM trainer with warmup/cosine decay and reproducible metrics."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Sized
import math
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype: str = "bfloat16"
    compile_model: bool = False
    compile_mode: str = "default"
    allow_tf32: bool = True
    non_blocking_transfers: bool = True
    log_every: int = 50
    validation_max_batches: int | None = None
    max_train_tokens: int | None = None
    max_consecutive_skipped_updates: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        finite = (self.learning_rate, self.min_learning_rate_ratio, self.weight_decay,
                  self.beta1, self.beta2, self.gradient_clip)
        if any(not math.isfinite(float(value)) for value in finite):
            raise ValueError("training numeric settings must be finite")
        if self.epochs < 1 or (self.max_steps is not None and self.max_steps < 1):
            raise ValueError("epochs and max_steps must be positive")
        if self.learning_rate <= 0.0 or not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("learning-rate settings are invalid")
        if self.warmup_steps < 0 or self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("warmup, weight decay, and gradient clip are invalid")
        if not 0.0 < self.beta1 < 1.0 or not 0.0 < self.beta2 < 1.0:
            raise ValueError("Adam beta values must lie in (0, 1)")
        if self.amp_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float32, float16, or bfloat16")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("unsupported compile_mode")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.validation_max_batches is not None and self.validation_max_batches < 1:
            raise ValueError("validation_max_batches must be positive")
        if self.max_train_tokens is not None and self.max_train_tokens < 1:
            raise ValueError("max_train_tokens must be positive")
        if self.max_consecutive_skipped_updates < 1:
            raise ValueError("max_consecutive_skipped_updates must be positive")


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_warmup_multiplier(
    step: int | float,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    """Return a continuous linear-warmup, cosine-decay LR multiplier."""

    if total_steps < 1 or warmup_steps < 0 or not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("scheduler arguments are invalid")
    step = max(0.0, float(step))
    if warmup_steps > 0 and step < warmup_steps:
        return max(np.finfo(float).eps, (step + 1) / warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _loss_from_output(output: object) -> Tensor:
    loss = getattr(output, "loss", None)
    if isinstance(loss, Tensor):
        return loss
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor) and output[0].ndim == 0:
        return output[0]
    raise ValueError("model output does not contain a scalar loss")


def evaluate_language_model(
    model: nn.Module,
    dataloader: Iterable[dict[str, Tensor]],
    device: str | torch.device,
    *,
    max_batches: int | None = None,
    amp_dtype: str = "float32",
    non_blocking_transfers: bool = True,
) -> dict[str, float]:
    """Return token-weighted validation loss and perplexity."""

    target = torch.device(device)
    if amp_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float32, float16, or bfloat16")
    selected_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[amp_dtype]
    use_amp = target.type == "cuda" and selected_dtype != torch.float32
    module_modes = {module: bool(module.training) for module in model.modules()}
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                moved = {
                    key: value.to(
                        target,
                        non_blocking=bool(non_blocking_transfers and target.type == "cuda"),
                    )
                    for key, value in batch.items()
                }
                context = (
                    torch.autocast(device_type="cuda", dtype=selected_dtype)
                    if use_amp
                    else nullcontext()
                )
                with context:
                    output = model(**moved)
                loss = _loss_from_output(output)
                labels = moved.get("labels")
                if labels is None:
                    raise ValueError("evaluation batches must contain labels")
                valid = labels[:, 1:] != -100
                if "attention_mask" in moved:
                    mask = moved["attention_mask"].to(dtype=torch.bool)
                    valid = valid & mask[:, 1:] & mask[:, :-1]
                count = int(torch.count_nonzero(valid).item())
                if count > 0:
                    loss_value = float(loss.detach().cpu())
                    if not math.isfinite(loss_value):
                        raise FloatingPointError("evaluation produced a non-finite loss")
                    total_loss += loss_value * count
                    total_tokens += count
    finally:
        for module, training in module_modes.items():
            module.training = training
    if total_tokens == 0:
        raise ValueError("evaluation scored no valid next-token targets")
    mean_loss = total_loss / total_tokens
    if not math.isfinite(mean_loss):
        raise FloatingPointError("evaluation mean loss is non-finite")
    try:
        perplexity = math.exp(mean_loss)
        perplexity_status = "finite"
    except OverflowError:
        perplexity = float("inf")
        perplexity_status = "overflow"
    return {
        "loss": float(mean_loss),
        "log_perplexity": float(mean_loss),
        "perplexity": float(perplexity),
        "perplexity_status": perplexity_status,
        "tokens": float(total_tokens),
    }


class LanguageModelTrainer:
    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        seed_everything(config.seed)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
            torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")
        self.model.to(self.device)
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.use_amp = self.device.type == "cuda" and config.amp_dtype != "float32"
        self.amp_torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.amp_dtype]
        scaler_enabled = self.use_amp and self.amp_torch_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except AttributeError:  # compatibility with older supported torch builds
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.execution_model: nn.Module = self.model
        if config.compile_model:
            if self.device.type != "cuda":
                raise ValueError("compile_model is supported only for CUDA training in this pipeline")
            self.execution_model = torch.compile(self.model, mode=config.compile_mode)
        self.global_step = 0
        self.attempted_steps = 0
        self.skipped_steps = 0
        self.processed_train_tokens = 0
        self.attempted_target_tokens = 0
        self.forwarded_input_positions = 0
        self.history: list[dict[str, float | int]] = []
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_torch_dtype)

    def fit(
        self,
        train_dataloader: Iterable[dict[str, Tensor]],
        validation_dataloader: Iterable[dict[str, Tensor]] | None = None,
        *,
        callback: Callable[[dict[str, float | int]], None] | None = None,
    ) -> list[dict[str, float | int]]:
        """Train until the epoch or max-step boundary and return step metrics."""

        if not isinstance(train_dataloader, Sized):
            raise ValueError("train_dataloader must expose its finite length")
        available_steps = len(train_dataloader) * self.config.epochs
        token_budget = self.config.max_train_tokens
        if token_budget is None:
            total_steps = min(available_steps, self.config.max_steps or available_steps)
        else:
            # Token-budget runs recycle the finite loader, so one epoch cannot
            # cap their LR horizon.  The caller's successful-update estimate is
            # authoritative when supplied; skipped attempts never advance it.
            total_steps = (
                int(self.config.max_steps)
                if self.config.max_steps is not None
                else max(available_steps, 1)
            )
        if total_steps < 1:
            raise ValueError("train_dataloader contains no batches")
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: cosine_warmup_multiplier(
                (
                    self.processed_train_tokens / token_budget * total_steps
                    if token_budget is not None else step
                ),
                total_steps,
                self.config.warmup_steps,
                self.config.min_learning_rate_ratio,
            ),
        )
        self.model.train()
        self.execution_model.train()
        stop = False
        epoch = 0
        consecutive_skips = 0
        while True:
            made_progress = False
            for batch in train_dataloader:
                if token_budget is None and self.global_step >= total_steps:
                    stop = True
                    break
                if token_budget is not None and self.processed_train_tokens >= token_budget:
                    stop = True
                    break
                moved = {
                    key: value.to(
                        self.device,
                        non_blocking=bool(
                            self.config.non_blocking_transfers and self.device.type == "cuda"
                        ),
                    )
                    for key, value in batch.items()
                }
                labels = moved.get("labels")
                if labels is None:
                    raise ValueError("training batches must contain labels")
                valid = labels[:, 1:] != -100
                if "attention_mask" in moved:
                    mask = moved["attention_mask"].to(torch.bool)
                    valid = valid & mask[:, 1:] & mask[:, :-1]
                original_batch_tokens = int(torch.count_nonzero(valid).item())
                batch_tokens = original_batch_tokens
                if self.config.max_train_tokens is not None:
                    remaining = self.config.max_train_tokens - self.processed_train_tokens
                    if remaining <= 0:
                        stop = True
                        break
                    if batch_tokens > remaining:
                        labels = labels.clone()
                        valid_positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
                        drop = valid_positions[remaining:]
                        width = labels.shape[1] - 1
                        labels[drop // width, (drop % width) + 1] = -100
                        moved["labels"] = labels
                        # Trim rows/context after masking so a tiny final token
                        # remainder does not execute a full physical batch.
                        kept = torch.nonzero(labels[:, 1:] != -100, as_tuple=False)
                        if kept.numel():
                            row_stop = int(torch.max(kept[:, 0]).item()) + 1
                            col_stop = int(torch.max(kept[:, 1]).item()) + 2
                            for key, value in list(moved.items()):
                                if value.ndim >= 2 and value.shape[:2] == labels.shape:
                                    moved[key] = value[:row_stop, :col_stop]
                            labels = moved["labels"]
                        batch_tokens = remaining
                if batch_tokens < 1:
                    continue
                # Attempted work is the work actually presented to a forward,
                # after final-budget masking—not merely fetched/eligible work.
                self.attempted_target_tokens += batch_tokens
                input_ids = moved.get("input_ids")
                if input_ids is not None:
                    self.forwarded_input_positions += int(input_ids.numel())
                self.attempted_steps += 1
                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.execution_model(**moved)
                    loss = _loss_from_output(output)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {self.global_step}")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                parameters_with_grad = [
                    parameter for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                if parameters_with_grad:
                    # FP32 sum-of-squares can overflow even when every FP32
                    # gradient entry is finite.  FP64 safely spans that range.
                    component_norms = torch.stack([
                        torch.linalg.vector_norm(parameter.grad.detach().double())
                        for parameter in parameters_with_grad
                    ])
                    raw_gradient_norm = torch.linalg.vector_norm(component_norms)
                else:
                    raw_gradient_norm = torch.zeros((), device=self.device)
                gradients_finite = bool(torch.isfinite(raw_gradient_norm).item())
                scaler_enabled = bool(self.scaler.is_enabled())
                if gradients_finite:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                elif scaler_enabled:
                    # GradScaler recorded the overflow during unscale_.  Do not
                    # clip inf gradients (which can turn them into NaNs); let
                    # scaler.step/update skip the optimizer and back off.
                    gradient_norm = raw_gradient_norm
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"non-finite gradient norm at attempted step {self.attempted_steps}")
                applied_learning_rate = float(self.optimizer.param_groups[0]["lr"])
                previous_scale = float(self.scaler.get_scale())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                updated = (not self.scaler.is_enabled()
                           or float(self.scaler.get_scale()) >= previous_scale)
                if updated:
                    self.global_step += 1
                    self.processed_train_tokens += batch_tokens
                    # Token-budget LR decay follows successful target progress;
                    # skipped updates and variable/partial batches do not skew it.
                    self.scheduler.step()
                    made_progress = True
                    consecutive_skips = 0
                else:
                    self.skipped_steps += 1
                    consecutive_skips += 1
                    if consecutive_skips >= self.config.max_consecutive_skipped_updates:
                        raise RuntimeError("training made no progress across the skipped-update limit")
                record: dict[str, float | int] = {
                    "step": self.global_step,
                    "attempted_step": self.attempted_steps,
                    "optimizer_update": int(updated),
                    "skipped_steps": self.skipped_steps,
                    "train_tokens": self.processed_train_tokens,
                    "attempted_target_tokens": self.attempted_target_tokens,
                    "forwarded_input_positions": self.forwarded_input_positions,
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu()),
                    "learning_rate": applied_learning_rate,
                    "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                }
                budget_complete = (
                    token_budget is not None
                    and self.processed_train_tokens >= token_budget
                )
                should_log = (self.global_step % self.config.log_every == 0
                              or self.global_step == total_steps or budget_complete)
                if validation_dataloader is not None and should_log:
                    validation = evaluate_language_model(
                        self.execution_model,
                        validation_dataloader,
                        self.device,
                        max_batches=self.config.validation_max_batches,
                        amp_dtype=self.config.amp_dtype,
                        non_blocking_transfers=self.config.non_blocking_transfers,
                    )
                    record["validation_loss"] = validation["loss"]
                    record["validation_perplexity"] = validation["perplexity"]
                    self.model.train()
                self.history.append(record)
                if callback is not None:
                    callback(dict(record))
            if stop:
                break
            if not made_progress and consecutive_skips == 0:
                raise RuntimeError("one complete data pass produced no optimizer updates")
            # A pass containing only scaler-overflow skips is allowed to recycle;
            # GradScaler may now succeed at its backed-off scale.  The explicit
            # consecutive-skip limit still prevents an infinite loop.
            epoch += 1
            if token_budget is None and epoch >= self.config.epochs:
                break
        if token_budget is not None and self.processed_train_tokens != token_budget:
            raise RuntimeError("training ended before the requested target-token budget")
        return [dict(record) for record in self.history]

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "global_step": self.global_step,
                "attempted_steps": self.attempted_steps,
                "skipped_steps": self.skipped_steps,
                "processed_train_tokens": self.processed_train_tokens,
                "attempted_target_tokens": self.attempted_target_tokens,
                "forwarded_input_positions": self.forwarded_input_positions,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "train_config": asdict(self.config),
                "metadata": dict(metadata or {}),
            },
            temporary,
        )
        temporary.replace(destination)


__all__ = [
    "LanguageModelTrainer",
    "TrainConfig",
    "cosine_warmup_multiplier",
    "evaluate_language_model",
    "seed_everything",
]
