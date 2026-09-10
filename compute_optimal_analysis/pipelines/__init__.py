"""Data, training, activation, and lesion pipelines."""

from .activation_extractor import (
    ActivationExtractor, CovarianceAccumulator, CovarianceEstimate, compute_tensor_svd,
)
from .cli_config import (
    SpectralCLIConfig,
    add_pipeline_cli_arguments,
    add_rmt_cli_arguments,
    rmt_config_from_namespace,
)
from .dataset import CharTokenizer, TokenSequenceDataset
from .trainer import LanguageModelTrainer, TrainConfig, evaluate_language_model

__all__ = [
    "ActivationExtractor",
    "CharTokenizer",
    "CovarianceAccumulator",
    "CovarianceEstimate",
    "LanguageModelTrainer",
    "TokenSequenceDataset",
    "TrainConfig",
    "SpectralCLIConfig",
    "add_pipeline_cli_arguments",
    "add_rmt_cli_arguments",
    "compute_tensor_svd",
    "evaluate_language_model",
    "rmt_config_from_namespace",
]
