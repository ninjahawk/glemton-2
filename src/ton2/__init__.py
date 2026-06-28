"""ton-2 (Glemton-2): a small from-scratch chat LM."""

from .model import ModelConfig, Ton2, model_from_config

__all__ = ["ModelConfig", "Ton2", "model_from_config"]
