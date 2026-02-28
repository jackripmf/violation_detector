"""Инференс-слой: загрузка моделей и единая точка вызова предикта."""

from .inference_hub import InferenceHub
from .model_path_resolver import get_default_model_root, resolve_model_path

__all__ = ["InferenceHub", "get_default_model_root", "resolve_model_path"]
