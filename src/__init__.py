"""Публичные реэкспорты основных компонентов проекта."""

from .detectors import (
    BaseDetector,
    CVDetector,
    DarkAreaDetector,
    CameraMovementDetector,
    YOLODetector,
    ForbiddenItemsDetector,
    DMSDetector,
)
from .processing.base_processor import ProcessorConfig, ViolationResult
from .processing.universal_processor import UniversalProcessor
from .runtime.controller import RuntimeConfig, RuntimeController
from .inference import InferenceHub
from .utils import (
    FileManager,
    Visualizer,
)

__version__ = "2.0.0"
__all__ = [
    # Detectors
    "BaseDetector",
    "CVDetector",
    "DarkAreaDetector",
    "CameraMovementDetector",
    "YOLODetector",
    "ForbiddenItemsDetector",
    "DMSDetector",
    
    # Processors
    "UniversalProcessor",
    "ProcessorConfig",
    "ViolationResult",
    "RuntimeController",
    "RuntimeConfig",
    
    # Inference
    "InferenceHub",
    
    # Utils
    "FileManager",
    "Visualizer",
]
