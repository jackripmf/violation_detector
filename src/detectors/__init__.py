"""Файл: src/detectors/__init__.py
Тип: детектор нарушений.
Назначение: получает кадр, применяет алгоритм детекции и возвращает структурированный результат.
Связи: детекторы вызываются менеджером детекции и передают данные в слой сохранения/визуализации.
Импортируемые внутренние модули:
- .base_detector: используется для передачи данных или вызова связанной логики.
- .cv_detector: используется для передачи данных или вызова связанной логики.
- .dark_area_detector: используется для передачи данных или вызова связанной логики.
- .dms_detector: используется для передачи данных или вызова связанной логики.
- .forbidden_items_detector: используется для передачи данных или вызова связанной логики.
- .movement_detector: используется для передачи данных или вызова связанной логики."""
from .base_detector import BaseDetector
from .movement_detector import CameraMovementDetector
from .cv_detector import CVDetector
from .dark_area_detector import DarkAreaDetector
from .yolo_detector import YOLODetector
from .forbidden_items_detector import ForbiddenItemsDetector
from .dms_detector import DMSDetector

__all__ = [
    'BaseDetector',
    'CameraMovementDetector',
    'CVDetector',
    'DarkAreaDetector',
    'YOLODetector',
    'ForbiddenItemsDetector',
    'DMSDetector',
]