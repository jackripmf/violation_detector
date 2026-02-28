"""Файл: tests/test_universal_processor_saved_stats.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.stats_manager: используется для передачи данных или вызова связанной логики.
- src.processing.universal_processor: используется для передачи данных или вызова связанной логики."""

import numpy as np

from src.processing.stats_manager import StatsManager
from src.processing.universal_processor import UniversalProcessor


class _DetectionManagerStub:
    def __init__(self):
        self.dms_detector = None

    def detect_obstruction(self, frame, frame_count):
        return {"detected": True, "detectors_count": 1, "yolo_objects": []}

    def detect_movement(self, frame, is_obstructed, frame_count=0):
        return {"movement_detected": True, "filtered": True, "movement_duration": 2.0}

    def detect_forbidden(self, frame, frame_count):
        return {
            "current_violation": True,
            "objects": [{"class": "phone"}],
            "violation_info": {"objects": [{"class": "phone"}], "affected_classes": ["phone"]},
        }

    def detect_dms(self, frame, frame_count):
        return {"violations": [{"type": "eye_closed"}], "objects": []}


class _ViolationManagerStub:
    def __init__(self, succeed=True):
        self.succeed = succeed
        self.obstruction_start_time = None

    def process_obstruction(self, *args, **kwargs):
        return object() if self.succeed else None

    def process_movement(self, *args, **kwargs):
        return object() if self.succeed else None

    def process_forbidden(self, *args, **kwargs):
        return object() if self.succeed else None

    def process_dms(self, *args, **kwargs):
        return object() if self.succeed else None


class _VisualizerStub:
    def __init__(self):
        self.last_stats = None

    def draw(self, frame, result, movement_info, video_timestamp, total_duration, fps):
        self.last_stats = result.get("stats", {})
        return frame


def _build_processor(succeed=True):
    processor = UniversalProcessor.__new__(UniversalProcessor)
    processor.frame_count = 0
    processor.detection_manager = _DetectionManagerStub()
    processor.violation_manager = _ViolationManagerStub(succeed=succeed)
    processor.stats_manager = StatsManager()
    processor.stats_manager.start_session()
    processor.visualizer = _VisualizerStub()
    return processor


def test_process_frame_increments_saved_stats_on_success():
    processor = _build_processor(succeed=True)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    result, _, _ = processor.process_frame(frame, video_timestamp=1.0, processing_time=2.0)

    stats = processor.stats_manager.get_current_stats()
    assert result.detected is True
    assert stats["saved_violations"] == 4
    assert stats["obstruction_violations"] == 1
    assert stats["movement_violations_saved"] == 1
    assert stats["forbidden_items_saved"] == 1
    assert stats["dms_violations_saved"] == 1


def test_process_frame_does_not_increment_saved_stats_on_failed_save():
    processor = _build_processor(succeed=False)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    processor.process_frame(frame, video_timestamp=1.0, processing_time=2.0)
    stats = processor.stats_manager.get_current_stats()

    assert stats["saved_violations"] == 0
    assert stats["obstruction_violations"] == 0
    assert stats["movement_violations_saved"] == 0
    assert stats["forbidden_items_saved"] == 0
    assert stats["dms_violations_saved"] == 0


def test_alerts_count_matches_saved_violations():
    processor = _build_processor(succeed=True)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    result, total_duration, _ = processor.process_frame(frame, video_timestamp=1.0, processing_time=2.0)
    processor._visualize_frame(frame, result, video_timestamp=1.0, total_duration=total_duration)

    assert processor.visualizer.last_stats["alerts_count"] == processor.stats_manager.stats.saved_violations
