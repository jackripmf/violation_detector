"""Файл: tests/test_violation_manager_fps.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.violation_manager: используется для передачи данных или вызова связанной логики."""

import numpy as np

from src.processing.violation_manager import ViolationManager


class _VisualizerStub:
    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
    }

    def _draw_yolo_object(self, frame, obj):
        return None

    def _draw_forbidden_object(self, frame, obj):
        return None

    def _draw_dms_object(self, frame, obj):
        return None


def _movement_info():
    return {
        "movement_detected": True,
        "filtered": True,
        "movement_duration": 2.4,
        "translation": 0.2,
        "rotation": 0.0,
        "translation_x": 0.2,
        "translation_y": 0.0,
        "reason": "rotation",
    }


def test_movement_video_uses_source_fps(monkeypatch, tmp_path):
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=False,
        movement_pre_event_sec=0.0,
        movement_post_event_sec=0.0,
        movement_turn_delta_threshold=0.0,
        movement_confirm_changes=1,
        movement_min_clip_frames=1,
    )
    manager.set_source_fps(24.0)

    captured = {"fps": None}

    class _Writer:
        def write(self, frame):
            return None

        def release(self):
            return None

    def fake_writer(path, fourcc, fps, size):
        captured["fps"] = fps
        return _Writer()

    monkeypatch.setattr("cv2.VideoWriter", fake_writer)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    violation = manager.process_movement(_movement_info(), frame, video_timestamp=1.0)
    assert violation is not None
    assert captured["fps"] == 24.0


def test_movement_video_falls_back_to_30_when_source_fps_invalid(monkeypatch, tmp_path):
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=False,
        movement_pre_event_sec=0.0,
        movement_post_event_sec=0.0,
        movement_turn_delta_threshold=0.0,
        movement_confirm_changes=1,
        movement_min_clip_frames=1,
    )
    manager.set_source_fps(0)

    captured = {"fps": None}

    class _Writer:
        def write(self, frame):
            return None

        def release(self):
            return None

    def fake_writer(path, fourcc, fps, size):
        captured["fps"] = fps
        return _Writer()

    monkeypatch.setattr("cv2.VideoWriter", fake_writer)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    violation = manager.process_movement(_movement_info(), frame, video_timestamp=1.0)
    assert violation is not None
    assert captured["fps"] == 30.0
