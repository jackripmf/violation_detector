"""Файл: tests/test_violation_manager_movement_segments.py
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


def _movement_info(tx: float) -> dict:
    return {
        "movement_detected": True,
        "filtered": True,
        "movement_duration": 3.0,
        "translation": abs(float(tx)),
        "rotation": 0.0,
        "translation_x": float(tx),
        "translation_y": 0.0,
        "reason": "Movement latched",
        "latched": True,
    }


def test_movement_segment_does_not_spam_on_static_latched_delta(monkeypatch, tmp_path):
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=False,
        movement_pre_event_sec=0.0,
        movement_post_event_sec=0.0,
        movement_turn_delta_threshold=0.05,
        movement_confirm_changes=2,
        movement_segment_cooldown_sec=10.0,
        movement_min_clip_frames=1,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    writes = {"count": 0}

    class _Writer:
        def write(self, frame):
            return None

        def release(self):
            return None

    def fake_writer(path, fourcc, fps, size):
        writes["count"] += 1
        return _Writer()

    monkeypatch.setattr("cv2.VideoWriter", fake_writer)


    for _ in range(12):
        assert manager.process_movement(_movement_info(0.20), frame, video_timestamp=1.0) is None


    assert manager.process_movement(_movement_info(0.30), frame, video_timestamp=2.0) is not None


    for _ in range(12):
        assert manager.process_movement(_movement_info(0.30), frame, video_timestamp=3.0) is None

    assert writes["count"] == 1
