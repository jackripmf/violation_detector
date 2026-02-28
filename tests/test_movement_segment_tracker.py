"""Файл: tests/test_movement_segment_tracker.py
Тип: файл автотестов.
Назначение: проверяет state machine сегментации движения вне ViolationManager.
Связи: взаимодействует с movement_segment_tracker через публичные методы."""

import numpy as np

from src.processing.movement_segment_tracker import MovementSegmentTracker


def _movement_info(tx: float) -> dict:
    return {
        "movement_detected": True,
        "filtered": True,
        "movement_duration": 3.0,
        "translation": abs(float(tx)),
        "rotation": 0.0,
        "translation_x": float(tx),
        "translation_y": 0.0,
    }


def test_movement_segment_tracker_finalizes_segment_and_respects_writer_fps():
    tracker = MovementSegmentTracker(
        movement_pre_event_sec=1.0,
        movement_post_event_sec=0.0,
        movement_turn_delta_threshold=0.05,
        movement_change_window_sec=0.8,
        movement_confirm_changes=2,
        movement_segment_cooldown_sec=1.0,
        movement_max_clip_sec=12.0,
        movement_min_clip_frames=1,
    )
    tracker.set_source_fps(24.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert tracker.process_frame(frame, _movement_info(0.20), video_timestamp=1.0, captured_at=10.0) is None
    segment = tracker.process_frame(frame, _movement_info(0.30), video_timestamp=2.0, captured_at=10.5)

    assert segment is not None
    assert segment.clip_video_timestamp == 1.0
    assert segment.movement_info["segment_frames"] == 2
    assert tracker.get_writer_fps() == 24.0
