"""Файл: tests/test_movement_detector.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.detectors.movement_detector: используется для передачи данных или вызова связанной логики."""

import numpy as np
import cv2
import time

from src.detectors.movement_detector import CameraMovementDetector


def _make_feature_rich_frame(seed: int = 42):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)


    cv2.rectangle(frame, (100, 100), (320, 320), (255, 255, 255), 3)
    cv2.circle(frame, (700, 360), 120, (0, 0, 0), 4)
    cv2.line(frame, (50, 650), (1200, 680), (255, 255, 255), 2)
    return frame


def test_detect_no_movement_on_same_frame():
    detector = CameraMovementDetector(
        rotation_treshold=10.0,
        translation_treshold=0.2,
        min_movement_duration=0.0,
        confirmation_frames=1,
    )
    frame = _make_feature_rich_frame()
    detector.set_reference_frame(frame)

    result = detector.detect(frame, is_obstructed=False)

    assert result["movement_detected"] is False
    assert result["filtered"] is True


def test_detect_translation_movement():
    detector = CameraMovementDetector(
        rotation_treshold=5.0,
        translation_treshold=0.05,
        min_movement_duration=0.0,
        confirmation_frames=1,
    )
    frame = _make_feature_rich_frame()
    detector.set_reference_frame(frame)

    matrix = np.float32([[1, 0, 160], [0, 1, 0]])
    shifted = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))

    result = detector.detect(shifted, is_obstructed=False)

    assert result["movement_detected"] is True
    assert result["filtered"] is True
    assert result["translation"] > 0.05


def test_detect_resets_tracking_after_large_frame_gap(monkeypatch):
    detector = CameraMovementDetector(
        min_movement_duration=0.0,
        confirmation_frames=3,
        max_frame_gap_sec=0.5,
    )
    frame = _make_feature_rich_frame()

    monkeypatch.setattr(
        detector,
        "_raw_movement_detection",
        lambda _: {
            "movement_detected": True,
            "rotation": 0.0,
            "translation": 0.2,
            "translation_x": 0.1,
            "translation_y": 0.0,
            "reason": "translation",
            "matches_count": 50,
        },
    )

    timeline = iter([1000.0, 1000.0, 1000.1, 1000.1, 1001.0, 1001.0])
    monkeypatch.setattr(time, "time", lambda: next(timeline))

    r1 = detector.detect(frame, is_obstructed=False)
    r2 = detector.detect(frame, is_obstructed=False)
    r3 = detector.detect(frame, is_obstructed=False)

    assert r1["movement_detected"] is False
    assert r2["movement_detected"] is False

    assert r3["movement_detected"] is False
    assert detector.consecutive_movement_frames == 1


def test_detect_latches_movement_until_reference_reset():
    detector = CameraMovementDetector(
        rotation_treshold=5.0,
        translation_treshold=0.05,
        min_movement_duration=0.0,
        confirmation_frames=1,
    )
    frame = _make_feature_rich_frame(seed=123)
    detector.set_reference_frame(frame)

    matrix = np.float32([[1, 0, 180], [0, 1, 0]])
    shifted = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))

    first = detector.detect(shifted, is_obstructed=False)
    second = detector.detect(shifted, is_obstructed=False)
    assert first["movement_detected"] is True
    assert second["movement_detected"] is True
    assert second.get("latched", False) is True

    detector.set_reference_frame(shifted)
    after_reset = detector.detect(shifted, is_obstructed=False)
    assert after_reset["movement_detected"] is False


def test_latched_state_updates_direction_relative_to_reference():
    detector = CameraMovementDetector(
        rotation_treshold=5.0,
        translation_treshold=0.05,
        min_movement_duration=0.0,
        confirmation_frames=1,
    )
    frame = _make_feature_rich_frame(seed=777)
    detector.set_reference_frame(frame)

    right_shift = cv2.warpAffine(
        frame,
        np.float32([[1, 0, 160], [0, 1, 0]]),
        (frame.shape[1], frame.shape[0]),
    )
    left_shift = cv2.warpAffine(
        frame,
        np.float32([[1, 0, -160], [0, 1, 0]]),
        (frame.shape[1], frame.shape[0]),
    )

    r1 = detector.detect(right_shift, is_obstructed=False)
    r2 = detector.detect(left_shift, is_obstructed=False)

    assert r1["movement_detected"] is True
    assert r2["movement_detected"] is True
    assert r2.get("latched", False) is True
    assert r1["translation_x"] > 0
    assert r2["translation_x"] < 0
