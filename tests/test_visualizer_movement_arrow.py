"""Файл: tests/test_visualizer_movement_arrow.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.utils.media.visualizer: используется для передачи данных или вызова связанной логики."""

import cv2
import numpy as np

from src.utils.media.visualizer import Visualizer


def _movement_info(tx: float, ty: float) -> dict:
    return {
        "filtered": True,
        "movement_detected": True,
        "rotation": 0.0,
        "translation": float(np.hypot(tx, ty)),
        "translation_x": tx,
        "translation_y": ty,
        "consecutive_frames": 10,
        "movement_duration": 3.0,
    }


def test_movement_arrow_endpoint_stays_inside_frame(monkeypatch):
    calls = []

    def _fake_arrowed_line(img, pt1, pt2, color, thickness, tipLength=0.1):
        calls.append({"pt1": pt1, "pt2": pt2, "tip": tipLength, "thickness": thickness})
        return img

    monkeypatch.setattr(cv2, "arrowedLine", _fake_arrowed_line)

    viz = Visualizer(config={})
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    viz._draw_movement_info(frame, _movement_info(tx=5.0, ty=5.0))

    assert calls, "arrowedLine must be called for movement_detected=True"
    end_x, end_y = calls[-1]["pt2"]
    assert 0 <= end_x < frame.shape[1]
    assert 0 <= end_y < frame.shape[0]


def test_movement_arrow_head_size_is_stable_while_length_changes(monkeypatch):
    calls = []

    def _fake_arrowed_line(img, pt1, pt2, color, thickness, tipLength=0.1):
        calls.append({"pt1": pt1, "pt2": pt2, "tip": tipLength})
        return img

    monkeypatch.setattr(cv2, "arrowedLine", _fake_arrowed_line)

    viz = Visualizer(config={})
    frame = np.zeros((1200, 1600, 3), dtype=np.uint8)

    viz._draw_movement_info(frame, _movement_info(tx=0.10, ty=0.0))
    viz._draw_movement_info(frame, _movement_info(tx=0.90, ty=0.0))

    short = calls[-2]
    long = calls[-1]

    short_len = float(np.hypot(short["pt2"][0] - short["pt1"][0], short["pt2"][1] - short["pt1"][1]))
    long_len = float(np.hypot(long["pt2"][0] - long["pt1"][0], long["pt2"][1] - long["pt1"][1]))
    short_head = short_len * float(short["tip"])
    long_head = long_len * float(long["tip"])

    assert long_len > short_len
    assert abs(long_head - short_head) <= 3.5
