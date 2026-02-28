"""Файл: tests/test_dark_detector.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.detectors.dark_area_detector: используется для передачи данных или вызова связанной логики."""

import pytest
import numpy as np
from src.detectors.dark_area_detector import DarkAreaDetector


class TestDarkAreaDetector:

    def test_all_dark_pixels_detected(self, dark_frame):
        detector = DarkAreaDetector(dark_area_threshold=0.8)
        result = detector.detect(dark_frame)

        assert result["detected"] == True
        assert result["metrics"]["dark_ratio"] == 1.0

    def test_all_bright_pixels_not_detected(self, bright_frame):
        detector = DarkAreaDetector()
        result = detector.detect(bright_frame)

        assert result["detected"] == False
        assert result["metrics"]["dark_ratio"] == 0.0

    def test_partial_dark_area(self):

        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        frame[:, :640] = 0

        detector = DarkAreaDetector(dark_area_threshold=0.8)
        result = detector.detect(frame)


        assert result["detected"] == False
        assert result["metrics"]["dark_ratio"] == pytest.approx(0.5, rel=0.1)

    def test_threshold_behavior(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)


        detector_low = DarkAreaDetector(dark_area_threshold=0.5)
        result_low = detector_low.detect(frame)
        assert result_low["detected"] == True


        detector_high = DarkAreaDetector(dark_area_threshold=0.95)
        result_high = detector_high.detect(frame)
        assert result_high["detected"] == True

    def test_detector_name(self):
        detector = DarkAreaDetector()
        assert detector.name == "DarkAreaDetector"
