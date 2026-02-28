"""Файл: tests/test_cv_detector.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.detectors.cv_detector: используется для передачи данных или вызова связанной логики."""

import pytest
import numpy as np
from src.detectors.cv_detector import CVDetector


class TestCVDetector:

    def test_dark_frame_detected(self, dark_frame):
        detector = CVDetector(brightness_thresh=25)
        result = detector.detect(dark_frame)

        assert result["detected"] == True
        assert result["metrics"]["Brightness"] < 25
        assert result["reasons"]["Dark"] == True

    def test_bright_frame_not_detected(self, bright_frame):
        detector = CVDetector()
        result = detector.detect(bright_frame)



        assert result["metrics"]["Brightness"] > 25

        assert result["reasons"]["Dark"] == False

    def test_blurry_frame_detected(self, sample_frame):

        blurry = cv2.GaussianBlur(sample_frame, (99, 99), 0)

        detector = CVDetector(sharpness_thresh=20)
        result = detector.detect(blurry)

        assert result["metrics"]["Sharpness"] < 20
        assert result["reasons"]["Blurry"] == True

    def test_low_contrast_detected(self):

        low_contrast = np.full((720, 1280, 3), 128, dtype=np.uint8)

        detector = CVDetector(contrast_thresh=10)
        result = detector.detect(low_contrast)

        assert result["metrics"]["Contrast"] < 10
        assert result["reasons"]["Low_contrast"] == True

    def test_metrics_values(self, sample_frame):
        detector = CVDetector()
        result = detector.detect(sample_frame)

        assert "Brightness" in result["metrics"]
        assert "Contrast" in result["metrics"]
        assert "Sharpness" in result["metrics"]


        assert isinstance(result["metrics"]["Brightness"], (int, float, np.number))
        assert isinstance(result["metrics"]["Contrast"], (int, float, np.number))
        assert isinstance(result["metrics"]["Sharpness"], (int, float, np.number))

    def test_enable_disable(self, sample_frame):
        detector = CVDetector()

        detector.disable()
        assert detector.is_active == False

        detector.enable()
        assert detector.is_active == True



import cv2
