import cv2
import numpy as np

from .base_detector import BaseDetector


class CVDetector(BaseDetector):
    """Детектор на основе компьютерного зрения"""

    def __init__(self, brightness_thresh=25, contrast_thresh=10, sharpness_thresh=20):
        super().__init__("CVDetector")
        self.brightness_thresh = brightness_thresh
        self.contrast_thresh = contrast_thresh
        self.sharpness_thresh = sharpness_thresh

    def detect(self, frame):
        """Детекция по метрикам"""
        brightness, contrast, sharpness = self._get_metrics(frame)
        is_dark = brightness < self.brightness_thresh
        is_low_contrast = contrast < self.contrast_thresh
        is_blurry = sharpness < self.sharpness_thresh

        cv_detected = is_dark or is_blurry or is_low_contrast

        return {
            "detected": cv_detected,
            "metrics": {
                "Brightness": brightness,
                "Contrast": contrast,
                "Sharpness": sharpness,
            },
            "reasons": {
                "Dark": is_dark,
                "Low_contrast": is_low_contrast,
                "Blurry": is_blurry,
            },
        }

    def _get_metrics(self, frame):
        """Получение метрик изображения"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return brightness, contrast, sharpness
