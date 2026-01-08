import cv2
import numpy as np

from .base_detector import BaseDetector


class DarkAreaDetector(BaseDetector):
    """Детектор темных областей"""

    def __init__(self, dark_area_threshold=0.8):
        super().__init__("DarkAreaDetector")
        self.dark_area_threshold = dark_area_threshold

    def detect(self, frame):
        """Детекция больших темных областей"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        dark_pixels = np.sum(dark_mask == 255)
        total_pixels = frame.shape[0] * frame.shape[1]
        dark_ratio = dark_pixels / total_pixels

        dark_detected = dark_ratio > self.dark_area_threshold

        return {"detected": dark_detected, "metrics": {"dark_ratio": dark_ratio}}
