import cv2
import numpy as np
import time
import logging
from .base_detector import BaseDetector


class CameraMovementDetector(BaseDetector):
    """Детектор движения камеры"""

    def __init__(
        self,
        rotation_treshold=8.0,  # Увеличили с 5.0 до 8.0 градусов
        translation_treshold=0.12,  # Увеличили с 0.08 до 0.12
        min_matches=20,  # Увеличили с 15 до 20
        min_movement_duration=2.0,  # Увеличили с 1.5 до 2.0 секунд
        confirmation_frames=8,  # Увеличили с 5 до 8 кадров
    ):
        super().__init__("CameraMovementDetector")
        self.rotation_treshold = rotation_treshold  # порог поворота
        self.translation_treshold = translation_treshold  # порог смещения
        self.min_matches = min_matches
        # Параметры фильтрации
        self.min_movement_duration = (
            min_movement_duration  # минимальная продолжительность движения
        )
        self.confirmation_frames = confirmation_frames  # кадры с движением
        self.movement_start_time = None  # начало движения
        self.consecutive_movement_frames = 0  # последовательность кадров с движением
        self.last_movement_detected = False  # конец движения
        # Инициализация детектора особенностей
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Референсный кадр
        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None

        logging.info(
            f"Movement detector initialized: rotation_treshold={rotation_treshold}°, translation_treshold={translation_treshold}"
        )

    def detect(self, current_frame, is_obstructed=False):
        """Основной метод детекции движений"""
        if not self.is_active or is_obstructed:
            self._reset_movement_tracking()
            return self._get_default_movement_info(
                "Camera obstructed or detector disabled"
            )

        raw_movement_info = self._raw_movement_detection(current_frame)
        filtered_movement_info = self._filter_movement(raw_movement_info)
        return filtered_movement_info

    def set_reference_frame(self, frame):
        """Установка референсного кадра"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_frame = gray
        self.reference_kp, self.reference_des = self.orb.detectAndCompute(gray, None)
        logging.info("Reference frame set for camera movement detection")

    def _raw_movement_detection(self, current_frame):
        """Сырая детекция движений"""
        if self.reference_frame is None or self.reference_des is None:
            self.set_reference_frame(current_frame)
            return self._get_default_movement_info("No reference")

        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        kp_current, des_current = self.orb.detectAndCompute(gray_current, None)

        if des_current is None or len(des_current) < self.min_matches:
            return self._get_default_movement_info("Not enough features")

        matches = self.bf_matcher.match(self.reference_des, des_current)

        if len(matches) < self.min_matches:
            return self._get_default_movement_info("Not enough matches")

        # Отбираем только хорошие матчи (расстояние < 50)
        good_matches = [m for m in matches if m.distance < 50]

        if len(good_matches) < self.min_matches:
            return self._get_default_movement_info("Not enough good matches")

        matches = sorted(good_matches, key=lambda x: x.distance)
        good_matches = matches[: min(50, len(matches))]

        src_pts = np.float32(
            [self.reference_kp[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_current[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return self._get_default_movement_info("Homography failed")

            h, w = current_frame.shape[:2]
            rotation_angle = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
            translation_x = H[0, 2] / w
            translation_y = H[1, 2] / h
            translation_magnitude = np.sqrt(translation_x**2 + translation_y**2)

            rotation_detected = abs(rotation_angle) > self.rotation_treshold
            translation_detected = translation_magnitude > self.translation_treshold
            movement_detected = rotation_detected or translation_detected

            reason = "No movement"
            if movement_detected:
                reasons = []
                if rotation_detected:
                    reasons.append(f"rotation({rotation_angle:.1f}°)")
                if translation_detected:
                    reasons.append(f"translation({translation_magnitude:.3f})")
                reason = " + ".join(reasons)

                logging.info(f"Movement detected: {reason}")

            return {
                "movement_detected": movement_detected,
                "rotation": rotation_angle,
                "translation": translation_magnitude,
                "translation_x": translation_x,
                "translation_y": translation_y,
                "reason": reason,
                "matches_count": len(good_matches),
            }
        except Exception as e:
            logging.error(f"Movement detection error: {e}")
            return self._get_default_movement_info(f"Error: {str(e)}")

    def _filter_movement(self, movement_info):
        """Фильтрация ложных срабатываний"""
        current_time = time.time()
        movement_detected = movement_info.get("movement_detected", False)

        if movement_detected:
            self.consecutive_movement_frames += 1
            if self.movement_start_time is None:
                self.movement_start_time = current_time
                logging.info(f"Movement started: {movement_info['reason']}")
            movement_duration = current_time - self.movement_start_time

            frames_condition = (
                self.consecutive_movement_frames >= self.confirmation_frames
            )
            duration_condition = movement_duration >= self.min_movement_duration

            if frames_condition and duration_condition:
                self.last_movement_detected = True
                result = {
                    **movement_info,
                    "movement_detected": True,
                    "filtered": True,
                    "consecutive_frames": self.consecutive_movement_frames,
                    "movement_duration": movement_duration,
                }
                logging.info(
                    f"Movement confirmed: {movement_info['reason']}(duration: {movement_duration:.1f}s"
                )
                return result
            else:
                return {
                    **movement_info,
                    "movement_detected": False,
                    "filtered": True,
                    "reason": f"Movement not confirmed ({self.consecutive_movement_frames}/{self.confirmation_frames} frames, {movement_duration:.1f}/{self.min_movement_duration}s",
                }

        else:
            if self.consecutive_movement_frames > 0:
                logging.info(
                    f"Movement stopped after {self.consecutive_movement_frames} frames"
                )
            self._reset_movement_tracking()
            return {
                **movement_info,
                "movement_detected": False,
                "filtered": True,
                "consecutive_frames": 0,
                "movement_duration": 0,
            }

    def _reset_movement_tracking(self):
        """Сброс трекеров движения"""
        self.movement_start_time = None
        self.consecutive_movement_frames = 0
        self.last_movement_detected = False

    def _get_default_movement_info(self, reason):
        """Информация о движении по умолчанию"""
        return {
            "movement_detected": False,
            "rotation": 0,
            "translation": 0,
            "translation_x": 0,
            "translation_y": 0,
            "reason": reason,
            "matches_count": 0,
            "filtered": True,
            "consecutive_frames": 0,
            "movement_duration": 0,
        }
