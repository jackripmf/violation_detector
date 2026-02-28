"""Файл: src/detectors/movement_detector.py
Тип: детектор нарушений.
Назначение: получает кадр, применяет алгоритм детекции и возвращает структурированный результат.
Связи: детекторы вызываются менеджером детекции и передают данные в слой сохранения/визуализации.
Импортируемые внутренние модули:
- .base_detector: используется для передачи данных или вызова связанной логики."""
import cv2
import numpy as np
import time
import logging
from .base_detector import BaseDetector


class CameraMovementDetector(BaseDetector):
    """Класс: CameraMovementDetector
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `bf_matcher` (`Any`): экземпляр BFMatcher для сопоставления дескрипторов признаков.
- `confirmation_frames` (`Any`): кадр/изображение, которое передается на обработку текущему этапу.
- `consecutive_movement_frames` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `last_confirmed_movement_info` (`None`): данные детектора нарушений, используемые для итогового решения по кадру.
- `last_frame_time` (`None`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `last_movement_detected` (`bool`): данные детектора нарушений, используемые для итогового решения по кадру.
- `last_output_info` (`None`): параметр источника/выхода данных, задающий направление потока обработки.
- `last_valid_relative_info` (`None`): последние валидные относительные параметры движения.
- `latched_movement_detected` (`bool`): данные детектора нарушений, используемые для итогового решения по кадру.
- `latched_since_time` (`None`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `max_frame_gap_sec` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `min_matches` (`Any`): минимум совпадений ключевых точек для подтверждения движения камеры.
- `min_movement_duration` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `movement_start_time` (`None`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `orb` (`Any`): экземпляр ORB-детектора ключевых точек для оценки движения камеры.
- `processing_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
Ключевые методы:
- `__init__()`, `_prepare_frame()`, `detect()`, `set_reference_frame()`, `_raw_movement_detection()`, `_filter_movement()`, `_reset_movement_tracking()`, `_build_latched_movement_info_from_base()`, `_build_latched_movement_info_live()`, `get_last_movement_info()`"""

    def __init__(
        self,
        rotation_treshold=8.0,
        translation_treshold=0.12,
        min_matches=20,
        min_movement_duration=2.0,
        confirmation_frames=8,
        processing_width=640,
        max_frame_gap_sec=1.0,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `rotation_treshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `translation_treshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `min_matches` (`Any`): минимум совпадений ключевых точек для подтверждения движения камеры.
- `min_movement_duration` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `confirmation_frames` (`Any`): кадр/изображение, которое передается на обработку текущему этапу.
- `processing_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `max_frame_gap_sec` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        super().__init__("CameraMovementDetector")
        self.rotation_treshold = rotation_treshold
        self.translation_treshold = translation_treshold
        self.min_matches = min_matches
        self.processing_width = int(processing_width)
        self.max_frame_gap_sec = float(max_frame_gap_sec)

        self.min_movement_duration = (
            min_movement_duration
        )
        self.confirmation_frames = confirmation_frames
        self.movement_start_time = None
        self.consecutive_movement_frames = 0
        self.last_movement_detected = False
        self.latched_movement_detected = False
        self.latched_since_time = None
        self.last_confirmed_movement_info = None
        self.last_valid_relative_info = None
        self.last_output_info = None

        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.reference_frame = None
        self.reference_kp = None
        self.reference_des = None
        self.reference_shape = None
        self.last_frame_time = None

        logging.info(
            f"Movement detector initialized: rotation_treshold={rotation_treshold}°, "
            f"translation_treshold={translation_treshold}, processing_width={self.processing_width}"
        )

    def _prepare_frame(self, frame):
        """Функция: _prepare_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if self.processing_width > 0 and w > self.processing_width:
            scale = self.processing_width / float(w)
            new_h = max(1, int(h * scale))
            gray = cv2.resize(gray, (self.processing_width, new_h), interpolation=cv2.INTER_AREA)
        return gray

    def detect(self, current_frame, is_obstructed=False):
        """Функция: detect()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `current_frame` (`Any`): текущий кадр, доступный на данном этапе обработки.
- `is_obstructed` (`Any`): логический флаг, включающий/отключающий соответствующее поведение.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        now_ts = time.time()
        if self.last_frame_time is not None:
            dt = now_ts - self.last_frame_time
            if dt > self.max_frame_gap_sec:
                self._reset_movement_tracking()
        self.last_frame_time = now_ts

        if not self.is_active or is_obstructed:
            self._reset_movement_tracking()
            result = self._get_default_movement_info(
                "Camera obstructed or detector disabled"
            )
            self.last_output_info = result
            return result

        raw_movement_info = self._raw_movement_detection(current_frame)
        if int(raw_movement_info.get("matches_count", 0)) > 0:
            self.last_valid_relative_info = dict(raw_movement_info)

        if self.latched_movement_detected:
            latched = self._build_latched_movement_info_live(raw_movement_info, now_ts)
            self.last_output_info = latched
            return latched

        filtered_movement_info = self._filter_movement(raw_movement_info)
        self.last_output_info = filtered_movement_info
        return filtered_movement_info

    def set_reference_frame(self, frame):
        """Функция: set_reference_frame()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        gray = self._prepare_frame(frame)
        self.reference_frame = gray
        self.reference_kp, self.reference_des = self.orb.detectAndCompute(gray, None)
        self.reference_shape = gray.shape[:2]
        self.latched_movement_detected = False
        self.latched_since_time = None
        self.last_confirmed_movement_info = None
        self.last_valid_relative_info = None
        self.last_output_info = self._get_default_movement_info("Reference updated")
        logging.info("Reference frame set for camera movement detection")

    def _raw_movement_detection(self, current_frame):
        """Функция: _raw_movement_detection()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `current_frame` (`Any`): текущий кадр, доступный на данном этапе обработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.reference_frame is None or self.reference_des is None:
            self.set_reference_frame(current_frame)
            return self._get_default_movement_info("No reference")

        gray_current = self._prepare_frame(current_frame)
        if self.reference_shape is not None and gray_current.shape[:2] != self.reference_shape:
            gray_current = cv2.resize(
                gray_current,
                (self.reference_shape[1], self.reference_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        kp_current, des_current = self.orb.detectAndCompute(gray_current, None)

        if des_current is None or len(des_current) < self.min_matches:
            return self._get_default_movement_info("Not enough features")

        matches = self.bf_matcher.match(self.reference_des, des_current)

        if len(matches) < self.min_matches:
            return self._get_default_movement_info("Not enough matches")

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

            h, w = gray_current.shape[:2]
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
        """Функция: _filter_movement()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `movement_info` (`Any`): данные детектора нарушений, используемые для итогового решения по кадру.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
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
                self.latched_movement_detected = True
                self.latched_since_time = current_time
                self.last_confirmed_movement_info = dict(result)
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
        """Функция: _reset_movement_tracking()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        self.movement_start_time = None
        self.consecutive_movement_frames = 0
        self.last_movement_detected = False

    def _build_latched_movement_info_from_base(self, base_info=None, now_ts=None):
        """Функция: _build_latched_movement_info_from_base()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `base_info` (`Any`): базовые данные о состоянии движения для последующего сравнения.
- `now_ts` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        base = (
            base_info
            or self.last_valid_relative_info
            or self.last_confirmed_movement_info
            or {}
        )
        reason = str(base.get("reason", "No movement"))
        latched_duration = 0.0
        if self.latched_since_time is not None and now_ts is not None:
            latched_duration = max(0.0, float(now_ts) - float(self.latched_since_time))

        return {
            "movement_detected": True,
            "rotation": base.get("rotation", 0),
            "translation": base.get("translation", 0),
            "translation_x": base.get("translation_x", 0),
            "translation_y": base.get("translation_y", 0),
            "reason": f"Movement latched | current delta: {reason}",
            "matches_count": base.get("matches_count", 0),
            "filtered": True,
            "consecutive_frames": max(1, int(base.get("consecutive_frames", 1))),
            "movement_duration": max(
                float(base.get("movement_duration", 0) or 0),
                latched_duration,
            ),
            "latched": True,
        }

    def _build_latched_movement_info_live(self, raw_info, now_ts):
        """Функция: _build_latched_movement_info_live()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `raw_info` (`Any`): сырые промежуточные данные до постобработки.
- `now_ts` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if int(raw_info.get("matches_count", 0)) > 0:
            return self._build_latched_movement_info_from_base(base_info=raw_info, now_ts=now_ts)
        return self._build_latched_movement_info_from_base(now_ts=now_ts)

    def get_last_movement_info(self):
        """Функция: get_last_movement_info()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.last_output_info is not None:
            return dict(self.last_output_info)
        if self.latched_movement_detected:
            return self._build_latched_movement_info_from_base(now_ts=time.time())
        return self._get_default_movement_info("No movement history")

    def _get_default_movement_info(self, reason):
        """Функция: _get_default_movement_info()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `reason` (`Any`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
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
