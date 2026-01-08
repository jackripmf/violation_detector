import os.path
import cv2
from datetime import datetime
import time
import logging

from ..detectors.movement_detector import CameraMovementDetector
from ..detectors.cv_detector import CVDetector
from ..detectors.dark_area_detector import DarkAreaDetector
from ..detectors.yolo_detector import YOLODetector
from ..detectors.forbidden_items_detector import ForbiddenItemsDetector
from ..utils.file_manager import FileManager
from ..utils.visualizer import Visualizer
from ..utils.utils import safe_release_video_writer, check_file_size
from ..detectors.dms_detector import DMSDetector


class VideoViolationProcessor:
    """Основной процессор для обработки видео"""

    def __init__(self, save_dir="violations",
                 enabled_detectors=["all"], disabled_detectors=[]):

        # Сохраняем настройки детекторов
        self.enabled_detectors = enabled_detectors
        self.disabled_detectors = disabled_detectors

        # Определяем какие детекторы включить
        self.detector_config = self._parse_detector_config()

        logging.info(f"Конфигурация детекторов: {self.detector_config}")

        # Инициализация детекторов (добавляем forbidden_items_detector)
        self.movement_detector = CameraMovementDetector() if self.detector_config.get(
            "movement", True) else None
        self.cv_detector = CVDetector() if self.detector_config.get("cv",
                                                                    True) else None
        self.dark_area_detector = DarkAreaDetector() if self.detector_config.get(
            "dark", True) else None
        self.yolo_detector = YOLODetector() if self.detector_config.get("yolo",
                                                                        True) else None

        # Детектор запрещенных объектов с перклассовым кулдауном
        if self.detector_config.get("forbidden", True):
            self.forbidden_items_detector = ForbiddenItemsDetector(
                confidence_threshold=0.4,
                min_detection_duration=3.0,
                violation_cooldown=30.0,
                min_object_area_ratio=0.005,
                class_specific_cooldown=True
            )
        else:
            self.forbidden_items_detector = None

        # Управление файлами
        self.file_manager = FileManager(save_dir)
        self.visualizer = Visualizer()

        # Состояние системы (добавляем для запрещенных объектов)
        self.alert_count = 0
        self.is_obstruction_active = False
        self.obstruction_start_time = None
        self.last_violation_time = None
        self.violation_cooldown = 5  # секунд кулдаун между нарушениями
        self.min_obstruction_duration = 5
        self.gap_threshold = 3
        self.capture_frame_at = 3

        # Для запрещенных объектов
        self.forbidden_violation_count = 0
        self.last_forbidden_violation_time = None
        self.forbidden_violation_cooldown = 10
        self.current_forbidden_violation = None

        # Статистика (обновляем)
        self.stats = {
            "total_detections": 0,
            "saved_violations": 0,
            "camera_movements": 0,
            "movement_violations_saved": 0,
            "forbidden_items_violations": 0,
            "forbidden_items_saved": 0,
            "obstruction_violations": 0,  # Новое: нарушения перекрытия
            "movement_violations": 0,     # Новое: нарушения движения
            "current_obstruction_duration": 0,
            "current_movement_duration": 0,
            "max_obstruction_duration": 0,
            "max_movement_duration": 0,
            "dms_violations": 0,
            "dms_violations_saved": 0,
            "current_dms_violations": [],
            "eye_state": "unknown",
            "seatbelt_state": "unknown"
        }

        self.frame_count = 0

        # Сегменты нарушений
        self.obstruction_segments = []
        self.current_segment_start = None
        self.frame_to_save = None
        self.saved_for_current_violation = False

        # Для запрещенных объектов
        self.forbidden_frame_to_save = None
        self.forbidden_violation_info = None

        # Запись видео
        self.movement_video_writer = None
        self.movement_video_start_time = None
        self.is_recording_movement = False
        self.movement_violation_count = 0
        self.current_video_path = None

        # Превью
        self.preview_writer = None

        # Для трекинга сегментов движения
        self.movement_segments = []  # Список сегментов движения: [(start_time, end_time, reason)]
        self.current_movement_start = None
        self.current_movement_reason = None
        self.movement_report_saved = False  # Флаг что отчет уже сохранен для текущего видео
        # Для обработки пауз в движении
        self.movement_pause_buffer = 1.0  # секунд, паузы короче этого не прерывают запись
        self.last_movement_time = None
        # Оптимизация записи движения
        self.min_movement_video_duration = 3.0  # Увеличили с 2.0 до 3.0 секунд
        self.movement_start_threshold = 0.5  # Начинаем запись после 0.5с движения
        self.movement_stop_delay = 2.0  # Останавливаем через 2.0с без движения

        # Для гистерезиса
        self.movement_confidence_frames = 0  # Счетчик подтверждающих кадров
        self.movement_confidence_threshold = 3  # Нужно 3 кадра подряд с движением

        possible_model_paths = [
            # Основной ожидаемый путь
            os.path.join("src", "utils", "models", "dms_model.pt"),
            # Относительный путь
            "dms_model.pt",
            # Абсолютный путь от рабочей директории
            os.path.join(os.getcwd(), "dms_model.pt"),
        ]

        # Выбираем существующий путь
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                logging.info(f"Found DMS model at: {path}")
                break

        if model_path is None:
            logging.warning(
                "DMS model not found. DMS functionality will be disabled.")
            model_path = None  # Будет использован fallback в DMSDetector

        # DMS детектор
        if self.detector_config.get("dms", True):
            # Код инициализации DMS детектора...
            possible_model_paths = [
                os.path.join("src", "utils", "models", "dms_model.pt"),
                "dms_model.pt",
                os.path.join(os.getcwd(), "dms_model.pt"),
            ]

            model_path = None
            for path in possible_model_paths:
                if os.path.exists(path):
                    model_path = path
                    logging.info(f"Found DMS model at: {path}")
                    break

            if model_path is None:
                logging.warning(
                    "DMS model not found. DMS functionality will be disabled.")
                model_path = None

            self.dms_detector = DMSDetector(
                model_path=model_path,
                confidence_threshold=0.4,
                eye_closed_threshold=5.0,
                seatbelt_check_interval=30.0,
                phone_threshold=4.0,
                cigarette_threshold=3.0,
                violation_cooldown=60.0
            )
        else:
            self.dms_detector = None

        # Для сохранения DMS нарушений
        self.dms_frame_to_save = None
        self.dms_violation_info = None
        self.dms_violation_count = 0
        self.last_dms_violation_time = None
        self.dms_violation_cooldown = 30.0

    def _parse_detector_config(self):
        """Парсинг конфигурации детекторов из аргументов"""
        config = {
            "cv": True,
            "yolo": True,
            "dark": True,
            "movement": True,
            "forbidden": True,
            "dms": True
        }

        # Если указан "all" или ничего не указано - все включены
        if "all" in self.enabled_detectors:
            # Все включены по умолчанию
            pass
        else:
            # Сначала отключаем все
            for key in config:
                config[key] = False

            # Затем включаем только указанные
            for detector_arg in self.enabled_detectors:
                # Нормализуем имя (убираем 'detector' если есть)
                normalized_arg = detector_arg.lower().replace("detector",
                                                              "").strip()

                for detector_name in config.keys():
                    if detector_name.startswith(
                            normalized_arg) or normalized_arg.startswith(
                            detector_name):
                        config[detector_name] = True
                        break

        # Применяем отключенные детекторы (имеют приоритет)
        for disabled_detector in self.disabled_detectors:
            config[disabled_detector] = False

        return config

    def process_frame(self, frame, current_time, video_timestamp):
        """Обработка одного кадра с учетом включенных детекторов"""
        self.frame_count += 1

        # ===== 1. ИНИЦИАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ПО УМОЛЧАНИЮ =====
        dms_res = {
            "detected": False,
            "objects": [],
            "violations": [],
            "current_violations": [],
            "stats": {
                "eye_state": "unknown",
                "seatbelt_state": "unknown",
                "phone_detected": False,
                "cigarette_detected": False,
                "current_objects": [],
                "violation_count": 0
            }
        }

        cv_res = {
            "detected": False,
            "metrics": {"Brightness": 0, "Contrast": 0, "Sharpness": 0},
            "reasons": {"Dark": False, "Low_contrast": False, "Blurry": False}
        }

        dark_res = {
            "detected": False,
            "metrics": {"dark_ratio": 0}
        }

        yolo_res = {
            "detected": False,
            "objects": [],
            "metrics": {"large_objects_area": 0, "large_objects_ratio": 0}
        }

        forbidden_res = {
            "detected": False,
            "objects": [],
            "current_violation": False,
            "violation_info": None,
            "stats": {}
        }

        movement_info = {
            "movement_detected": False,
            "reason": "Movement detector disabled",
            "filtered": True,
            "rotation": 0,
            "translation": 0,
            "consecutive_frames": 0,
            "movement_duration": 0
        }

        # ===== 2. ДЕТЕКЦИЯ DMS НАРУШЕНИЙ =====
        if self.dms_detector:
            try:
                dms_res = self.dms_detector.detect(frame)

                # Обновляем статистику DMS
                self.stats["eye_state"] = dms_res["stats"].get("eye_state",
                                                               "unknown")
                self.stats["seatbelt_state"] = dms_res["stats"].get(
                    "seatbelt_state", "unknown")

                # Проверяем DMS нарушения
                dms_violations = dms_res.get("violations", [])
                current_dms_violations = dms_res.get("current_violations", [])

                if current_dms_violations:
                    self.stats[
                        "current_dms_violations"] = current_dms_violations
                    self._handle_dms_violation(frame, dms_res, current_time,
                                               video_timestamp)

            except Exception as e:
                logging.error(f"DMS detection error: {e}")

        # ===== 3. ДЕТЕКЦИЯ ПЕРЕКРЫТИЯ =====
        # Детекторы должны работать вместе для определения перекрытия
        detectors_available_for_obstruction = (
                self.cv_detector is not None and
                self.dark_area_detector is not None and
                self.yolo_detector is not None
        )

        if detectors_available_for_obstruction:
            try:
                if self.cv_detector:
                    cv_res = self.cv_detector.detect(frame)

                if self.dark_area_detector:
                    dark_res = self.dark_area_detector.detect(frame)

                if self.yolo_detector:
                    yolo_res = self.yolo_detector.detect(frame)

            except Exception as e:
                logging.error(f"Obstruction detection error: {e}")

        # Определяем есть ли перекрытие (только если все детекторы доступны)
        is_obstructed = False
        if detectors_available_for_obstruction:
            is_obstructed = cv_res["detected"] or dark_res["detected"] or \
                            yolo_res["detected"]

        # ===== 4. ДЕТЕКЦИЯ ДВИЖЕНИЯ =====
        if self.movement_detector:
            try:
                movement_info = self.movement_detector.detect(frame,
                                                              is_obstructed)

                # Трекинг сегментов движения
                if movement_info.get("movement_detected",
                                     False) and movement_info.get("filtered",
                                                                  False):
                    # Обновляем время последнего движения
                    self.last_movement_time = current_time

                    # Начало движения
                    if self.current_movement_start is None:
                        self.current_movement_start = current_time
                        self.current_movement_reason = movement_info.get(
                            "reason", "Unknown")
                        self.movement_report_saved = False
                        logging.info(
                            f"Movement segment started: {self.current_movement_reason}")

                    # Управление записью видео - начинаем только если движение длится > 0.5с
                    if not self.is_recording_movement:
                        movement_duration = current_time - self.current_movement_start
                        if movement_duration > 0.5:  # Ждем 0.5 секунды движения
                            self._start_movement_video_recording(frame,
                                                                 video_timestamp)
                            logging.info(
                                f"CAMERA MOVEMENT VIOLATION DETECTED: {movement_info.get('reason', 'Unknown')}")

                    if self.is_recording_movement:
                        self._save_movement_video_frame(frame)

                elif movement_info.get("filtered",
                                       False) and not movement_info.get(
                        "movement_detected", False):
                    # Конец движения
                    if self.current_movement_start is not None:
                        movement_duration = current_time - self.current_movement_start

                        if movement_duration > 0.5:
                            self.movement_segments.append({
                                "start_time": self.current_movement_start,
                                "end_time": current_time,
                                "duration": movement_duration,
                                "reason": self.current_movement_reason,
                                "start_timestamp": video_timestamp - movement_duration
                            })
                            logging.info(
                                f"Movement segment ended: {movement_duration:.1f}s")

                        self.current_movement_start = None
                        self.current_movement_reason = None

                    # Проверяем нужно ли останавливать запись
                    if self.is_recording_movement and self.last_movement_time:
                        time_since_last_movement = current_time - self.last_movement_time

                        # Останавливаем если нет движения более 1.5 секунд
                        if time_since_last_movement > 1.5:
                            self._stop_movement_video_recording(movement_info)

            except Exception as e:
                logging.error(f"Movement detection failed: {e}")
                movement_info = {"movement_detected": False,
                                 "reason": f"Error: {str(e)}"}

        # ===== 5. ДЕТЕКЦИЯ ЗАПРЕЩЕННЫХ ОБЪЕКТОВ =====
        if self.forbidden_items_detector:
            try:
                forbidden_res = self.forbidden_items_detector.detect(frame)

                # логируем что возвращает детектор
                if forbidden_res.get("current_violation", False):
                    violation_info = forbidden_res.get('violation_info')
                    if violation_info:
                        objects = violation_info.get('objects', [])
                        logging.info(
                            f"Forbidden violation detected at frame {self.frame_count}: "
                            f"ID={violation_info.get('violation_id')}, "
                            f"objects={len(objects)}"
                        )

                    # Обработка нарушения
                    objects = violation_info.get("objects",
                                                 []) if violation_info else []
                    if objects:
                        # Сохраняем кадр с нарушением
                        self._handle_forbidden_violation(frame, forbidden_res,
                                                         current_time,
                                                         video_timestamp)

            except Exception as e:
                logging.error(f"Forbidden items detection error: {e}")

        # ===== 6. АНАЛИЗ РЕЗУЛЬТАТОВ ПЕРЕКРЫТИЯ =====
        detectors_count = 0
        confidence = "LOW"

        if detectors_available_for_obstruction:
            detectors_count = sum([cv_res["detected"], dark_res["detected"],
                                   yolo_res["detected"]])
            confidence = (
                "HIGH" if detectors_count >= 2
                else "MEDIUM" if detectors_count == 1
                else "LOW"
            )

        # ===== 7. ОБНОВЛЕНИЕ СТАТИСТИКИ =====
        # Обновляем статистику движения
        if self.movement_detector:
            self.update_movement_stats(movement_info)

        # Обновляем статистику перекрытия
        if detectors_available_for_obstruction:
            total_duration = self._get_total_obstruction_duration(current_time)
            self.stats["current_obstruction_duration"] = total_duration
            self.stats["max_obstruction_duration"] = max(
                self.stats["max_obstruction_duration"],
                total_duration
            )
        else:
            total_duration = 0

        # ===== 8. ФОРМИРОВАНИЕ РЕЗУЛЬТАТА =====
        result = {
            "obstructed": is_obstructed,
            "confidence": confidence,
            "detectors_count": detectors_count,
            "forbidden_violation": forbidden_res.get("current_violation",
                                                     False),
            "forbidden_objects": forbidden_res.get("objects", []),
            "details": {
                "cv": cv_res,
                "dark_area": dark_res,
                "yolo": yolo_res,
                "forbidden": forbidden_res
            },
            "alert_count": self.alert_count,
            "frame_count": self.frame_count,
            "stats": self.get_detection_stats(),
            "dms_violations": dms_res.get("violations", []),
            "current_dms_violations": dms_res.get("current_violations", []),
            "dms_stats": dms_res["stats"],
            "dms_objects": dms_res["objects"]
        }

        # ===== 9. ОБНОВЛЕНИЕ СЕГМЕНТОВ НАРУШЕНИЙ ПЕРЕКРЫТИЯ =====
        if detectors_available_for_obstruction:
            self._update_obstruction_segments(is_obstructed, current_time)
            total_duration = self._get_total_obstruction_duration(current_time)

            # Сохранение кадра перекрытия - ТОЛЬКО HIGH!
            if (
                    is_obstructed
                    and total_duration >= self.capture_frame_at
                    and confidence == "HIGH"
                    and not self.saved_for_current_violation
                    and self.current_segment_start is not None
            ):
                # Просто сохраняем текущий кадр
                self.frame_to_save = frame.copy()
                self.saved_for_current_violation = True
                logging.info(
                    f"Frame captured for obstruction violation at {video_timestamp:.1f}s "
                    f"(confidence: {confidence}, duration: {total_duration:.1f}s)"
                )

            # Обработка нарушений перекрытия
            self._handle_violation_saving(result, current_time, movement_info,
                                          video_timestamp)

        return result, total_duration, movement_info

    def get_detection_stats(self):
        """Получение статистики детекции с учетом включенных детекторов"""
        stats = {
            "frame_count": self.frame_count,
            "alerts_count": self.alert_count
        }

        # Статистика перекрытия
        if self.cv_detector or self.dark_area_detector or self.yolo_detector:
            stats["obstruction"] = {
                "total": self.stats["saved_violations"],
                "current_duration": self.stats["current_obstruction_duration"],
                "max_duration": self.stats["max_obstruction_duration"],
                "detected": self.stats["total_detections"]
            }

        # Статистика движения
        if self.movement_detector:
            stats["movement"] = {
                "total": self.stats["movement_violations_saved"],
                "current_duration": self.stats["current_movement_duration"],
                "max_duration": self.stats["max_movement_duration"],
                "detected": self.stats["camera_movements"]
            }

        # Статистика запрещенных объектов
        if self.forbidden_items_detector:
            current_objects = 0
            if hasattr(self.forbidden_items_detector, 'current_detections'):
                current_objects = len(
                    self.forbidden_items_detector.current_detections)

            active_cooldowns = {}
            if hasattr(self.forbidden_items_detector, 'get_cooldown_info'):
                active_cooldowns = self.forbidden_items_detector.get_cooldown_info()

            stats["forbidden_items"] = {
                "total": self.stats["forbidden_items_saved"],
                "current_objects": current_objects,
                "detected": self.stats["forbidden_items_violations"],
                "active_cooldowns": active_cooldowns
            }

        # Статистика DMS
        if self.dms_detector:
            stats["dms"] = {
                "total": self.stats.get("dms_violations_saved", 0),
                "detected": self.stats.get("dms_violations", 0),
                "eye_state": self.stats.get("eye_state", "unknown"),
                "seatbelt_state": self.stats.get("seatbelt_state", "unknown")
            }

        # Подсчитываем общее количество нарушений
        total_violations = 0
        if "obstruction" in stats:
            total_violations += stats["obstruction"]["total"]
        if "movement" in stats:
            total_violations += stats["movement"]["total"]
        if "forbidden_items" in stats:
            total_violations += stats["forbidden_items"]["total"]
        if "dms" in stats:
            total_violations += stats["dms"]["total"]

        stats["total_violations"] = total_violations

        return stats

    def update_movement_stats(self, movement_info):
        """Обновление статистики движения (только если детектор включен)"""
        if not self.movement_detector:
            return

        if movement_info.get("movement_detected", False):
            duration = movement_info.get("movement_duration", 0)
            self.stats["current_movement_duration"] = duration
            self.stats["max_movement_duration"] = max(
                self.stats["max_movement_duration"],
                duration
            )
        else:
            self.stats["current_movement_duration"] = 0


    def _update_obstruction_segments(self, is_obstructed, current_time):
        """Обновление сегментов нарушений"""
        if is_obstructed:
            if self.current_segment_start is None:
                self.current_segment_start = current_time
                self.saved_for_current_violation = False
        else:
            if self.current_segment_start is not None:
                if self.obstruction_segments:
                    last_segment_end = self.obstruction_segments[-1][1]
                    gap = self.current_segment_start - last_segment_end
                    if gap <= self.gap_threshold:
                        self.obstruction_segments[-1][1] = current_time
                    else:
                        self.obstruction_segments.append(
                            [self.current_segment_start, current_time]
                        )
                else:
                    self.obstruction_segments.append(
                        [self.current_segment_start, current_time]
                    )
                self.current_segment_start = None

    def _get_total_obstruction_duration(self, current_time):
        """Вычисление общей длительности нарушения"""
        if not self.obstruction_segments:
            return 0
        total_duration = 0
        for segment in self.obstruction_segments:
            total_duration += segment[1] - segment[0]
        if self.current_segment_start is not None:
            total_duration += current_time - self.current_segment_start
        return total_duration

    def _handle_violation_saving(
            self, result, current_time, movement_info, video_timestamp
    ):
        """Обработка сохранения нарушений перекрытия - ТОЛЬКО HIGH!"""
        is_overlapped = result["obstructed"]
        confidence = result["confidence"]

        # Только HIGH уверенность!
        if is_overlapped and confidence == "HIGH":
            total_duration = self._get_total_obstruction_duration(current_time)

            if (
                    not self.is_obstruction_active
                    and total_duration >= self.min_obstruction_duration
                    and self.frame_to_save is not None
                    and self.saved_for_current_violation
            ):
                if (
                        self.last_violation_time is None
                        or current_time - self.last_violation_time >= self.violation_cooldown
                ):
                    self.is_obstruction_active = True
                    self.alert_count += 1
                    self.stats["total_detections"] += 1

                    violation_info = self._get_violation_info(
                        result, total_duration, movement_info, video_timestamp
                    )

                    if self._save_violation_frame(
                            self.frame_to_save, result, violation_info,
                            movement_info
                    ):
                        self.last_violation_time = current_time
                        logging.info(
                            f"===VIOLATION #{self.alert_count} SAVED "
                            f"(confidence: {confidence}, duration: {total_duration:.1f}s)==="
                        )
                        self.frame_to_save = None
                        self.saved_for_current_violation = False
        else:
            if self.is_obstruction_active:
                self.is_obstruction_active = False
                self.obstruction_segments = []
                self.current_segment_start = None
                self.frame_to_save = None
                self.saved_for_current_violation = False


    def _handle_forbidden_violation(self, frame, forbidden_res, current_time,
                                    video_timestamp):
        """Обработка нарушения с запрещенными объектами"""
        # Проверяем кулдаун
        if self.last_forbidden_violation_time is not None:
            time_since_last = current_time - self.last_forbidden_violation_time
            if time_since_last < self.forbidden_violation_cooldown:
                logging.info(
                    f"Forbidden items: Cooldown active ({self.forbidden_violation_cooldown - time_since_last:.1f}s remaining)")
                return

        violation_info = forbidden_res.get("violation_info")
        if not violation_info:
            logging.warning("Forbidden items: No violation info in response")
            return

        # Проверяем есть ли объекты в violation_info
        objects = violation_info.get("objects", [])
        if not objects:
            logging.warning(
                "Forbidden items: Violation info exists but no objects found")
            logging.warning(f"Violation info keys: {violation_info.keys()}")
            return

        logging.info(f"=== FORBIDDEN ITEMS VIOLATION DETECTED ===")
        logging.info(f"Violation ID: {violation_info.get('violation_id')}")
        logging.info(f"Number of objects: {len(objects)}")
        logging.info(
            f"Objects: {[obj.get('class', 'Unknown') for obj in objects]}")

        # Сохраняем кадр
        self.forbidden_frame_to_save = frame.copy()

        # Копируем violation_info и добавляем дополнительные данные
        self.forbidden_violation_info = {
            **violation_info,  # Копируем все из violation_info
            "video_timestamp": video_timestamp,
            "frame_count": self.frame_count,
            "current_time": current_time,
            "image_filename": None  # Будет заполнено при сохранении
        }

        # Немедленно пытаемся сохранить
        if self._save_forbidden_violation():
            self.last_forbidden_violation_time = current_time
            self.forbidden_violation_count += 1
            self.stats["forbidden_items_violations"] += 1
            self.stats["forbidden_items_saved"] += 1

            logging.info(
                f"=== FORBIDDEN ITEMS VIOLATION #{self.forbidden_violation_count} SAVED SUCCESSFULLY ===")
        else:
            logging.error("Failed to save forbidden items violation!")


    def _get_violation_info(
        self, result, total_duration, movement_info, video_timestamp
    ):
        """Получение информации о нарушении"""
        info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "violation_id": self.alert_count + 1,
            "confidence": result["confidence"],
            "detectors_count": result["detectors_count"],
            "total_duration": total_duration,
            "frame_number": self.frame_count,
            "video_timestamp": video_timestamp,
            "camera_movement": movement_info.get("movement_detected", False),
            "movement_reason": movement_info.get("reason", "Unknown"),
            "movement_rotation": movement_info.get("rotation", 0),
            "movement_translation": movement_info.get("translation", 0),
            "movement_duration": movement_info.get("movement_duration", 0),
        }

        cv_details = result["details"]["cv"]
        dark_details = result["details"]["dark_area"]
        yolo_details = result["details"]["yolo"]

        reasons = []
        if cv_details["detected"]:
            for reason, detected in cv_details["reasons"].items():
                if detected:
                    reasons.append(reason)

        if dark_details["detected"]:
            reasons.append("dark_area")

        if yolo_details["detected"]:
            reasons.append(f"yolo_object({len(yolo_details['objects'])} objects)")

        info["reasons"] = reasons
        info["metrics"] = {
            "brightness": cv_details["metrics"]["Brightness"],
            "contrast": cv_details["metrics"]["Contrast"],
            "sharpness": cv_details["metrics"]["Sharpness"],
            "dark_ratio": dark_details["metrics"]["dark_ratio"],
            "yolo_objects_ratio": yolo_details["metrics"]["large_objects_ratio"],
            "yolo_objects_count": len(yolo_details["objects"]),
        }

        info["yolo_objects"] = []
        for obj in yolo_details["objects"]:
            info["yolo_objects"].append(
                {
                    "class": obj["class"],
                    "confidence": obj["confidence"],
                    "area_ratio": obj["area_ratio"],
                }
            )
        return info

    def _save_forbidden_violation(self):
        """Сохранение нарушения с запрещенными объектами"""
        if self.forbidden_frame_to_save is None or self.forbidden_violation_info is None:
            return False

        try:
            # Создаем отображаемый кадр с bounding boxes
            display_frame = self.forbidden_frame_to_save.copy()

            # Рисуем bounding boxes для запрещенных объектов
            objects = self.forbidden_violation_info.get("objects", [])
            for obj in objects:
                self.visualizer._draw_forbidden_object(display_frame, obj)

            # Добавляем текст с информацией о нарушении
            violation_id = self.forbidden_violation_info.get("violation_id", 0)

            # Основной заголовок
            cv2.putText(
                display_frame,
                f"FORBIDDEN ITEMS VIOLATION #{violation_id}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                self.visualizer.colors["purple"],
                3,
            )

            # Перечисляем обнаруженные классы
            classes = list(
                set([obj.get('class', 'Unknown') for obj in objects]))
            class_text = f"Classes: {', '.join(classes)}"

            cv2.putText(
                display_frame,
                class_text,
                (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.visualizer.colors["cyan"],
                2,
            )

            # Сохраняем изображение
            img_filename = self.file_manager.generate_filename(
                "forbidden_items", "jpg")
            img_path = self.file_manager.get_full_path(img_filename,
                                                       "forbidden_items")
            success = cv2.imwrite(img_path, display_frame)

            if not success:
                logging.error(f"Failed to save image: {img_path}")
                return False

            # Обновляем информацию о нарушении
            self.forbidden_violation_info.update({
                "image_filename": img_filename,
                "frame_count": self.frame_count,
                "detected_classes": classes,
                "total_objects": len(objects)
            })

            # Сохраняем отчет
            self._save_forbidden_items_report(self.forbidden_violation_info)

            logging.info(f"Forbidden items violation saved: {img_filename}")
            logging.info(f"Classes detected: {classes}")

            # Сбрасываем
            self.forbidden_frame_to_save = None
            self.forbidden_violation_info = None

            return True

        except Exception as e:
            logging.error(f"Error saving forbidden items violation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_forbidden_items_report(self, violation_info):
        """Сохранение отчета о нарушении с запрещенными объектами"""
        try:
            if not violation_info.get("image_filename"):
                logging.error("No image filename in violation info")
                return

            report_filename = violation_info["image_filename"].replace(".jpg",
                                                                       ".txt")
            report_path = self.file_manager.get_full_path(report_filename,
                                                          "forbidden_items")

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("FORBIDDEN ITEMS VIOLATION REPORT\n")
                f.write("=" * 40 + "\n\n")

                f.write(
                    f"Violation ID: {violation_info.get('violation_id', 'N/A')}\n")
                f.write(
                    f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(violation_info.get('timestamp', time.time())))}\n")
                f.write(
                    f"Video Time: {violation_info.get('video_timestamp', 0):.1f}s\n")
                f.write(f"Frame: {violation_info.get('frame_count', 0)}\n")
                f.write(
                    f"Current Time: {time.strftime('%H:%M:%S', time.localtime(violation_info.get('current_time', time.time())))}\n\n")

                f.write("DETECTED OBJECTS:\n")
                f.write("-" * 40 + "\n")

                objects = violation_info.get('objects', [])
                if not objects:
                    f.write("No objects detected\n")
                else:
                    for i, obj in enumerate(objects, 1):
                        # ОСНОВНАЯ ИНФОРМАЦИЯ О КЛАССЕ
                        class_name = obj.get('class', 'Unknown')
                        confidence = obj.get('confidence', 0)
                        duration = obj.get('duration', 0)

                        f.write(f"\n{i}. OBJECT: {class_name.upper()}\n")
                        f.write(f"   Class: {class_name}\n")
                        f.write(f"   Confidence: {confidence:.2f}\n")
                        f.write(f"   Detection Duration: {duration:.1f}s\n")

                        # Координаты bounding box
                        if 'bbox' in obj and len(obj['bbox']) == 4:
                            x1, y1, x2, y2 = obj['bbox']
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height

                            f.write(
                                f"   Bounding Box: [{x1}, {y1}, {x2}, {y2}]\n")
                            f.write(
                                f"   Dimensions: {width}x{height} pixels\n")
                            f.write(f"   Area: {area} pixels\n")

                        # Дополнительная информация если есть
                        if 'area_ratio' in obj:
                            f.write(
                                f"   Area Ratio: {obj['area_ratio']:.3f}\n")

                f.write(f"\n" + "=" * 40 + "\n")
                f.write(f"VIOLATION SUMMARY:\n")
                f.write(f"Total Objects: {len(objects)}\n")

                # Группировка по классам
                class_counts = {}
                for obj in objects:
                    class_name = obj.get('class', 'Unknown')
                    class_counts[class_name] = class_counts.get(class_name,
                                                                0) + 1

                f.write(f"Objects by Class:\n")
                for class_name, count in class_counts.items():
                    f.write(f"  - {class_name}: {count}\n")

                f.write(
                    f"\nCooldown remaining: {violation_info.get('cooldown_remaining', 0):.1f}s\n")
                f.write(f"Image File: {violation_info['image_filename']}\n")
                f.write(f"Report File: {report_filename}\n")

            logging.info(f"Forbidden items report saved: {report_filename}")
            logging.info(f"Detected classes: {list(class_counts.keys())}")

        except Exception as e:
            logging.error(f"Error saving forbidden items report: {e}")
            import traceback
            traceback.print_exc()

    def _save_violation_frame(self, frame, result, violation_info,
                              movement_info):
        """Сохранение кадра с нарушением"""
        try:
            # Создаем отображаемый кадр
            display_frame = self.visualizer.draw_detection_info(
                frame,
                result,
                violation_info["total_duration"],
                movement_info,
                violation_info["video_timestamp"],
            )

            # Сохраняем изображение
            img_filename = self.file_manager.generate_filename("obstruction",
                                                               "jpg")
            img_path = self.file_manager.get_full_path(img_filename)
            cv2.imwrite(img_path, display_frame)

            # Сохраняем отчет
            self.file_manager.save_violation_report(violation_info,
                                                    movement_info)

            # Увеличиваем счетчик нарушений перекрытия
            self.stats["obstruction_violations"] += 1

            logging.info(
                f"Obstruction violation #{violation_info['violation_id']} saved: {img_filename}"
            )
            self.stats["saved_violations"] += 1
            return True
        except Exception as exc:
            logging.error(f"Save error: {exc}")
            return False

    def _start_movement_video_recording(self, frame, video_timestamp):
        """Начало записи движения"""
        if self.is_recording_movement:
            return
        self.is_recording_movement = True
        self.movement_violation_count += 1
        self.movement_video_start_time = time.time()
        # Создаем VideoWriter
        video_filename = self.file_manager.generate_filename("movement", "mp4")
        self.current_video_path = self.file_manager.get_full_path(video_filename)

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.movement_video_writer = cv2.VideoWriter(
            self.current_video_path, fourcc, 20.0, (width, height)
        )
        logging.info(f"Started recording movement violation video: {video_filename}")
        self.movement_video_start_timestamp = video_timestamp

    def _save_movement_video_frame(self, frame):
        """Сохранение кадра движения"""
        if self.is_recording_movement and self.movement_video_writer is not None:
            try:
                self.movement_video_writer.write(frame.copy())
            except Exception as e:
                logging.error(f"Error writing video frame: {e}")

    def _stop_movement_video_recording(self, movement_info):
        """Остановка записи видео движения - фильтрация коротких видео"""
        if not self.is_recording_movement:
            return

        video_duration = time.time() - self.movement_video_start_time

        # ВАЖНО: Фильтруем видео короче 2 секунд
        if video_duration < 2.0:
            logging.info(
                f"Movement video too short ({video_duration:.1f}s < 2s), deleting")

            # Освобождаем VideoWriter
            safe_release_video_writer(self.movement_video_writer)
            self.movement_video_writer = None

            # Удаляем файл если он был создан
            if self.current_video_path and os.path.exists(
                    self.current_video_path):
                try:
                    os.remove(self.current_video_path)
                    logging.info(
                        f"Deleted short video: {os.path.basename(self.current_video_path)}")
                except Exception as e:
                    logging.error(f"Error deleting short video: {e}")

            # Очищаем сегменты
            self.movement_segments = []
            self.is_recording_movement = False
            return

        # Всегда освобождаем VideoWriter
        safe_release_video_writer(self.movement_video_writer)
        self.movement_video_writer = None

        # Проверяем размер файла
        if self.current_video_path and os.path.exists(self.current_video_path):
            if not check_file_size(self.current_video_path):
                try:
                    os.remove(self.current_video_path)
                    logging.info("Deleted corrupted video file")
                    return
                except Exception as e:
                    logging.error(f"Error deleting corrupted video: {e}")
                    return

        # Сохраняем отчет о движении
        if not self.movement_report_saved and self.movement_segments:
            video_filename = os.path.basename(self.current_video_path)

            # Консолидируем информацию из всех сегментов
            consolidated_info = self._consolidate_movement_segments(
                movement_info, video_duration)

            self.file_manager.save_movement_report(
                consolidated_info, video_filename, video_duration
            )
            self.movement_report_saved = True

            logging.info(
                f"Movement violation video saved, duration: {video_duration:.1f}s")
            logging.info(
                f"Consolidated {len(self.movement_segments)} movement segments")

        self.is_recording_movement = False
        self.stats["movement_violations_saved"] += 1

        # Очищаем сегменты после сохранения отчета
        self.movement_segments = []

    def start_preview_recording(self, frame, fps=None):
        """Начало записи превью с возможностью автоматического определения FPS"""
        if self.preview_writer is not None:
            return

        # Если fps не указан, пытаемся получить реальный FPS обработки
        if fps is None and hasattr(self,
                                   'real_processing_fps') and self.real_processing_fps:
            fps = self.real_processing_fps
        elif fps is None:
            fps = 15.0  # Дефолтное значение

        preview_filename = self.file_manager.generate_filename("preview",
                                                               "mp4")
        preview_path = self.file_manager.get_full_path(preview_filename)
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Ограничиваем FPS разумными пределами
        fps = max(5.0, min(fps, 60.0))  # Не меньше 5, не больше 60 FPS

        self.preview_writer = cv2.VideoWriter(
            preview_path, fourcc, fps, (width, height)
        )

        logging.info(
            f"Started preview recording: {preview_filename} (FPS: {fps:.1f})")

    def save_preview_frame(self, display_frame):
        """Сохранение кадра превью"""
        if self.preview_writer is not None:
            try:
                self.preview_writer.write(display_frame.copy())
            except Exception as e:
                logging.error(f"Error writing preview frame: {e}")

    def stop_preview_recording(self):
        """Остановка записи первью"""
        safe_release_video_writer(self.preview_writer)
        self.preview_writer = None
        logging.info("Preview recording stopped")

    def cleanup(self):
        """Очистка ресурсов"""
        # Если все еще записываем движение, останавливаем
        if self.is_recording_movement:
            self._stop_movement_video_recording({
                "movement_detected": False,
                "reason": "Processing stopped",
                "filtered": True
            })

        # Очищаем сегменты
        self.movement_segments = []
        self.current_movement_start = None
        self.current_movement_reason = None

        self.stop_preview_recording()


    def _consolidate_movement_segments(self, movement_info, total_duration):
        """Консолидация информации из всех сегментов движения"""
        if not self.movement_segments:
            return movement_info

        # Собираем общую статистику по всем сегментам
        total_segments_duration = sum(
            segment["duration"] for segment in self.movement_segments)
        all_reasons = list(
            set(segment["reason"] for segment in self.movement_segments))

        # Находим самый частый тип движения
        reason_counts = {}
        for segment in self.movement_segments:
            reason = segment["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        most_common_reason = max(reason_counts.items(), key=lambda x: x[1])[
            0] if reason_counts else "Unknown"

        # Собираем временные метки сегментов
        segments_info = []
        for i, segment in enumerate(self.movement_segments, 1):
            segments_info.append({
                "segment": i,
                "start": segment["start_time"],
                "end": segment["end_time"],
                "duration": segment["duration"],
                "reason": segment["reason"],
                "video_start": segment.get("start_timestamp", 0)
            })

        # Создаем консолидированную информацию
        consolidated = {
            **movement_info,
            "total_duration": total_duration,
            "movement_segments": len(self.movement_segments),
            "active_movement_time": total_segments_duration,
            "inactive_time": total_duration - total_segments_duration,
            "consolidated_reason": most_common_reason,
            "all_reasons": all_reasons,
            "segments": segments_info,
            "segment_count": len(self.movement_segments),
            "is_consolidated": True
        }

        logging.info(
            f"Consolidated {len(self.movement_segments)} segments into one report")
        logging.info(
            f"Active movement: {total_segments_duration:.1f}s of {total_duration:.1f}s")

        return consolidated


    def _handle_dms_violation(self, frame, dms_res, current_time,
                              video_timestamp):
        """Обработка DMS нарушений"""
        violations = dms_res.get("current_violations", [])
        if not violations:
            return

        logging.debug(
            f"DMS: Found {len(violations)} violations: {[v.get('type', 'unknown') for v in violations]}")

        # Сохраняем все нарушения независимо от глобального кулдауна
        # DMS уже имеет свои кулдауны на уровне классов
        self.dms_frame_to_save = frame.copy()

        # Создаем информацию о нарушении
        self.dms_violation_info = {
            "timestamp": current_time,
            "video_timestamp": video_timestamp,
            "frame_count": self.frame_count,
            "violations": violations,
            "dms_stats": dms_res["stats"],
            "objects": dms_res["objects"]
        }

        # Немедленно пытаемся сохранить
        if self._save_dms_violation():
            self.dms_violation_count += 1
            self.stats["dms_violations"] += 1
            self.stats["dms_violations_saved"] += 1

            logging.info(
                f"=== DMS VIOLATION #{self.dms_violation_count} SAVED ===")
            logging.info(
                f"DMS violations total: {self.stats['dms_violations_saved']}")
        else:
            logging.error("Failed to save DMS violation!")

    def _save_dms_violation(self):
        """Сохранение DMS нарушения"""
        if self.dms_frame_to_save is None or self.dms_violation_info is None:
            return False

        try:
            # Создаем отображаемый кадр
            display_frame = self.dms_frame_to_save.copy()

            # Рисуем bounding boxes для DMS объектов
            for obj in self.dms_violation_info.get("objects", []):
                self.visualizer._draw_dms_object(display_frame, obj)

            # Добавляем информацию о нарушениях
            violations = self.dms_violation_info.get("violations", [])

            # Основной заголовок
            cv2.putText(
                display_frame,
                f"DMS VIOLATION #{self.dms_violation_count}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                self.visualizer.colors["red"],
                3,
            )

            # Перечисляем нарушения
            y_offset = 90
            for i, violation in enumerate(
                    violations[:3]):  # Показываем первые 3 нарушения
                message = violation.get('message', 'Unknown violation')
                cv2.putText(
                    display_frame,
                    f"{i + 1}. {message}",
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    self.visualizer.colors["yellow"],
                    2,
                )
                y_offset += 40

            # Сохраняем изображение
            img_filename = self.file_manager.generate_filename("dms_violation",
                                                               "jpg")
            img_path = self.file_manager.get_full_path(img_filename)
            success = cv2.imwrite(img_path, display_frame)

            if not success:
                logging.error(f"Failed to save DMS image: {img_path}")
                return False

            # Сохраняем отчет
            self._save_dms_report(self.dms_violation_info, img_filename)

            logging.info(f"DMS violation saved: {img_filename}")

            # Сбрасываем
            self.dms_frame_to_save = None
            self.dms_violation_info = None

            return True

        except Exception as e:
            logging.error(f"Error saving DMS violation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_dms_report(self, violation_info, image_filename):
        """Сохранение отчета о DMS нарушении"""
        try:
            report_filename = image_filename.replace(".jpg", ".txt")
            report_path = self.file_manager.get_full_path(report_filename)

            with open(report_path, "w", encoding="utf-8") as f:
                f.write("DMS VIOLATION REPORT\n")
                f.write("=" * 40 + "\n\n")

                f.write(f"Violation ID: {self.dms_violation_count}\n")
                f.write(
                    f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(violation_info.get('timestamp')))}\n")
                f.write(
                    f"Video Time: {violation_info.get('video_timestamp', 0):.1f}s\n")
                f.write(f"Frame: {violation_info.get('frame_count', 0)}\n\n")

                f.write("VIOLATIONS DETECTED:\n")
                f.write("-" * 40 + "\n")

                violations = violation_info.get('violations', [])
                for i, violation in enumerate(violations, 1):
                    f.write(
                        f"\n{i}. {violation.get('type', 'unknown').upper()} VIOLATION\n")
                    f.write(f"   Class: {violation.get('class', 'unknown')}\n")
                    f.write(
                        f"   Message: {violation.get('message', 'No message')}\n")
                    f.write(
                        f"   Duration: {violation.get('duration', 0):.1f}s\n")
                    f.write(
                        f"   Severity: {violation.get('severity', 'medium')}\n")

                f.write(f"\n" + "=" * 40 + "\n")
                f.write("SYSTEM STATUS:\n")

                stats = violation_info.get('dms_stats', {})
                f.write(f"Eye State: {stats.get('eye_state', 'unknown')}\n")
                f.write(
                    f"Seatbelt State: {stats.get('seatbelt_state', 'unknown')}\n")
                f.write(
                    f"Phone Detected: {stats.get('phone_detected', False)}\n")
                f.write(
                    f"Cigarette Detected: {stats.get('cigarette_detected', False)}\n")

                f.write(f"\nImage File: {image_filename}\n")
                f.write(f"Report File: {report_filename}\n")

            logging.info(f"DMS report saved: {report_filename}")

        except Exception as e:
            logging.error(f"Error saving DMS report: {e}")
            import traceback
            traceback.print_exc()