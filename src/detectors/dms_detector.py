# dms_detector.py
import cv2
import numpy as np
import time
import logging
import os
from collections import defaultdict
from ultralytics import YOLO
from .base_detector import BaseDetector


class DMSDetector(BaseDetector):
    """DMS детектор с правильной логикой нарушений"""

    def __init__(self,
                 model_path=None,
                 confidence_threshold=0.25,
                 eye_closed_threshold=5.0,  # 5 секунд для глаз
                 seatbelt_check_interval=15.0,  # Проверка каждые 15 секунд
                 phone_threshold=4.0,  # 4 секунды для телефона
                 cigarette_threshold=3.0,  # 3 секунды для сигареты
                 violation_cooldown=60.0,
                 # 60 секунд кулдаун для одного класса
                 min_object_area_ratio=0.0005,
                 iou_threshold=0.5):

        super().__init__("DMSDetector")

        # Настройки по требованиям
        self.confidence_threshold = confidence_threshold
        self.eye_closed_threshold = eye_closed_threshold
        self.seatbelt_check_interval = seatbelt_check_interval
        self.phone_threshold = phone_threshold
        self.cigarette_threshold = cigarette_threshold
        self.min_object_area_ratio = min_object_area_ratio
        self.iou_threshold = iou_threshold

        # Кулдауны для каждого класса нарушений
        self.class_cooldowns = {
            'eye_closed': violation_cooldown,
            'no_seatbelt': violation_cooldown,
            'phone': violation_cooldown,
            'cigarette': violation_cooldown
        }

        # Время последнего нарушения для каждого класса
        self.last_violation_times = {
            'eye_closed': 0,
            'no_seatbelt': 0,
            'phone': 0,
            'cigarette': 0
        }

        # Загружаем модель
        if model_path is None:
            model_path = self._find_model_file()

        logging.info(f"Loading DMS model from: {model_path}")

        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            logging.info(f"Model classes: {self.class_names}")
        except Exception as e:
            logging.error(f"Failed to load DMS model: {e}")
            self.model = None

        # Трекинг объектов для каждого класса
        self.tracked_objects = {
            'phone': {},  # {object_id: {start_time, last_seen, bbox}}
            'cigarette': {},  # {object_id: {start_time, last_seen, bbox}}
            'seatbelt': {},  # {object_id: {start_time, last_seen, bbox}}
            'eye': {}  # {object_id: {start_time, last_seen, is_closed}}
        }

        # Время для проверки ремня
        self.last_seatbelt_check_time = time.time()
        self.seatbelt_detected_in_interval = False

        # Время для глаз
        self.eye_closed_start_time = None
        self.last_eye_open_time = time.time()

        # Статистика
        self.stats = {
            "total_detections": 0,
            "detections_by_class": defaultdict(int),
            "eye_state": "unknown",
            "seatbelt_state": "unknown",
            "phone_detected": False,
            "cigarette_detected": False,
            "current_objects": [],
            "violation_count": 0
        }

        logging.info("DMSDetector initialized with proper violation logic")

    def detect(self, frame):
        """Основной метод детекции"""
        if self.model is None:
            return self._get_default_result()

        try:
            current_time = time.time()
            height, width = frame.shape[:2]
            frame_area = height * width

            # Детекция объектов
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                max_det=50
            )

            # Парсинг результатов
            detections = []
            eye_detections = {'open': 0, 'closed': 0}
            seatbelt_detected = False

            for r in results:
                if r.boxes is None:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.class_names.get(cls_id,
                                                      f"Unknown_{cls_id}")

                    # Проверка размера объекта
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / frame_area

                    if area_ratio < self.min_object_area_ratio:
                        continue

                    # Собираем информацию
                    obj_info = {
                        "bbox": [x1, y1, x2, y2],
                        "class": class_name,
                        "class_id": cls_id,
                        "confidence": conf,
                        "area_ratio": area_ratio,
                        "timestamp": current_time
                    }

                    detections.append(obj_info)

                    # Анализ по классам
                    if class_name == "Open Eye":
                        eye_detections['open'] += 1
                    elif class_name == "Closed Eye":
                        eye_detections['closed'] += 1
                    elif class_name == "Seatbelt":
                        seatbelt_detected = True
                    elif class_name == "Phone":
                        self._update_object_tracking('phone', obj_info,
                                                     current_time)
                    elif class_name == "Cigarette":
                        self._update_object_tracking('cigarette', obj_info,
                                                     current_time)

            # Обновление состояния глаз
            self._update_eye_state(eye_detections, current_time)

            # Обновление состояния ремня
            self._update_seatbelt_state(seatbelt_detected, current_time)

            # Проверка нарушений
            violations = self._check_all_violations(current_time)

            # Обновление статистики
            self._update_stats(detections, violations)

            return {
                "detected": len(detections) > 0,
                "objects": detections,
                "violations": violations,
                "current_violations": violations,  # Текущие нарушения
                "stats": self._get_stats_dict(),
                "eye_states": eye_detections,
                "frame_info": {"width": width, "height": height}
            }

        except Exception as e:
            logging.error(f"DMS detection error: {e}")
            return self._get_default_result()

    def _update_object_tracking(self, obj_class, obj_info, current_time):
        """Обновление трекинга объектов (телефон, сигарета)"""
        if obj_class not in ['phone', 'cigarette']:
            return

        # Ищем существующий объект по позиции
        obj_id = self._find_matching_object(obj_class, obj_info)

        if obj_id:
            # Обновляем существующий объект
            self.tracked_objects[obj_class][obj_id].update({
                "last_seen": current_time,
                "bbox": obj_info["bbox"],
                "confidence": obj_info["confidence"]
            })
        else:
            # Новый объект
            new_id = f"{obj_class}_{int(current_time * 1000)}"
            self.tracked_objects[obj_class][new_id] = {
                "start_time": current_time,
                "last_seen": current_time,
                "bbox": obj_info["bbox"],
                "confidence": obj_info["confidence"],
                "class": obj_info["class"]
            }
            logging.debug(f"New {obj_class} detected: {new_id}")

    def _find_matching_object(self, obj_class, new_obj):
        """Поиск существующего объекта по схожести"""
        tracked = self.tracked_objects[obj_class]
        if not tracked:
            return None

        new_bbox = new_obj["bbox"]
        new_center = self._get_bbox_center(new_bbox)

        for obj_id, obj_data in tracked.items():
            # Проверяем время (не старше 2 секунд)
            if new_obj["timestamp"] - obj_data["last_seen"] > 2.0:
                continue

            # Проверяем позицию
            old_center = self._get_bbox_center(obj_data["bbox"])
            distance = np.sqrt((new_center[0] - old_center[0]) ** 2 +
                               (new_center[1] - old_center[1]) ** 2)

            # Максимальное расстояние для сопоставления (20% от ширины кадра)
            max_distance = new_bbox[2] * 0.2

            if distance < max_distance:
                return obj_id

        return None

    def _get_bbox_center(self, bbox):
        """Получение центра bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _update_eye_state(self, eye_detections, current_time):
        """Обновление состояния глаз"""
        total_eyes = eye_detections['open'] + eye_detections['closed']

        if total_eyes > 0:
            if eye_detections['open'] > 0:
                # Есть открытые глаза
                self.stats["eye_state"] = "open"
                self.last_eye_open_time = current_time

                # Сбрасываем таймер закрытых глаз
                if self.eye_closed_start_time:
                    closed_duration = current_time - self.eye_closed_start_time
                    logging.debug(
                        f"Eyes opened after {closed_duration:.1f}s closed")
                    self.eye_closed_start_time = None

            elif eye_detections['closed'] > 0:
                # Только закрытые глаза
                self.stats["eye_state"] = "closed"

                # Запускаем таймер если еще не запущен
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = current_time
                    logging.debug("Eye closed timer started")
        else:
            # Глаза не обнаружены
            self.stats["eye_state"] = "unknown"
            self.eye_closed_start_time = None

    def _update_seatbelt_state(self, seatbelt_detected, current_time):
        """Обновление состояния ремня"""
        if seatbelt_detected:
            self.stats["seatbelt_state"] = "on"
            self.seatbelt_detected_in_interval = True
        else:
            self.stats["seatbelt_state"] = "off"

    def _check_all_violations(self, current_time):
        """Проверка всех типов нарушений"""
        violations = []

        # 1. Проверка глаз
        if (self.eye_closed_start_time and
                self.stats["eye_state"] == "closed"):

            eye_closed_duration = current_time - self.eye_closed_start_time

            if eye_closed_duration >= self.eye_closed_threshold:
                # Проверяем кулдаун
                if self._check_cooldown('eye_closed', current_time):
                    violation = {
                        "class": "eye_closed",
                        "type": "eye_closed",
                        "duration": eye_closed_duration,
                        "start_time": self.eye_closed_start_time,
                        "message": f"Driver fatigue! Eyes closed for {eye_closed_duration:.1f} seconds",
                        "severity": "high"
                    }
                    violations.append(violation)
                    self._record_violation('eye_closed', current_time)

        # 2. Проверка ремня (раз в 15 секунд)
        if current_time - self.last_seatbelt_check_time >= self.seatbelt_check_interval:
            self.last_seatbelt_check_time = current_time

            # Если за последние 15 секунд ремень не был обнаружен ни разу
            if not self.seatbelt_detected_in_interval:
                # Проверяем кулдаун
                if self._check_cooldown('no_seatbelt', current_time):
                    violation = {
                        "class": "no_seatbelt",
                        "type": "seatbelt_off",
                        "duration": self.seatbelt_check_interval,
                        "start_time": current_time - self.seatbelt_check_interval,
                        "message": "Seatbelt not fastened!",
                        "severity": "high"
                    }
                    violations.append(violation)
                    self._record_violation('no_seatbelt', current_time)

            # Сбрасываем флаг для следующего интервала
            self.seatbelt_detected_in_interval = False

        # 3. Проверка телефона
        phone_violation = self._check_object_violation('phone',
                                                       self.phone_threshold,
                                                       current_time)
        if phone_violation:
            violations.append(phone_violation)
            self._record_violation('phone', current_time)

        # 4. Проверка сигареты
        cigarette_violation = self._check_object_violation('cigarette',
                                                           self.cigarette_threshold,
                                                           current_time)
        if cigarette_violation:
            violations.append(cigarette_violation)
            self._record_violation('cigarette', current_time)

        # Очистка старых объектов
        self._cleanup_old_objects(current_time)

        return violations

    def _check_object_violation(self, obj_class, threshold, current_time):
        """Проверка нарушений для объектов (телефон, сигарета)"""
        tracked = self.tracked_objects[obj_class]

        if not tracked:
            logging.debug(f"No tracked {obj_class} objects")
            return None

        # Ищем объект с максимальной длительностью
        max_duration = 0
        violating_obj = None

        for obj_id, obj_data in tracked.items():
            duration = current_time - obj_data["start_time"]

            if duration > max_duration:
                max_duration = duration
                violating_obj = obj_data

        logging.debug(
            f"{obj_class}: max_duration={max_duration:.1f}s, threshold={threshold}s, cooldown_check={self._check_cooldown(obj_class, current_time)}")

        # Проверяем порог и кулдаун
        if max_duration >= threshold:
            if self._check_cooldown(obj_class, current_time):
                class_name = "Phone" if obj_class == 'phone' else "Cigarette"

                violation = {
                    "class": obj_class,
                    "type": f"{obj_class}_usage",
                    "duration": max_duration,
                    "start_time": violating_obj["start_time"],
                     "message": f"{class_name} usage detected for {max_duration:.1f} seconds",
                    "severity": "medium",
                    "object_data": violating_obj
                }

                logging.info(
                    f"{obj_class} VIOLATION DETECTED: {max_duration:.1f}s")
                return violation
            else:
                logging.debug(
                    f"{obj_class}: Cooldown active, skipping violation")
        else:
            pass
        return None


    def _check_cooldown(self, violation_class, current_time):
        """Проверка кулдауна для класса нарушений"""
        last_time = self.last_violation_times.get(violation_class, 0)
        cooldown = self.class_cooldowns.get(violation_class, 60.0)

        if last_time == 0:
            return True

        time_since_last = current_time - last_time
        if time_since_last >= cooldown:
            return True

        # Кулдаун активен - логируем только если близко к окончанию
        remaining = cooldown - time_since_last
        if remaining < 10:  # Только последние 10 секунд
            logging.debug(f"{violation_class}: Cooldown active, {remaining:.1f}s left")
        return False

    def _record_violation(self, violation_class, current_time):
        """Запись времени нарушения"""
        self.last_violation_times[violation_class] = current_time
        self.stats["violation_count"] += 1
        logging.info(f"DMS violation recorded: {violation_class}")

    def _cleanup_old_objects(self, current_time):
        """Очистка старых объектов из трекинга"""
        max_age = 10.0  # Максимальный возраст объекта в секундах

        for obj_class in ['phone', 'cigarette']:
            to_remove = []

            for obj_id, obj_data in self.tracked_objects[obj_class].items():
                if current_time - obj_data["last_seen"] > max_age:
                    to_remove.append(obj_id)

            for obj_id in to_remove:
                del self.tracked_objects[obj_class][obj_id]

    def _update_stats(self, detections, violations):
        """Обновление статистики"""
        self.stats["total_detections"] += len(detections)

        for obj in detections:
            self.stats["detections_by_class"][obj["class"]] += 1

        # Текущие объекты
        self.stats["current_objects"] = [
            {"class": obj["class"], "confidence": obj["confidence"]}
            for obj in detections
        ]

        # Флаги обнаружения
        self.stats["phone_detected"] = any(
            obj["class"] == "Phone" for obj in detections)
        self.stats["cigarette_detected"] = any(
            obj["class"] == "Cigarette" for obj in detections)

    def _get_stats_dict(self):
        """Получение статистики в виде словаря"""
        return {
            "total_detections": self.stats["total_detections"],
            "detections_by_class": dict(self.stats["detections_by_class"]),
            "eye_state": self.stats["eye_state"],
            "seatbelt_state": self.stats["seatbelt_state"],
            "phone_detected": self.stats["phone_detected"],
            "cigarette_detected": self.stats["cigarette_detected"],
            "current_objects": self.stats["current_objects"],
            "violation_count": self.stats["violation_count"]
        }

    def _get_default_result(self):
        """Результат по умолчанию"""
        return {
            "detected": False,
            "objects": [],
            "violations": [],
            "current_violations": [],
            "stats": self._get_stats_dict()
        }

    def _find_model_file(self):
        """Поиск файла модели"""
        model_filename = "dms_model.pt"
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        possible_paths = [
            os.path.join(current_file_dir, model_filename),
            os.path.join(current_file_dir, "models", model_filename),
            os.path.join(current_file_dir, "..", "utils", "models",
                         model_filename),
            os.path.join(os.getcwd(), model_filename),
            os.path.join(os.getcwd(), "src", "utils", "models",
                         model_filename),
            model_filename
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path

        return model_filename