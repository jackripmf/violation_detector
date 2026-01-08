import cv2
import os
import numpy as np
import time
import logging
from collections import defaultdict
from ultralytics import YOLO
from .base_detector import BaseDetector


class ForbiddenItemsDetector(BaseDetector):
    """Детектор запрещенных объектов (еда, напитки, телефоны)"""

    # Классы из COCO dataset, которые считаем запрещенными
    FORBIDDEN_CLASSES = {
        # Еда
        'banana': 46, 'apple': 47, 'orange': 49, 'sandwich': 48, 'carrot': 51,
        'broccoli': 50, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55,
        # Напитки
        'bottle': 39, 'wine glass': 40, 'cup': 41,
        # Телефоны
        'cell phone': 67,
        # Дополнительно
        'book': 73, 'laptop': 63
    }

    def __init__(self,
                 model_name=None,  # Изменено с "yolov8n.pt" на None
                 confidence_threshold=0.5,
                 min_detection_duration=3.0,
                 violation_cooldown=60.0,
                 min_object_area_ratio=0.01,
                 max_object_age=2.0,
                 class_specific_cooldown=True):

        super().__init__("ForbiddenItemsDetector")

        self.confidence_threshold = confidence_threshold
        self.min_detection_duration = min_detection_duration
        self.violation_cooldown = violation_cooldown
        self.min_object_area_ratio = min_object_area_ratio
        self.max_object_age = max_object_age
        self.class_specific_cooldown = class_specific_cooldown

        # Определяем путь к модели
        if model_name is None:
            model_name = self._find_yolo_model()

        logging.info(f"Loading forbidden items model from: {model_name}")

        try:
            self.yolo_model = YOLO(model_name)
            logging.info(f"✓ Forbidden items model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            # Пробуем стандартную модель
            try:
                self.yolo_model = YOLO("yolov8n.pt")
                logging.info("✓ Loaded default YOLOv8n model")
            except Exception as e2:
                logging.error(f"Failed to load any model: {e2}")
                self.yolo_model = None

        # Трекинг объектов
        self.detected_objects = {}  # {object_id: {class, start_time, last_seen, bbox, total_duration}}
        self.violation_count = 0

        # ПЕРКЛАССОВЫЙ кулдаун: {class_name: last_violation_time}
        self.last_violation_times = {}

        # Статистика
        self.current_detections = []
        self.stats = {
            "total_violations": 0,
            "violations_by_class": defaultdict(int),
            "last_violation_time": None
        }

        # Установка пути к модели
        if model_name == "yolov8n.pt":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_name = os.path.join(current_dir, "..", "utils", "models",
                                      "yolov8n.pt")

        logging.info(
            f"ForbiddenItemsDetector initialized with {len(self.FORBIDDEN_CLASSES)} forbidden classes")
        logging.info(f"Class-specific cooldown: {class_specific_cooldown}")

    def _find_yolo_model(self):
        """Поиск файла модели YOLO"""
        model_filename = "yolov8n.pt"

        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        possible_paths = [
            os.path.join(current_file_dir, model_filename),
            os.path.join(current_file_dir, "models", model_filename),
            os.path.join(current_file_dir, "..", "utils", "models",
                         model_filename),
            os.path.join(current_file_dir, "..", "..", "..", "src", "utils",
                         "models", model_filename),
            os.path.join(os.getcwd(), model_filename),
            os.path.join(os.getcwd(), "src", "utils", "models",
                         model_filename),
            "yolov8n.pt"  # Скачает автоматически
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                logging.info(f"Found model at: {abs_path}")
                return abs_path

        logging.info("Model not found locally, will download automatically")
        return "yolov8n.pt"

    def detect(self, frame):
        """Детекция запрещенных объектов в кадре"""
        try:
            current_time = time.time()

            # Получаем детекции от YOLO
            results = self.yolo_model(frame, conf=self.confidence_threshold,
                                      verbose=False)

            # Сбрасываем текущие детекции
            self.current_detections = []

            # Анализируем результаты
            forbidden_objects = []
            frame_area = frame.shape[0] * frame.shape[1]

            for r in results:
                if r.boxes is None:
                    continue

                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(r.boxes.cls.cpu().numpy()[i])
                    confidence = float(r.boxes.conf.cpu().numpy()[i])

                    # Проверяем, является ли класс запрещенным
                    class_name = self.yolo_model.names[class_id]
                    if class_name not in self.FORBIDDEN_CLASSES:
                        continue

                    # Проверяем площадь объекта
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / frame_area

                    if area_ratio < self.min_object_area_ratio:
                        continue

                    # Создаем объект детекции
                    obj = {
                        "bbox": [x1, y1, x2, y2],
                        "class": class_name,
                        "confidence": confidence,
                        "area_ratio": area_ratio,
                        "timestamp": current_time
                    }

                    forbidden_objects.append(obj)
                    self.current_detections.append(obj)

            # Обновляем трекинг объектов
            self._update_object_tracking(forbidden_objects, current_time)

            # Проверяем нарушения
            violation_detected, violation_info = self._check_violations(
                current_time)

            # Получаем информацию о кулдаунах
            cooldown_info = self.get_cooldown_info()

            return {
                "detected": len(forbidden_objects) > 0,
                "objects": forbidden_objects,
                "current_violation": violation_detected,
                "violation_info": violation_info,
                "stats": {
                    "total_violations": self.stats["total_violations"],
                    "current_objects": len(forbidden_objects),
                    "tracked_objects": len(self.detected_objects),
                    "violations_by_class": dict(
                        self.stats["violations_by_class"]),
                    "active_cooldowns": cooldown_info
                    # Добавляем информацию о кулдаунах
                }
            }

        except Exception as e:
            logging.error(f"ForbiddenItemsDetector error: {e}")
            return {
                "detected": False,
                "objects": [],
                "current_violation": False,
                "violation_info": None,
                "stats": self.stats
            }

    def _update_object_tracking(self, objects, current_time):
        """Обновление трекинга объектов с учетом сходства"""
        # Создаем список отслеживаемых ID
        tracked_ids = list(self.detected_objects.keys())

        # Для каждого существующего объекта проверяем, не устарел ли он
        for obj_id in tracked_ids:
            obj_data = self.detected_objects[obj_id]

            logging.debug(
                f"Removing old object: {obj_data['class']} (age: {current_time - obj_data['last_seen']:.1f}s)")

            # Увеличиваем общую длительность детекции
            if "total_duration" not in obj_data:
                obj_data["total_duration"] = 0
            obj_data["total_duration"] = current_time - obj_data["start_time"]

        # Добавляем/обновляем текущие объекты
        for obj in objects:
            # Пытаемся найти существующий объект по схожести (класс + позиция)
            matched_obj_id = self._find_matching_object(obj, current_time)

            if matched_obj_id is not None:
                # Обновляем существующий объект
                self.detected_objects[matched_obj_id].update({
                    "last_seen": current_time,
                    "bbox": obj["bbox"],
                    "total_duration": current_time -
                                      self.detected_objects[matched_obj_id][
                                          "start_time"]
                })
            else:
                # Новый объект
                obj_id = f"{obj['class']}_{int(current_time * 1000)}_{len(self.detected_objects)}"
                self.detected_objects[obj_id] = {
                    "class": obj["class"],
                    "bbox": obj["bbox"],
                    "start_time": current_time,
                    "last_seen": current_time,
                    "total_duration": 0,
                    "confidence": obj["confidence"]
                }
                logging.debug(f"New forbidden object: {obj['class']}")

    def _find_matching_object(self, new_obj, current_time):
        """Находит существующий объект по схожести"""
        if not self.detected_objects:
            return None

        new_bbox = new_obj["bbox"]
        new_class = new_obj["class"]
        new_center = self._get_bbox_center(new_bbox)

        best_match = None
        best_distance = float('inf')

        for obj_id, obj_data in self.detected_objects.items():
            # Проверяем класс
            if obj_data["class"] != new_class:
                continue

            # Проверяем время (не обновлять слишком старые объекты)
            if current_time - obj_data["last_seen"] > 1.0:  # 1 секунда макс
                continue

            # Проверяем позицию
            old_bbox = obj_data["bbox"]
            old_center = self._get_bbox_center(old_bbox)

            # Расстояние между центрами
            distance = ((new_center[0] - old_center[0]) ** 2 +
                        (new_center[1] - old_center[1]) ** 2) ** 0.5

            # Максимальное расстояние для сопоставления (10% от ширины кадра)
            max_distance = new_obj.get("bbox", [0, 0, 640, 480])[2] * 0.1

            if distance < max_distance and distance < best_distance:
                best_distance = distance
                best_match = obj_id

        return best_match

    def _get_bbox_center(self, bbox):
        """Получает центр bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _check_violations(self, current_time):
        """Проверка нарушений с перклассовым кулдауном"""
        violation_objects = []
        valid_objects = []  # Объекты, которые прошли кулдаун

        # Группируем объекты по классам
        objects_by_class = defaultdict(list)
        for obj_id, obj_data in self.detected_objects.items():
            class_name = obj_data["class"]
            objects_by_class[class_name].append((obj_id, obj_data))

        # Проверяем каждый класс отдельно
        for class_name, class_objects in objects_by_class.items():
            # Находим объект с максимальной длительностью в этом классе
            max_duration_obj = None
            max_duration = 0

            for obj_id, obj_data in class_objects:
                total_duration = current_time - obj_data["start_time"]

                if total_duration > max_duration:
                    max_duration = total_duration
                    max_duration_obj = (obj_id, obj_data, total_duration)

            # Если есть объект с достаточной длительностью
            if max_duration_obj and max_duration >= self.min_detection_duration:
                obj_id, obj_data, total_duration = max_duration_obj

                # Проверяем кулдаун для этого класса
                if self._check_class_cooldown(class_name, current_time):
                    violation_objects.append({
                        "class": class_name,
                        "duration": total_duration,
                        "bbox": obj_data["bbox"],
                        "object_id": obj_id,
                        "confidence": obj_data.get("confidence", 0.5),
                        "start_time": obj_data["start_time"],
                        "last_seen": obj_data["last_seen"]
                    })
                    valid_objects.append(obj_data)
                else:
                    # Объект есть, но кулдаун активен
                    last_time = self.last_violation_times.get(class_name, 0)
                    cooldown_left = self.violation_cooldown - (
                                current_time - last_time)
                    logging.info(
                        logging.debug(f"Cooldown for {class_name}: {cooldown_left:.1f}s left"))

        if violation_objects:
            # Фиксируем нарушение
            self._record_violation(violation_objects, current_time)

            # Обновляем время последнего нарушения для каждого класса
            for obj_data in valid_objects:
                class_name = obj_data["class"]
                self.last_violation_times[class_name] = current_time

            # НЕ удаляем объекты! Они могут продолжать существовать
            # Это позволяет фиксировать тот же объект повторно после кулдауна

            logging.info(
                f"Forbidden items violation triggered with {len(violation_objects)} objects!")
            logging.info(
                f"Keeping {len(self.detected_objects)} tracked objects (not removing)")
            return True, self._create_violation_info(violation_objects,
                                                     current_time)

        return False, None

    def _record_violation(self, violation_objects, current_time):
        """Запись нарушения"""
        self.violation_count += 1
        self.stats["total_violations"] += 1

        for obj in violation_objects:
            self.stats["violations_by_class"][obj["class"]] += 1

        self.stats["last_violation_time"] = current_time

        obj_names = [obj['class'] for obj in violation_objects]
        durations = [obj['duration'] for obj in violation_objects]

        logging.info(
            f"=== FORBIDDEN ITEMS VIOLATION #{self.violation_count} ===")
        logging.info(f"Objects: {obj_names}")
        logging.info(f"Durations: {[f'{d:.1f}s' for d in durations]}")
        logging.info(f"Total detection time: {sum(durations):.1f}s")


    def get_stats(self):
        """Получение статистики"""
        return {
            "total_violations": self.stats["total_violations"],
            "violations_by_class": dict(self.stats["violations_by_class"]),
            "last_violation": self.stats["last_violation_time"],
            "current_objects": len(self.current_detections)
        }

    def _check_class_cooldown(self, class_name, current_time):
        """Проверка кулдауна для конкретного класса"""
        if not self.class_specific_cooldown:
            # Старая логика: общий кулдаун
            return True

        last_time = self.last_violation_times.get(class_name, 0)

        # Если никогда не было нарушения для этого класса - ок
        if last_time == 0:
            logging.info(f"No cooldown for {class_name} (first violation)")
            return True

        # Проверяем прошло ли достаточно времени
        time_since_last = current_time - last_time
        if time_since_last < self.violation_cooldown:
            cooldown_left = self.violation_cooldown - time_since_last
            logging.info(
                f"Cooldown active for {class_name}: {cooldown_left:.1f}s left")
            return False

        logging.info(
            f"Cooldown passed for {class_name}: {time_since_last:.1f}s since last")
        return True

    def get_cooldown_info(self):
        """Получение информации о текущих кулдаунах"""
        current_time = time.time()
        cooldown_info = {}

        for class_name, last_time in self.last_violation_times.items():
            time_since_last = current_time - last_time
            if time_since_last < self.violation_cooldown:
                cooldown_left = self.violation_cooldown - time_since_last
                cooldown_info[class_name] = {
                    "cooldown_left": cooldown_left,
                    "last_violation": last_time
                }

        return cooldown_info


    def _create_violation_info(self, violation_objects, current_time):
        """Создание информации о нарушении"""
        if not violation_objects:
            return None

        # Группируем по классам для кулдауна
        classes = list(set([obj["class"] for obj in violation_objects]))

        return {
            "violation_id": self.violation_count,
            "timestamp": current_time,
            "objects": violation_objects.copy(),
            "cooldown_remaining": self.violation_cooldown,
            "affected_classes": classes,
            "is_class_specific": self.class_specific_cooldown
        }