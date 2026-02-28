"""Файл: src/detectors/forbidden_items_detector.py
Тип: детектор нарушений.
Назначение: получает кадр, применяет алгоритм детекции и возвращает структурированный результат.
Связи: детекторы вызываются менеджером детекции и передают данные в слой сохранения/визуализации.
Импортируемые внутренние модули:
- .base_detector: используется для передачи данных или вызова связанной логики."""
import time
import logging
from collections import defaultdict

from .base_detector import BaseDetector
from ..inference.model_path_resolver import resolve_model_path


class ForbiddenItemsDetector(BaseDetector):
    """Класс: ForbiddenItemsDetector
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `FORBIDDEN_CLASSES` (`dict`): словарь запрещенных классов и их `class_id` для фильтрации.
- `class_names` (`Any`): словарь соответствия `class_id -> человекочитаемое имя класса`.
- `class_specific_cooldown` (`Any`): флаг применения отдельного кулдауна для каждого класса.
- `confidence_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `current_detections` (`list`): данные обнаруженного объекта или его служебного представления.
- `detected_objects` (`dict`): данные обнаруженного объекта или его служебного представления.
- `hub` (`Any`): общий объект инференса, который кэширует модели и выполняет predict.
- `imgsz` (`Any`): размер входного изображения для инференса модели.
- `iou_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `last_violation_times` (`dict`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `max_det` (`Any`): максимальное число детекций, возвращаемых моделью на кадр.
- `max_object_age` (`Any`): максимальный возраст трека объекта без обновления перед удалением.
- `min_detection_duration` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `min_object_area_ratio` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `model_key` (`str`): логический ключ модели в InferenceHub для повторного использования экземпляра.
- `model_path` (`Any`): путь к файлу весов модели на диске.
Ключевые методы:
- `__init__()`, `detect()`, `postprocess_shared_results()`, `_update_object_tracking()`, `_find_matching_object()`, `_get_bbox_center()`, `_check_violations()`, `_record_violation()`, `_check_class_cooldown()`"""

    FORBIDDEN_CLASSES = {
        'banana': 46, 'apple': 47, 'orange': 49, 'sandwich': 48, 'carrot': 51,
        'broccoli': 50, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55,
        'bottle': 39, 'wine glass': 40, 'cup': 41,
        'cell phone': 67,
        'book': 73, 'laptop': 63
    }

    def __init__(
        self,
        hub,
        model_name=None,
        model_key: str = "yolo_main",
        confidence_threshold: float = 0.5,
        min_detection_duration: float = 3.0,
        violation_cooldown: float = 60.0,
        min_object_area_ratio: float = 0.01,
        max_object_age: float = 2.0,
        class_specific_cooldown: bool = True,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        max_det: int = 50,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `hub` (`Any`): общий объект инференса, который кэширует модели и выполняет predict.
- `model_name` (`Any`): читаемое имя/идентификатор модели из конфигурации.
- `model_key` (`str`): логический ключ модели в InferenceHub для повторного использования экземпляра.
- `confidence_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `min_detection_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `violation_cooldown` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `min_object_area_ratio` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `max_object_age` (`float`): максимальный возраст трека объекта без обновления перед удалением.
- `class_specific_cooldown` (`bool`): флаг применения отдельного кулдауна для каждого класса.
- `iou_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `max_det` (`int`): максимальное число детекций, возвращаемых моделью на кадр.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        super().__init__("ForbiddenItemsDetector")

        self.hub = hub
        self.model_key = model_key
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.confidence_threshold = float(confidence_threshold)
        self.min_detection_duration = float(min_detection_duration)
        self.violation_cooldown = float(violation_cooldown)
        self.min_object_area_ratio = float(min_object_area_ratio)
        self.max_object_age = float(max_object_age)
        self.class_specific_cooldown = bool(class_specific_cooldown)

        self.model_path = resolve_model_path(model_name)
        logging.info(f"ForbiddenItemsDetector will use shared model: {self.model_path}")

        try:
            model = self.hub.get_model(self.model_key, self.model_path)
            self.class_names = model.names
        except Exception:
            self.class_names = {}

        self.detected_objects = {}
        self.violation_count = 0
        self.last_violation_times = {}

        self.current_detections = []
        self.stats = {
            "total_violations": 0,
            "violations_by_class": defaultdict(int),
            "last_violation_time": None
        }

        logging.info(f"ForbiddenItemsDetector initialized with {len(self.FORBIDDEN_CLASSES)} forbidden classes")
        logging.info(f"Class-specific cooldown: {self.class_specific_cooldown}")

    def detect(self, frame, frame_id: int = None):
        """Функция: detect()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_id` (`int`): уникальный индекс кадра, используется для кэша и синхронизации этапов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            results = self.hub.predict(
                model_key=self.model_key,
                weights_path=self.model_path,
                frame_bgr=frame,
                frame_id=frame_id,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False,
            )
            return self.postprocess_shared_results(frame, results)

        except Exception as e:
            logging.error(f"ForbiddenItemsDetector error: {e}")
            return {
                "detected": False,
                "objects": [],
                "current_violation": False,
                "violation_info": None,
                "stats": self.stats
            }

    def postprocess_shared_results(self, frame, raw_results):
        """Функция: postprocess_shared_results()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `raw_results` (`Any`): сырые выходные данные модели до доменной постобработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            current_time = time.time()
            results = raw_results
            self.current_detections = []
            forbidden_objects = []
            frame_area = frame.shape[0] * frame.shape[1]

            names = self.class_names
            if not names:
                try:
                    names = self.hub.get_model(self.model_key, self.model_path).names
                except Exception:
                    names = {}

            for r in results:
                if r.boxes is None:
                    continue

                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                for box, cls_id, conf in zip(boxes_xyxy, cls_ids, confs):
                    class_name = names.get(int(cls_id), str(cls_id))
                    if class_name not in self.FORBIDDEN_CLASSES:
                        continue

                    x1, y1, x2, y2 = box.astype(int)
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    area_ratio = area / frame_area if frame_area > 0 else 0.0
                    if area_ratio < self.min_object_area_ratio:
                        continue

                    obj = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "class": class_name,
                        "confidence": float(conf),
                        "area_ratio": float(area_ratio),
                        "timestamp": current_time
                    }
                    forbidden_objects.append(obj)
                    self.current_detections.append(obj)

            self._update_object_tracking(forbidden_objects, current_time)
            violation_detected, violation_info = self._check_violations(current_time)
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
                    "violations_by_class": dict(self.stats["violations_by_class"]),
                    "active_cooldowns": cooldown_info,
                }
            }

        except Exception as e:
            logging.error(f"ForbiddenItemsDetector shared postprocess error: {e}")
            return {
                "detected": False,
                "objects": [],
                "current_violation": False,
                "violation_info": None,
                "stats": self.stats
            }

    def _update_object_tracking(self, objects, current_time):
        """Функция: _update_object_tracking()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `objects` (`Any`): список обнаруженных объектов в кадре.
- `current_time` (`Any`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        tracked_ids = list(self.detected_objects.keys())
        for obj_id in tracked_ids:
            obj_data = self.detected_objects[obj_id]
            if "total_duration" not in obj_data:
                obj_data["total_duration"] = 0
            obj_data["total_duration"] = current_time - obj_data["start_time"]

        for obj in objects:
            matched_obj_id = self._find_matching_object(obj, current_time)
            if matched_obj_id is not None:
                self.detected_objects[matched_obj_id].update({
                    "last_seen": current_time,
                    "bbox": obj["bbox"],
                    "total_duration": current_time - self.detected_objects[matched_obj_id]["start_time"]
                })
            else:
                obj_id = f"{obj['class']}_{int(current_time * 1000)}_{len(self.detected_objects)}"
                self.detected_objects[obj_id] = {
                    "class": obj["class"],
                    "bbox": obj["bbox"],
                    "start_time": current_time,
                    "last_seen": current_time,
                    "total_duration": 0,
                    "confidence": obj["confidence"]
                }

    def _find_matching_object(self, new_obj, current_time):
        """Функция: _find_matching_object()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `new_obj` (`Any`): данные обнаруженного объекта или его служебного представления.
- `current_time` (`Any`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.detected_objects:
            return None

        new_bbox = new_obj["bbox"]
        new_class = new_obj["class"]
        new_center = self._get_bbox_center(new_bbox)

        best_match = None
        best_distance = float('inf')

        for obj_id, obj_data in self.detected_objects.items():
            if obj_data["class"] != new_class:
                continue
            if current_time - obj_data["last_seen"] > 1.0:
                continue

            old_center = self._get_bbox_center(obj_data["bbox"])
            distance = ((new_center[0] - old_center[0]) ** 2 + (new_center[1] - old_center[1]) ** 2) ** 0.5
            max_distance = new_bbox[2] * 0.1

            if distance < max_distance and distance < best_distance:
                best_distance = distance
                best_match = obj_id

        return best_match

    def _get_bbox_center(self, bbox):
        """Функция: _get_bbox_center()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `bbox` (`Any`): ограничивающий прямоугольник объекта в формате `[x1, y1, x2, y2]`.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _check_violations(self, current_time):
        """Функция: _check_violations()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `current_time` (`Any`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        violation_objects = []
        valid_objects = []

        objects_by_class = defaultdict(list)
        for obj_id, obj_data in self.detected_objects.items():
            objects_by_class[obj_data["class"]].append((obj_id, obj_data))

        for class_name, class_objects in objects_by_class.items():
            max_duration_obj = None
            max_duration = 0

            for obj_id, obj_data in class_objects:
                total_duration = current_time - obj_data["start_time"]
                if total_duration > max_duration:
                    max_duration = total_duration
                    max_duration_obj = (obj_id, obj_data, total_duration)

            if max_duration_obj and max_duration >= self.min_detection_duration:
                obj_id, obj_data, total_duration = max_duration_obj

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

        if violation_objects:
            self._record_violation(violation_objects, current_time)
            for obj_data in valid_objects:
                self.last_violation_times[obj_data["class"]] = current_time
            return True, self._create_violation_info(violation_objects, current_time)

        return False, None

    def _record_violation(self, violation_objects, current_time):
        """Функция: _record_violation()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_objects` (`Any`): объекты, из-за которых сформировано событие нарушения.
- `current_time` (`Any`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        self.violation_count += 1
        self.stats["total_violations"] += 1
        for obj in violation_objects:
            self.stats["violations_by_class"][obj["class"]] += 1
        self.stats["last_violation_time"] = current_time

    def _check_class_cooldown(self, class_name, current_time):
        """Функция: _check_class_cooldown()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `class_name` (`Any`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `current_time` (`Any`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.class_specific_cooldown:
            return True
        last_time = self.last_violation_times.get(class_name, 0)
        if last_time == 0:
            return True
        return (current_time - last_time) >= self.violation_cooldown

    def get_cooldown_info(self):
        """Функция: get_cooldown_info()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        current_time = time.time()
        cooldown_info = {}
        for class_name, last_time in self.last_violation_times.items():
            time_since_last = current_time - last_time
            if time_since_last < self.violation_cooldown:
                cooldown_info[class_name] = {
                    "cooldown_left": self.violation_cooldown - time_since_last,
                    "last_violation": last_time
                }
        return cooldown_info

    def _create_violation_info(self, violation_objects, current_time):
        """Функция: _create_violation_info()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_objects` (`Any`): объекты, из-за которых сформировано событие нарушения.
- `current_time` (`Any`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if not violation_objects:
            return None
        classes = list(set([obj["class"] for obj in violation_objects]))
        return {
            "violation_id": self.violation_count,
            "timestamp": current_time,
            "objects": violation_objects.copy(),
            "cooldown_remaining": self.violation_cooldown,
            "affected_classes": classes,
            "is_class_specific": self.class_specific_cooldown
        }
