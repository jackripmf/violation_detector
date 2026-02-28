"""Файл: src/detectors/dms_detector.py
Тип: детектор нарушений.
Назначение: получает кадр, применяет алгоритм детекции и возвращает структурированный результат.
Связи: детекторы вызываются менеджером детекции и передают данные в слой сохранения/визуализации.
Импортируемые внутренние модули:
- .base_detector: используется для передачи данных или вызова связанной логики."""
import time
import logging
from collections import defaultdict
from typing import Optional, Dict, Any, List

import numpy as np
from .base_detector import BaseDetector
from ..inference.model_path_resolver import resolve_model_path


class DMSDetector(BaseDetector):
    """Класс: DMSDetector
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `_allowed_ids_logged_once` (`bool`): флаг одноразового логирования списка разрешенных class-id.
- `allowed_cls_ids` (`Any`): набор class-id, которые разрешены для обработки детектором.
- `cigarette_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `class_cooldowns` (`dict`): настройка кулдаунов по типам нарушений (секунды на класс).
- `class_names` (`Any`): словарь соответствия `class_id -> человекочитаемое имя класса`.
- `cls_cigarette` (`int`): class-id сигареты в модели DMS.
- `cls_closed_eye` (`int`): class-id закрытого глаза в модели DMS.
- `cls_open_eye` (`int`): class-id открытого глаза в модели DMS.
- `cls_seatbelt` (`int`): class-id ремня безопасности в модели DMS.
- `confidence_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `eye_closed_start_time` (`None`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `eye_closed_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `hub` (`Any`): общий объект инференса, который кэширует модели и выполняет predict.
- `iou_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `last_eye_open_time` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `last_seatbelt_check_time` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Ключевые методы:
- `__init__()`, `detect()`, `postprocess_shared_results()`, `_resolve_dms_class()`, `_update_object_tracking()`, `_find_matching_object()`, `_get_bbox_center()`, `_update_eye_state()`, `_update_seatbelt_state()`, `_check_all_violations()`"""
    _allowed_ids_logged_once = False

    def __init__(
        self,
        hub,
        model_path: Optional[str] = None,
        model_key: str = "yolo_main",
        confidence_threshold: float = 0.25,
        eye_closed_threshold: float = 5.0,
        seatbelt_check_interval: float = 15.0,
        phone_threshold: float = 4.0,
        cigarette_threshold: float = 3.0,
        violation_cooldown: float = 60.0,
        min_object_area_ratio: float = 0.005,
        iou_threshold: float = 0.5,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `hub` (`Any`): общий объект инференса, который кэширует модели и выполняет predict.
- `model_path` (`Optional[str]`): путь к файлу весов модели на диске.
- `model_key` (`str`): логический ключ модели в InferenceHub для повторного использования экземпляра.
- `confidence_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `eye_closed_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `seatbelt_check_interval` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `phone_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `cigarette_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `violation_cooldown` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `min_object_area_ratio` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `iou_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        super().__init__("DMSDetector")

        self.hub = hub
        self.model_key = model_key

        # настройки
        self.confidence_threshold = float(confidence_threshold)
        self.eye_closed_threshold = float(eye_closed_threshold)
        self.seatbelt_check_interval = float(seatbelt_check_interval)
        self.phone_threshold = float(phone_threshold)
        self.cigarette_threshold = float(cigarette_threshold)
        self.min_object_area_ratio = float(min_object_area_ratio)
        self.iou_threshold = float(iou_threshold)

        # кулдауны
        self.class_cooldowns = {
            "eye_closed": float(violation_cooldown),
            "no_seatbelt": float(violation_cooldown),
            "phone": float(violation_cooldown),
            "cigarette": float(violation_cooldown),
        }
        self.last_violation_times = {
            "eye_closed": 0.0,
            "no_seatbelt": 0.0,
            "phone": 0.0,
            "cigarette": 0.0,
        }

        # путь к общей модели
        self.model_path = resolve_model_path(model_path)
        logging.info(f"[dms:model] shared_model={self.model_path}")
        try:
            model = self.hub.get_model(self.model_key, self.model_path)
            self.class_names = model.names
        except Exception:
            self.class_names = {}

        # --------------------------
        # ЖЁСТКО: class_id из обученной модели
        # --------------------------
        self.cls_open_eye = 80
        self.cls_closed_eye = 81
        self.cls_cigarette = 82
        self.cls_seatbelt = 84

        # phone объединяем (COCO phone=67 и DMS phone=83)
        self.phone_ids = {67, 83}

        self.allowed_cls_ids = {
            int(self.cls_open_eye),
            int(self.cls_closed_eye),
            int(self.cls_cigarette),
            int(self.cls_seatbelt),
            *{int(x) for x in self.phone_ids},
        }

        if not DMSDetector._allowed_ids_logged_once:
            logging.info(
                f"[dms:allowed_classes] open_eye={self.cls_open_eye} "
                f"closed_eye={self.cls_closed_eye} seatbelt={self.cls_seatbelt} "
                f"cigarette={self.cls_cigarette} phone={sorted(self.phone_ids)}"
            )
            DMSDetector._allowed_ids_logged_once = True

        self.tracked_objects = {
            "phone": {},      # {object_id: {start_time, last_seen, bbox, confidence, class_id, class}}
            "cigarette": {},  # same
        }

        # ремень
        self.last_seatbelt_check_time = time.time()
        self.seatbelt_detected_in_interval = False

        # глаза
        self.eye_closed_start_time = None
        self.last_eye_open_time = time.time()

        # статистика
        self.stats = {
            "total_detections": 0,
            "detections_by_class": defaultdict(int),
            "eye_state": "unknown",
            "seatbelt_state": "unknown",
            "phone_detected": False,
            "cigarette_detected": False,
            "current_objects": [],
            "violation_count": 0,
        }

        logging.info("[dms:init] detector initialized (full-frame, shared model)")


    def detect(self, frame, frame_id: int = None):
        """Функция: detect()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_id` (`int`): уникальный индекс кадра, используется для кэша и синхронизации этапов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            allowed_ids = sorted(self.allowed_cls_ids)

            results = self.hub.predict(
                model_key=self.model_key,
                weights_path=self.model_path,
                frame_bgr=frame,
                frame_id=frame_id,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=50,
                imgsz=None,
                verbose=False,
                classes=allowed_ids,
                cache_tag="dms_full_frame",
            )
            return self.postprocess_shared_results(frame, results, frame_id=frame_id)

        except Exception as e:
            logging.error(f"[dms:detect_failed] error={e}")
            return self._get_default_result()

    def postprocess_shared_results(self, frame, raw_results, frame_id: int = None):
        """Функция: postprocess_shared_results()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `raw_results` (`Any`): сырые выходные данные модели до доменной постобработки.
- `frame_id` (`int`): уникальный индекс кадра, используется для кэша и синхронизации этапов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            current_time = time.time()
            height, width = frame.shape[:2]
            frame_area = float(height * width)
            results = raw_results
            detections = []
            eye_detections = {"open": 0, "closed": 0}
            seatbelt_detected = False

            for r in results:
                if r.boxes is None:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    cid = int(cls_id)
                    if cid not in self.allowed_cls_ids:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    area_ratio = (area / frame_area) if frame_area > 0 else 0.0
                    if area_ratio < self.min_object_area_ratio:
                        continue

                    class_name, dms_kind = self._resolve_dms_class(cid)

                    obj = {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(conf),
                        "class_id": cid,
                        "class": class_name,
                        "dms_kind": dms_kind,
                        "area_ratio": float(area_ratio),
                        "timestamp": current_time,
                    }
                    detections.append(obj)

                    if dms_kind == "open_eye":
                        eye_detections["open"] += 1
                    elif dms_kind == "closed_eye":
                        eye_detections["closed"] += 1
                    elif dms_kind == "seatbelt":
                        seatbelt_detected = True
                    elif dms_kind == "phone":
                        self._update_object_tracking("phone", obj, current_time)
                    elif dms_kind == "cigarette":
                        self._update_object_tracking("cigarette", obj, current_time)

            self._update_eye_state(eye_detections, current_time)
            self._update_seatbelt_state(seatbelt_detected, current_time)

            violations = self._check_all_violations(current_time)
            self._update_stats(detections, violations)

            return {
                "detected": len(detections) > 0,
                "objects": detections,
                "violations": violations,
                "current_violations": violations,
                "stats": self._get_stats_dict(),
                "eye_states": eye_detections,
                "frame_info": {"width": width, "height": height},
            }

        except Exception as e:
            logging.error(f"[dms:postprocess_failed] error={e}")
            return self._get_default_result()

    def _resolve_dms_class(self, class_id: int):
        """Функция: _resolve_dms_class()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `class_id` (`int`): идентификатор/индекс для адресации и сопоставления сущностей.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if class_id in self.phone_ids:
            return "Phone", "phone"
        if class_id == int(self.cls_open_eye):
            return "Open Eye", "open_eye"
        if class_id == int(self.cls_closed_eye):
            return "Closed Eye", "closed_eye"
        if class_id == int(self.cls_seatbelt):
            return "Seatbelt", "seatbelt"
        if class_id == int(self.cls_cigarette):
            return "Cigarette", "cigarette"

        raw_name = self.class_names.get(int(class_id), f"Unknown_{class_id}")
        normalized = str(raw_name).strip().lower()
        if normalized in {"phone", "cell phone"}:
            return "Phone", "phone"
        if normalized in {"open eye", "open_eye"}:
            return "Open Eye", "open_eye"
        if normalized in {"closed eye", "closed_eye"}:
            return "Closed Eye", "closed_eye"
        if normalized in {"seatbelt", "seat belt"}:
            return "Seatbelt", "seatbelt"
        if normalized in {"cigarette", "cigar"}:
            return "Cigarette", "cigarette"
        return str(raw_name), str(raw_name).lower()

    def _update_object_tracking(self, obj_class: str, obj_info: Dict[str, Any], current_time: float):
        """Функция: _update_object_tracking()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `obj_class` (`str`): данные обнаруженного объекта или его служебного представления.
- `obj_info` (`Dict[str, Any]`): данные обнаруженного объекта или его служебного представления.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if obj_class not in ["phone", "cigarette"]:
            return

        obj_id = self._find_matching_object(obj_class, obj_info)
        if obj_id:
            self.tracked_objects[obj_class][obj_id].update(
                {
                    "last_seen": current_time,
                    "bbox": obj_info["bbox"],
                    "confidence": obj_info["confidence"],
                    "class_id": obj_info["class_id"],
                    "class": obj_info["class"],
                }
            )
        else:
            new_id = f"{obj_class}_{int(current_time * 1000)}"
            self.tracked_objects[obj_class][new_id] = {
                "start_time": current_time,
                "last_seen": current_time,
                "bbox": obj_info["bbox"],
                "confidence": obj_info["confidence"],
                "class_id": obj_info["class_id"],
                "class": obj_info["class"],
            }

    def _find_matching_object(self, obj_class: str, new_obj: Dict[str, Any]) -> Optional[str]:
        """Функция: _find_matching_object()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `obj_class` (`str`): данные обнаруженного объекта или его служебного представления.
- `new_obj` (`Dict[str, Any]`): данные обнаруженного объекта или его служебного представления.
Возвращаемое значение: Optional[str]: результат шага обработки, который используется следующим этапом пайплайна."""
        tracked = self.tracked_objects.get(obj_class, {})
        if not tracked:
            return None

        new_bbox = new_obj["bbox"]
        new_center = self._get_bbox_center(new_bbox)
        now_ts = new_obj["timestamp"]

        for obj_id, obj_data in tracked.items():
            if now_ts - obj_data["last_seen"] > 2.0:
                continue

            old_center = self._get_bbox_center(obj_data["bbox"])
            distance = float(np.sqrt((new_center[0] - old_center[0]) ** 2 + (new_center[1] - old_center[1]) ** 2))

            max_distance = float(new_bbox[2]) * 0.2
            if distance < max_distance:
                return obj_id

        return None

    def _get_bbox_center(self, bbox: List[int]):
        """Функция: _get_bbox_center()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `bbox` (`List[int]`): ограничивающий прямоугольник объекта в формате `[x1, y1, x2, y2]`.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _update_eye_state(self, eye_detections: Dict[str, int], current_time: float):
        """Функция: _update_eye_state()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `eye_detections` (`Dict[str, int]`): данные обнаруженного объекта или его служебного представления.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        total_eyes = eye_detections["open"] + eye_detections["closed"]

        if total_eyes > 0:
            if eye_detections["open"] > 0:
                self.stats["eye_state"] = "open"
                self.last_eye_open_time = current_time
                self.eye_closed_start_time = None
            elif eye_detections["closed"] > 0:
                self.stats["eye_state"] = "closed"
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = current_time
        else:
            self.stats["eye_state"] = "unknown"
            self.eye_closed_start_time = None

    def _update_seatbelt_state(self, seatbelt_detected: bool, current_time: float):
        """Функция: _update_seatbelt_state()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `seatbelt_detected` (`bool`): флаг наличия детекции ремня безопасности на текущем кадре.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if seatbelt_detected:
            self.stats["seatbelt_state"] = "on"
            self.seatbelt_detected_in_interval = True
        else:
            self.stats["seatbelt_state"] = "off"

    def _check_all_violations(self, current_time: float):
        """Функция: _check_all_violations()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        violations = []

        if self.eye_closed_start_time and self.stats["eye_state"] == "closed":
            eye_closed_duration = current_time - self.eye_closed_start_time
            if eye_closed_duration >= self.eye_closed_threshold:
                if self._check_cooldown("eye_closed", current_time):
                    violations.append(
                        {
                            "class": "eye_closed",
                            "type": "eye_closed",
                            "duration": eye_closed_duration,
                            "start_time": self.eye_closed_start_time,
                            "message": f"Driver fatigue! Eyes closed for {eye_closed_duration:.1f} seconds",
                            "severity": "high",
                        }
                    )
                    self._record_violation("eye_closed", current_time)

        if current_time - self.last_seatbelt_check_time >= self.seatbelt_check_interval:
            self.last_seatbelt_check_time = current_time

            if not self.seatbelt_detected_in_interval:
                if self._check_cooldown("no_seatbelt", current_time):
                    violations.append(
                        {
                            "class": "no_seatbelt",
                            "type": "seatbelt_off",
                            "duration": self.seatbelt_check_interval,
                            "start_time": current_time - self.seatbelt_check_interval,
                            "message": "Seatbelt not fastened!",
                            "severity": "high",
                        }
                    )
                    self._record_violation("no_seatbelt", current_time)

            self.seatbelt_detected_in_interval = False

        phone_violation = self._check_object_violation("phone", self.phone_threshold, current_time)
        if phone_violation:
            violations.append(phone_violation)
            self._record_violation("phone", current_time)

        cigarette_violation = self._check_object_violation("cigarette", self.cigarette_threshold, current_time)
        if cigarette_violation:
            violations.append(cigarette_violation)
            self._record_violation("cigarette", current_time)

        self._cleanup_old_objects(current_time)
        return violations

    def _check_object_violation(self, obj_class: str, threshold: float, current_time: float):
        """Функция: _check_object_violation()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `obj_class` (`str`): данные обнаруженного объекта или его служебного представления.
- `threshold` (`float`): пороговое значение принятия решения в алгоритме.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        tracked = self.tracked_objects.get(obj_class, {})
        if not tracked:
            return None

        max_duration = 0.0
        violating_obj = None

        for _, obj_data in tracked.items():
            duration = current_time - obj_data["start_time"]
            if duration > max_duration:
                max_duration = duration
                violating_obj = obj_data

        if max_duration >= threshold:
            if self._check_cooldown(obj_class, current_time):
                class_name = "Phone" if obj_class == "phone" else "Cigarette"
                return {
                    "class": obj_class,
                    "type": f"{obj_class}_usage",
                    "duration": max_duration,
                    "start_time": violating_obj["start_time"],
                    "message": f"{class_name} usage detected for {max_duration:.1f} seconds",
                    "severity": "medium",
                    "object_data": violating_obj,
                }

        return None

    def _check_cooldown(self, violation_class: str, current_time: float) -> bool:
        """Функция: _check_cooldown()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_class` (`str`): данные детектора нарушений, используемые для итогового решения по кадру.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        last_time = self.last_violation_times.get(violation_class, 0.0)
        cooldown = self.class_cooldowns.get(violation_class, 60.0)
        if last_time == 0:
            return True
        return (current_time - last_time) >= cooldown

    def _record_violation(self, violation_class: str, current_time: float):
        """Функция: _record_violation()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_class` (`str`): данные детектора нарушений, используемые для итогового решения по кадру.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        self.last_violation_times[violation_class] = current_time
        self.stats["violation_count"] += 1
        logging.info(f"[dms:violation_recorded] type={violation_class}")

    def _cleanup_old_objects(self, current_time: float):
        """Функция: _cleanup_old_objects()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        max_age = 10.0
        for obj_class in ["phone", "cigarette"]:
            tracked = self.tracked_objects.get(obj_class, {})
            to_remove = [obj_id for obj_id, obj_data in tracked.items() if (current_time - obj_data["last_seen"]) > max_age]
            for obj_id in to_remove:
                del tracked[obj_id]

    def _update_stats(self, detections, violations):
        """Функция: _update_stats()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detections` (`Any`): данные обнаруженного объекта или его служебного представления.
- `violations` (`Any`): список найденных нарушений за кадр/интервал.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats["total_detections"] += len(detections)
        for obj in detections:
            self.stats["detections_by_class"][obj["class"]] += 1

        self.stats["current_objects"] = [{"class": obj["class"], "confidence": obj["confidence"]} for obj in detections]
        self.stats["phone_detected"] = any((obj["class_id"] in self.phone_ids) for obj in detections)
        self.stats["cigarette_detected"] = any((obj["class_id"] == int(self.cls_cigarette)) for obj in detections)

    def _get_stats_dict(self):
        """Функция: _get_stats_dict()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        return {
            "total_detections": self.stats["total_detections"],
            "detections_by_class": dict(self.stats["detections_by_class"]),
            "eye_state": self.stats["eye_state"],
            "seatbelt_state": self.stats["seatbelt_state"],
            "phone_detected": self.stats["phone_detected"],
            "cigarette_detected": self.stats["cigarette_detected"],
            "current_objects": self.stats["current_objects"],
            "violation_count": self.stats["violation_count"],
        }

    def _get_default_result(self):
        """Функция: _get_default_result()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        return {
            "detected": False,
            "objects": [],
            "violations": [],
            "current_violations": [],
            "stats": self._get_stats_dict(),
        }
