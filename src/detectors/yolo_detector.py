"""Файл: src/detectors/yolo_detector.py
Тип: детектор нарушений.
Назначение: получает кадр, применяет алгоритм детекции и возвращает структурированный результат.
Связи: детекторы вызываются менеджером детекции и передают данные в слой сохранения/визуализации.
Импортируемые внутренние модули:
- .base_detector: используется для передачи данных или вызова связанной логики."""
import logging
from .base_detector import BaseDetector
from ..inference.model_path_resolver import resolve_model_path


class YOLODetector(BaseDetector):
    """Класс: YOLODetector
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `area_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `class_names` (`Any`): словарь соответствия `class_id -> человекочитаемое имя класса`.
- `conf_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `hub` (`Any`): общий объект инференса, который кэширует модели и выполняет predict.
- `imgsz` (`Any`): размер входного изображения для инференса модели.
- `iou_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `max_det` (`Any`): максимальное число детекций, возвращаемых моделью на кадр.
- `model_key` (`str`): логический ключ модели в InferenceHub для повторного использования экземпляра.
- `model_path` (`Any`): путь к файлу весов модели на диске.
Ключевые методы:
- `__init__()`, `detect()`, `postprocess_shared_results()`"""

    def __init__(
        self,
        hub,
        model_name=None,
        model_key: str = "yolo_main",
        area_threshold: float = 0.7,
        conf_threshold: float = 0.5,
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
- `area_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `conf_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `iou_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `max_det` (`int`): максимальное число детекций, возвращаемых моделью на кадр.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        super().__init__("YOLODetector")

        self.hub = hub
        self.model_key = model_key
        self.area_threshold = float(area_threshold)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)

        self.model_path = resolve_model_path(model_name)
        logging.info(f"YOLODetector will use shared model: {self.model_path}")

        try:
            model = self.hub.get_model(self.model_key, self.model_path)
            self.class_names = model.names
        except Exception:
            self.class_names = {}

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
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False,
            )
            return self.postprocess_shared_results(frame, results)

        except Exception as e:
            logging.error(f"YOLO detection error: {e}")
            return {
                "detected": False,
                "objects": [],
                "metrics": {"large_objects_area": 0.0, "large_objects_ratio": 0.0},
            }

    def postprocess_shared_results(self, frame, raw_results):
        """Функция: postprocess_shared_results()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `raw_results` (`Any`): сырые выходные данные модели до доменной постобработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            results = raw_results

            image_area = frame.shape[0] * frame.shape[1]
            large_objects = []
            total_large_area = 0

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
                    x1, y1, x2, y2 = box.astype(int)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    area = w * h
                    area_ratio = area / image_area if image_area > 0 else 0.0

                    class_name = names.get(int(cls_id), str(cls_id))
                    confidence = float(conf)

                    if area_ratio > self.area_threshold:
                        total_large_area += area
                        large_objects.append(
                            {
                                "bbox": [x1, y1, x2, y2],
                                "class": class_name,
                                "confidence": confidence,
                                "area_ratio": area_ratio,
                            }
                        )

            yolo_detected = total_large_area > image_area * self.area_threshold
            return {
                "detected": bool(yolo_detected),
                "objects": large_objects,
                "metrics": {
                    "large_objects_area": float(total_large_area),
                    "large_objects_ratio": float(total_large_area / image_area) if image_area > 0 else 0.0,
                },
            }

        except Exception as e:
            logging.error(f"YOLO shared postprocess error: {e}")
            return {
                "detected": False,
                "objects": [],
                "metrics": {"large_objects_area": 0.0, "large_objects_ratio": 0.0},
            }
