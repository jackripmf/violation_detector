import logging
import os

from ultralytics import YOLO
from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """Детектор объектов YOLO"""

    def __init__(self, model_name=None, area_threshold=0.7,
                 conf_threshold=0.5):
        super().__init__("YOLODetector")

        # Если путь не указан, ищем модель
        if model_name is None:
            model_name = self._find_yolo_model()

        logging.info(f"Loading YOLO model from: {model_name}")

        try:
            self.yolo_model = YOLO(model_name)
            logging.info(f"✓ YOLO model loaded successfully")
        except Exception as e:
            logging.error(f"YOLO model loading error: {e}")
            # Пробуем стандартное имя
            try:
                self.yolo_model = YOLO("yolov8n.pt")
                logging.info("✓ Loaded default YOLOv8n model")
            except Exception as e2:
                logging.error(f"Failed to load any YOLO model: {e2}")
                self.yolo_model = None

        self.area_threshold = area_threshold
        self.conf_threshold = conf_threshold

    def _find_yolo_model(self):
        """Поиск файла модели YOLO"""
        model_filename = "yolov8n.pt"

        possible_paths = [
            # Рядом с текущим файлом
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         model_filename),
            # В папке models
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models",
                         model_filename),
            # В src/utils/models
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "utils", "models", model_filename),
            # В проекте (3 уровня вверх)
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "..", "..", "src", "utils", "models", model_filename),
            # В рабочей директории
            os.path.join(os.getcwd(), model_filename),
            # В src/utils/models относительно рабочей директории
            os.path.join(os.getcwd(), "src", "utils", "models",
                         model_filename),
            # Стандартное имя (скачает автоматически)
            "yolov8n.pt"
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                logging.info(f"Found YOLO model at: {abs_path}")
                return abs_path

        # Если не нашли, вернем стандартное имя (YOLO скачает сам)
        logging.warning(
            "YOLO model not found locally, will download automatically")
        return "yolov8n.pt"

    def detect(self, frame):
        """Детекция объектов YOLO"""
        try:
            results = self.yolo_model(frame, conf=self.conf_threshold)
            image_area = frame.shape[0] * frame.shape[1]

            large_objects = []
            total_large_area = 0

            for r in results:
                if r.boxes is None:
                    continue
                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    area_ratio = area / image_area

                    class_id = int(r.boxes.cls.cpu().numpy()[i])
                    class_name = self.yolo_model.names[class_id]
                    confidence = float(r.boxes.conf.cpu().numpy()[i])

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
                "detected": yolo_detected,
                "objects": large_objects,
                "metrics": {
                    "large_objects_area": total_large_area,
                    "large_objects_ratio": total_large_area / image_area,
                },
            }
        except Exception as e:
            logging.error(f"YOLO detection error: {e}")
            return {
                "detected": False,
                "objects": [],
                "metrics": {"large_objects_area": 0, "large_objects_ratio": 0},
            }
