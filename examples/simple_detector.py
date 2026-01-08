import cv2
import time
import logging
from collections import defaultdict
from ultralytics import YOLO


class SimpleForbiddenDetector:
    """Упрощенный детектор для отладки"""

    FORBIDDEN_CLASSES = {
        'bottle': 39,
        'cell phone': 67,
        'cup': 41,
        'wine glass': 40
    }

    def __init__(self):
        self.yolo_model = YOLO("yolov8n.pt")
        self.last_violation_time = 0
        self.violation_cooldown = 5.0
        self.min_detection_duration = 2.0

        self.tracked_objects = {}
        self.violation_count = 0

        print("Simple detector initialized")

    def detect(self, frame):
        try:
            current_time = time.time()

            # Детекция YOLO
            results = self.yolo_model(frame, conf=0.3, verbose=False)

            forbidden_objects = []

            for r in results:
                if r.boxes is None:
                    continue

                for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(r.boxes.cls.cpu().numpy()[i])
                    confidence = float(r.boxes.conf.cpu().numpy()[i])

                    class_name = self.yolo_model.names[class_id]
                    if class_name not in self.FORBIDDEN_CLASSES:
                        continue

                    obj = {
                        "bbox": [x1, y1, x2, y2],
                        "class": class_name,
                        "confidence": confidence,
                        "timestamp": current_time
                    }

                    forbidden_objects.append(obj)

            # Обновляем трекинг
            self._update_tracking(forbidden_objects, current_time)

            # Проверяем нарушения
            violation_detected, violation_info = self._check_violations(
                current_time)

            return {
                "objects": forbidden_objects,
                "current_violation": violation_detected,
                "violation_info": violation_info
            }

        except Exception as e:
            print(f"Detector error: {e}")
            return {"objects": [], "current_violation": False,
                    "violation_info": None}

    def _update_tracking(self, objects, current_time):
        # Простой трекинг
        if objects:
            for obj in objects:
                obj_id = f"{obj['class']}_{obj['bbox'][0]}_{obj['bbox'][1]}"

                if obj_id not in self.tracked_objects:
                    self.tracked_objects[obj_id] = {
                        "class": obj["class"],
                        "bbox": obj["bbox"],
                        "start_time": current_time,
                        "last_seen": current_time
                    }
                    print(f"New object: {obj['class']}")
                else:
                    self.tracked_objects[obj_id]["last_seen"] = current_time
                    self.tracked_objects[obj_id]["bbox"] = obj["bbox"]

        # Удаляем старые
        to_remove = []
        for obj_id, data in self.tracked_objects.items():
            if current_time - data["last_seen"] > 1.0:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

    def _check_violations(self, current_time):
        # Проверяем кулдаун
        if current_time - self.last_violation_time < self.violation_cooldown:
            return False, None

        violation_objects = []

        for obj_id, data in self.tracked_objects.items():
            duration = current_time - data["start_time"]

            if duration >= self.min_detection_duration:
                violation_objects.append({
                    "class": data["class"],
                    "duration": duration,
                    "bbox": data["bbox"]
                })

        if violation_objects:
            self.violation_count += 1
            self.last_violation_time = current_time

            violation_info = {
                "violation_id": self.violation_count,
                "timestamp": current_time,
                "objects": violation_objects,
                "cooldown_remaining": 0
            }

            print(
                f"VIOLATION #{self.violation_count}: {[obj['class'] for obj in violation_objects]}")

            # Очищаем после нарушения
            self.tracked_objects.clear()

            return True, violation_info

        return False, None


# Тест
if __name__ == "__main__":
    detector = SimpleForbiddenDetector()

    # Тестовое изображение
    img = cv2.imread("test.jpg")  # или используйте веб-камеру

    if img is not None:
        result = detector.detect(img)
        print(f"Result: {result}")
    else:
        print("Create test.jpg with forbidden object first")