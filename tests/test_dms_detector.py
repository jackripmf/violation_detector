"""Файл: tests/test_dms_detector.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.detectors.dms_detector: используется для передачи данных или вызова связанной логики."""

import numpy as np

from src.detectors.dms_detector import DMSDetector


class _ArrayWrapper:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _ArrayWrapper(np.array(xyxy, dtype=np.float32))
        self.conf = _ArrayWrapper(np.array(conf, dtype=np.float32))
        self.cls = _ArrayWrapper(np.array(cls, dtype=np.float32))


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _ModelStub:
    names = {
        67: "cell phone",
        80: "Open Eye",
        81: "Closed Eye",
        82: "Cigarette",
        84: "Seatbelt",
    }


class _HubStub:
    def __init__(self):
        self.calls = []

    def get_model(self, model_key, weights_path):
        return _ModelStub()

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes = _Boxes(
            xyxy=[[10, 10, 80, 80], [100, 100, 160, 160], [200, 120, 260, 190]],
            conf=[0.9, 0.85, 0.8],
            cls=[67, 81, 84],
        )
        return [_Result(boxes)]


def test_dms_detect_uses_full_frame_shared_predict_with_allowed_classes():
    hub = _HubStub()
    detector = DMSDetector(hub=hub)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    result = detector.detect(frame, frame_id=42)

    assert result["detected"] is True
    assert len(hub.calls) == 1

    call = hub.calls[0]
    assert call["frame_bgr"] is frame
    assert call["frame_id"] == 42
    assert sorted(call["classes"]) == sorted([67, 83, 80, 81, 82, 84])

    classes = [obj["class"] for obj in result["objects"]]
    assert "Phone" in classes
    assert "Closed Eye" in classes
    assert "Seatbelt" in classes
