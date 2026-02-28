"""Файл: tests/test_detection_scheduler.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.detection_manager: используется для передачи данных или вызова связанной логики."""

from src.processing.detection_manager import DetectionManager


class _ForbiddenStub:
    def __init__(self):
        self.calls = 0
        self.is_active = True

    def detect(self, frame, frame_id=None):
        self.calls += 1
        return {
            "detected": True,
            "objects": [{"class": "phone", "frame_id": frame_id}],
            "current_violation": True,
            "violation_info": {"objects": [{"class": "phone"}]},
            "stats": {},
        }

    def enable(self):
        self.is_active = True

    def disable(self):
        self.is_active = False


class _MovementStub:
    def __init__(self):
        self.calls = 0
        self.last = {
            "movement_detected": False,
            "filtered": True,
            "reason": "none",
            "consecutive_frames": 0,
            "movement_duration": 0.0,
        }

    def detect(self, frame, is_obstructed=False):
        self.calls += 1
        self.last = {
            "movement_detected": True,
            "filtered": True,
            "reason": "movement",
            "consecutive_frames": self.calls,
            "movement_duration": 1.0,
        }
        return dict(self.last)

    def get_last_movement_info(self):
        return dict(self.last)


class _HubSharedStub:
    def __init__(self):
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        return [{"shared": True, "kwargs": kwargs}]


class _SharedYoloStub:
    model_key = "yolo_main"
    model_path = "/tmp/mock.pt"
    conf_threshold = 0.5
    iou_threshold = 0.5
    max_det = 50
    imgsz = 640

    def postprocess_shared_results(self, frame, raw_results):
        assert raw_results[0]["shared"] is True
        return {
            "detected": True,
            "objects": [{"class": "large_object"}],
            "metrics": {"shared": True},
        }


class _SharedForbiddenStub:
    model_key = "yolo_main"
    model_path = "/tmp/mock.pt"
    confidence_threshold = 0.4
    iou_threshold = 0.5
    max_det = 50
    imgsz = 640

    def postprocess_shared_results(self, frame, raw_results):
        assert raw_results[0]["shared"] is True
        return {
            "detected": True,
            "objects": [{"class": "phone"}],
            "current_violation": True,
            "violation_info": {"objects": [{"class": "phone"}]},
            "stats": {},
        }


class _SharedDmsStub:
    model_key = "yolo_main"
    model_path = "/tmp/mock.pt"
    confidence_threshold = 0.05
    iou_threshold = 0.5
    max_det = 50
    imgsz = 640

    def postprocess_shared_results(self, frame, raw_results, frame_id=None):
        assert raw_results[0]["shared"] is True
        return {
            "detected": True,
            "objects": [{"class": "Phone"}],
            "violations": [{"type": "phone_usage"}],
            "current_violations": [{"type": "phone_usage"}],
            "stats": {"frame_id": frame_id},
        }


def test_forbidden_scheduler_uses_cache_until_ttl_expires():
    manager = DetectionManager(
        hub=None,
        enabled_detectors=[],
        detector_schedule={
            "forbidden": {
                "every_n_frames": 5,
                "result_ttl_frames": 1,
            }
        },
    )
    manager.forbidden_detector = _ForbiddenStub()

    r0 = manager.detect_forbidden(frame=None, frame_count=0)
    r1 = manager.detect_forbidden(frame=None, frame_count=1)
    r2 = manager.detect_forbidden(frame=None, frame_count=2)
    r5 = manager.detect_forbidden(frame=None, frame_count=5)

    assert manager.forbidden_detector.calls == 2
    assert r0["current_violation"] is True
    assert r1["current_violation"] is True
    assert r2["current_violation"] is False
    assert r5["current_violation"] is True


def test_movement_scheduler_returns_cached_result_when_skipped():
    manager = DetectionManager(
        hub=None,
        enabled_detectors=[],
        detector_schedule={
            "movement": {
                "every_n_frames": 3,
                "result_ttl_frames": 2,
            }
        },
    )
    manager.movement_detector = _MovementStub()

    r0 = manager.detect_movement(frame=None, is_obstructed=False, frame_count=0)
    r1 = manager.detect_movement(frame=None, is_obstructed=False, frame_count=1)
    r2 = manager.detect_movement(frame=None, is_obstructed=False, frame_count=2)
    r3 = manager.detect_movement(frame=None, is_obstructed=False, frame_count=3)

    assert manager.movement_detector.calls == 2
    assert r0["movement_detected"] is True
    assert r1["movement_detected"] is True
    assert r2["movement_detected"] is True
    assert r3["movement_detected"] is True


def test_shared_yolo_raw_infer_is_single_call_per_frame():
    hub = _HubSharedStub()
    manager = DetectionManager(
        hub=hub,
        enabled_detectors=[],
        detector_schedule={
            "yolo": {"every_n_frames": 1, "result_ttl_frames": 0},
            "forbidden": {"every_n_frames": 1, "result_ttl_frames": 0},
            "dms": {"every_n_frames": 1, "result_ttl_frames": 0},
        },
    )
    manager.cv_detector = None
    manager.dark_detector = None
    manager.movement_detector = None
    manager.yolo_detector = _SharedYoloStub()
    manager.forbidden_detector = _SharedForbiddenStub()
    manager.dms_detector = _SharedDmsStub()

    frame = object()
    r_yolo = manager.detect_obstruction(frame=frame, frame_count=10)
    r_forbidden = manager.detect_forbidden(frame=frame, frame_count=10)
    r_dms = manager.detect_dms(frame=frame, frame_count=10)

    assert hub.calls == 1
    assert r_yolo["detected"] is True
    assert r_forbidden["current_violation"] is True
    assert len(r_dms["violations"]) == 1


def test_movement_scheduler_uses_last_state_when_ttl_zero():
    manager = DetectionManager(
        hub=None,
        enabled_detectors=[],
        detector_schedule={
            "movement": {
                "every_n_frames": 3,
                "result_ttl_frames": 0,
            }
        },
    )
    manager.movement_detector = _MovementStub()

    r0 = manager.detect_movement(frame=None, is_obstructed=False, frame_count=0)
    r1 = manager.detect_movement(frame=None, is_obstructed=False, frame_count=1)

    assert manager.movement_detector.calls == 1
    assert r0["movement_detected"] is True
    assert r1["movement_detected"] is True


def test_runtime_detector_update_disables_forbidden_on_the_fly():
    manager = DetectionManager(
        hub=None,
        enabled_detectors=[],
        detector_schedule={
            "forbidden": {"every_n_frames": 1, "result_ttl_frames": 0},
        },
    )
    manager.forbidden_detector = _ForbiddenStub()
    manager.set_runtime_detectors(enabled_detectors=["forbidden"])

    r_enabled = manager.detect_forbidden(frame=None, frame_count=0)
    assert r_enabled["current_violation"] is True
    assert manager.forbidden_detector.calls == 1

    manager.set_runtime_detectors(enabled_detectors=[])
    r_disabled = manager.detect_forbidden(frame=None, frame_count=1)
    assert r_disabled["current_violation"] is False
    assert manager.forbidden_detector.calls == 1
