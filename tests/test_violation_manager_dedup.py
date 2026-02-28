"""Файл: tests/test_violation_manager_dedup.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.violation_manager: используется для передачи данных или вызова связанной логики."""

import numpy as np

from src.processing.violation_manager import ViolationManager


class _VisualizerStub:
    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
    }

    def _draw_yolo_object(self, frame, obj):
        return None

    def _draw_forbidden_object(self, frame, obj):
        return None

    def _draw_dms_object(self, frame, obj):
        return None


def _forbidden_payload(violation_id: int) -> dict:
    return {
        "current_violation": True,
        "violation_info": {
            "violation_id": violation_id,
            "objects": [
                {
                    "class": "knife",
                    "object_id": 1,
                    "bbox": [10, 20, 100, 200],
                    "duration": 4.0,
                }
            ],
            "affected_classes": ["knife"],
            "cooldown_remaining": 30.0,
            "is_class_specific": True,
        },
    }


def _dms_payload() -> dict:
    return {
        "violations": [
            {
                "type": "seatbelt_off",
                "class": "no_seatbelt",
                "severity": "high",
                "start_time": 123.0,
                "message": "Seatbelt not fastened!",
            }
        ],
        "objects": [],
        "stats": {},
    }


def test_forbidden_deduplicates_same_source_violation_id(monkeypatch, tmp_path):
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=False,
        forbidden_duplicate_window_sec=2.0,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    writes = {"count": 0}

    def fake_imwrite(path, image):
        writes["count"] += 1
        return True

    monkeypatch.setattr("cv2.imwrite", fake_imwrite)
    monkeypatch.setattr(manager.file_manager, "save_violation_report", lambda *args, **kwargs: True)

    v1 = manager.process_forbidden(_forbidden_payload(violation_id=7), frame, video_timestamp=1.0)
    v2 = manager.process_forbidden(_forbidden_payload(violation_id=7), frame, video_timestamp=1.1)
    v3 = manager.process_forbidden(_forbidden_payload(violation_id=8), frame, video_timestamp=2.0)

    assert v1 is not None
    assert v2 is None
    assert v3 is not None
    assert writes["count"] == 2


def test_dms_deduplicates_same_signature_in_window(monkeypatch, tmp_path):
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=False,
        dms_duplicate_window_sec=1.0,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    writes = {"count": 0}

    def fake_imwrite(path, image):
        writes["count"] += 1
        return True

    monkeypatch.setattr("cv2.imwrite", fake_imwrite)
    monkeypatch.setattr(manager.file_manager, "save_violation_report", lambda *args, **kwargs: True)

    payload = _dms_payload()
    v1 = manager.process_dms(payload, frame, video_timestamp=1.0, processing_time=1000.0, frame_count=1)
    v2 = manager.process_dms(payload, frame, video_timestamp=1.1, processing_time=1000.2, frame_count=2)
    v3 = manager.process_dms(payload, frame, video_timestamp=2.0, processing_time=1001.2, frame_count=3)

    assert v1 is not None
    assert v2 is None
    assert v3 is not None
    assert writes["count"] == 2

