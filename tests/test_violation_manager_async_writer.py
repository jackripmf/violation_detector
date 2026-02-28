"""Файл: tests/test_violation_manager_async_writer.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.violation_manager: используется для передачи данных или вызова связанной логики."""

import threading
import time

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


def _dms_payload(violation_type="phone_usage"):
    return {
        "violations": [{"type": violation_type, "message": "Phone usage"}],
        "objects": [],
        "stats": {},
    }


def test_async_writer_flushes_on_cleanup(monkeypatch, tmp_path):
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=True,
        writer_queue_max_size=8,
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    threads_seen = []
    lock = threading.Lock()

    def fake_imwrite(path, image):
        with lock:
            threads_seen.append(threading.current_thread().name)
        return True

    monkeypatch.setattr("cv2.imwrite", fake_imwrite)
    monkeypatch.setattr(manager.file_manager, "save_violation_report", lambda *args, **kwargs: True)

    v1 = manager.process_dms(_dms_payload("phone_usage"), frame, video_timestamp=1.0, processing_time=time.time(), frame_count=1)
    v2 = manager.process_dms(_dms_payload("seatbelt_off"), frame, video_timestamp=2.0, processing_time=time.time(), frame_count=2)
    assert v1 is not None
    assert v2 is not None

    manager.cleanup()

    assert len(threads_seen) == 2
    assert all(name == "ViolationWriter" for name in threads_seen)


def test_async_writer_drop_newest_when_queue_is_full(monkeypatch, tmp_path):
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=True,
        writer_queue_max_size=1,
        writer_overflow_strategy="drop_newest",
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    first_started = threading.Event()
    first_release = threading.Event()
    calls = {"count": 0}
    lock = threading.Lock()

    def fake_imwrite(path, image):
        with lock:
            calls["count"] += 1
            idx = calls["count"]
        if idx == 1:
            first_started.set()
            first_release.wait(timeout=2.0)
        return True

    monkeypatch.setattr("cv2.imwrite", fake_imwrite)
    monkeypatch.setattr(manager.file_manager, "save_violation_report", lambda *args, **kwargs: True)

    v1 = manager.process_dms(_dms_payload("phone_usage"), frame, video_timestamp=1.0, processing_time=time.time(), frame_count=1)
    assert v1 is not None
    assert first_started.wait(timeout=1.0)

    v2 = manager.process_dms(_dms_payload("seatbelt_off"), frame, video_timestamp=2.0, processing_time=time.time(), frame_count=2)
    v3 = manager.process_dms(_dms_payload("eye_closed"), frame, video_timestamp=3.0, processing_time=time.time(), frame_count=3)

    assert v2 is not None
    assert v3 is None

    first_release.set()
    manager.cleanup()


def test_async_writer_emits_violation_saved_after_successful_write(monkeypatch, tmp_path):
    events = []
    manager = ViolationManager(
        save_dir=str(tmp_path),
        visualizer=_VisualizerStub(),
        async_writes=True,
        writer_queue_max_size=8,
        log_source_id="cam0",
        event_callback=lambda **payload: events.append(payload),
    )
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    first_started = threading.Event()
    first_release = threading.Event()

    def fake_imwrite(path, image):
        first_started.set()
        first_release.wait(timeout=2.0)
        return True

    monkeypatch.setattr("cv2.imwrite", fake_imwrite)
    monkeypatch.setattr(manager.file_manager, "save_violation_report", lambda *args, **kwargs: True)

    violation = manager.process_dms(
        _dms_payload("phone_usage"),
        frame,
        video_timestamp=1.0,
        processing_time=time.time(),
        frame_count=1,
    )

    assert violation is not None
    assert first_started.wait(timeout=1.0)
    assert events == []

    first_release.set()
    manager.cleanup()

    assert len(events) == 1
    assert events[0]["event_type"] == "violation_saved"
    assert events[0]["source_id"] == "cam0"
    assert events[0]["data"]["violation_kind"] == "dms"
