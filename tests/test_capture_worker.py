"""Файл: tests/test_capture_worker.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.capture_worker: используется для передачи данных или вызова связанной логики.
- src.runtime.topology: используется для передачи данных или вызова связанной логики."""

import queue
import threading
import time
import numpy as np
import pytest

from src.processing.capture_worker import CaptureWorker
from src.runtime.topology import SourceConfig


class _SourceManagerStub:
    def __init__(self, total_frames: int = 5):
        self.total_frames = total_frames
        self.index = 0
        self.released = False
        self.opened_with = {}

    def open_source(self, input_source, camera_id=None):
        self.opened_with = {"input_source": input_source, "camera_id": camera_id}
        return True

    def read_frame(self):
        if self.index >= self.total_frames:
            return False, None
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.index += 1
        return True, frame

    def get_timestamp(self, frame_index, start_time):
        return frame_index / 10.0

    def release(self):
        self.released = True


class _ReconnectSourceManagerStub:
    def __init__(self):
        self.session_idx = -1
        self.frame_idx = 0
        self.released = False
        self.open_calls = 0
        self.frames_per_session = [1, 2]

    def open_source(self, input_source, camera_id=None):
        self.open_calls += 1
        self.session_idx += 1
        self.frame_idx = 0
        return True

    def read_frame(self):
        if self.session_idx < 0:
            return False, None
        current_limit = (
            self.frames_per_session[self.session_idx]
            if self.session_idx < len(self.frames_per_session)
            else 0
        )
        if self.frame_idx >= current_limit:
            return False, None
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.frame_idx += 1
        return True, frame

    def get_timestamp(self, frame_index, start_time):
        return frame_index / 10.0

    def release(self):
        self.released = True


def _drain_indices(q):
    indices = []
    while not q.empty():
        packet = q.get_nowait()
        indices.append(packet.frame_index)
    return indices


def test_capture_worker_start_stop_and_release():
    output_q = queue.Queue(maxsize=8)
    source_cfg = SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)
    source_stub = _SourceManagerStub(total_frames=3)
    worker = CaptureWorker(
        source_config=source_cfg,
        output_queue=output_q,
        source_manager=source_stub,
        stop_event=threading.Event(),
    )

    assert worker.start() is True
    worker.join(timeout=1.0)

    stats = worker.get_stats()
    assert stats.read_frames == 3
    assert stats.queued_frames == 3
    assert stats.dropped_frames == 0
    assert source_stub.released is True
    assert worker.is_running() is False


def test_capture_worker_drop_oldest_keeps_latest_frames():
    output_q = queue.Queue(maxsize=2)
    source_cfg = SourceConfig(
        source_id="cam0",
        camera_id=0,
        capture_queue_size=2,
        drop_policy="drop_oldest",
    )
    source_stub = _SourceManagerStub(total_frames=5)
    worker = CaptureWorker(
        source_config=source_cfg,
        output_queue=output_q,
        source_manager=source_stub,
        stop_event=threading.Event(),
    )

    worker.start()
    worker.join(timeout=1.0)

    stats = worker.get_stats()
    assert stats.read_frames == 5
    assert stats.queued_frames == 5
    assert stats.dropped_frames == 3
    assert _drain_indices(output_q) == [3, 4]


def test_capture_worker_drop_newest_preserves_old_queue():
    output_q = queue.Queue(maxsize=2)
    source_cfg = SourceConfig(
        source_id="cam0",
        camera_id=0,
        capture_queue_size=2,
        drop_policy="drop_newest",
    )
    source_stub = _SourceManagerStub(total_frames=5)
    worker = CaptureWorker(
        source_config=source_cfg,
        output_queue=output_q,
        source_manager=source_stub,
        stop_event=threading.Event(),
    )

    worker.start()
    worker.join(timeout=1.0)

    stats = worker.get_stats()
    assert stats.read_frames == 5
    assert stats.queued_frames == 2
    assert stats.dropped_frames == 3
    assert _drain_indices(output_q) == [0, 1]


def test_capture_worker_reconnects_for_camera_when_enabled():
    output_q = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    source_cfg = SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)
    source_stub = _ReconnectSourceManagerStub()
    worker = CaptureWorker(
        source_config=source_cfg,
        output_queue=output_q,
        source_manager=source_stub,
        stop_event=stop_event,
        reconnect_on_loss=True,
        reconnect_backoff_sec=0.01,
    )

    worker.start()
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if worker.get_stats().read_frames >= 3 and source_stub.open_calls >= 2:
            break
        time.sleep(0.01)
    stop_event.set()
    worker.join(timeout=1.0)

    stats = worker.get_stats()
    assert stats.read_frames >= 3
    assert source_stub.open_calls >= 2


def test_sync_file_playback_waits_when_ahead_of_video_time():
    output_q = queue.Queue(maxsize=2)
    source_cfg = SourceConfig(source_id="file0", input_source="/tmp/a.mp4", capture_queue_size=2)

    class _StopEventStub:
        def __init__(self):
            self.waited = 0.0

        def wait(self, timeout):
            self.waited = float(timeout)
            return False

        def is_set(self):
            return False

        def set(self):
            return None

    stop_event = _StopEventStub()
    worker = CaptureWorker(
        source_config=source_cfg,
        output_queue=output_q,
        source_manager=_SourceManagerStub(total_frames=1),
        stop_event=stop_event,
    )
    started_at = time.monotonic() - 0.15
    worker._sync_file_playback(frame_index=5, source_started_at=started_at, source_fps=20.0)

    assert stop_event.waited == pytest.approx(0.10, abs=0.05)
