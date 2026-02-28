"""Файл: tests/test_source_manager.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.source_manager: используется для передачи данных или вызова связанной логики."""

import pytest
import numpy as np
import cv2
from src.processing.source_manager import SourceManager


class TestSourceManager:

    def test_open_video_file(self, test_video_path):
        manager = SourceManager()
        result = manager.open_source(test_video_path)

        assert result == True
        assert manager.is_file == True
        assert manager.is_rtsp == False
        assert manager.fps == 30.0
        assert manager.width == 1280
        assert manager.height == 720
        assert manager.total_frames == 10
        assert manager.duration_sec == pytest.approx(10.0 / 30.0, rel=0.05)

        manager.release()

    def test_read_frames_from_video(self, test_video_path):
        manager = SourceManager()
        manager.open_source(test_video_path)

        frames_read = 0
        while True:
            ret, frame = manager.read_frame()
            if not ret:
                break
            frames_read += 1
            assert frame is not None
            assert frame.shape == (720, 1280, 3)

        assert frames_read == 10
        manager.release()

    def test_get_timestamp_for_file(self, test_video_path):
        manager = SourceManager()
        manager.open_source(test_video_path)


        timestamp = manager.get_timestamp(frame_index=30, start_time=0)
        assert timestamp == pytest.approx(1.0, rel=0.01)

        manager.release()

    def test_get_timestamp_for_stream(self):
        manager = SourceManager()
        manager.is_file = False

        import time
        start_time = time.time()
        time.sleep(0.1)

        timestamp = manager.get_timestamp(frame_index=0, start_time=start_time)
        assert timestamp >= 0.1

        manager.release()

    def test_get_info(self, test_video_path):
        manager = SourceManager()
        manager.open_source(test_video_path)

        info = manager.get_info()

        assert info["is_file"] == True
        assert info["is_rtsp"] == False
        assert info["fps"] == 30.0
        assert info["width"] == 1280
        assert info["height"] == 720
        assert info["total_frames"] == 10
        assert info["duration_sec"] == pytest.approx(10.0 / 30.0, rel=0.05)

        manager.release()

    def test_release(self, test_video_path):
        manager = SourceManager()
        manager.open_source(test_video_path)

        assert manager.cap is not None
        manager.release()
        assert manager.cap is None

    def test_invalid_file_path(self):
        manager = SourceManager()
        result = manager.open_source("/nonexistent/path/video.mp4")

        assert result == False
        manager.release()

    def test_ambiguous_numeric_string_requires_explicit_camera_mode(self):
        manager = SourceManager()
        result = manager.open_source("123")

        assert result == False
        manager.release()

    def test_explicit_camera_id_has_priority_over_input(self, monkeypatch):
        manager = SourceManager()
        captured = {"camera_id": None}

        def fake_open_camera(camera_id):
            captured["camera_id"] = camera_id
            return True

        monkeypatch.setattr(manager, "_open_camera", fake_open_camera)

        result = manager.open_source("123", camera_id=7)

        assert result == True
        assert captured["camera_id"] == 7

    def test_rtsps_url_routed_to_rtsp(self, monkeypatch):
        manager = SourceManager()
        captured = {"url": None}

        def fake_open_rtsp(url):
            captured["url"] = url
            return True

        monkeypatch.setattr(manager, "_open_rtsp", fake_open_rtsp)

        result = manager.open_source("rtsps://example.local/stream")

        assert result == True
        assert captured["url"] == "rtsps://example.local/stream"

    def test_open_source_redacts_rtsp_credentials_in_logs(self, monkeypatch, caplog):
        manager = SourceManager()

        def fake_open_rtsp(url):
            return False

        monkeypatch.setattr(manager, "_open_rtsp", fake_open_rtsp)

        with caplog.at_level("INFO"):
            result = manager.open_source("rtsp://user:secret@example.local:554/live")

        assert result == False
        assert "user:secret@" not in caplog.text
        assert "rtsp://***:***@example.local:554/live" in caplog.text

    def test_open_camera_uses_dshow_on_windows(self, monkeypatch):
        manager = SourceManager()

        class _CapStub:
            def __init__(self):
                self.props = {}

            def set(self, prop_id, value):
                self.props[prop_id] = value
                return True

            def isOpened(self):
                return True

            def get(self, prop_id):
                if prop_id == cv2.CAP_PROP_FPS:
                    return 30.0
                if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                    return 1280
                if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 720
                return 0

        captured = {"api": None}

        def fake_video_capture(camera_id, api):
            captured["api"] = api
            return _CapStub()

        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)

        assert manager._open_camera(0) is True
        assert captured["api"] == cv2.CAP_DSHOW

    def test_open_camera_uses_v4l2_on_linux(self, monkeypatch):
        manager = SourceManager()

        class _CapStub:
            def set(self, prop_id, value):
                return True

            def isOpened(self):
                return True

            def get(self, prop_id):
                if prop_id == cv2.CAP_PROP_FPS:
                    return 30.0
                if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                    return 1280
                if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 720
                return 0

        captured = {"api": None}

        def fake_video_capture(camera_id, api):
            captured["api"] = api
            return _CapStub()

        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr(cv2, "VideoCapture", fake_video_capture)

        assert manager._open_camera(1) is True
        assert captured["api"] == cv2.CAP_V4L2
