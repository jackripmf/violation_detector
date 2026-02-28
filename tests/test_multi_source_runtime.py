"""Файл: tests/test_multi_source_runtime.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.multi_source_runtime: используется для передачи данных или вызова связанной логики.
- src.runtime.topology: используется для передачи данных или вызова связанной логики.
- src.processing.stats_manager: используется для передачи данных или вызова связанной логики."""

import time
import queue

import numpy as np

from src.processing.multi_source_runtime import MultiSourceRuntime
from src.runtime.contracts import PreviewFramePayload, SourceVisualConfig
from src.runtime.topology import CapturedFrame, RuntimeTopology, SchedulerConfig, SourceConfig
from src.processing.stats_manager import StatsManager


class _SourceManagerStub:
    def __init__(self, frames_count: int = 4, fps: float = 25.0):
        self.frames_count = frames_count
        self.index = 0
        self.released = False
        self.fps = fps

    def open_source(self, input_source, camera_id=None):
        return True

    def read_frame(self):
        if self.index >= self.frames_count:
            return False, None
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.index += 1
        return True, frame

    def get_timestamp(self, frame_index, start_time):
        return float(frame_index) / max(1.0, float(self.fps))

    def release(self):
        self.released = True


class _HubStub:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, **kwargs):
        self.predict_calls += 1
        return [{"ok": True, "frame_id": kwargs.get("frame_id")}]

    def get_metrics(self):
        return {"predict_calls": self.predict_calls, "cache_hits": 0}


class _SlowHubStub(_HubStub):
    def predict(self, **kwargs):
        time.sleep(0.02)
        return super().predict(**kwargs)


class _DetectionManagerStub:
    def __init__(self, hub=None, enabled_detectors=None, detector_schedule=None):
        self.hub = hub
        self.enabled = list(enabled_detectors or [])
        self.enabled_detectors = list(enabled_detectors or [])
        self.detector_schedule = dict(detector_schedule or {
            "movement": {"every_n_frames": 2, "min_interval_ms": 0, "priority": 95, "result_ttl_frames": 0},
            "forbidden": {"every_n_frames": 2, "min_interval_ms": 0, "priority": 100, "result_ttl_frames": 1},
        })
        self.seeded_frames = []
        self.dms_detector = None
        self.force_forbidden_violation = False
        self.movement_detector = None

    def get_shared_yolo_infer_params(self):
        if self.hub is None:
            return None
        return {
            "model_key": "mock",
            "weights_path": "mock.pt",
            "conf": 0.2,
            "iou": 0.5,
            "max_det": 20,
            "imgsz": 64,
        }

    def build_shared_yolo_infer_key(self, params):
        return f"{params['model_key']}|{params['weights_path']}"

    def seed_shared_yolo_results(self, frame_count, infer_key, raw_results):
        self.seeded_frames.append((int(frame_count), infer_key, raw_results))

    def detect_obstruction(self, frame, frame_count):
        return {
            "detected": False,
            "reasons": [],
            "metrics": {},
            "detectors_count": 0,
            "yolo_objects": [],
        }

    def detect_movement(self, frame, is_obstructed, frame_count=0):
        return {
            "movement_detected": False,
            "filtered": True,
            "reason": "stable",
            "consecutive_frames": 0,
            "movement_duration": 0.0,
        }

    def detect_forbidden(self, frame, frame_count):
        if self.force_forbidden_violation:
            return {
                "detected": True,
                "objects": [{"class": "phone"}],
                "current_violation": True,
                "violation_info": {
                    "objects": [{"class": "phone"}],
                    "affected_classes": ["phone"],
                    "violation_id": int(frame_count),
                    "cooldown_remaining": 0.0,
                    "is_class_specific": True,
                },
                "stats": {},
            }
        return {
            "detected": False,
            "objects": [],
            "current_violation": False,
            "violation_info": None,
            "stats": {},
        }

    def detect_dms(self, frame, frame_count):
        return {
            "detected": False,
            "objects": [],
            "violations": [],
            "current_violations": [],
            "stats": {},
        }

    def set_runtime_detectors(self, enabled_detectors=None):
        if enabled_detectors is not None:
            self.enabled = list(enabled_detectors)
            self.enabled_detectors = list(enabled_detectors)
        return {
            "enabled_detectors": list(self.enabled),
        }

    def get_detector_status(self):
        return [
            type("DetectorStatus", (), {"name": "CameraMovementDetector", "enabled": True, "active": True})(),
            type("DetectorStatus", (), {"name": "ForbiddenItemsDetector", "enabled": self.force_forbidden_violation or ("forbidden" in self.enabled_detectors), "active": True})(),
        ]

    def get_detector_schedule_snapshot(self):
        return {name: dict(cfg) for name, cfg in self.detector_schedule.items()}

    def set_detector_schedule(self, detector_schedule=None):
        self.detector_schedule = {
            str(name): dict(cfg)
            for name, cfg in dict(detector_schedule or {}).items()
        }
        return {"detector_schedule": self.get_detector_schedule_snapshot()}


class _MovementResetStub:
    def __init__(self):
        self.reference_frames = []

    def set_reference_frame(self, frame):
        self.reference_frames.append(frame)


class _ViolationManagerStub:
    def __init__(self, **kwargs):
        self.source_fps = 30.0
        self.obstruction_start_time = None
        self.flush_called = False
        self.cleanup_called = False
        self.visualizer = kwargs.get("visualizer")
        self.event_callback = kwargs.get("event_callback")
        self.source_id = kwargs.get("log_source_id")

    def set_source_fps(self, source_fps):
        self.source_fps = float(source_fps)

    def process_obstruction(self, result, frame, video_timestamp, processing_time):
        return None

    def process_movement(self, movement_info, frame, video_timestamp):
        return None

    def process_forbidden(self, forbidden_result, frame, video_timestamp):
        if forbidden_result.get("current_violation"):
            if self.event_callback is not None:
                self.event_callback(
                    event_type="violation_saved",
                    source_id=self.source_id,
                    severity="info",
                    data={
                        "violation_kind": "forbidden",
                        "violation_id": int(forbidden_result.get("violation_info", {}).get("violation_id", 0)),
                        "media_file": "forbidden_stub.jpg",
                    },
                )
            return {"saved": True}
        return None

    def process_dms(self, dms_result, frame, video_timestamp, processing_time, frame_count):
        return None

    def flush_writes(self, timeout=None):
        self.flush_called = True
        return True

    def get_writer_queue_size(self):
        return 0

    def cleanup(self):
        self.cleanup_called = True


def _build_runtime(
    *,
    sources,
    source_frames=3,
    hub=None,
    detection_factory=_DetectionManagerStub,
    violation_factory=_ViolationManagerStub,
    infer_queue_size=64,
    dispatch_sleep_sec=0.001,
    infer_workers=1,
    postprocess_workers=1,
    show_preview=False,
    preview_callback=None,
    event_callback=None,
    command_timeout_sec=1.0,
):
    topology = RuntimeTopology(
        sources=sources,
        scheduler=SchedulerConfig(
            infer_queue_size=infer_queue_size,
            dispatch_sleep_sec=dispatch_sleep_sec,
        ),
        infer_workers=infer_workers,
        postprocess_workers=postprocess_workers,
    )
    return MultiSourceRuntime(
        topology=topology,
        source_manager_factory=lambda: _SourceManagerStub(frames_count=source_frames),
        hub=hub,
        detection_manager_factory=detection_factory,
        stats_manager_factory=StatsManager,
        violation_manager_factory=violation_factory,
        show_preview=show_preview,
        preview_callback=preview_callback,
        event_callback=event_callback,
        command_timeout_sec=command_timeout_sec,
    )


def test_multi_source_runtime_runs_two_sources_and_stops():
    runtime = _build_runtime(
        sources=[
            SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=4),
            SourceConfig(source_id="cam1", camera_id=1, capture_queue_size=4),
        ],
        source_frames=3,
        hub=None,
    )

    assert runtime.start() is True
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    state = runtime.get_state()
    assert runtime.is_running() is False
    assert state["sources"]["cam0"]["served_count"] > 0
    assert state["sources"]["cam1"]["served_count"] > 0
    assert state["sources"]["cam0"]["processed_frames"] > 0
    assert state["sources"]["cam1"]["processed_frames"] > 0


def test_inference_stage_calls_hub_once_per_frame():
    hub = _HubStub()
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=5,
        hub=hub,
    )

    runtime.start()
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    stats = runtime.get_stats()
    processed = stats["sources"]["cam0"]["processed_frames"]
    assert processed == 5
    assert hub.predict_calls == 5
    assert stats["hub_metrics"]["predict_calls"] == 5


def test_runtime_runs_with_cpu_only_path_without_hub():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=4,
        hub=None,
    )

    runtime.start()
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    stats = runtime.get_stats()
    assert stats["sources"]["cam0"]["processed_frames"] == 4
    assert stats["sources"]["cam0"]["avg_infer_time_ms"] == 0.0


def test_runtime_reports_saved_counters_per_source():
    def detection_factory(**kwargs):
        manager = _DetectionManagerStub(**kwargs)
        manager.force_forbidden_violation = True
        return manager

    runtime = _build_runtime(
        sources=[
            SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8),
            SourceConfig(source_id="cam1", camera_id=1, capture_queue_size=8),
        ],
        source_frames=2,
        hub=None,
        detection_factory=detection_factory,
    )

    runtime.start()
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    stats = runtime.get_stats()
    assert stats["sources"]["cam0"]["runtime_stats"]["saved_violations"] > 0
    assert stats["sources"]["cam1"]["runtime_stats"]["saved_violations"] > 0


def test_stop_drains_violation_pipeline_cleanup():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=30,
        hub=None,
    )

    runtime.start()
    time.sleep(0.05)
    runtime.stop(timeout=2.0)

    manager = runtime.source_contexts["cam0"].violation_manager
    assert manager.flush_called is True
    assert manager.cleanup_called is True


def test_runtime_collects_queue_depth_and_latency_metrics():
    hub = _HubStub()
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=2)],
        source_frames=20,
        hub=hub,
        infer_queue_size=4,
        dispatch_sleep_sec=0.01,
    )

    runtime.start()
    runtime.wait(timeout=4.0)
    runtime.stop(timeout=2.0)

    stats = runtime.get_stats()
    source_stats = stats["sources"]["cam0"]

    assert stats["max_infer_queue_depth"] >= 0
    assert stats["max_postprocess_queue_depth"] >= 0
    assert source_stats["max_source_queue_depth"] >= 1
    assert source_stats["avg_scheduling_latency_ms"] >= 0.0
    assert source_stats["avg_end_to_end_lag_ms"] >= 0.0
    assert source_stats["capture_dropped_frames"] > 0


def test_runtime_preview_callback_receives_frames_when_enabled():
    preview_calls = []

    def on_preview(source_id, frame):
        preview_calls.append((source_id, frame.shape[1], frame.shape[0]))

    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=3,
        hub=None,
        show_preview=True,
        preview_callback=on_preview,
    )

    runtime.start()
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    assert len(preview_calls) > 0
    assert preview_calls[0][0] == "cam0"


def test_runtime_propagates_show_fps_to_visualizer():
    runtime_no_fps = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=1,
        hub=None,
    )

    visualizer_no_fps = runtime_no_fps.source_contexts["cam0"].violation_manager.visualizer
    assert getattr(visualizer_no_fps, "show_fps", None) is False

    runtime_with_fps = MultiSourceRuntime(
        topology=RuntimeTopology(
            sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
            scheduler=SchedulerConfig(infer_queue_size=8, dispatch_sleep_sec=0.001),
        ),
        source_manager_factory=lambda: _SourceManagerStub(frames_count=1),
        hub=None,
        detection_manager_factory=_DetectionManagerStub,
        stats_manager_factory=StatsManager,
        violation_manager_factory=_ViolationManagerStub,
        show_preview=True,
        show_fps=True,
    )
    visualizer_with_fps = runtime_with_fps.source_contexts["cam0"].violation_manager.visualizer
    assert getattr(visualizer_with_fps, "show_fps", None) is True


def test_runtime_configures_indicator_conditions_for_preview():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=1,
        hub=None,
    )

    visualizer = runtime.source_contexts["cam0"].violation_manager.visualizer
    indicators = visualizer.indicators_config.indicators

    sample_result = {
        "detected": True,
        "forbidden_objects": [{"class": "knife"}],
        "dms_violations": [{"type": "no_seatbelt"}],
        "dms_objects": [{"class": "phone"}, {"class": "cigarette"}],
    }
    sample_movement = {"movement_detected": True}

    assert indicators["obstruction"].condition is not None
    assert indicators["movement"].condition is not None
    assert indicators["forbidden"].condition is not None
    assert indicators["dms"].condition is not None
    assert indicators["cigarette"].condition is not None
    assert indicators["phone"].condition is not None

    assert indicators["obstruction"].condition(sample_result, sample_movement) is True
    assert indicators["movement"].condition(sample_result, sample_movement) is True
    assert indicators["forbidden"].condition(sample_result, sample_movement) is True
    assert indicators["dms"].condition(sample_result, sample_movement) is True
    assert indicators["cigarette"].condition(sample_result, sample_movement) is True
    assert indicators["phone"].condition(sample_result, sample_movement) is True


def test_preview_key_target_is_active_source_only_for_multi_window():
    runtime = _build_runtime(
        sources=[
            SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8),
            SourceConfig(source_id="cam1", camera_id=1, capture_queue_size=8),
        ],
        source_frames=1,
        hub=None,
    )
    runtime.source_contexts["cam0"].preview_closed = False
    runtime.source_contexts["cam1"].preview_closed = False


    assert runtime._resolve_preview_key_source("cam0") is None

    runtime._active_preview_source_id = "cam1"
    assert runtime._resolve_preview_key_source("cam0") == "cam1"


def test_request_stop_source_stops_only_selected_context():
    runtime = _build_runtime(
        sources=[
            SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8),
            SourceConfig(source_id="cam1", camera_id=1, capture_queue_size=8),
        ],
        source_frames=1,
        hub=None,
    )

    packet = CapturedFrame(
        source_id="cam0",
        frame_index=0,
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        video_timestamp=0.0,
        captured_at=time.time(),
    )
    runtime.source_contexts["cam0"].frame_queue.put_nowait(packet)

    runtime._request_stop_source("cam0")

    cam0 = runtime.source_contexts["cam0"]
    cam1 = runtime.source_contexts["cam1"]
    assert cam0.stop_requested is True
    assert cam0.preview_closed is True
    assert cam0.frame_queue.empty() is True
    assert cam0.capture_worker.stop_event.is_set() is True
    assert cam1.stop_requested is False


def test_runtime_state_and_stats_expose_stable_source_contract_fields():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=2,
        hub=None,
    )

    runtime.start()
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    state = runtime.get_state()
    stats = runtime.get_stats()

    source_state = state["sources"]["cam0"]
    source_stats = stats["sources"]["cam0"]

    assert {"running", "infer_queue_size", "postprocess_queue_size", "max_infer_queue_depth", "max_postprocess_queue_depth", "sources", "last_error"} <= set(state)
    assert {"source_queue_size", "served_count", "dropped_before_infer", "captured_frames", "capture_dropped_frames", "processed_frames", "max_source_queue_depth", "stop_requested", "preview_closed", "enabled_detectors"} <= set(source_state)
    assert {"running", "total_sources", "total_processed_frames", "infer_queue_size", "postprocess_queue_size", "max_infer_queue_depth", "max_postprocess_queue_depth", "hub_metrics", "sources", "last_error"} <= set(stats)
    assert {"captured_frames", "capture_dropped_frames", "processed_frames", "served_count", "dropped_before_infer", "postprocess_errors", "max_source_queue_depth", "avg_scheduling_latency_ms", "avg_infer_time_ms", "avg_end_to_end_lag_ms", "writer_queue_size", "runtime_stats", "enabled_detectors"} <= set(source_stats)
    assert source_state["visual_config"]["show_boxes"] is True
    assert source_state["health_status"] in {"running", "idle", "degraded"}
    assert source_state["last_error"] is None
    assert source_state["detector_statuses"]
    assert "movement" in source_state["detector_schedule"]
    assert source_stats["visual_config"]["show_boxes"] is True
    assert source_stats["health_status"] in {"running", "idle", "degraded"}
    assert source_stats["last_error"] is None
    assert source_stats["detector_statuses"]
    assert "movement" in source_stats["detector_schedule"]


def test_runtime_supports_per_source_detector_updates_without_touching_neighbors():
    runtime = _build_runtime(
        sources=[
            SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8),
            SourceConfig(source_id="cam1", camera_id=1, capture_queue_size=8),
        ],
        source_frames=1,
        hub=None,
    )

    result = runtime.set_source_detectors("cam0", ["movement"])
    assert result.success is True
    assert result.data["enabled_detectors"] == ["movement"]
    assert runtime.get_source_detectors("cam0")["enabled_detectors"] == ["movement"]
    assert runtime.get_source_detectors("cam1")["enabled_detectors"] != ["movement"]

    bulk = runtime.update_detectors(enabled_detectors=["cv"])
    assert bulk["cam0"]["enabled_detectors"] == ["cv"]
    assert bulk["cam1"]["enabled_detectors"] == ["cv"]


def test_runtime_stop_resume_and_reset_source_commands_work():
    def detection_factory(**kwargs):
        manager = _DetectionManagerStub(**kwargs)
        manager.movement_detector = _MovementResetStub()
        return manager

    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=20,
        hub=None,
        detection_factory=detection_factory,
    )

    runtime.start()
    time.sleep(0.05)

    stop_result = runtime.stop_source("cam0")
    assert stop_result.success is True
    assert runtime.source_contexts["cam0"].stop_requested is True

    resume_result = runtime.resume_source("cam0")
    assert resume_result.success is True
    assert runtime.source_contexts["cam0"].stop_requested is False

    runtime.source_contexts["cam0"].latest_frame_for_reset = np.zeros((8, 8, 3), dtype=np.uint8)
    reset_result = runtime.reset_movement_reference("cam0")
    assert reset_result.success is True
    movement_detector = runtime.source_contexts["cam0"].detection_manager.movement_detector
    assert len(movement_detector.reference_frames) == 1

    runtime.stop(timeout=2.0)


def test_runtime_resume_file_source_restarts_capture_from_beginning():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="file0", input_source="/tmp/sample.mp4", capture_queue_size=8)],
        source_frames=3,
        hub=None,
    )

    runtime.start()
    time.sleep(0.05)
    runtime.stop_source("file0")
    resumed = runtime.resume_source("file0")

    assert resumed.success is True
    assert runtime.source_contexts["file0"].capture_worker is not None
    assert runtime.source_contexts["file0"].generation == 1

    runtime.stop(timeout=2.0)


def test_runtime_preview_callback_supports_structured_payload():
    preview_calls = []

    def on_preview(payload):
        preview_calls.append(payload)

    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=3,
        hub=None,
        show_preview=True,
        preview_callback=on_preview,
    )

    runtime.start()
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    assert preview_calls
    payload = preview_calls[0]
    assert isinstance(payload, PreviewFramePayload)
    assert payload.source_id == "cam0"
    assert payload.frame_index >= 0
    assert payload.video_timestamp >= 0.0
    assert {"processed_frames", "writer_queue_size", "show_fps", "preview_width", "overlays_enabled"} <= set(payload.metadata)
    assert payload.processed_at >= payload.captured_at


def test_runtime_visual_config_is_per_source():
    runtime = _build_runtime(
        sources=[
            SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8),
            SourceConfig(source_id="cam1", camera_id=1, capture_queue_size=8),
        ],
        source_frames=1,
        hub=None,
    )

    result = runtime.set_source_visual_config(
        "cam0",
        SourceVisualConfig(show_fps=True, show_movement_arrow=False, show_boxes=False),
    )
    assert result.success is True

    cam0_visual = runtime.get_source_visual_config("cam0")["visual_config"]
    cam1_visual = runtime.get_source_visual_config("cam1")["visual_config"]
    assert cam0_visual["show_fps"] is True
    assert cam0_visual["show_movement_arrow"] is False
    assert cam0_visual["show_boxes"] is False
    assert cam1_visual["show_fps"] is False
    assert cam1_visual["show_boxes"] is True


def test_runtime_emits_events_for_lifecycle_and_violation_save():
    events = []

    def detection_factory(**kwargs):
        manager = _DetectionManagerStub(**kwargs)
        manager.force_forbidden_violation = True
        return manager

    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=2)],
        source_frames=8,
        hub=None,
        detection_factory=detection_factory,
        event_callback=lambda event: events.append(event),
    )

    runtime.start()
    time.sleep(0.05)
    runtime.stop_source("cam0")
    runtime.resume_source("cam0")
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    event_types = [event.event_type for event in events]
    assert "source_started" in event_types
    assert "source_stopped" in event_types
    assert "source_resumed" in event_types
    assert "violation_saved" in event_types
    violation_event = next(event for event in events if event.event_type == "violation_saved")
    assert violation_event.source_id == "cam0"
    assert violation_event.severity == "info"
    assert {"violation_kind", "violation_id", "media_file"} <= set(violation_event.data)


def test_runtime_reports_source_last_error_in_snapshot():
    class _FailingDetectionManager(_DetectionManagerStub):
        def detect_movement(self, frame, is_obstructed, frame_count=0):
            raise RuntimeError("movement pipeline failed")

    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=2,
        hub=None,
        detection_factory=_FailingDetectionManager,
    )

    runtime.start()
    runtime.wait(timeout=3.0)
    runtime.stop(timeout=2.0)

    state = runtime.get_state()
    stats = runtime.get_stats()

    assert state["sources"]["cam0"]["last_error"] == "movement pipeline failed"
    assert state["sources"]["cam0"]["health_status"] == "error"
    assert stats["sources"]["cam0"]["last_error"] == "movement pipeline failed"
    assert stats["sources"]["cam0"]["health_status"] == "error"
    assert stats["sources"]["cam0"]["postprocess_errors"] > 0


def test_runtime_serializes_commands_while_running():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=15,
        hub=None,
    )

    runtime.start()
    time.sleep(0.05)
    first = runtime.set_source_detectors("cam0", ["movement"])
    second = runtime.set_source_visual_config("cam0", {"show_fps": True})
    third = runtime.stop_source("cam0")

    assert first.success is True
    assert second.success is True
    assert third.success is True

    runtime.stop(timeout=2.0)


def test_runtime_returns_command_queue_full_when_ui_queue_is_saturated():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=1,
        hub=None,
    )

    class _AliveThread:
        def is_alive(self):
            return True

    runtime._running = True
    runtime._scheduler_thread = _AliveThread()
    runtime._command_queue = queue.Queue(maxsize=1)
    runtime._command_queue.put_nowait(("noop", (), {}, queue.Queue(maxsize=1)))

    result = runtime.stop_source("cam0")

    assert result.success is False
    assert result.error == "command_queue_full"
    assert result.message == "Runtime command queue is full."


def test_runtime_returns_command_timeout_when_scheduler_does_not_drain_queue():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=1,
        hub=None,
        command_timeout_sec=0.05,
    )

    class _AliveThread:
        def is_alive(self):
            return True

    runtime._running = True
    runtime._scheduler_thread = _AliveThread()
    runtime._command_queue = queue.Queue(maxsize=8)

    result = runtime.stop_source("cam0")

    assert result.success is False
    assert result.error == "command_timeout"


def test_runtime_applies_hot_topology_updates_and_reports_restart_required_fields():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8, base_priority=1.0)],
        source_frames=20,
        hub=None,
    )

    runtime.start()
    time.sleep(0.05)

    result = runtime.apply_topology_updates(
        {
            "scheduler": {
                "dispatch_sleep_sec": 0.02,
                "aging_factor": 1.4,
                "infer_queue_size": 99,
            },
            "sources": [
                {
                    "source_id": "cam0",
                    "detectors": ["movement"],
                    "detector_schedule": {
                        "movement": {
                            "every_n_frames": 5,
                            "min_interval_ms": 10,
                            "priority": 90,
                            "result_ttl_frames": 1,
                        }
                    },
                    "visual_config": {"show_fps": True, "show_boxes": False},
                    "base_priority": 2.5,
                    "camera_id": 3,
                }
            ],
            "queue_limits": {"postprocess_queue_size": 128},
        }
    )
    runtime.stop(timeout=2.0)

    state = runtime.get_state()

    assert result.success is True
    assert result.applied["scheduler.dispatch_sleep_sec"] == 0.02
    assert result.applied["scheduler.aging_factor"] == 1.4
    assert result.applied["sources.cam0.detectors"] == ["movement"]
    assert result.applied["sources.cam0.detector_schedule"]["movement"]["every_n_frames"] == 5
    assert result.applied["sources.cam0.visual_config"]["show_fps"] is True
    assert result.applied["sources.cam0.base_priority"] == 2.5
    assert result.rejected["scheduler.infer_queue_size"] == "restart_required"
    assert result.rejected["sources.cam0.camera_id"] == "restart_required"
    assert result.rejected["queue_limits.postprocess_queue_size"] == "restart_required"
    assert "scheduler.infer_queue_size" in result.restart_required
    assert state["sources"]["cam0"]["enabled_detectors"] == ["movement"]
    assert state["sources"]["cam0"]["detector_schedule"]["movement"]["every_n_frames"] == 5
    assert state["sources"]["cam0"]["visual_config"]["show_fps"] is True
    assert state["sources"]["cam0"]["visual_config"]["show_boxes"] is False


def test_runtime_apply_topology_updates_returns_contract_while_stopped():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=1,
        hub=None,
    )

    result = runtime.apply_topology_updates({"scheduler": {"dispatch_sleep_sec": 0.01}})

    assert result.success is True
    assert result.applied["scheduler.dispatch_sleep_sec"] == 0.01
    assert result.rejected == {}


def test_runtime_exposes_configuration_and_capabilities_for_ui():
    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=1,
        hub=None,
        show_preview=True,
        preview_callback=lambda payload: None,
        event_callback=lambda event: None,
    )

    runtime_config = runtime.get_runtime_configuration()
    capabilities = runtime.describe_ui_capabilities()

    assert runtime_config["engine"] == "centralized"
    assert runtime_config["sources"]["cam0"]["source_type"] == "camera"
    assert "movement" in runtime_config["sources"]["cam0"]["detector_schedule"]
    assert runtime_config["preview_callback_attached"] is True
    assert runtime_config["event_callback_attached"] is True
    assert {
        "policy",
        "aging_factor",
        "backlog_factor",
        "starvation_threshold_sec",
        "starvation_boost",
        "dispatch_sleep_sec",
        "infer_queue_size",
        "infer_overflow_strategy",
    } <= set(runtime_config["scheduler"])
    assert capabilities["supports_runtime_configuration_snapshot"] is True
    assert capabilities["supports_detector_schedule_updates"] is True
    assert "detector_schedule" in capabilities["mutable_source_fields"]
    assert capabilities["source_command_names"] == [
        "set_source_detectors",
        "get_source_detectors",
        "stop_source",
        "resume_source",
        "reset_movement_reference",
        "set_source_visual_config",
        "get_source_visual_config",
    ]


def test_runtime_emits_source_queue_drop_event():
    events = []

    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=1)],
        source_frames=20,
        hub=None,
        dispatch_sleep_sec=0.02,
        event_callback=lambda event: events.append(event),
    )

    runtime.start()
    runtime.wait(timeout=4.0)
    runtime.stop(timeout=2.0)

    drop_event = next(event for event in events if event.event_type == "source_queue_drop")
    assert drop_event.source_id == "cam0"
    assert drop_event.severity == "warning"
    assert {"drop_policy", "dropped_frames"} <= set(drop_event.data)


def test_runtime_emits_infer_queue_overflow_event():
    events = []

    runtime = _build_runtime(
        sources=[SourceConfig(source_id="cam0", camera_id=0, capture_queue_size=8)],
        source_frames=25,
        hub=_SlowHubStub(),
        infer_queue_size=1,
        dispatch_sleep_sec=0.0005,
        event_callback=lambda event: events.append(event),
    )

    runtime.start()
    runtime.wait(timeout=5.0)
    runtime.stop(timeout=2.0)

    overflow_event = next(event for event in events if event.event_type == "infer_queue_overflow")
    assert overflow_event.source_id == "cam0"
    assert overflow_event.severity == "warning"
    assert {"infer_queue_size", "strategy"} <= set(overflow_event.data)
