"""Файл: tests/test_runtime_controller.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.runtime.controller: используется для передачи данных или вызова связанной логики."""

import time

from src.runtime.controller import RuntimeController, RuntimeConfig


class _DetectionManagerStub:
    def __init__(self):
        self.last_enabled = []

    def set_runtime_detectors(self, enabled_detectors=None):
        if enabled_detectors is not None:
            self.last_enabled = list(enabled_detectors)
        return {"enabled_detectors": list(self.last_enabled)}


class _StatsManagerStub:
    def get_current_stats(self):
        return {"saved_violations": 2, "total_frames": 10}


class _ViolationManagerStub:
    def get_writer_queue_size(self):
        return 3


class _SourceManagerStub:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _ProcessorStub:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.frame_count = 0
        self.is_running = False
        self.detection_manager = _DetectionManagerStub()
        self.stats_manager = _StatsManagerStub()
        self.violation_manager = _ViolationManagerStub()
        self.source_manager = _SourceManagerStub()

    def process_source(self, show_preview=True, max_duration=None):
        self.is_running = True
        started_at = time.time()
        while self.is_running:
            self.frame_count += 1
            time.sleep(0.01)
            if max_duration is not None and (time.time() - started_at) >= max_duration:
                break
        self.is_running = False


class _MultiSourceRuntimeStub:
    def __init__(self, topology, **kwargs):
        self.topology = topology
        self.init_kwargs = kwargs
        self.running = False
        self.stopped = False
        self.visual_config = {"show_fps": False}

    def start(self):
        self.running = True
        return True

    def wait(self, timeout=None):
        time.sleep(0.02)
        self.running = False

    def stop(self, timeout=5.0):
        self.running = False
        self.stopped = True

    def update_detectors(self, enabled_detectors=None):
        return {
            "camera_1": {
                "enabled_detectors": list(enabled_detectors or []),
            }
        }

    def set_source_detectors(self, source_id, detectors):
        return type("Command", (), {
            "success": True,
            "source_id": source_id,
            "error": None,
            "message": "ok",
            "data": {"enabled_detectors": list(detectors)},
        })()

    def get_source_detectors(self, source_id):
        return {
            "source_id": source_id,
            "enabled_detectors": ["forbidden"],
            "detector_statuses": [{"name": "ForbiddenItemsDetector", "enabled": True, "active": True}],
            "detector_schedule": {"forbidden": {"every_n_frames": 2}},
        }

    def stop_source(self, source_id):
        return type("Command", (), {
            "success": True,
            "source_id": source_id,
            "error": None,
            "message": "stopped",
            "data": {},
        })()

    def resume_source(self, source_id):
        return type("Command", (), {
            "success": True,
            "source_id": source_id,
            "error": None,
            "message": "resumed",
            "data": {},
        })()

    def reset_movement_reference(self, source_id):
        return type("Command", (), {
            "success": True,
            "source_id": source_id,
            "error": None,
            "message": "reset",
            "data": {},
        })()

    def set_source_visual_config(self, source_id, config):
        self.visual_config = dict(config)
        return type("Command", (), {
            "success": True,
            "source_id": source_id,
            "error": None,
            "message": "visual",
            "data": {"visual_config": dict(config)},
        })()

    def get_source_visual_config(self, source_id):
        return {"source_id": source_id, "visual_config": dict(self.visual_config)}

    def get_stats(self):
        return {
            "running": False,
            "sources": {
                "camera_1": {
                    "runtime_stats": {"saved_violations": 1, "total_frames": 11},
                    "enabled_detectors": ["forbidden"],
                    "detector_statuses": [{"name": "ForbiddenItemsDetector", "enabled": True, "active": True}],
                    "detector_schedule": {"forbidden": {"every_n_frames": 2}},
                    "visual_config": {"show_fps": False},
                    "health_status": "running",
                    "last_error": None,
                    "writer_queue_size": 3,
                }
            },
            "infer_queue_size": 4,
            "postprocess_queue_size": 0,
            "max_infer_queue_depth": 4,
            "max_postprocess_queue_depth": 0,
            "command_queue_size": 0,
            "total_sources": 1,
            "total_processed_frames": 11,
            "hub_metrics": {},
            "last_error": None,
        }

    def get_state(self):
        return {
            "running": self.running,
            "infer_queue_size": 4,
            "postprocess_queue_size": 0,
            "max_infer_queue_depth": 4,
            "max_postprocess_queue_depth": 0,
            "sources": {
                "camera_1": {
                    "captured_frames": 11,
                    "served_count": 10,
                    "source_queue_size": 1,
                    "dropped_before_infer": 0,
                    "capture_dropped_frames": 0,
                    "processed_frames": 11,
                    "max_source_queue_depth": 2,
                    "stop_requested": False,
                    "preview_closed": False,
                    "enabled_detectors": ["forbidden"],
                    "detector_statuses": [{"name": "ForbiddenItemsDetector", "enabled": True, "active": True}],
                    "detector_schedule": {"forbidden": {"every_n_frames": 2}},
                    "visual_config": {"show_fps": False},
                    "writer_queue_size": 3,
                    "capture_running": True,
                    "health_status": "running",
                    "last_error": None,
                }
            },
            "command_queue_size": 0,
            "last_error": None,
        }

    def apply_topology_updates(self, updates):
        return type("TopologyUpdateResult", (), {
            "success": True,
            "applied": {"scheduler.dispatch_sleep_sec": updates["scheduler"]["dispatch_sleep_sec"]},
            "rejected": {"sources.camera_1.camera_id": "restart_required"},
            "restart_required": ["sources.camera_1.camera_id"],
            "message": "Topology updates applied.",
        })()

    def describe_restart_required_updates(self):
        return {
            "runtime": ["scheduler.infer_queue_size"],
            "source": ["camera_id"],
        }

    def get_runtime_configuration(self):
        return {
            "engine": "centralized",
            "running": self.running,
            "save_dir": "violations",
            "show_preview": True,
            "default_visual_config": {"show_fps": False},
            "async_violation_writes": True,
            "writer_queue_max_size": 256,
            "writer_overflow_strategy": "drop_newest",
            "preview_callback_attached": True,
            "event_callback_attached": True,
            "command_timeout_sec": 1.0,
            "infer_workers": 1,
            "postprocess_workers": 1,
            "scheduler": {"dispatch_sleep_sec": 0.003},
            "sources": {
                "camera_1": {
                    "source_id": "camera_1",
                    "source_type": "camera",
                    "input_source": None,
                    "camera_id": 1,
                    "base_priority": 1.0,
                    "capture_queue_size": 8,
                    "drop_policy": "drop_oldest",
                    "enabled_detectors": ["forbidden"],
                    "detector_schedule": {"forbidden": {"every_n_frames": 2}},
                    "visual_config": {"show_fps": False},
                    "output_dir": "violations/camera_1",
                }
            },
        }

    def describe_ui_capabilities(self):
        return {
            "engine": "centralized",
            "supports_per_source_control": True,
            "supports_preview_callback": True,
            "supports_event_callback": True,
            "supports_hot_topology_updates": True,
            "supports_runtime_configuration_snapshot": True,
            "supports_detector_schedule_updates": True,
            "detector_catalog": ["cv", "dark", "yolo", "movement", "forbidden", "dms"],
            "visual_config_fields": ["show_fps", "show_boxes"],
            "source_command_names": ["set_source_detectors", "set_source_visual_config"],
            "mutable_runtime_fields": ["scheduler.dispatch_sleep_sec"],
            "mutable_source_fields": ["detector_schedule", "visual_config"],
            "restart_required_fields": {"runtime": ["scheduler.infer_queue_size"], "source": ["camera_id"]},
        }


def test_runtime_controller_start_update_stop_flow():
    config = RuntimeConfig(
        input_source="dummy.mp4",
        enabled_detectors=["all"],
        runtime_engine="legacy",
        show_preview=False,
        max_duration=0.5,
    )
    controller = RuntimeController(config=config, processor_factory=_ProcessorStub)

    assert controller.start() is True
    time.sleep(0.05)
    state = controller.get_state()
    assert state["running"] is True
    assert state["frame_count"] > 0
    assert state["writer_queue_size"] == 3
    assert state["last_error"] is None

    detectors_state = controller.update_detectors(enabled_detectors=["forbidden", "dms"])
    assert detectors_state["enabled_detectors"] == ["forbidden", "dms"]

    stats = controller.get_stats()
    assert stats["saved_violations"] == 2

    assert controller.stop(timeout=2.0) is True
    controller.wait(timeout=2.0)
    assert controller.is_running() is False


def test_runtime_controller_supports_centralized_engine_and_contract_fields():
    config = RuntimeConfig(
        input_source=None,
        camera_id=1,
        runtime_engine="centralized",
        max_duration=0.05,
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )

    assert controller.start() is True
    controller.wait(timeout=1.0)

    state = controller.get_state()
    stats = controller.get_stats()

    assert state["running"] is False
    assert state["frame_count"] == 11
    assert state["uptime_sec"] >= 0.0
    assert state["writer_queue_size"] == 3
    assert state["last_error"] is None
    assert state["sources"]["camera_1"]["visual_config"]["show_fps"] is False
    assert state["sources"]["camera_1"]["health_status"] == "running"
    assert stats["total_sources"] == 1
    assert stats["sources"]["camera_1"]["enabled_detectors"] == ["forbidden"]
    assert stats["sources"]["camera_1"]["writer_queue_size"] == 3


def test_runtime_controller_proxies_per_source_runtime_api():
    config = RuntimeConfig(
        input_source=None,
        camera_id=1,
        runtime_engine="centralized",
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )
    controller.start()

    set_result = controller.set_source_detectors("camera_1", ["forbidden"])
    assert set_result.success is True
    assert set_result.data["enabled_detectors"] == ["forbidden"]

    get_result = controller.get_source_detectors("camera_1")
    assert get_result["enabled_detectors"] == ["forbidden"]
    assert get_result["detector_schedule"]["forbidden"]["every_n_frames"] == 2

    stop_result = controller.stop_source("camera_1")
    resume_result = controller.resume_source("camera_1")
    reset_result = controller.reset_movement_reference("camera_1")
    visual_result = controller.set_source_visual_config("camera_1", {"show_fps": True})
    visual_state = controller.get_source_visual_config("camera_1")

    assert stop_result.success is True
    assert resume_result.success is True
    assert reset_result.success is True
    assert visual_result.success is True
    assert visual_state["visual_config"]["show_fps"] is True


def test_runtime_controller_proxies_topology_hot_reload_api():
    config = RuntimeConfig(
        input_source=None,
        camera_id=1,
        runtime_engine="centralized",
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )
    controller.start()

    result = controller.apply_topology_updates(
        {
            "scheduler": {"dispatch_sleep_sec": 0.02},
            "sources": [{"source_id": "camera_1", "camera_id": 2}],
        }
    )
    restart_required = controller.describe_restart_required_updates()

    assert result.success is True
    assert result.applied["scheduler.dispatch_sleep_sec"] == 0.02
    assert result.rejected["sources.camera_1.camera_id"] == "restart_required"
    assert result.restart_required == ["sources.camera_1.camera_id"]
    assert "scheduler.infer_queue_size" in restart_required["runtime"]


def test_runtime_controller_exposes_runtime_configuration_and_capabilities():
    config = RuntimeConfig(
        input_source=None,
        camera_id=1,
        runtime_engine="centralized",
        preview_callback=lambda payload: None,
        event_callback=lambda event: None,
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )
    controller.start()

    runtime_config = controller.get_runtime_configuration()
    capabilities = controller.describe_ui_capabilities()

    assert runtime_config["sources"]["camera_1"]["source_type"] == "camera"
    assert runtime_config["sources"]["camera_1"]["detector_schedule"]["forbidden"]["every_n_frames"] == 2
    assert runtime_config["preview_callback_attached"] is True
    assert capabilities["supports_detector_schedule_updates"] is True
    assert "detector_schedule" in capabilities["mutable_source_fields"]


def test_runtime_controller_builds_configuration_snapshot_without_started_runtime():
    config = RuntimeConfig(
        input_source="rtsp://camera.local/stream",
        camera_id=None,
        runtime_engine="centralized",
        save_dir="violations",
        enabled_detectors=["movement", "forbidden"],
        show_preview=False,
        show_fps=True,
        source_queue_max_size=12,
        source_drop_policy="drop_newest",
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )

    runtime_config = controller.get_runtime_configuration()
    capabilities = controller.describe_ui_capabilities()

    assert runtime_config["engine"] == "centralized"
    assert runtime_config["running"] is False
    assert runtime_config["show_preview"] is False
    assert runtime_config["sources"]["source_rtsp"]["source_type"] == "rtsp"
    assert runtime_config["sources"]["source_rtsp"]["capture_queue_size"] == 12
    assert runtime_config["sources"]["source_rtsp"]["drop_policy"] == "drop_newest"
    assert runtime_config["sources"]["source_rtsp"]["enabled_detectors"] == ["movement", "forbidden"]
    assert runtime_config["scheduler"]["infer_queue_size"] == config.scheduler_infer_queue_size
    assert capabilities["supports_runtime_configuration_snapshot"] is True
    assert capabilities["supports_hot_topology_updates"] is True
    assert "scheduler.dispatch_sleep_sec" in capabilities["mutable_runtime_fields"]


def test_runtime_controller_legacy_capabilities_match_contract():
    config = RuntimeConfig(
        input_source="dummy.mp4",
        runtime_engine="legacy",
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )

    capabilities = controller.describe_ui_capabilities()

    assert capabilities["engine"] == "legacy"
    assert capabilities["source_command_names"] == []
    assert capabilities["mutable_runtime_fields"] == []
    assert capabilities["mutable_source_fields"] == []
    assert capabilities["restart_required_fields"] == {"runtime": [], "source": []}


def test_runtime_controller_builds_file_source_id_from_filename():
    config = RuntimeConfig(
        input_source="/data/videos/front_cam.mp4",
        camera_id=None,
        runtime_engine="centralized",
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )

    topology = controller._build_single_source_topology()
    assert topology.sources[0].source_id == "source_front_cam"


def test_runtime_controller_derives_preview_width_from_imgsz_for_legacy():
    config = RuntimeConfig(
        input_source="dummy.mp4",
        runtime_engine="legacy",
        imgsz=810,
        preview_width=None,
    )
    controller = RuntimeController(config=config, processor_factory=_ProcessorStub)

    processor = controller._create_runtime_engine()
    assert processor.init_kwargs["preview_width"] == 1440


def test_runtime_controller_derives_preview_width_and_callbacks_for_centralized():
    preview_calls = []
    event_calls = []

    config = RuntimeConfig(
        input_source=None,
        camera_id=1,
        runtime_engine="centralized",
        imgsz=900,
        preview_width=None,
        preview_callback=lambda *args: preview_calls.append(args),
        event_callback=lambda event: event_calls.append(event),
    )
    controller = RuntimeController(
        config=config,
        processor_factory=_ProcessorStub,
        multi_source_runtime_factory=_MultiSourceRuntimeStub,
    )

    runtime = controller._create_runtime_engine()
    assert runtime.init_kwargs["preview_width"] == 1600
    assert runtime.init_kwargs["preview_callback"] is config.preview_callback
    assert runtime.init_kwargs["event_callback"] is config.event_callback
