"""Файл: tests/test_runtime_snapshot_builders.py
Тип: файл автотестов.
Назначение: проверяет стабильность общих builder-ов runtime snapshots.
Связи: взаимодействует с runtime_snapshot_builders и runtime_models через публичные контракты."""

from src.runtime.topology import RuntimeTopology, SchedulerConfig, SourceConfig
from src.runtime.snapshot_builders import (
    build_runtime_capabilities_snapshot,
    build_runtime_configuration_snapshot,
    build_source_configuration_snapshot,
)


def test_runtime_snapshot_builders_create_consistent_configuration_snapshot():
    topology = RuntimeTopology(
        sources=[
            SourceConfig(
                source_id="cam0",
                camera_id=0,
                capture_queue_size=8,
                enabled_detectors=["movement"],
            )
        ],
        scheduler=SchedulerConfig(
            dispatch_sleep_sec=0.01,
            infer_queue_size=32,
            infer_overflow_strategy="drop_oldest",
        ),
        infer_workers=2,
        postprocess_workers=3,
    )
    source_snapshot = build_source_configuration_snapshot(
        source_cfg=topology.sources[0],
        enabled_detectors=["movement"],
        detector_schedule={"movement": {"every_n_frames": 2}},
        visual_config={"show_fps": True},
        save_dir="violations",
    )

    snapshot = build_runtime_configuration_snapshot(
        engine="centralized",
        running=True,
        save_dir="violations",
        show_preview=True,
        default_visual_config={"show_fps": False},
        async_violation_writes=True,
        writer_queue_max_size=128,
        writer_overflow_strategy="drop_newest",
        preview_callback_attached=True,
        event_callback_attached=False,
        command_timeout_sec=1.5,
        topology=topology,
        sources={"cam0": source_snapshot},
    ).to_dict()

    assert snapshot["engine"] == "centralized"
    assert snapshot["scheduler"]["dispatch_sleep_sec"] == 0.01
    assert snapshot["scheduler"]["infer_queue_size"] == 32
    assert snapshot["infer_workers"] == 2
    assert snapshot["postprocess_workers"] == 3
    assert snapshot["sources"]["cam0"]["source_type"] == "camera"
    assert snapshot["sources"]["cam0"]["output_dir"] == "violations/cam0"


def test_runtime_snapshot_builders_expose_expected_capabilities_for_legacy_and_centralized():
    legacy_caps = build_runtime_capabilities_snapshot(
        engine="legacy",
        supports_per_source_control=False,
        supports_preview_callback=False,
        supports_event_callback=False,
        supports_hot_topology_updates=False,
        supports_detector_schedule_updates=False,
    ).to_dict()
    centralized_caps = build_runtime_capabilities_snapshot(
        engine="centralized",
        supports_per_source_control=True,
        supports_preview_callback=True,
        supports_event_callback=True,
        supports_hot_topology_updates=True,
        supports_detector_schedule_updates=True,
        restart_required_fields={"runtime": ["infer_workers"], "source": ["camera_id"]},
    ).to_dict()

    assert legacy_caps["supports_runtime_configuration_snapshot"] is True
    assert legacy_caps["source_command_names"] == []
    assert legacy_caps["mutable_runtime_fields"] == []
    assert centralized_caps["supports_detector_schedule_updates"] is True
    assert "set_source_visual_config" in centralized_caps["source_command_names"]
    assert "detector_schedule" in centralized_caps["mutable_source_fields"]
    assert centralized_caps["restart_required_fields"]["runtime"] == ["infer_workers"]
