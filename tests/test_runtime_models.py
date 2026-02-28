"""Файл: tests/test_runtime_models.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.runtime.topology: используется для передачи данных или вызова связанной логики."""

import pytest

from src.runtime.topology import (
    RuntimeTopology,
    SourceConfig,
    apply_profile_defaults_to_topology_payload,
    auto_select_runtime_profile,
    get_runtime_profile,
    resolve_runtime_profile,
)


def test_runtime_topology_from_dict_parses_sources_and_scheduler():
    topology = RuntimeTopology.from_dict(
        {
            "sources": [
                {"camera_id": 0, "base_priority": 2.0},
                {"input": "/tmp/cam1.mp4", "source_id": "front-door"},
            ],
            "scheduler": {
                "policy": "lag_aware",
                "aging_factor": 3.0,
                "infer_queue_size": 12,
            },
            "infer_workers": 1,
            "postprocess_workers": 2,
        }
    )

    assert len(topology.sources) == 2
    assert topology.sources[0].source_id == "camera_0"
    assert topology.sources[0].camera_id == 0
    assert topology.sources[0].base_priority == 2.0
    assert topology.sources[1].source_id == "front-door"
    assert topology.scheduler.policy == "lag_aware"
    assert topology.scheduler.aging_factor == 3.0
    assert topology.scheduler.infer_queue_size == 12
    assert topology.postprocess_workers == 2


def test_runtime_topology_defaults_file_source_id_from_input_filename():
    topology = RuntimeTopology.from_dict(
        {
            "sources": [
                {"input": "/tmp/archive/front-view.mp4"},
            ],
        }
    )
    assert topology.sources[0].source_id == "source_front-view"


def test_source_config_requires_input_or_camera_id():
    with pytest.raises(ValueError):
        SourceConfig(source_id="x", input_source=None, camera_id=None)


def test_source_config_rejects_invalid_drop_policy():
    with pytest.raises(ValueError):
        SourceConfig(
            source_id="cam0",
            camera_id=0,
            drop_policy="drop_random",
        )


def test_runtime_topology_rejects_duplicate_source_ids():
    with pytest.raises(ValueError):
        RuntimeTopology.from_dict(
            {
                "sources": [
                    {"source_id": "cam", "camera_id": 0},
                    {"source_id": "cam", "camera_id": 1},
                ]
            }
        )


def test_get_runtime_profile_and_apply_defaults_to_payload():
    profile = get_runtime_profile("laptop")
    assert profile.name == "laptop"
    assert profile.source_queue_size > 0

    payload = {
        "cameras": [{"camera_id": 0}],
        "scheduler": {},
    }
    patched = apply_profile_defaults_to_topology_payload(payload, profile)
    assert patched["scheduler"]["infer_queue_size"] == profile.infer_queue_size
    assert patched["scheduler"]["dispatch_sleep_sec"] == profile.dispatch_sleep_sec
    assert patched["infer_workers"] == profile.infer_workers
    assert patched["postprocess_workers"] == profile.postprocess_workers
    assert patched["cameras"][0]["capture_queue_size"] == profile.source_queue_size


def test_auto_select_runtime_profile_server_by_resources():
    decision = auto_select_runtime_profile(
        requested_device="cuda:0",
        cpu_count=16,
        total_ram_gb=32.0,
        cuda_available=True,
    )
    assert decision.profile.name == "server"


def test_auto_select_runtime_profile_laptop_by_resources():
    decision = auto_select_runtime_profile(
        requested_device="cpu",
        cpu_count=4,
        total_ram_gb=8.0,
        cuda_available=False,
    )
    assert decision.profile.name == "laptop"


def test_auto_select_runtime_profile_balanced_mid_tier():
    decision = auto_select_runtime_profile(
        requested_device="cpu",
        cpu_count=8,
        total_ram_gb=16.0,
        cuda_available=False,
    )
    assert decision.profile.name == "balanced"


def test_resolve_runtime_profile_respects_manual_profile_request():
    decision = resolve_runtime_profile(
        requested_profile="server",
        requested_device="cpu",
        cpu_count=4,
        total_ram_gb=8.0,
        cuda_available=False,
    )
    assert decision.profile.name == "server"
    assert decision.requested_device == "cpu"
