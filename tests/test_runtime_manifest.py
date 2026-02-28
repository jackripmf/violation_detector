"""Файл: tests/test_runtime_manifest.py
Тип: файл автотестов.
Назначение: проверяет сборку и запись resolved runtime manifest.
Связи: взаимодействует с runtime manifest утилитой и runtime profile моделями через публичные API.
Импортируемые внутренние модули:
- src.utils.io.runtime_manifest: используется для передачи данных или вызова связанной логики.
- src.runtime.topology: используется для передачи данных или вызова связанной логики."""

import json

from src.runtime.topology import RuntimeTopology, resolve_runtime_profile
from src.utils.io.runtime_manifest import (
    build_runtime_manifest,
    summarize_topology,
    write_runtime_manifest,
)


def test_summarize_topology_single_source_camera_mode():
    summary = summarize_topology(topology=None, input_source=None, camera_id=3)

    assert summary["source_count"] == 1
    assert summary["source_ids"] == ["camera_3"]
    assert summary["source_kinds"] == ["camera"]
    assert summary["mode"] == "single_source"


def test_build_and_write_runtime_manifest(tmp_path):
    topology = RuntimeTopology.from_dict(
        {
            "sources": [
                {"source_id": "front", "input": "/tmp/front.mp4"},
                {"source_id": "rear", "camera_id": 1},
            ],
        }
    )
    profile_decision = resolve_runtime_profile(
        requested_profile="server",
        requested_device="cuda:0",
        cpu_count=16,
        total_ram_gb=32.0,
        cuda_available=True,
    )
    manifest = build_runtime_manifest(
        runtime_engine="centralized",
        runtime_profile_name=profile_decision.profile.name,
        runtime_profile_mode="manual",
        requested_runtime_profile="server",
        device="cuda:0",
        use_half=True,
        imgsz=960,
        source_queue_size=16,
        scheduler_infer_queue_size=256,
        infer_workers=1,
        postprocess_workers=2,
        scheduler_dispatch_sleep_sec=0.0015,
        output_dir=str(tmp_path / "violations"),
        save_dir=str(tmp_path / "violations"),
        log_file=str(tmp_path / "violations" / "logs" / "runtime.log"),
        topology_summary=summarize_topology(topology=topology, input_source=None, camera_id=None),
        source_drop_policy="drop_oldest",
        show_preview=False,
        show_fps=True,
        profile_decision=profile_decision,
        shared_model_path="/repo/src/utils/models/best_auto.pt",
    )

    manifest_path = write_runtime_manifest(manifest, str(tmp_path / "violations"))
    saved_manifest = json.loads((tmp_path / "violations" / "runtime_manifest.json").read_text(encoding="utf-8"))

    assert manifest_path.endswith("runtime_manifest.json")
    assert saved_manifest["runtime"]["engine"] == "centralized"
    assert saved_manifest["runtime_profile"]["selected"] == "server"
    assert saved_manifest["runtime_profile"]["selection_mode"] == "manual"
    assert saved_manifest["topology"]["source_count"] == 2
    assert saved_manifest["paths"]["shared_model_path"] == "/repo/src/utils/models/best_auto.pt"
