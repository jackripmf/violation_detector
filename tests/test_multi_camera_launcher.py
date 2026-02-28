"""Файл: tests/test_multi_camera_launcher.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

import importlib.util
import json
from pathlib import Path


def _load_launcher_module():
    project_root = Path(__file__).resolve().parents[1]
    launcher_path = project_root / "start_scripts" / "multi_camera_launcher.py"
    spec = importlib.util.spec_from_file_location("multi_camera_launcher", launcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_camera_command_contains_expected_flags():
    launcher = _load_launcher_module()
    cmd = launcher.build_camera_command(
        python_executable="/usr/bin/python3",
        main_script="/repo/start_scripts/single_source_launcher.py",
        camera_cfg={"camera_id": 2, "device": "cuda:1", "show_fps": True},
        global_cfg={
            "output": "violations",
            "imgsz": 640,
            "no_preview": True,
            "detectors": ["all"],
        },
    )

    assert cmd[:2] == ["/usr/bin/python3", "/repo/start_scripts/single_source_launcher.py"]
    assert "--camera-id" in cmd and "2" in cmd
    assert "--output" in cmd and "violations" in cmd
    assert "--log-dir" not in cmd
    assert "--device" in cmd and "cuda:1" in cmd
    assert "--imgsz" in cmd and "640" in cmd
    assert "--show-fps" in cmd
    assert "--no-preview" in cmd


def test_build_camera_command_ignores_log_dir_key():
    launcher = _load_launcher_module()
    cmd = launcher.build_camera_command(
        python_executable="/usr/bin/python3",
        main_script="/repo/start_scripts/single_source_launcher.py",
        camera_cfg={"camera_id": 2},
        global_cfg={"output": "violations", "log_dir": "custom_logs"},
    )
    assert "--log-dir" not in cmd


def test_build_camera_command_supports_runtime_engine_override():
    launcher = _load_launcher_module()
    cmd = launcher.build_camera_command(
        python_executable="/usr/bin/python3",
        main_script="/repo/start_scripts/single_source_launcher.py",
        camera_cfg={"camera_id": 2},
        global_cfg={"runtime_engine": "legacy"},
    )
    assert "--runtime-engine" in cmd and "legacy" in cmd
    assert "--runtime-profile" not in cmd


def test_load_config_requires_non_empty_cameras(tmp_path):
    launcher = _load_launcher_module()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"global": {}, "cameras": []}), encoding="utf-8")
    try:
        launcher.load_config(str(cfg_path))
        assert False, "Expected ValueError for empty cameras"
    except ValueError:
        assert True


def test_load_config_success(tmp_path):
    launcher = _load_launcher_module()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps({"global": {"output": "violations"}, "cameras": [{"camera_id": 0}]}),
        encoding="utf-8",
    )
    cfg = launcher.load_config(str(cfg_path))
    assert "cameras" in cfg
    assert cfg["cameras"][0]["camera_id"] == 0


def test_build_centralized_command_contains_topology_runtime_flags():
    launcher = _load_launcher_module()
    cmd = launcher.build_centralized_command(
        python_executable="/usr/bin/python3",
        main_script="/repo/start_scripts/single_source_launcher.py",
        config_path="/repo/config.json",
        global_cfg={
            "output": "violations",
            "device": "cpu",
            "imgsz": 640,
            "source_queue_size": 16,
            "scheduler_infer_queue_size": 48,
            "source_drop_policy": "drop_oldest",
        },
    )

    assert cmd[:2] == ["/usr/bin/python3", "/repo/start_scripts/single_source_launcher.py"]
    assert "--runtime-engine" in cmd and "centralized" in cmd
    assert "--topology-config" in cmd and "/repo/config.json" in cmd
    assert "--source-queue-size" in cmd and "16" in cmd
    assert "--scheduler-infer-queue-size" in cmd and "48" in cmd
    assert "--source-drop-policy" in cmd and "drop_oldest" in cmd


def test_launch_processes_dry_run_prints_commands(capsys):
    launcher = _load_launcher_module()
    commands = [
        ["/usr/bin/python3", "/repo/start_scripts/single_source_launcher.py", "--camera-id", "0"],
        ["/usr/bin/python3", "/repo/start_scripts/single_source_launcher.py", "--camera-id", "1"],
    ]

    code = launcher.launch_processes(commands, dry_run=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "[DRY-RUN #0]" in out
    assert "--camera-id 0" in out
    assert "[DRY-RUN #1]" in out


def test_launch_processes_dry_run_redacts_rtsp_credentials(capsys):
    launcher = _load_launcher_module()
    commands = [
        [
            "/usr/bin/python3",
            "/repo/start_scripts/single_source_launcher.py",
            "--input",
            "rtsp://user:secret@example.local:554/live",
        ],
    ]

    code = launcher.launch_processes(commands, dry_run=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "user:secret@" not in out
    assert "rtsp://***:***@example.local:554/live" in out


def test_build_process_meta_defaults_to_restart_for_camera():
    launcher = _load_launcher_module()
    meta = launcher.build_process_meta(
        camera_cfg={"camera_id": 0},
        global_cfg={},
    )
    assert meta["source_kind"] == "camera"
    assert meta["restart_on_failure"] is True


def test_build_process_meta_defaults_to_no_restart_for_file():
    launcher = _load_launcher_module()
    meta = launcher.build_process_meta(
        camera_cfg={"input": "/tmp/video.mp4"},
        global_cfg={},
    )
    assert meta["source_kind"] == "file"
    assert meta["restart_on_failure"] is False
