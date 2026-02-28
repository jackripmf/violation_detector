"""Файл: tests/test_main_cli.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- start_scripts: используется для передачи данных или вызова связанной логики."""

import sys

import pytest

from start_scripts import single_source_launcher as launcher


def test_parse_cli_args_requires_input_or_camera_id(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["single_source_launcher.py"])
    with pytest.raises(SystemExit) as exc:
        launcher.parse_cli_args()
    assert exc.value.code == 2


def test_parse_cli_args_accepts_camera_mode_without_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["single_source_launcher.py", "--camera-id", "2"])
    args = launcher.parse_cli_args()
    assert args.camera_id == 2
    assert args.input is None
    assert args.runtime_engine == "centralized"


def test_parse_cli_args_normalizes_detector_aliases(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_source_launcher.py",
            "--camera-id",
            "1",
            "--detectors",
            "cvdetector",
            "yolodetector",
            "cv",
        ],
    )
    args = launcher.parse_cli_args()
    assert args.detectors == ["cv", "yolo"]


def test_parse_cli_args_accepts_centralized_runtime_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "single_source_launcher.py",
            "--camera-id",
            "0",
            "--runtime-engine",
            "centralized",
            "--runtime-profile",
            "server",
            "--source-queue-size",
            "12",
            "--scheduler-infer-queue-size",
            "40",
            "--source-drop-policy",
            "drop_newest",
            "--infer-workers",
            "2",
            "--postprocess-workers",
            "3",
            "--scheduler-dispatch-sleep",
            "0.01",
        ],
    )
    args = launcher.parse_cli_args()
    assert args.runtime_engine == "centralized"
    assert args.runtime_profile == "server"
    assert args.source_queue_size == 12
    assert args.scheduler_infer_queue_size == 40
    assert args.source_drop_policy == "drop_newest"
    assert args.infer_workers == 2
    assert args.postprocess_workers == 3
    assert args.scheduler_dispatch_sleep == 0.01


def test_parse_cli_args_allows_topology_config_without_input_or_camera(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["single_source_launcher.py", "--topology-config", "/tmp/topology.json"],
    )
    args = launcher.parse_cli_args()
    assert args.topology_config == "/tmp/topology.json"


def test_parse_cli_args_defaults_runtime_profile_to_auto(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["single_source_launcher.py", "--camera-id", "0"])
    args = launcher.parse_cli_args()
    assert args.runtime_profile == "auto"
