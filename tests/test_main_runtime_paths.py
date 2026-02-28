"""Файл: tests/test_main_runtime_paths.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- start_scripts.single_source_launcher: используется для передачи данных или вызова связанной логики."""

import os

from start_scripts.single_source_launcher import (
    resolve_runtime_manifest_output_dir,
    resolve_runtime_paths,
)


def test_resolve_runtime_paths_camera_specific_output_and_log():
    output_dir, log_file = resolve_runtime_paths(
        output_root="violations",
        camera_id=7,
        input_source=None,
    )
    assert output_dir.endswith(os.path.join("violations", "camera_7"))
    assert log_file.endswith(os.path.join("violations", "logs", "camera_7.log"))


def test_resolve_runtime_paths_file_source_uses_output_logs():
    output_dir, log_file = resolve_runtime_paths(
        output_root="violations",
        camera_id=None,
        input_source="/tmp/my_video.mp4",
    )
    assert output_dir.endswith(os.path.join("violations", "source_my_video"))
    assert log_file.endswith(os.path.join("violations", "logs", "my_video.log"))


def test_resolve_runtime_paths_default_logs_inside_output_root():
    output_dir, log_file = resolve_runtime_paths(
        output_root="violations",
        camera_id=0,
        input_source=None,
    )
    assert output_dir.endswith(os.path.join("violations", "camera_0"))
    assert log_file.endswith(os.path.join("violations", "logs", "camera_0.log"))


def test_resolve_runtime_paths_without_source_keeps_output_root():
    output_dir, log_file = resolve_runtime_paths(
        output_root="violations",
        camera_id=None,
        input_source=None,
    )
    assert output_dir.endswith("violations")
    assert log_file.endswith(os.path.join("violations", "logs", "source.log"))


def test_resolve_runtime_manifest_output_dir_for_legacy_uses_source_output():
    manifest_dir = resolve_runtime_manifest_output_dir(
        output_root="violations",
        runtime_engine="legacy",
        resolved_output_dir="/tmp/violations/source_front",
    )
    assert manifest_dir == os.path.abspath("/tmp/violations/source_front")


def test_resolve_runtime_manifest_output_dir_for_centralized_uses_output_root():
    manifest_dir = resolve_runtime_manifest_output_dir(
        output_root="violations",
        runtime_engine="centralized",
        resolved_output_dir="/tmp/violations/source_front",
    )
    assert manifest_dir == os.path.abspath("violations")
