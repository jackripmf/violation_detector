"""Файл: tests/test_source_redaction.py
Тип: файл автотестов.
Назначение: проверяет маскирование чувствительных источников и команд логирования.
Связи: взаимодействует с утилитами redaction через публичные функции модуля.
Импортируемые внутренние модули:
- src.utils.io.source_redaction: используется для передачи данных или вызова связанной логики."""

from src.utils.io.source_redaction import (
    format_command_for_logging,
    format_path_for_logging,
    format_source_for_logging,
    redact_rtsp_url,
)


def test_redact_rtsp_url_masks_credentials():
    safe_url = redact_rtsp_url("rtsp://user:secret@example.local:554/live")

    assert safe_url == "rtsp://***:***@example.local:554/live"
    assert "user" not in safe_url
    assert "secret" not in safe_url


def test_format_source_for_logging_reduces_file_path_to_filename():
    safe_source = format_source_for_logging("/very/secret/folder/front_camera.mp4")

    assert safe_source == ".../front_camera.mp4"


def test_format_path_for_logging_handles_plain_filename():
    assert format_path_for_logging("camera_1.log") == "camera_1.log"


def test_format_command_for_logging_masks_rtsp_and_paths():
    command = [
        "/usr/bin/python3",
        "/repo/start_scripts/single_source_launcher.py",
        "--input",
        "rtsp://user:secret@example.local:554/live",
        "--output",
        "/srv/violations",
        "--topology-config",
        "/srv/configs/prod_topology.json",
    ]

    safe_command = format_command_for_logging(command)

    assert "user:secret@" not in safe_command
    assert "rtsp://***:***@example.local:554/live" in safe_command
    assert ".../single_source_launcher.py" in safe_command
    assert "--output .../violations" in safe_command
    assert "--topology-config .../prod_topology.json" in safe_command
