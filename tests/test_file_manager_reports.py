"""Файл: tests/test_file_manager_reports.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.utils.io.file_manager: используется для передачи данных или вызова связанной логики."""

from pathlib import Path

from src.utils.io.file_manager import FileManager


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_save_obstruction_report_with_legacy_schema(tmp_path):
    manager = FileManager(base_dir=str(tmp_path))

    ok = manager.save_violation_report(
        {
            "violation_id": 1,
            "timestamp": 123.45,
            "video_timestamp": 10.5,
            "reasons": ["dark_frame"],
            "detectors_count": 2,
            "metrics": {"brightness": 0.1},
        }
    )

    assert ok is True
    reports = list(tmp_path.glob("report_*.txt"))
    assert len(reports) == 1
    content = _read_text(reports[0])
    assert "Type: obstruction" in content
    assert "reasons:" in content
    assert "detectors_count:" in content


def test_save_movement_report_with_defaults(tmp_path):
    manager = FileManager(base_dir=str(tmp_path))

    ok = manager.save_violation_report(
        {
            "violation_type": "movement",
            "violation_id": 9,
            "video_timestamp": 5.0,
            "details": {"duration": 3.2, "movement_info": {"reason": "rotation"}},
        },
        report_filename="movement_unified.txt",
    )

    assert ok is True
    report_path = tmp_path / "movement_unified.txt"
    assert report_path.exists()
    content = _read_text(report_path)
    assert "Type: movement" in content
    assert "duration: 3.200" in content
    assert "movement_info:" in content


def test_save_forbidden_report_to_forbidden_dir(tmp_path):
    manager = FileManager(base_dir=str(tmp_path))

    ok = manager.save_violation_report(
        {
            "violation_type": "forbidden",
            "violation_id": 2,
            "video_timestamp": 2.5,
            "media_file": "forbidden_items_1.jpg",
            "details": {
                "objects": [{"class": "phone", "duration": 4.1}],
                "affected_classes": ["phone"],
            },
        },
        report_filename="forbidden_items_1.txt",
        event_type="forbidden_items",
    )

    assert ok is True
    report_path = tmp_path / "forbidden_items" / "forbidden_items_1.txt"
    assert report_path.exists()
    content = _read_text(report_path)
    assert "Type: forbidden" in content
    assert "objects:" in content
    assert "affected_classes:" in content


def test_save_dms_report_with_string_timestamp_fallback(tmp_path):
    manager = FileManager(base_dir=str(tmp_path))

    ok = manager.save_violation_report(
        {
            "violation_type": "dms",
            "violation_id": 3,
            "timestamp": "invalid-ts",
            "video_timestamp": "invalid-vt",
            "frame_number": "bad-frame",
            "details": {"violations": [{"type": "eye_closed"}]},
        },
        report_filename="dms.txt",
    )

    assert ok is True
    report_path = tmp_path / "dms.txt"
    assert report_path.exists()
    content = _read_text(report_path)
    assert "Type: dms" in content
    assert "Frame: -1" in content
    assert "violations:" in content


def test_detector_dirs_are_created_lazily(tmp_path):
    manager = FileManager(base_dir=str(tmp_path))


    assert (tmp_path / "movement").exists() is False
    assert (tmp_path / "obstruction").exists() is False
    assert (tmp_path / "dms").exists() is False
    assert (tmp_path / "forbidden_items").exists() is False


    ok = manager.save_violation_report(
        {
            "violation_type": "movement",
            "violation_id": 1,
            "video_timestamp": 1.0,
            "media_file": "camera_movement_test.mp4",
            "details": {"duration": 1.2},
        },
        report_filename="movement_test.txt",
        event_type="movement",
    )
    assert ok is True
    assert (tmp_path / "movement").exists() is True
    assert (tmp_path / "obstruction").exists() is False
