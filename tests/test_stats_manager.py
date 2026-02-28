"""Файл: tests/test_stats_manager.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.stats_manager: используется для передачи данных или вызова связанной логики."""

import pytest
import time
from src.processing.stats_manager import StatsManager, SessionStats


class TestStatsManager:

    def test_initial_stats(self):
        manager = StatsManager()
        stats = manager.get_current_stats()

        assert stats["total_frames"] == 0
        assert stats["total_detections"] == 0
        assert stats["saved_violations"] == 0
        assert stats["obstruction_violations"] == 0
        assert stats["dms_violations"] == 0

    def test_increment_frames(self):
        manager = StatsManager()
        manager.start_session()

        manager.increment_frames(5)
        stats = manager.get_current_stats()

        assert stats["total_frames"] == 5

        manager.increment_frames(10)
        stats = manager.get_current_stats()

        assert stats["total_frames"] == 15

    def test_record_obstruction(self):
        manager = StatsManager()
        manager.start_session()

        manager.record_obstruction(duration=3.0)
        stats = manager.get_current_stats()

        assert stats["total_detections"] == 1
        assert stats["current_obstruction_duration"] == 3.0

    def test_record_obstruction_updates_max(self):
        manager = StatsManager()
        manager.start_session()

        manager.record_obstruction(duration=3.0)
        manager.record_obstruction(duration=5.0)
        manager.record_obstruction(duration=2.0)

        stats = manager.get_current_stats()
        assert stats["max_obstruction_duration"] == 5.0

    def test_record_obstruction_violation(self):
        manager = StatsManager()
        manager.start_session()

        manager.record_obstruction_violation()
        stats = manager.get_current_stats()

        assert stats["obstruction_violations"] == 1
        assert stats["saved_violations"] == 1

    def test_record_movement(self):
        manager = StatsManager()
        manager.start_session()

        manager.record_movement(duration=4.0)
        stats = manager.get_current_stats()

        assert stats["camera_movements"] == 1
        assert stats["movement_violations"] == 1
        assert stats["current_movement_duration"] == 4.0

    def test_record_dms_violation(self):
        manager = StatsManager()
        manager.start_session()

        manager.record_dms_violation("eye_closed")
        manager.record_dms_violation("no_seatbelt")
        manager.record_dms_violation("eye_closed")

        stats = manager.get_current_stats()
        assert stats["dms_violations"] == 3
        assert stats["dms_violation_types"]["eye_closed"] == 2
        assert stats["dms_violation_types"]["no_seatbelt"] == 1

    def test_record_saved_counters_increment_total_saved(self):
        manager = StatsManager()
        manager.start_session()

        manager.record_obstruction_violation()
        manager.record_movement_saved()
        manager.record_forbidden_saved()
        manager.record_dms_saved()

        stats = manager.get_current_stats()
        assert stats["saved_violations"] == 4
        assert stats["obstruction_violations"] == 1
        assert stats["movement_violations_saved"] == 1
        assert stats["forbidden_items_saved"] == 1
        assert stats["dms_violations_saved"] == 1

    def test_update_dms_state(self):
        manager = StatsManager()

        manager.update_dms_state(eye_state="closed", seatbelt_state="off")
        stats = manager.get_current_stats()

        assert stats["eye_state"] == "closed"
        assert stats["seatbelt_state"] == "off"

    def test_update_fps(self):
        manager = StatsManager()
        manager.start_session()


        for _ in range(10):
            manager.update_fps(0.033)

        stats = manager.get_current_stats()
        assert stats["current_fps"] == pytest.approx(30.0, rel=0.5)

    def test_reset(self):
        manager = StatsManager()
        manager.start_session()

        manager.increment_frames(100)
        manager.record_obstruction_violation()
        manager.record_dms_violation("phone")

        manager.reset()
        stats = manager.get_current_stats()

        assert stats["total_frames"] == 0
        assert stats["saved_violations"] == 0
        assert stats["dms_violations"] == 0

    def test_get_summary(self):
        manager = StatsManager()
        manager.start_session()

        manager.increment_frames(100)
        manager.record_obstruction_violation()

        summary = manager.get_summary()

        assert "SESSION SUMMARY" in summary
        assert "Frames" in summary and "total=100" in summary
        assert "Saved" in summary and "total=1" in summary and "obstruction=1" in summary
