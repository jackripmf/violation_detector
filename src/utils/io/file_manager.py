"""Файл: src/utils/io/file_manager.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- src.utils.common.utils: используется для передачи данных или вызова связанной логики."""
import logging
import os
import json
from datetime import datetime
from ..common.utils import create_timestamp


class FileManager:
    """Класс: FileManager
Назначение: координирует подсистему и управляет ее состоянием во время обработки.
Поля класса:
- `base_dir` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `event_dirs` (`dict`): событие синхронизации или тип события для переключения ветки обработки.
Ключевые методы:
- `__init__()`, `_ensure_directories()`, `generate_filename()`, `get_full_path()`, `_normalize_violation_info()`, `save_violation_report()`, `save_movement_report()`"""

    def __init__(self, base_dir="violations"):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `base_dir` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.base_dir = base_dir
        self.event_dirs = {
            "forbidden_items": "forbidden_items",
            "movement": "movement",
            "obstruction": "obstruction",
            "dms": "dms",
        }
        self._ensure_directories()

    def _ensure_directories(self):
        """Функция: _ensure_directories()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        os.makedirs(self.base_dir, exist_ok=True)

    def generate_filename(self, event_type, extension, prefix=None):
        """Функция: generate_filename()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `event_type` (`Any`): тип события/нарушения для выбора ветки обработки.
- `extension` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `prefix` (`Any`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        timestamp = create_timestamp()
        prefixes = {
            "preview": "preview",
            "movement": "camera_movement",
            "obstruction": "camera_obstruction",
            "combined_movement": "combined_movements",
            "report": "report",
            "forbidden_items": "forbidden_items",  # Новое
        }
        file_prefix = prefixes.get(event_type, event_type)
        if prefix:
            file_prefix = f"{prefix}_{file_prefix}"
        return f"{file_prefix}_{timestamp}.{extension}"

    def get_full_path(self, filename, event_type=None):
        """Функция: get_full_path()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `filename` (`Any`): имя файла без/с расширением для сохранения или чтения.
- `event_type` (`Any`): тип события/нарушения для выбора ветки обработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        if event_type in self.event_dirs:
            event_dir = os.path.join(self.base_dir, self.event_dirs[event_type])
            os.makedirs(event_dir, exist_ok=True)
            return os.path.join(event_dir, filename)
        return os.path.join(self.base_dir, filename)


    def _normalize_violation_info(self, violation_info):
        """Функция: _normalize_violation_info()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_info` (`Any`): детали зафиксированного нарушения (тип, время, метаданные).
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        raw = dict(violation_info or {})
        violation_type = str(raw.get("violation_type", "obstruction")).lower()

        details = raw.get("details")
        if not isinstance(details, dict):
            details = {}

        # Обратная совместимость с текущими структурами
        if "reasons" in raw and "reasons" not in details:
            details["reasons"] = raw.get("reasons", [])
        if "metrics" in raw and "metrics" not in details:
            details["metrics"] = raw.get("metrics", {})
        if "detectors_count" in raw and "detectors_count" not in details:
            details["detectors_count"] = raw.get("detectors_count", 0)
        if "objects" in raw and "objects" not in details:
            details["objects"] = raw.get("objects", [])
        if "affected_classes" in raw and "affected_classes" not in details:
            details["affected_classes"] = raw.get("affected_classes", [])
        if "violations" in raw and "violations" not in details:
            details["violations"] = raw.get("violations", [])
        if "dms_stats" in raw and "dms_stats" not in details:
            details["dms_stats"] = raw.get("dms_stats", {})
        if "movement_info" in raw and "movement_info" not in details:
            details["movement_info"] = raw.get("movement_info", {})
        if "duration" in raw and "duration" not in details:
            details["duration"] = raw.get("duration", 0.0)

        timestamp = raw.get("timestamp", datetime.now().timestamp())
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            timestamp = datetime.now().timestamp()

        video_timestamp = raw.get("video_timestamp", 0.0)
        try:
            video_timestamp = float(video_timestamp)
        except (TypeError, ValueError):
            video_timestamp = 0.0

        frame_number = raw.get("frame_number", raw.get("frame_count", -1))
        try:
            frame_number = int(frame_number)
        except (TypeError, ValueError):
            frame_number = -1

        return {
            "violation_type": violation_type,
            "violation_id": raw.get("violation_id", "unknown"),
            "timestamp": timestamp,
            "video_timestamp": video_timestamp,
            "frame_number": frame_number,
            "media_file": raw.get("media_file", "unknown"),
            "summary": raw.get("summary", ""),
            "details": details,
        }

    def save_violation_report(self, violation_info, report_filename=None, event_type=None):
        """Функция: save_violation_report()
Назначение: сохраняет данные и артефакты в целевое хранилище.
Параметры функции:
- `violation_info` (`Any`): детали зафиксированного нарушения (тип, время, метаданные).
- `report_filename` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `event_type` (`Any`): тип события/нарушения для выбора ветки обработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            normalized = self._normalize_violation_info(violation_info)
            filename = report_filename or self.generate_filename("report", "txt")
            filepath = self.get_full_path(filename)
            if event_type:
                filepath = self.get_full_path(filename, event_type=event_type)

            with open(filepath, "w", encoding="utf8") as f:
                f.write("VIOLATION REPORT\n")
                f.write("==================\n\n")
                f.write(f"Type: {normalized['violation_type']}\n")
                f.write(f"Violation ID: {normalized['violation_id']}\n")
                f.write(
                    f"Timestamp: {datetime.fromtimestamp(normalized['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Video Time: {normalized['video_timestamp']:.1f}s\n")
                f.write(f"Frame: {normalized['frame_number']}\n")
                f.write(f"Media File: {normalized['media_file']}\n")
                if normalized["summary"]:
                    f.write(f"Summary: {normalized['summary']}\n")

                f.write("\nDETAILS:\n")
                f.write("METRICS:\n")
                for key, value in normalized["details"].items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    elif isinstance(value, (list, dict)):
                        f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            logging.info(f"Violation report saved: {filename}")
            return True
        except Exception as e:
            logging.error(f"Error saving violation report: {e}")
            return False

    def save_movement_report(self, movement_info, video_filename, duration, event_type=None):
        """Функция: save_movement_report()
Назначение: сохраняет данные и артефакты в целевое хранилище.
Параметры функции:
- `movement_info` (`Any`): данные детектора нарушений, используемые для итогового решения по кадру.
- `video_filename` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `duration` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `event_type` (`Any`): тип события/нарушения для выбора ветки обработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            filename = video_filename.replace(".mp4", ".txt")
            filepath = self.get_full_path(filename, event_type=event_type)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("CAMERA MOVEMENT REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Video File: {video_filename}\n")
                f.write(
                    f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Video Duration: {duration:.1f}s\n")

                # Проверяем консолидированный ли отчет
                if movement_info.get("is_consolidated", False):
                    f.write(
                        f"Movement Segments: {movement_info.get('segment_count', 1)}\n")
                    f.write(
                        f"Consolidated Reason: {movement_info.get('consolidated_reason', 'Unknown')}\n")
                    f.write(
                        f"Active Movement Time: {movement_info.get('active_movement_time', 0):.1f}s\n")
                    f.write(
                        f"Inactive Time: {movement_info.get('inactive_time', 0):.1f}s\n")
                    f.write(
                        f"Movement Ratio: {movement_info.get('active_movement_time', 0) / duration * 100:.1f}%\n\n")

                    # Детали по сегментам
                    f.write("MOVEMENT SEGMENTS DETAILS:\n")
                    f.write("-" * 40 + "\n")

                    segments = movement_info.get("segments", [])
                    for i, segment in enumerate(segments, 1):
                        f.write(f"\nSegment #{i}:\n")
                        f.write(
                            f"  Start: {segment.get('video_start', 0):.1f}s\n")
                        f.write(
                            f"  Duration: {segment.get('duration', 0):.1f}s\n")
                        f.write(
                            f"  Reason: {segment.get('reason', 'Unknown')}\n")
                else:
                    # Старый формат для обратной совместимости
                    f.write(
                        f"Reason: {movement_info.get('reason', 'Unknown')}\n")
                    f.write(
                        f"Rotation: {movement_info.get('rotation', 0):.1f}°\n")
                    f.write(
                        f"Translation: {movement_info.get('translation', 0):.3f}\n")
                    f.write(
                        f"Movement Duration: {movement_info.get('movement_duration', 0):.1f}s\n")

                f.write("\n" + "=" * 50 + "\n")
                f.write("DETECTION PARAMETERS:\n")
                f.write(f"Filtered: {movement_info.get('filtered', False)}\n")
                f.write(
                    f"Consecutive Frames: {movement_info.get('consecutive_frames', 0)}\n")

            logging.info(f"Movement report saved: {filename}")
            if movement_info.get("is_consolidated", False):
                logging.debug(
                    f"Consolidated {movement_info.get('segment_count', 1)} movement segments")
            return True

        except Exception as e:
            logging.error(f"Error saving movement report: {e}")
            return False
