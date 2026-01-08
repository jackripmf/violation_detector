import logging
import os
from datetime import datetime
from src.utils.utils import create_timestamp


class FileManager:
    """Управление файлами и неймингом"""

    def __init__(self, base_dir="violations"):
        self.base_dir = base_dir
        self.forbidden_dir = os.path.join(base_dir,
                                          "forbidden_items")  # Новая папка
        self._ensure_directories()

    def _ensure_directories(self):
        """Создание необходимых директорий"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.forbidden_dir,
                    exist_ok=True)  # Создаем папку для запрещенных объектов

    def generate_filename(self, event_type, extension, prefix=None):
        """Генерация имени файла"""
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
        """Получение полного пути к файлу"""
        if event_type == "forbidden_items":
            return os.path.join(self.forbidden_dir, filename)
        return os.path.join(self.base_dir, filename)


    def save_violation_report(self, violation_info, movement_info=None):
        """сохранение отчета о нарушении"""
        try:
            filename = self.generate_filename("report", "txt")
            filepath = self.get_full_path(filename)

            with open(filepath, "w", encoding="utf8") as f:
                f.write("VIOLATION REPORT\n")
                f.write("==================\n\n")
                f.write(
                    f"Violation ID: {violation_info['violation_id']}\n")  # Убрали #
                f.write(
                    f"Timestamp: {violation_info['timestamp']}\n")  # Убрали #
                f.write(
                    f"Video Time: {violation_info['video_timestamp']:.1f}s\n")  # Убрали #
                f.write(
                    f"Confidence: {violation_info['confidence']}\n")  # Убрали #
                f.write(
                    f"Detectors: {violation_info['detectors_count']}/3\n")  # Убрали #
                f.write(
                    f"Duration: {violation_info['total_duration']:.1f}s\n")  # Убрали #
                f.write(
                    f"Frame: {violation_info['frame_number']}\n")  # Убрали #

                f.write("DETECTION DETAILS:\n")
                f.write(
                    f"Reasons: {', '.join(violation_info['reasons'])}\n\n")  # Убрали '

                if movement_info:
                    f.write("CAMERA MOVEMENT:\n")
                    f.write(
                        f"Detected: {movement_info.get('movement_detected', False)}\n"
                    )
                    f.write(f"Reason: {movement_info.get('reason', 'Unknown')}\n")
                    f.write(f"Rotation: {movement_info.get('rotation', 0):.1f}°\n")
                    f.write(
                        f"Translation: {movement_info.get('translation', 0):.3f}\n\n"
                    )

                f.write("METRICS:\n")
                for key, value in violation_info["metrics"].items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            logging.info(f"Violation report saved: {filename}")
            return True
        except Exception as e:
            logging.error(f"Error saving violation report: {e}")
            return False

    def save_movement_report(self, movement_info, video_filename, duration):
        """Сохранение отчета о движении камеры с поддержкой консолидации"""
        try:
            filename = video_filename.replace(".mp4", ".txt")
            filepath = self.get_full_path(filename)

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