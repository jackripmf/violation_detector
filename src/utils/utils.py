import cv2
import os
import logging
from datetime import datetime


def setup_logging(log_level=logging.INFO):
    """Настройка логирования"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_timestamp():
    """Создание временной метки для файлов"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def format_video_time(seconds):
    """Форматирование времени в видео в XX:XX.xxx"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def add_timestamp_to_frame(frame, video_timestamp):
    """Добавление временной метки на кадр"""
    display_frame = frame.copy()
    timestamp_str = format_video_time(video_timestamp)
    cv2.putText(
        display_frame,
        timestamp_str,
        (frame.shape[1] - 200, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return display_frame


def get_video_info(video_path):
    """Получение информации о видео"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": total_frames / fps if fps > 0 else 0,
    }


def safe_release_video_writer(writer):
    """Безопасное освобождение VideoWriter"""
    if writer is not None:
        try:
            writer.release()
        except Exception as e:
            logging.error(f"Error releasing video writer: {e}")


def check_file_size(file_path, min_size_kb=1):
    """Проверка размера файла"""
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    return file_size >= min_size_kb * 1024
