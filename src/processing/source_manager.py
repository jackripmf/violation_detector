"""Файл: src/processing/source_manager.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

import logging
import os
import platform
from typing import Optional, Tuple, Dict, Any
import cv2

from ..utils.io.source_redaction import format_source_for_logging


class SourceManager:
    """Класс: SourceManager
Назначение: координирует подсистему и управляет ее состоянием во время обработки.
Поля класса:
- `cap` (`Optional[cv2.VideoCapture]`): объект `cv2.VideoCapture`, читающий кадры из источника.
- `duration_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `fps` (`float`): частота кадров (кадров/с), используемая для таймингов и видео-вывода.
- `height` (`int`): высота кадра или области в пикселях.
- `is_file` (`bool`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `is_rtsp` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `total_frames` (`int`): общее число кадров в источнике или тестовом сценарии.
- `width` (`int`): ширина кадра или области в пикселях.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `open_source()`, `_resolve_source()`, `_open_camera()`, `_configure_camera()`, `_read_camera_properties()`, `_open_rtsp()`, `_open_video_file()`, `_read_video_properties()`, `read_frame()`"""
    
    def __init__(self):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_file: bool = False
        self.is_rtsp: bool = False
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0
        self.duration_sec: float = 0.0
        
    def open_source(self, input_source: Any, camera_id: Optional[int] = None) -> bool:
        """Функция: open_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `camera_id` (`Optional[int]`): идентификатор/индекс для адресации и сопоставления сущностей.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            source_kind, source_value = self._resolve_source(input_source=input_source, camera_id=camera_id)
            logging.info(
                f"[source:resolve] kind={source_kind} value={format_source_for_logging(source_value)}"
            )

            if source_kind == "camera":
                return self._open_camera(source_value)
            if source_kind == "rtsp":
                return self._open_rtsp(source_value)
            return self._open_video_file(source_value)
        except ValueError as e:
            logging.error(f"[source:resolve_failed] error={e}")
            return False
        except Exception as e:
            logging.error(
                f"[source:open_failed] input={format_source_for_logging(input_source)} error={e}"
            )
            return False

    def _resolve_source(self, input_source: Any, camera_id: Optional[int] = None) -> Tuple[str, Any]:
        """Функция: _resolve_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `camera_id` (`Optional[int]`): идентификатор/индекс для адресации и сопоставления сущностей.
Возвращаемое значение: Tuple[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if camera_id is not None:
            return "camera", int(camera_id)

        if isinstance(input_source, int):
            return "camera", input_source

        source_text = "" if input_source is None else str(input_source).strip()
        if not source_text:
            raise ValueError("Input source is not set. Use --input <path_or_rtsp> or --camera-id <id>.")

        source_lower = source_text.lower()
        if source_lower.startswith(("rtsp://", "rtsps://")):
            return "rtsp", source_text

        if os.path.isfile(source_text):
            return "file", source_text

        if source_text.isdigit():
            raise ValueError(
                f"Ambiguous numeric source '{source_text}'. "
                f"Use --camera-id {source_text} for camera input or provide an existing file path."
            )

        return "file", source_text
    
    def _open_camera(self, camera_id: int) -> bool:
        """Функция: _open_camera()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `camera_id` (`int`): идентификатор/индекс для адресации и сопоставления сущностей.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        # Выбор API в зависимости от ОС
        if platform.system() == "Windows":
            api = cv2.CAP_DSHOW
        else:
            api = cv2.CAP_V4L2
        
        self.cap = cv2.VideoCapture(camera_id, api)
        
        # Настройка параметров
        self._configure_camera()
        
        if not self.cap or not self.cap.isOpened():
            logging.error(f"[source:camera_open_failed] camera_id={camera_id}")
            return False
        
        self._read_camera_properties()
        self.is_file = False
        self.is_rtsp = False
        self.total_frames = 0
        self.duration_sec = 0.0
        
        logging.info(
            f"[source:camera_opened] camera_id={camera_id} "
            f"resolution={self.width}x{self.height} fps={self.fps:.2f}"
        )
        return True
    
    def _configure_camera(self) -> None:
        """Функция: _configure_camera()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.cap is None:
            return
            
        # Уменьшение буферизации
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        
        # Разрешение 1280x720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Включение MJPG (если поддерживается)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        
        # Запрос FPS
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def _read_camera_properties(self) -> None:
        """Функция: _read_camera_properties()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.cap is None:
            return
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    
    def _open_rtsp(self, url: str) -> bool:
        """Функция: _open_rtsp()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `url` (`str`): URL-адрес видеоисточника (например RTSP-поток).
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        self.cap = cv2.VideoCapture(url)
        
        if not self.cap or not self.cap.isOpened():
            logging.error(f"[source:rtsp_open_failed] url={format_source_for_logging(url)}")
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        
        self.is_file = False
        self.is_rtsp = True
        self.total_frames = 0
        self.duration_sec = 0.0
        
        logging.info(
            f"[source:rtsp_opened] resolution={self.width}x{self.height} fps={self.fps:.2f}"
        )
        return True
    
    def _open_video_file(self, path: str) -> bool:
        """Функция: _open_video_file()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `path` (`str`): путь к файлу или директории, участвующей в текущей операции.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if not os.path.isfile(path):
            logging.error(f"[source:file_not_found] path={format_source_for_logging(path)}")
            return False

        self.cap = cv2.VideoCapture(path)
        
        if not self.cap or not self.cap.isOpened():
            logging.error(f"[source:file_open_failed] path={format_source_for_logging(path)}")
            return False
        
        self._read_video_properties()
        self.is_file = True
        self.is_rtsp = False
        duration_human = self._format_duration(self.duration_sec)
        logging.info(
            f"[source:file_opened] path={format_source_for_logging(path)} resolution={self.width}x{self.height} "
            f"fps={self.fps:.2f} total_frames={self.total_frames} "
            f"duration={self.duration_sec:.2f}s ({duration_human})"
        )
        return True
    
    def _read_video_properties(self) -> None:
        """Функция: _read_video_properties()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.cap is None:
            return
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_sec = float(self.total_frames) / float(self.fps) if self.fps > 0 else 0.0
    
    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """Функция: read_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Tuple[bool, Optional[Any]]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def get_timestamp(self, frame_index: int, start_time: float) -> float:
        """Функция: get_timestamp()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `frame_index` (`int`): порядковый номер кадра внутри источника.
- `start_time` (`float`): время начала текущей операции/сессии.
Возвращаемое значение: float: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.is_file:
            return frame_index / self.fps
        else:
            import time
            return time.time() - start_time
    
    def release(self) -> None:
        """Функция: release()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logging.error(f"[source:release_failed] error={e}")
            finally:
                self.cap = None
    
    def get_info(self) -> Dict[str, Any]:
        """Функция: get_info()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        return {
            "is_file": self.is_file,
            "is_rtsp": self.is_rtsp,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
        }

    @staticmethod
    def _format_duration(total_seconds: float) -> str:
        """Функция: _format_duration()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `total_seconds` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        seconds = max(0, int(total_seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
