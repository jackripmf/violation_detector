"""Файл: src/processing/preview_manager.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Импортируемые внутренние модули:
- ..utils.file_manager: используется для передачи данных или вызова связанной логики."""

import cv2
import logging
from typing import Optional, Any
from ..utils.io.file_manager import FileManager


class PreviewManager:
    """Класс: PreviewManager
Назначение: координирует подсистему и управляет ее состоянием во время обработки.
Поля класса:
- `file_manager` (`FileManager`): объект файловых операций: создание путей, сохранение артефактов.
- `filename` (`Optional[str]`): имя файла без/с расширением для сохранения или чтения.
- `fps` (`float`): частота кадров (кадров/с), используемая для таймингов и видео-вывода.
- `is_recording` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `writer` (`Optional[cv2.VideoWriter]`): объект подсистемы, через который вызывается профильная логика этого этапа.
Ключевые методы:
- `__init__()`, `start_recording()`, `write_frame()`, `stop_recording()`, `cleanup()`, `is_active()`"""
    
    def __init__(self, file_manager: FileManager, fps: float = 30.0):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `file_manager` (`FileManager`): объект файловых операций: создание путей, сохранение артефактов.
- `fps` (`float`): частота кадров (кадров/с), используемая для таймингов и видео-вывода.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.file_manager = file_manager
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.filename: Optional[str] = None
        
    def start_recording(self, frame: Any) -> bool:
        """Функция: start_recording()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.writer is not None:
            return True
        
        try:
            self.filename = self.file_manager.generate_filename("preview", "mp4")
            preview_path = self.file_manager.get_full_path(self.filename)
            
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                preview_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                (width, height)
            )
            
            self.is_recording = True
            logging.info(f"[preview:recording_started] file={self.filename}")
            return True
            
        except Exception as e:
            logging.error(f"[preview:recording_start_failed] error={e}")
            return False
    
    def write_frame(self, frame: Any) -> bool:
        """Функция: write_frame()
Назначение: сохраняет данные и артефакты в целевое хранилище.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.writer is None:
            if not self.start_recording(frame):
                return False
        
        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            logging.error(f"[preview:frame_write_failed] error={e}")
            return False
    
    def stop_recording(self) -> None:
        """Функция: stop_recording()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception as e:
                logging.error(f"[preview:recording_stop_failed] error={e}")
            finally:
                self.writer = None
                self.is_recording = False
                logging.info(f"[preview:recording_stopped] file={self.filename}")
    
    def cleanup(self) -> None:
        """Функция: cleanup()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stop_recording()
    
    def is_active(self) -> bool:
        """Функция: is_active()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
        return self.is_recording and self.writer is not None
