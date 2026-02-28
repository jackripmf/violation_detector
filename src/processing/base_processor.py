"""Файл: src/processing/base_processor.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Импортируемые внутренние модули:
- ..utils.detector_aliases: используется для передачи данных или вызова связанной логики."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..utils.config.detector_aliases import CANONICAL_DETECTORS

@dataclass
class ViolationResult:
    """Класс: ViolationResult
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `detected` (`bool`): флаг наличия обнаруженного события/объекта в текущем результате.
- `dms` (`Dict[str, Any]`): результат DMS-детектора (глаза, ремень, телефон и сопутствующие нарушения).
- `dms_objects` (`List[Dict]`): список объектов, относящихся к DMS-сценарию.
- `forbidden` (`Dict[str, Any]`): результат детектора запрещенных предметов.
- `forbidden_objects` (`List[Dict]`): список запрещенных объектов, подтвержденных правилами детектора.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
- `obstruction` (`Dict[str, Any]`): результат проверки перекрытия объектива/обструкции камеры.
- `yolo_objects` (`List[Dict]`): список объектов, полученных из YOLO перед доменной фильтрацией.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    detected: bool = False
    obstruction: Dict[str, Any] = field(default_factory=dict)
    movement: Dict[str, Any] = field(default_factory=dict)
    forbidden: Dict[str, Any] = field(default_factory=dict)
    dms: Dict[str, Any] = field(default_factory=dict)
    yolo_objects: List[Dict] = field(default_factory=list)
    forbidden_objects: List[Dict] = field(default_factory=list)
    dms_objects: List[Dict] = field(default_factory=list)


@dataclass
class ProcessorConfig:
    """Класс: ProcessorConfig
Назначение: описывает структуру конфигурации и типизированные данные компонента.
Поля класса:
- `device` (`str`): вычислительное устройство для инференса (`cpu`, `cuda`, `cuda:N`).
- `dms_cigarette_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `dms_cooldown` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `dms_eye_closed_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `dms_phone_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `dms_seatbelt_interval` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `enabled_detectors` (`List[str]`): список активных детекторов, участвующих в обработке кадра.
- `forbidden_cooldown` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `forbidden_min_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `max_duration` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `min_movement_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `min_obstruction_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `movement_confirmation_frames` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `obstruction_cooldown` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `save_dir` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    enabled_detectors: List[str] = field(default_factory=lambda: ["all"])
    save_dir: str = "violations"
    device: str = "cuda:0"
    use_half: bool = True
    imgsz: int = 720
    show_preview: bool = True
    max_duration: Optional[float] = None
    
    # Параметры обструкции
    min_obstruction_duration: float = 5.0
    obstruction_cooldown: float = 5.0
    
    # Параметры движения
    min_movement_duration: float = 2.0
    movement_confirmation_frames: int = 8
    
    # Параметры forbidden items
    forbidden_min_duration: float = 3.0
    forbidden_cooldown: float = 30.0
    
    # Параметры DMS
    dms_eye_closed_threshold: float = 5.0
    dms_seatbelt_interval: float = 15.0
    dms_phone_threshold: float = 4.0
    dms_cigarette_threshold: float = 3.0
    dms_cooldown: float = 60.0


class BaseProcessor(ABC):
    """Класс: BaseProcessor
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `config` (`ProcessorConfig`): структура конфигурации компонента/подсистемы.
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
- `is_running` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
Ключевые методы:
- `__init__()`, `process_source()`, `process_frame()`, `cleanup()`, `enable_detector()`, `disable_detector()`, `is_detector_enabled()`"""
    
    def __init__(self, config: ProcessorConfig):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `config` (`ProcessorConfig`): структура конфигурации компонента/подсистемы.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.config = config
        self.frame_count = 0
        self.is_running = False
        
    @abstractmethod
    def process_source(self, input_source: Any, show_preview: bool = True) -> None:
        """Функция: process_source()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `show_preview` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        pass
    
    @abstractmethod
    def process_frame(self, frame: Any, timestamp: float) -> ViolationResult:
        """Функция: process_frame()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `timestamp` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: ViolationResult: результат шага обработки, который используется следующим этапом пайплайна."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Функция: cleanup()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        pass
    
    def enable_detector(self, detector_name: str) -> None:
        """Функция: enable_detector()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if detector_name not in self.config.enabled_detectors:
            self.config.enabled_detectors.append(detector_name)
            logging.info(f"Детектор {detector_name} активирован")
    
    def disable_detector(self, detector_name: str) -> None:
        """Функция: disable_detector()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if "all" in self.config.enabled_detectors:
            self.config.enabled_detectors = [name for name in CANONICAL_DETECTORS if name != detector_name]
        if detector_name in self.config.enabled_detectors:
            self.config.enabled_detectors.remove(detector_name)
            logging.info(f"Детектор {detector_name} деактивирован")
    
    def is_detector_enabled(self, detector_name: str) -> bool:
        """Функция: is_detector_enabled()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
        if "all" in self.config.enabled_detectors:
            return True
        return detector_name in self.config.enabled_detectors
