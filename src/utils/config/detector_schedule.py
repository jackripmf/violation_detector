"""Файл: src/utils/config/detector_schedule.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DetectorScheduleConfig:
    """Класс: DetectorScheduleConfig
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `every_n_frames` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `min_interval_ms` (`int`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `priority` (`int`): идентификатор/индекс для адресации и сопоставления сущностей.
- `result_ttl_frames` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    every_n_frames: int = 1
    min_interval_ms: int = 0
    priority: int = 100
    result_ttl_frames: int = 0


DEFAULT_DETECTOR_SCHEDULE: Dict[str, DetectorScheduleConfig] = {
    "cv": DetectorScheduleConfig(
        every_n_frames=1,
        min_interval_ms=0,
        priority=95,
        result_ttl_frames=0,
    ),
    "dark": DetectorScheduleConfig(
        every_n_frames=1,
        min_interval_ms=0,
        priority=95,
        result_ttl_frames=0,
    ),
    "yolo": DetectorScheduleConfig(
        every_n_frames=2,
        min_interval_ms=0,
        priority=90,
        result_ttl_frames=1,
    ),
    "movement": DetectorScheduleConfig(
        every_n_frames=2,
        min_interval_ms=0,
        priority=95,
        result_ttl_frames=0,
    ),
    "forbidden": DetectorScheduleConfig(
        every_n_frames=2,
        min_interval_ms=0,
        priority=100,
        result_ttl_frames=1,
    ),
    "dms": DetectorScheduleConfig(
        every_n_frames=3,
        min_interval_ms=0,
        priority=100,
        result_ttl_frames=1,
    ),
}
