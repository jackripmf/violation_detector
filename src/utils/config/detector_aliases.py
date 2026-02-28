"""Файл: src/utils/config/detector_aliases.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from typing import Iterable, List

CANONICAL_DETECTORS = ["cv", "dark", "yolo", "movement", "forbidden", "dms"]

DETECTOR_ALIASES = {
    "all": "all",
    "cv": "cv",
    "cvdetector": "cv",
    "dark": "dark",
    "darkdetector": "dark",
    "yolo": "yolo",
    "yolodetector": "yolo",
    "movement": "movement",
    "movementdetector": "movement",
    "forbidden": "forbidden",
    "forbiddendetector": "forbidden",
    "forbiddenitemsdetector": "forbidden",
    "dms": "dms",
    "dmsdetector": "dms",
}


def normalize_detector_name(name: str) -> str:
    """Функция: normalize_detector_name()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
    normalized = name.strip().lower()
    return DETECTOR_ALIASES.get(normalized, normalized)


def normalize_detector_list(detectors: Iterable[str]) -> List[str]:
    """Функция: normalize_detector_list()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detectors` (`Iterable[str]`): список детекторов из CLI/конфига до или после нормализации алиасов.
Возвращаемое значение: List[str]: результат шага обработки, который используется следующим этапом пайплайна."""
    result: List[str] = []

    for detector in detectors:
        if detector is None:
            continue

        normalized = normalize_detector_name(detector)
        if not normalized:
            continue
        if normalized not in result:
            result.append(normalized)

    return result


def get_cli_detector_choices(include_all: bool = True) -> List[str]:
    """Функция: get_cli_detector_choices()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `include_all` (`bool`): флаг добавления всех алиасов/значений без дополнительной фильтрации.
Возвращаемое значение: List[str]: результат шага обработки, который используется следующим этапом пайплайна."""
    choices = list(DETECTOR_ALIASES.keys())
    if not include_all:
        choices = [choice for choice in choices if choice != "all"]
    return choices
