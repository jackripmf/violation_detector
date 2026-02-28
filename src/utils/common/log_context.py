"""Файл: src/utils/common/log_context.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator, Optional


_LOG_SOURCE_ID: ContextVar[Optional[str]] = ContextVar("log_source_id", default=None)


@contextmanager
def bind_log_source(source_id: Optional[str]) -> Iterator[None]:
    """Функция: bind_log_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `source_id` (`Optional[str]`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: Iterator[None]: результат шага обработки, который используется следующим этапом пайплайна."""
    token: Token[Optional[str]] = _LOG_SOURCE_ID.set(None if source_id is None else str(source_id))
    try:
        yield
    finally:
        _LOG_SOURCE_ID.reset(token)


def current_log_source() -> Optional[str]:
    """Функция: current_log_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Optional[str]: результат шага обработки, который используется следующим этапом пайплайна."""
    return _LOG_SOURCE_ID.get()

