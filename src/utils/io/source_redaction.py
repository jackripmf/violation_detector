"""Файл: src/utils/io/source_redaction.py
Тип: вспомогательный модуль.
Назначение: маскирует чувствительные источники и команды перед логированием.
Связи: используется launcher-скриптами и runtime-компонентами, которые работают с путями и RTSP URL.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

import os
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit


_COMMAND_VALUE_FLAGS = {
    "--config",
    "--input",
    "--main-script",
    "--output",
    "--topology-config",
}


def _is_rtsp_url(value: str) -> bool:
    """Функция: _is_rtsp_url()
Назначение: определяет, является ли строка RTSP/RTSPS URL.
Параметры функции:
- `value` (`str`): входное строковое значение для проверки.
Возвращаемое значение: bool: `True`, если значение является RTSP/RTSPS URL."""
    lowered = value.lower()
    return lowered.startswith(("rtsp://", "rtsps://"))


def format_path_for_logging(path: Any) -> str:
    """Функция: format_path_for_logging()
Назначение: возвращает безопасное представление пути для логов без полного раскрытия структуры директорий.
Параметры функции:
- `path` (`Any`): исходный путь или значение, похожее на путь.
Возвращаемое значение: str: безопасное строковое представление пути."""
    if path is None:
        return "<none>"

    text = str(path).strip()
    if not text:
        return "<empty>"

    normalized = text.replace("\\", "/").rstrip("/")
    if not normalized:
        return text

    file_name = os.path.basename(normalized)
    if not file_name:
        return normalized
    if normalized == file_name:
        return file_name
    return f".../{file_name}"


def redact_rtsp_url(url: Any) -> str:
    """Функция: redact_rtsp_url()
Назначение: маскирует credentials в RTSP/RTSPS URL, сохраняя схему, host и путь.
Параметры функции:
- `url` (`Any`): исходный RTSP/RTSPS URL.
Возвращаемое значение: str: безопасное строковое представление URL."""
    if url is None:
        return "<none>"

    text = str(url).strip()
    if not text:
        return "<empty>"
    if not _is_rtsp_url(text):
        return text

    parts = urlsplit(text)
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    if parts.username is not None and parts.password is not None:
        auth_prefix = "***:***@"
    elif parts.username is not None:
        auth_prefix = "***@"
    else:
        auth_prefix = ""

    safe_netloc = f"{auth_prefix}{hostname}{port}"
    return urlunsplit((parts.scheme, safe_netloc, parts.path, parts.query, parts.fragment))


def format_source_for_logging(source: Any) -> str:
    """Функция: format_source_for_logging()
Назначение: возвращает безопасное представление источника для логов.
Параметры функции:
- `source` (`Any`): путь к файлу, RTSP URL, camera id или другое входное значение.
Возвращаемое значение: str: безопасное строковое представление источника."""
    if source is None:
        return "<none>"
    if isinstance(source, int):
        return str(source)

    text = str(source).strip()
    if not text:
        return "<empty>"
    if _is_rtsp_url(text):
        return redact_rtsp_url(text)
    return format_path_for_logging(text)


def format_command_for_logging(command: Sequence[Any]) -> str:
    """Функция: format_command_for_logging()
Назначение: формирует безопасное строковое представление CLI-команды для dry-run и process logs.
Параметры функции:
- `command` (`Sequence[Any]`): список аргументов команды запуска.
Возвращаемое значение: str: безопасная строка команды без чувствительных источников и лишних путей."""
    rendered_parts: list[str] = []
    pending_flag: str | None = None

    for raw_part in command:
        part = str(raw_part)
        if pending_flag is not None:
            if pending_flag == "--input":
                rendered_parts.append(format_source_for_logging(part))
            else:
                rendered_parts.append(format_path_for_logging(part))
            pending_flag = None
            continue

        if part in _COMMAND_VALUE_FLAGS:
            rendered_parts.append(part)
            pending_flag = part
            continue

        if part.startswith("-"):
            rendered_parts.append(part)
            continue

        if _is_rtsp_url(part):
            rendered_parts.append(redact_rtsp_url(part))
            continue

        if os.path.sep in part or "/" in part or "\\" in part:
            rendered_parts.append(format_path_for_logging(part))
            continue

        rendered_parts.append(part)

    return " ".join(rendered_parts)
