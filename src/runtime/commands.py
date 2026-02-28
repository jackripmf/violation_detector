"""Файл: src/runtime/commands.py
Тип: helper-модуль command handling centralized runtime.
Назначение: содержит dispatch/apply/get helper-ы для per-source команд runtime.
Связи: используется MultiSourceRuntime и runtime_state_serializers."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any, Dict

from .contracts import SourceCommandResult
from .state_serializers import (
    copy_visual_config,
    extract_detector_schedule,
    extract_detector_statuses,
    extract_enabled_detectors,
)
from .topology_updates import apply_topology_updates_command

if TYPE_CHECKING:
    from ..processing.multi_source_runtime import MultiSourceRuntime
    from .contracts import SourceVisualConfig
    from .topology import CapturedFrame


def build_source_command_error(source_id: str, error: str, message: str) -> SourceCommandResult:
    """Функция: build_source_command_error()
Назначение: формирует унифицированный результат неуспешной команды источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
- `error` (`str`): код ошибки.
- `message` (`str`): человекочитаемое сообщение.
Возвращаемое значение: SourceCommandResult: результат команды."""
    return SourceCommandResult(success=False, source_id=source_id, error=error, message=message)


def dispatch_source_command(
    runtime: "MultiSourceRuntime",
    command_name: str,
    *args: Any,
    allow_when_stopped: bool = False,
    **kwargs: Any,
) -> SourceCommandResult:
    """Функция: dispatch_source_command()
Назначение: выполняет команду источника напрямую или через command queue runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `command_name` (`str`): имя внутренней команды.
- `args` (`Any`): позиционные аргументы команды.
- `allow_when_stopped` (`bool`): флаг разрешения прямого вызова вне запущенного runtime.
- `kwargs` (`Any`): именованные аргументы команды.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    source_id = str(args[0]) if args else str(kwargs.get("source_id", ""))
    if not runtime.is_running():
        if not allow_when_stopped:
            return build_source_command_error(source_id=source_id, error="runtime_not_running", message="Runtime is not running.")
        return execute_source_command(runtime, command_name, *args, **kwargs)
    if runtime._scheduler_thread is None or not runtime._scheduler_thread.is_alive():
        return execute_source_command(runtime, command_name, *args, **kwargs)

    response_queue: "queue.Queue[SourceCommandResult]" = queue.Queue(maxsize=1)
    try:
        runtime._command_queue.put_nowait((command_name, args, kwargs, response_queue))
    except queue.Full:
        return build_source_command_error(source_id=source_id, error="command_queue_full", message="Runtime command queue is full.")
    try:
        return response_queue.get(timeout=runtime.command_timeout_sec)
    except queue.Empty:
        if runtime._scheduler_thread is None or not runtime._scheduler_thread.is_alive():
            return execute_source_command(runtime, command_name, *args, **kwargs)
        return build_source_command_error(
            source_id=source_id,
            error="command_timeout",
            message=f"Command '{command_name}' timed out.",
        )


def drain_command_queue(runtime: "MultiSourceRuntime") -> None:
    """Функция: drain_command_queue()
Назначение: обрабатывает накопившиеся команды UI перед очередной итерацией scheduler.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
Возвращаемое значение: None: helper дренирует очередь команд."""
    while True:
        try:
            command_name, args, kwargs, response_queue = runtime._command_queue.get_nowait()
        except queue.Empty:
            break

        result = execute_source_command(runtime, command_name, *args, **kwargs)
        try:
            response_queue.put_nowait(result)
        except queue.Full:
            pass
        runtime._command_queue.task_done()


def execute_source_command(runtime: "MultiSourceRuntime", command_name: str, *args: Any, **kwargs: Any) -> SourceCommandResult:
    """Функция: execute_source_command()
Назначение: выполняет внутреннюю команду источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `command_name` (`str`): имя внутренней команды.
- `args` (`Any`): позиционные аргументы команды.
- `kwargs` (`Any`): именованные аргументы команды.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    handlers = {
        "set_source_detectors": apply_set_source_detectors,
        "stop_source": apply_stop_source,
        "resume_source": apply_resume_source,
        "reset_movement_reference": apply_reset_movement_reference,
        "set_source_visual_config": apply_set_source_visual_config,
        "apply_topology_updates": apply_topology_updates_command,
    }
    handler = handlers.get(command_name)
    if handler is None:
        source_id = str(args[0]) if args else str(kwargs.get("source_id", ""))
        return build_source_command_error(source_id=source_id, error="unknown_command", message=f"Unknown command '{command_name}'.")
    return handler(runtime, *args, **kwargs)


def set_source_detectors(runtime: "MultiSourceRuntime", source_id: str, detectors: list[str]) -> SourceCommandResult:
    """Функция: set_source_detectors()
Назначение: обновляет набор детекторов для одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
- `detectors` (`list[str]`): целевой список активных детекторов.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    return dispatch_source_command(runtime, "set_source_detectors", source_id, list(detectors or []), allow_when_stopped=True)


def get_source_detectors(runtime: "MultiSourceRuntime", source_id: str) -> Dict[str, Any]:
    """Функция: get_source_detectors()
Назначение: возвращает набор активных детекторов для одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: Dict[str, Any]: словарь с текущими детекторами источника."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return {"source_id": source_id, "enabled_detectors": [], "detector_statuses": [], "detector_schedule": {}, "error": "source_not_found"}
    return {
        "source_id": source_id,
        "enabled_detectors": extract_enabled_detectors(runtime, context),
        "detector_statuses": extract_detector_statuses(runtime, context),
        "detector_schedule": extract_detector_schedule(runtime, context),
    }


def stop_source(runtime: "MultiSourceRuntime", source_id: str) -> SourceCommandResult:
    """Функция: stop_source()
Назначение: останавливает один источник во время работы runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    return dispatch_source_command(runtime, "stop_source", source_id)


def resume_source(runtime: "MultiSourceRuntime", source_id: str) -> SourceCommandResult:
    """Функция: resume_source()
Назначение: возобновляет обработку одного источника во время работы runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    return dispatch_source_command(runtime, "resume_source", source_id)


def reset_movement_reference(runtime: "MultiSourceRuntime", source_id: str) -> SourceCommandResult:
    """Функция: reset_movement_reference()
Назначение: сбрасывает movement reference для одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    return dispatch_source_command(runtime, "reset_movement_reference", source_id)


def set_source_visual_config(
    runtime: "MultiSourceRuntime",
    source_id: str,
    config: Dict[str, Any] | "SourceVisualConfig",
) -> SourceCommandResult:
    """Функция: set_source_visual_config()
Назначение: обновляет visual config для одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
- `config` (`Dict[str, Any] | SourceVisualConfig`): новый visual config источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    return dispatch_source_command(runtime, "set_source_visual_config", source_id, config, allow_when_stopped=True)


def get_source_visual_config(runtime: "MultiSourceRuntime", source_id: str) -> Dict[str, Any]:
    """Функция: get_source_visual_config()
Назначение: возвращает текущую visual config источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: Dict[str, Any]: словарь visual config."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return {"source_id": source_id, "visual_config": {}, "error": "source_not_found"}
    return {"source_id": source_id, "visual_config": copy_visual_config(runtime, context.visual_config).to_dict()}


def apply_set_source_detectors(runtime: "MultiSourceRuntime", source_id: str, detectors: list[str]) -> SourceCommandResult:
    """Функция: apply_set_source_detectors()
Назначение: применяет обновление детекторов для одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
- `detectors` (`list[str]`): целевой список детекторов.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return build_source_command_error(source_id=source_id, error="source_not_found", message=f"Unknown source '{source_id}'.")
    manager = context.detection_manager
    if manager is None or not hasattr(manager, "set_runtime_detectors"):
        return build_source_command_error(
            source_id=source_id,
            error="detector_manager_unavailable",
            message="Detection manager is unavailable for this source.",
        )
    updated = manager.set_runtime_detectors(enabled_detectors=list(detectors))
    context.source_config.enabled_detectors = list(updated.get("enabled_detectors", []))
    return SourceCommandResult(
        success=True,
        source_id=source_id,
        message="Source detectors updated.",
        data={"enabled_detectors": list(updated.get("enabled_detectors", []))},
    )


def apply_stop_source(runtime: "MultiSourceRuntime", source_id: str) -> SourceCommandResult:
    """Функция: apply_stop_source()
Назначение: применяет остановку одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return build_source_command_error(source_id=source_id, error="source_not_found", message=f"Unknown source '{source_id}'.")
    if context.stop_requested:
        return SourceCommandResult(success=True, source_id=source_id, message="Source already stopped.", data={"already_stopped": True})
    runtime._request_stop_source(source_id)
    runtime._emit_event(event_type="source_stopped", source_id=source_id, severity="info", data={})
    return SourceCommandResult(success=True, source_id=source_id, message="Source stopped.")


def apply_resume_source(runtime: "MultiSourceRuntime", source_id: str) -> SourceCommandResult:
    """Функция: apply_resume_source()
Назначение: применяет возобновление одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return build_source_command_error(source_id=source_id, error="source_not_found", message=f"Unknown source '{source_id}'.")
    worker = context.capture_worker
    if worker is not None and worker.is_running() and not context.stop_requested:
        return build_source_command_error(source_id=source_id, error="already_running", message="Source is already running.")
    if worker is not None:
        try:
            worker.stop()
            worker.join(timeout=max(0.0, runtime.command_timeout_sec))
        except Exception:
            pass

    frame_queue: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=context.source_config.capture_queue_size)
    new_worker = runtime._create_capture_worker(source_cfg=context.source_config, frame_queue=frame_queue)
    context.generation += 1
    context.frame_queue = frame_queue
    context.capture_worker = new_worker
    context.stop_requested = False
    context.preview_closed = False
    context.latest_frame_for_reset = None
    context.latest_frame_packet = None
    context.last_error = None
    new_worker.start()
    runtime._emit_event(event_type="source_resumed", source_id=source_id, severity="info", data={})
    return SourceCommandResult(success=True, source_id=source_id, message="Source resumed.")


def apply_reset_movement_reference(runtime: "MultiSourceRuntime", source_id: str) -> SourceCommandResult:
    """Функция: apply_reset_movement_reference()
Назначение: применяет сброс movement reference для одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return build_source_command_error(source_id=source_id, error="source_not_found", message=f"Unknown source '{source_id}'.")
    if context.latest_frame_for_reset is None:
        return build_source_command_error(
            source_id=source_id,
            error="no_reference_frame",
            message="No frame is available for movement reference reset.",
        )
    if not runtime._reset_movement_reference_for_source(source_id):
        return build_source_command_error(
            source_id=source_id,
            error="reset_unavailable",
            message="Movement detector reference reset is unavailable.",
        )
    return SourceCommandResult(success=True, source_id=source_id, message="Movement reference reset.")


def apply_set_source_visual_config(
    runtime: "MultiSourceRuntime",
    source_id: str,
    config: Dict[str, Any] | "SourceVisualConfig",
) -> SourceCommandResult:
    """Функция: apply_set_source_visual_config()
Назначение: применяет visual config для одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
- `config` (`Dict[str, Any] | SourceVisualConfig`): visual config источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return build_source_command_error(source_id=source_id, error="source_not_found", message=f"Unknown source '{source_id}'.")
    if isinstance(config, runtime._source_visual_config_cls):
        new_config = copy_visual_config(runtime, config)
    else:
        merged = copy_visual_config(runtime, context.visual_config).to_dict()
        merged.update(dict(config or {}))
        new_config = runtime._source_visual_config_cls(**merged)
    context.visual_config = new_config
    runtime._apply_visual_config_to_context(context)
    return SourceCommandResult(
        success=True,
        source_id=source_id,
        message="Source visual config updated.",
        data={"visual_config": new_config.to_dict()},
    )
