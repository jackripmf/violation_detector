"""Файл: src/runtime/topology_updates.py
Тип: helper-модуль hot topology updates centralized runtime.
Назначение: содержит безопасное применение scheduler/source topology patch-ей.
Связи: используется MultiSourceRuntime и SchedulerPolicy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from .contracts import SourceCommandResult, TopologyUpdateResult
from .scheduler_policy import SchedulerPolicy

if TYPE_CHECKING:
    from ..processing.multi_source_runtime import MultiSourceRuntime


def describe_restart_required_updates(runtime: "MultiSourceRuntime") -> Dict[str, list[str]]:
    """Функция: describe_restart_required_updates()
Назначение: возвращает список параметров, требующих полного restart runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
Возвращаемое значение: Dict[str, list[str]]: словарь с группами restart-required параметров."""
    del runtime
    return {
        "runtime": [
            "infer_workers",
            "postprocess_workers",
            "scheduler.infer_queue_size",
            "queue_limits.infer_queue_size",
            "queue_limits.postprocess_queue_size",
        ],
        "source": [
            "capture_queue_size",
            "drop_policy",
            "input_source",
            "camera_id",
        ],
    }


def apply_topology_updates(runtime: "MultiSourceRuntime", updates: Dict[str, Any]) -> TopologyUpdateResult:
    """Функция: apply_topology_updates()
Назначение: применяет ограниченный hot-reload topology/runtime настроек.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `updates` (`Dict[str, Any]`): патч настроек runtime/topology.
Возвращаемое значение: TopologyUpdateResult: результат применения обновлений."""
    if not runtime.is_running():
        result = apply_topology_updates_command(runtime, "__runtime__", updates)
        return TopologyUpdateResult(
            success=result.success,
            applied=dict(result.data.get("applied", {})),
            rejected=dict(result.data.get("rejected", {})),
            restart_required=list(result.data.get("restart_required", [])),
            message=result.message,
        )
    result = runtime._dispatch_source_command("apply_topology_updates", "__runtime__", updates)
    return TopologyUpdateResult(
        success=result.success,
        applied=dict(result.data.get("applied", {})),
        rejected=dict(result.data.get("rejected", {})),
        restart_required=list(result.data.get("restart_required", [])),
        message=result.message,
    )


def apply_topology_updates_command(runtime: "MultiSourceRuntime", _scope: str, updates: Dict[str, Any]) -> SourceCommandResult:
    """Функция: apply_topology_updates_command()
Назначение: применяет ограниченный hot-reload runtime/topology настроек как внутреннюю команду.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `_scope` (`str`): служебный идентификатор области команды.
- `updates` (`Dict[str, Any]`): патч обновлений runtime/topology.
Возвращаемое значение: SourceCommandResult: результат применения topology patch."""
    del _scope
    applied: Dict[str, Any] = {}
    rejected: Dict[str, str] = {}
    restart_required: list[str] = []

    payload = dict(updates or {})
    scheduler_updates = payload.get("scheduler", {})
    if isinstance(scheduler_updates, dict):
        apply_scheduler_updates(runtime, scheduler_updates, applied, rejected, restart_required)

    sources_updates = payload.get("sources", [])
    if isinstance(sources_updates, list):
        for raw_source_update in sources_updates:
            if not isinstance(raw_source_update, dict):
                continue
            apply_source_topology_update(runtime, raw_source_update, applied, rejected, restart_required)

    for field_name in ("infer_workers", "postprocess_workers"):
        if field_name in payload:
            rejected[field_name] = "restart_required"
            restart_required.append(field_name)

    queue_limits = payload.get("queue_limits", {})
    if isinstance(queue_limits, dict):
        for key in queue_limits:
            dotted_name = f"queue_limits.{key}"
            rejected[dotted_name] = "restart_required"
            restart_required.append(dotted_name)

    success = bool(applied) or not rejected
    message = "Topology updates applied." if success else "Topology updates rejected."
    return SourceCommandResult(
        success=success,
        source_id=None,
        message=message,
        data={
            "applied": applied,
            "rejected": rejected,
            "restart_required": sorted(set(restart_required)),
        },
    )


def apply_scheduler_updates(
    runtime: "MultiSourceRuntime",
    scheduler_updates: Dict[str, Any],
    applied: Dict[str, Any],
    rejected: Dict[str, str],
    restart_required: list[str],
) -> None:
    """Функция: apply_scheduler_updates()
Назначение: применяет безопасные обновления scheduler-конфига runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `scheduler_updates` (`Dict[str, Any]`): патч scheduler-полей.
- `applied` (`Dict[str, Any]`): накопитель успешно примененных изменений.
- `rejected` (`Dict[str, str]`): накопитель отклоненных изменений.
- `restart_required` (`list[str]`): накопитель restart-required полей.
Возвращаемое значение: None: helper обновляет переданные структуры."""
    safe_fields = {
        "policy",
        "aging_factor",
        "backlog_factor",
        "starvation_threshold_sec",
        "starvation_boost",
        "dispatch_sleep_sec",
        "infer_overflow_strategy",
    }
    for key, value in scheduler_updates.items():
        dotted_name = f"scheduler.{key}"
        if key == "infer_queue_size":
            rejected[dotted_name] = "restart_required"
            restart_required.append(dotted_name)
            continue
        if key not in safe_fields:
            rejected[dotted_name] = "unsupported"
            continue
        try:
            setattr(runtime.topology.scheduler, key, value)
            runtime.topology.scheduler.__post_init__()
            runtime.scheduler.scheduler_config = runtime.topology.scheduler
            runtime.scheduler.policy = SchedulerPolicy(config=runtime.topology.scheduler)
            applied[dotted_name] = getattr(runtime.topology.scheduler, key)
        except Exception as exc:
            rejected[dotted_name] = f"invalid:{exc}"


def apply_source_topology_update(
    runtime: "MultiSourceRuntime",
    source_update: Dict[str, Any],
    applied: Dict[str, Any],
    rejected: Dict[str, str],
    restart_required: list[str],
) -> None:
    """Функция: apply_source_topology_update()
Назначение: применяет безопасные topology-обновления одного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_update` (`Dict[str, Any]`): патч настроек одного источника.
- `applied` (`Dict[str, Any]`): накопитель успешно примененных изменений.
- `rejected` (`Dict[str, str]`): накопитель отклоненных изменений.
- `restart_required` (`list[str]`): накопитель restart-required полей.
Возвращаемое значение: None: helper обновляет переданные структуры."""
    source_id = str(source_update.get("source_id", "")).strip()
    if not source_id or source_id not in runtime.source_contexts:
        rejected[f"source.{source_id or 'unknown'}"] = "source_not_found"
        return
    context = runtime.source_contexts[source_id]

    if "detectors" in source_update:
        result = runtime._apply_set_source_detectors(source_id, list(source_update.get("detectors") or []))
        if result.success:
            applied[f"sources.{source_id}.detectors"] = list(result.data.get("enabled_detectors", []))
        else:
            rejected[f"sources.{source_id}.detectors"] = result.error or "failed"

    if "detector_schedule" in source_update:
        manager = context.detection_manager
        if manager is None or not hasattr(manager, "set_detector_schedule"):
            rejected[f"sources.{source_id}.detector_schedule"] = "unsupported"
        else:
            try:
                updated = manager.set_detector_schedule(detector_schedule=dict(source_update.get("detector_schedule") or {}))
                context.source_config.detector_schedule = dict(updated.get("detector_schedule", {}))
                applied[f"sources.{source_id}.detector_schedule"] = dict(updated.get("detector_schedule", {}))
            except Exception as exc:
                rejected[f"sources.{source_id}.detector_schedule"] = f"invalid:{exc}"

    if "visual_config" in source_update:
        result = runtime._apply_set_source_visual_config(source_id, dict(source_update.get("visual_config") or {}))
        if result.success:
            applied[f"sources.{source_id}.visual_config"] = dict(result.data.get("visual_config", {}))
        else:
            rejected[f"sources.{source_id}.visual_config"] = result.error or "failed"

    if "base_priority" in source_update:
        try:
            context.source_config.base_priority = float(source_update.get("base_priority"))
            if context.source_config.base_priority <= 0:
                raise ValueError("base_priority must be > 0")
            applied[f"sources.{source_id}.base_priority"] = context.source_config.base_priority
        except Exception as exc:
            rejected[f"sources.{source_id}.base_priority"] = f"invalid:{exc}"

    for restart_field in ("capture_queue_size", "drop_policy", "input_source", "camera_id"):
        if restart_field in source_update:
            dotted_name = f"sources.{source_id}.{restart_field}"
            rejected[dotted_name] = "restart_required"
            restart_required.append(dotted_name)
