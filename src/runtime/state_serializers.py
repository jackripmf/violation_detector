"""Файл: src/runtime/state_serializers.py
Тип: helper-модуль сериализации runtime state/stats/config.
Назначение: содержит общие функции построения snapshots для MultiSourceRuntime.
Связи: используется MultiSourceRuntime и runtime_snapshot_builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from .contracts import RuntimeStateSnapshot, RuntimeStatsSnapshot, SourceStateSnapshot, SourceStatsSnapshot
from .snapshot_builders import (
    build_runtime_capabilities_snapshot,
    build_runtime_configuration_snapshot,
    build_source_configuration_snapshot,
)
from ..utils.config.detector_schedule import DEFAULT_DETECTOR_SCHEDULE

if TYPE_CHECKING:
    from ..processing.multi_source_runtime import MultiSourceRuntime, SourceContext
    from .contracts import SourceVisualConfig


def extract_enabled_detectors(runtime: "MultiSourceRuntime", context: Optional["SourceContext"]) -> list[str]:
    """Функция: extract_enabled_detectors()
Назначение: возвращает текущий нормализованный набор детекторов source context.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: list[str]: список активных детекторов."""
    del runtime
    if context is None:
        return []
    manager = context.detection_manager
    enabled = getattr(manager, "enabled_detectors", None) if manager is not None else None
    if enabled is not None:
        return list(enabled)
    return list(context.source_config.enabled_detectors or [])


def extract_detector_statuses(runtime: "MultiSourceRuntime", context: Optional["SourceContext"]) -> list[dict[str, Any]]:
    """Функция: extract_detector_statuses()
Назначение: возвращает сериализуемый runtime-статус детекторов конкретного источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: list[dict[str, Any]]: список статусов детекторов."""
    del runtime
    if context is None or context.detection_manager is None:
        return []
    manager = context.detection_manager
    if not hasattr(manager, "get_detector_status"):
        return []
    try:
        return [
            {
                "name": str(item.name),
                "enabled": bool(item.enabled),
                "active": bool(item.active),
            }
            for item in manager.get_detector_status()
        ]
    except Exception:
        return []


def extract_detector_schedule(runtime: "MultiSourceRuntime", context: Optional["SourceContext"]) -> dict[str, dict[str, int]]:
    """Функция: extract_detector_schedule()
Назначение: возвращает сериализуемое расписание детекторов источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: dict[str, dict[str, int]]: расписание детекторов источника."""
    del runtime
    if context is None:
        return {}
    manager = context.detection_manager
    if manager is not None and hasattr(manager, "get_detector_schedule_snapshot"):
        try:
            schedule = manager.get_detector_schedule_snapshot()
            if isinstance(schedule, dict):
                return {
                    str(name): dict(cfg)
                    for name, cfg in schedule.items()
                    if isinstance(cfg, dict)
                }
        except Exception:
            pass
    if context.source_config.detector_schedule:
        return {
            str(name): dict(cfg)
            for name, cfg in context.source_config.detector_schedule.items()
            if isinstance(cfg, dict)
        }
    return {
        str(name): {
            "every_n_frames": int(cfg.every_n_frames),
            "min_interval_ms": int(cfg.min_interval_ms),
            "priority": int(cfg.priority),
            "result_ttl_frames": int(cfg.result_ttl_frames),
        }
        for name, cfg in DEFAULT_DETECTOR_SCHEDULE.items()
    }


def get_source_last_error(runtime: "MultiSourceRuntime", context: Optional["SourceContext"]) -> Optional[str]:
    """Функция: get_source_last_error()
Назначение: возвращает последнее известное сообщение об ошибке для источника.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: Optional[str]: сообщение об ошибке или `None`."""
    del runtime
    if context is None:
        return None
    worker_error = getattr(context.capture_worker, "last_error", None) if context.capture_worker is not None else None
    return worker_error or context.last_error


def get_source_health_status(runtime: "MultiSourceRuntime", context: Optional["SourceContext"]) -> str:
    """Функция: get_source_health_status()
Назначение: вычисляет агрегированный health-статус источника для UI snapshot.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: str: агрегированный health-статус источника."""
    if context is None:
        return "unknown"
    if get_source_last_error(runtime, context):
        return "error"
    if context.stop_requested:
        return "stopped"
    if context.postprocess_errors > 0 or context.dropped_before_infer > 0:
        return "degraded"
    if context.capture_worker is not None and context.capture_worker.is_running():
        return "running"
    return "idle"


def copy_visual_config(runtime: "MultiSourceRuntime", visual_config: Optional["SourceVisualConfig"]) -> "SourceVisualConfig":
    """Функция: copy_visual_config()
Назначение: создает копию visual config для безопасной передачи наружу.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `visual_config` (`Optional[SourceVisualConfig]`): исходная visual config.
Возвращаемое значение: SourceVisualConfig: копия visual config."""
    config = visual_config or runtime._build_default_visual_config()
    return runtime._source_visual_config_cls(**config.to_dict())


def build_source_state_snapshot(runtime: "MultiSourceRuntime", context: "SourceContext") -> SourceStateSnapshot:
    """Функция: build_source_state_snapshot()
Назначение: формирует state snapshot одного источника для UI.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`SourceContext`): контекст источника.
Возвращаемое значение: SourceStateSnapshot: снимок состояния источника."""
    capture_stats = context.capture_worker.get_stats() if context.capture_worker else None
    writer_queue_size = int(
        context.violation_manager.get_writer_queue_size()
        if context.violation_manager is not None and hasattr(context.violation_manager, "get_writer_queue_size")
        else 0
    )
    return SourceStateSnapshot(
        source_queue_size=int(context.frame_queue.qsize()),
        served_count=int(context.served_count),
        dropped_before_infer=int(context.dropped_before_infer),
        captured_frames=int(capture_stats.read_frames if capture_stats else 0),
        capture_dropped_frames=int(capture_stats.dropped_frames if capture_stats else 0),
        processed_frames=int(context.processed_frames),
        max_source_queue_depth=int(context.max_source_queue_depth),
        stop_requested=bool(context.stop_requested),
        preview_closed=bool(context.preview_closed),
        enabled_detectors=extract_enabled_detectors(runtime, context),
        detector_statuses=extract_detector_statuses(runtime, context),
        detector_schedule=extract_detector_schedule(runtime, context),
        visual_config=copy_visual_config(runtime, context.visual_config).to_dict(),
        writer_queue_size=writer_queue_size,
        capture_running=bool(context.capture_worker.is_running()) if context.capture_worker is not None else False,
        health_status=get_source_health_status(runtime, context),
        last_error=get_source_last_error(runtime, context),
    )


def build_source_stats_snapshot(runtime: "MultiSourceRuntime", context: "SourceContext") -> SourceStatsSnapshot:
    """Функция: build_source_stats_snapshot()
Назначение: формирует stats snapshot одного источника для UI.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`SourceContext`): контекст источника.
Возвращаемое значение: SourceStatsSnapshot: снимок статистики источника."""
    capture_stats = context.capture_worker.get_stats() if context.capture_worker else None
    runtime_stats = {}
    if context.stats_manager is not None and hasattr(context.stats_manager, "get_current_stats"):
        runtime_stats = context.stats_manager.get_current_stats()

    processed_frames = max(1, int(context.processed_frames))
    infer_runs = max(1, int(context.infer_runs))
    return SourceStatsSnapshot(
        captured_frames=int(capture_stats.read_frames if capture_stats else 0),
        capture_dropped_frames=int(capture_stats.dropped_frames if capture_stats else 0),
        processed_frames=int(context.processed_frames),
        served_count=int(context.served_count),
        dropped_before_infer=int(context.dropped_before_infer),
        postprocess_errors=int(context.postprocess_errors),
        max_source_queue_depth=int(context.max_source_queue_depth),
        avg_scheduling_latency_ms=context.scheduling_latency_sum_ms / processed_frames,
        avg_infer_time_ms=context.infer_time_sum_ms / infer_runs,
        avg_end_to_end_lag_ms=context.end_to_end_lag_sum_ms / processed_frames,
        writer_queue_size=int(
            context.violation_manager.get_writer_queue_size()
            if context.violation_manager is not None and hasattr(context.violation_manager, "get_writer_queue_size")
            else 0
        ),
        runtime_stats=runtime_stats,
        enabled_detectors=extract_enabled_detectors(runtime, context),
        detector_statuses=extract_detector_statuses(runtime, context),
        detector_schedule=extract_detector_schedule(runtime, context),
        visual_config=copy_visual_config(runtime, context.visual_config).to_dict(),
        capture_running=bool(context.capture_worker.is_running()) if context.capture_worker is not None else False,
        health_status=get_source_health_status(runtime, context),
        last_error=get_source_last_error(runtime, context),
    )


def build_source_configuration_snapshot_for_runtime(runtime: "MultiSourceRuntime", context: "SourceContext"):
    """Функция: build_source_configuration_snapshot_for_runtime()
Назначение: формирует configuration snapshot одного источника runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`SourceContext`): контекст источника.
Возвращаемое значение: SourceConfigurationSnapshot: snapshot конфигурации источника."""
    return build_source_configuration_snapshot(
        source_cfg=context.source_config,
        enabled_detectors=extract_enabled_detectors(runtime, context),
        detector_schedule=extract_detector_schedule(runtime, context),
        visual_config=copy_visual_config(runtime, context.visual_config).to_dict(),
        save_dir=runtime.save_dir,
    )


def get_runtime_configuration(runtime: "MultiSourceRuntime") -> Dict[str, Any]:
    """Функция: get_runtime_configuration()
Назначение: возвращает активную runtime-конфигурацию в сериализуемом виде для UI.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
Возвращаемое значение: Dict[str, Any]: словарь runtime-конфигурации."""
    sources = {
        source_id: build_source_configuration_snapshot_for_runtime(runtime, context)
        for source_id, context in runtime.source_contexts.items()
    }
    snapshot = build_runtime_configuration_snapshot(
        engine="centralized",
        running=runtime.is_running(),
        save_dir=runtime.save_dir,
        show_preview=bool(runtime.show_preview),
        default_visual_config=runtime._build_default_visual_config().to_dict(),
        async_violation_writes=bool(runtime.async_violation_writes),
        writer_queue_max_size=int(runtime.writer_queue_max_size),
        writer_overflow_strategy=str(runtime.writer_overflow_strategy),
        preview_callback_attached=bool(runtime.preview_callback is not None),
        event_callback_attached=bool(runtime.event_callback is not None),
        command_timeout_sec=float(runtime.command_timeout_sec),
        topology=runtime.topology,
        sources=sources,
    )
    return snapshot.to_dict()


def describe_ui_capabilities(runtime: "MultiSourceRuntime") -> Dict[str, Any]:
    """Функция: describe_ui_capabilities()
Назначение: возвращает UI-discovery контракт с доступными командами и mutable полями runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
Возвращаемое значение: Dict[str, Any]: словарь capabilities runtime."""
    snapshot = build_runtime_capabilities_snapshot(
        engine="centralized",
        supports_per_source_control=True,
        supports_preview_callback=True,
        supports_event_callback=True,
        supports_hot_topology_updates=True,
        supports_detector_schedule_updates=True,
        restart_required_fields=runtime.describe_restart_required_updates(),
    )
    return snapshot.to_dict()


def get_stats(runtime: "MultiSourceRuntime") -> Dict[str, Any]:
    """Функция: get_stats()
Назначение: формирует snapshot статистики centralized runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
Возвращаемое значение: Dict[str, Any]: snapshot статистики runtime."""
    sources: Dict[str, SourceStatsSnapshot] = {}
    total_processed = 0

    for source_id, context in runtime.source_contexts.items():
        sources[source_id] = build_source_stats_snapshot(runtime, context)
        total_processed += int(context.processed_frames)

    hub_metrics = {}
    if runtime.hub is not None and hasattr(runtime.hub, "get_metrics"):
        try:
            hub_metrics = runtime.hub.get_metrics()
        except Exception:
            hub_metrics = {}

    snapshot = RuntimeStatsSnapshot(
        running=runtime.is_running(),
        total_sources=len(runtime.source_contexts),
        total_processed_frames=total_processed,
        infer_queue_size=int(runtime.infer_queue.qsize()),
        postprocess_queue_size=int(runtime.postprocess_queue.qsize()),
        max_infer_queue_depth=int(runtime.scheduler.max_infer_queue_depth),
        max_postprocess_queue_depth=int(runtime.max_postprocess_queue_depth),
        command_queue_size=int(runtime._command_queue.qsize()),
        hub_metrics=hub_metrics,
        sources=sources,
        last_error=runtime._last_error,
    )
    return snapshot.to_dict()


def get_state(runtime: "MultiSourceRuntime") -> Dict[str, Any]:
    """Функция: get_state()
Назначение: формирует snapshot текущего состояния centralized runtime.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
Возвращаемое значение: Dict[str, Any]: snapshot состояния runtime."""
    source_stats: Dict[str, SourceStateSnapshot] = {}
    for source_id, context in runtime.source_contexts.items():
        source_stats[source_id] = build_source_state_snapshot(runtime, context)
    snapshot = RuntimeStateSnapshot(
        running=runtime.is_running(),
        infer_queue_size=int(runtime.infer_queue.qsize()),
        postprocess_queue_size=int(runtime.postprocess_queue.qsize()),
        max_infer_queue_depth=int(runtime.scheduler.max_infer_queue_depth),
        max_postprocess_queue_depth=int(runtime.max_postprocess_queue_depth),
        command_queue_size=int(runtime._command_queue.qsize()),
        sources=source_stats,
        last_error=runtime._last_error,
    )
    return snapshot.to_dict()
