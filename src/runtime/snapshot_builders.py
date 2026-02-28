"""Файл: src/runtime/snapshot_builders.py
Тип: вспомогательный модуль runtime snapshots.
Назначение: содержит общие builder-функции для runtime configuration/capabilities snapshots.
Связи: используется RuntimeController и MultiSourceRuntime для стабилизации публичного API.
Критичность: модуль влияет на API-контракт runtime, поэтому изменения нужно сопровождать тестами."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .contracts import (
    RuntimeCapabilitiesSnapshot,
    RuntimeConfigurationSnapshot,
    SourceConfigurationSnapshot,
)
from .topology import RuntimeTopology, SchedulerConfig, SourceConfig
from ..utils.config.detector_aliases import CANONICAL_DETECTORS


VISUAL_CONFIG_FIELDS: List[str] = [
    "preview_width",
    "show_fps",
    "show_indicators",
    "show_stats_panel",
    "show_movement_arrow",
    "show_violation_labels",
    "show_boxes",
]

CENTRALIZED_SOURCE_COMMAND_NAMES: List[str] = [
    "set_source_detectors",
    "get_source_detectors",
    "stop_source",
    "resume_source",
    "reset_movement_reference",
    "set_source_visual_config",
    "get_source_visual_config",
]

CENTRALIZED_MUTABLE_RUNTIME_FIELDS: List[str] = [
    "scheduler.policy",
    "scheduler.aging_factor",
    "scheduler.backlog_factor",
    "scheduler.starvation_threshold_sec",
    "scheduler.starvation_boost",
    "scheduler.dispatch_sleep_sec",
    "scheduler.infer_overflow_strategy",
]

CENTRALIZED_MUTABLE_SOURCE_FIELDS: List[str] = [
    "detectors",
    "detector_schedule",
    "visual_config",
    "base_priority",
]


def resolve_source_type(source_cfg: SourceConfig) -> str:
    """Функция: resolve_source_type()
Назначение: определяет тип источника для configuration snapshot.
Параметры функции:
- `source_cfg` (`SourceConfig`): конфигурация источника.
Возвращаемое значение: str: тип источника (`camera`, `rtsp`, `file`, `unknown`)."""
    if source_cfg.camera_id is not None:
        return "camera"
    source_text = str(source_cfg.input_source or "").strip().lower()
    if source_text.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    if source_text:
        return "file"
    return "unknown"


def build_scheduler_snapshot(scheduler_config: SchedulerConfig) -> Dict[str, Any]:
    """Функция: build_scheduler_snapshot()
Назначение: формирует сериализуемый snapshot конфигурации scheduler.
Параметры функции:
- `scheduler_config` (`SchedulerConfig`): конфигурация scheduler.
Возвращаемое значение: Dict[str, Any]: словарь scheduler-настроек."""
    return {
        "policy": str(scheduler_config.policy),
        "aging_factor": float(scheduler_config.aging_factor),
        "backlog_factor": float(scheduler_config.backlog_factor),
        "starvation_threshold_sec": float(scheduler_config.starvation_threshold_sec),
        "starvation_boost": float(scheduler_config.starvation_boost),
        "dispatch_sleep_sec": float(scheduler_config.dispatch_sleep_sec),
        "infer_queue_size": int(scheduler_config.infer_queue_size),
        "infer_overflow_strategy": str(scheduler_config.infer_overflow_strategy),
    }


def build_source_configuration_snapshot(
    source_cfg: SourceConfig,
    enabled_detectors: List[str],
    detector_schedule: Dict[str, Dict[str, int]],
    visual_config: Dict[str, Any],
    save_dir: str,
) -> SourceConfigurationSnapshot:
    """Функция: build_source_configuration_snapshot()
Назначение: формирует configuration snapshot одного источника.
Параметры функции:
- `source_cfg` (`SourceConfig`): конфигурация источника.
- `enabled_detectors` (`List[str]`): активные детекторы источника.
- `detector_schedule` (`Dict[str, Dict[str, int]]`): расписание детекторов.
- `visual_config` (`Dict[str, Any]`): visual config источника.
- `save_dir` (`str`): базовая директория артефактов runtime.
Возвращаемое значение: SourceConfigurationSnapshot: snapshot конфигурации источника."""
    input_source = None if source_cfg.input_source is None else str(source_cfg.input_source)
    return SourceConfigurationSnapshot(
        source_id=source_cfg.source_id,
        source_type=resolve_source_type(source_cfg),
        input_source=input_source,
        camera_id=source_cfg.camera_id,
        base_priority=float(source_cfg.base_priority),
        capture_queue_size=int(source_cfg.capture_queue_size),
        drop_policy=str(source_cfg.drop_policy),
        enabled_detectors=list(enabled_detectors),
        detector_schedule=dict(detector_schedule),
        visual_config=dict(visual_config),
        output_dir=os.path.join(save_dir, source_cfg.source_id),
    )


def build_runtime_configuration_snapshot(
    engine: str,
    running: bool,
    save_dir: str,
    show_preview: bool,
    default_visual_config: Dict[str, Any],
    async_violation_writes: bool,
    writer_queue_max_size: int,
    writer_overflow_strategy: str,
    preview_callback_attached: bool,
    event_callback_attached: bool,
    command_timeout_sec: float,
    topology: RuntimeTopology,
    sources: Dict[str, SourceConfigurationSnapshot],
) -> RuntimeConfigurationSnapshot:
    """Функция: build_runtime_configuration_snapshot()
Назначение: формирует единый configuration snapshot runtime.
Параметры функции:
- `engine` (`str`): тип runtime-движка.
- `running` (`bool`): флаг активного runtime.
- `save_dir` (`str`): базовая директория сохранения.
- `show_preview` (`bool`): глобальный флаг preview.
- `default_visual_config` (`Dict[str, Any]`): visual config по умолчанию.
- `async_violation_writes` (`bool`): режим асинхронной записи артефактов.
- `writer_queue_max_size` (`int`): лимит writer queue.
- `writer_overflow_strategy` (`str`): стратегия переполнения writer queue.
- `preview_callback_attached` (`bool`): подключен ли preview callback.
- `event_callback_attached` (`bool`): подключен ли event callback.
- `command_timeout_sec` (`float`): timeout команд UI.
- `topology` (`RuntimeTopology`): активная topology runtime.
- `sources` (`Dict[str, SourceConfigurationSnapshot]`): snapshot конфигурации источников.
Возвращаемое значение: RuntimeConfigurationSnapshot: snapshot конфигурации runtime."""
    return RuntimeConfigurationSnapshot(
        engine=str(engine),
        running=bool(running),
        save_dir=str(save_dir),
        show_preview=bool(show_preview),
        default_visual_config=dict(default_visual_config),
        async_violation_writes=bool(async_violation_writes),
        writer_queue_max_size=int(writer_queue_max_size),
        writer_overflow_strategy=str(writer_overflow_strategy),
        preview_callback_attached=bool(preview_callback_attached),
        event_callback_attached=bool(event_callback_attached),
        command_timeout_sec=float(command_timeout_sec),
        infer_workers=int(topology.infer_workers),
        postprocess_workers=int(topology.postprocess_workers),
        scheduler=build_scheduler_snapshot(topology.scheduler),
        sources=dict(sources),
    )


def build_runtime_capabilities_snapshot(
    engine: str,
    supports_per_source_control: bool,
    supports_preview_callback: bool,
    supports_event_callback: bool,
    supports_hot_topology_updates: bool,
    supports_detector_schedule_updates: bool,
    restart_required_fields: Optional[Dict[str, List[str]]] = None,
) -> RuntimeCapabilitiesSnapshot:
    """Функция: build_runtime_capabilities_snapshot()
Назначение: формирует единый capabilities snapshot runtime.
Параметры функции:
- `engine` (`str`): тип runtime-движка.
- `supports_per_source_control` (`bool`): поддержка per-source команд.
- `supports_preview_callback` (`bool`): поддержка preview callback.
- `supports_event_callback` (`bool`): поддержка event callback.
- `supports_hot_topology_updates` (`bool`): поддержка hot topology updates.
- `supports_detector_schedule_updates` (`bool`): поддержка hot-update detector schedule.
- `restart_required_fields` (`Optional[Dict[str, List[str]]]`): поля, требующие restart.
Возвращаемое значение: RuntimeCapabilitiesSnapshot: capabilities snapshot runtime."""
    centralized = bool(supports_per_source_control)
    return RuntimeCapabilitiesSnapshot(
        engine=str(engine),
        supports_per_source_control=centralized,
        supports_preview_callback=bool(supports_preview_callback),
        supports_event_callback=bool(supports_event_callback),
        supports_hot_topology_updates=bool(supports_hot_topology_updates),
        supports_runtime_configuration_snapshot=True,
        supports_detector_schedule_updates=bool(supports_detector_schedule_updates),
        detector_catalog=list(CANONICAL_DETECTORS),
        visual_config_fields=list(VISUAL_CONFIG_FIELDS),
        source_command_names=list(CENTRALIZED_SOURCE_COMMAND_NAMES) if centralized else [],
        mutable_runtime_fields=list(CENTRALIZED_MUTABLE_RUNTIME_FIELDS) if centralized else [],
        mutable_source_fields=list(CENTRALIZED_MUTABLE_SOURCE_FIELDS) if centralized else [],
        restart_required_fields=dict(restart_required_fields or {"runtime": [], "source": []}),
    )
