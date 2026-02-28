"""Файл: src/runtime/contracts.py
Тип: слой оркестрации обработки видеопотока.
Назначение: содержит стабильные dataclass-контракты runtime для UI, тестов и внешних интеграций.
Связи: используется runtime_controller, multi_source_runtime и смежными модулями как типизированная база публичных ответов.
Критичность: модуль формализует внешние контракты runtime, поэтому изменения нужно сопровождать тестами совместимости.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SourceCommandResult:
    """Класс: SourceCommandResult
Назначение: описывает результат выполнения per-source команды runtime.
Поля класса:
- `success` (`bool`): признак успешного выполнения команды.
- `source_id` (`Optional[str]`): идентификатор источника, к которому относилась команда.
- `error` (`Optional[str]`): код ошибки или причина отказа.
- `message` (`Optional[str]`): человекочитаемое описание результата.
- `data` (`Dict[str, Any]`): дополнительная полезная нагрузка команды.
Ключевые методы:
- `to_dict()`"""

    success: bool
    source_id: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление результата команды.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь с полями результата команды."""
        return asdict(self)


@dataclass(frozen=True)
class SourceStateSnapshot:
    """Класс: SourceStateSnapshot
Назначение: описывает стабильное состояние одного источника runtime.
Поля класса:
- `source_queue_size` (`int`): текущая глубина очереди источника.
- `served_count` (`int`): количество кадров, переданных в infer pipeline.
- `dropped_before_infer` (`int`): количество кадров, отброшенных до инференса.
- `captured_frames` (`int`): количество считанных кадров источника.
- `capture_dropped_frames` (`int`): количество кадров, отброшенных на этапе захвата.
- `processed_frames` (`int`): количество полностью обработанных кадров.
- `max_source_queue_depth` (`int`): максимальная глубина очереди источника.
- `stop_requested` (`bool`): признак остановки источника пользовательской командой.
- `preview_closed` (`bool`): признак закрытого preview для источника.
- `enabled_detectors` (`List[str]`): текущий набор активных детекторов для источника.
- `detector_statuses` (`List[Dict[str, Any]]`): состояние runtime-экземпляров детекторов для источника.
- `detector_schedule` (`Dict[str, Dict[str, int]]`): текущее расписание запуска детекторов.
- `visual_config` (`Dict[str, Any]`): текущая visual config источника.
- `writer_queue_size` (`int`): текущий размер writer queue источника.
- `capture_running` (`bool`): признак активного capture worker.
- `health_status` (`str`): агрегированный health-статус источника.
- `last_error` (`Optional[str]`): последнее сообщение об ошибке по источнику.
Ключевые методы:
- `to_dict()`"""

    source_queue_size: int
    served_count: int
    dropped_before_infer: int
    captured_frames: int
    capture_dropped_frames: int
    processed_frames: int
    max_source_queue_depth: int
    stop_requested: bool
    preview_closed: bool
    enabled_detectors: List[str] = field(default_factory=list)
    detector_statuses: List[Dict[str, Any]] = field(default_factory=list)
    detector_schedule: Dict[str, Dict[str, int]] = field(default_factory=dict)
    visual_config: Dict[str, Any] = field(default_factory=dict)
    writer_queue_size: int = 0
    capture_running: bool = False
    health_status: str = "unknown"
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление состояния источника.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь состояния источника."""
        return asdict(self)


@dataclass(frozen=True)
class RuntimeStateSnapshot:
    """Класс: RuntimeStateSnapshot
Назначение: описывает стабильное состояние всего runtime.
Поля класса:
- `running` (`bool`): признак активного runtime.
- `infer_queue_size` (`int`): текущая глубина infer queue.
- `postprocess_queue_size` (`int`): текущая глубина postprocess queue.
- `max_infer_queue_depth` (`int`): максимальная глубина infer queue.
- `max_postprocess_queue_depth` (`int`): максимальная глубина postprocess queue.
- `command_queue_size` (`int`): текущая глубина command queue.
- `sources` (`Dict[str, SourceStateSnapshot]`): состояние по каждому источнику.
- `last_error` (`Optional[str]`): последнее сообщение об ошибке runtime.
Ключевые методы:
- `to_dict()`"""

    running: bool
    infer_queue_size: int
    postprocess_queue_size: int
    max_infer_queue_depth: int
    max_postprocess_queue_depth: int
    command_queue_size: int
    sources: Dict[str, SourceStateSnapshot] = field(default_factory=dict)
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление состояния runtime.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь состояния runtime."""
        payload = asdict(self)
        payload["sources"] = {
            source_id: snapshot.to_dict()
            for source_id, snapshot in self.sources.items()
        }
        return payload


@dataclass(frozen=True)
class SourceStatsSnapshot:
    """Класс: SourceStatsSnapshot
Назначение: описывает стабильную статистику одного источника runtime.
Поля класса:
- `captured_frames` (`int`): количество считанных кадров источника.
- `capture_dropped_frames` (`int`): количество кадров, отброшенных на этапе захвата.
- `processed_frames` (`int`): количество обработанных кадров.
- `served_count` (`int`): количество кадров, переданных в infer pipeline.
- `dropped_before_infer` (`int`): количество кадров, отброшенных до инференса.
- `postprocess_errors` (`int`): количество ошибок постобработки.
- `max_source_queue_depth` (`int`): максимальная глубина source queue.
- `avg_scheduling_latency_ms` (`float`): средняя задержка между capture и dispatch.
- `avg_infer_time_ms` (`float`): среднее время shared inference.
- `avg_end_to_end_lag_ms` (`float`): средний end-to-end lag.
- `writer_queue_size` (`int`): текущий размер writer queue.
- `runtime_stats` (`Dict[str, Any]`): статистика domain runtime менеджеров.
- `enabled_detectors` (`List[str]`): текущий набор активных детекторов.
- `detector_statuses` (`List[Dict[str, Any]]`): состояние runtime-экземпляров детекторов.
- `detector_schedule` (`Dict[str, Dict[str, int]]`): текущее расписание запуска детекторов.
- `visual_config` (`Dict[str, Any]`): текущая visual config источника.
- `capture_running` (`bool`): признак активного capture worker.
- `health_status` (`str`): агрегированный health-статус источника.
- `last_error` (`Optional[str]`): последнее сообщение об ошибке по источнику.
Ключевые методы:
- `to_dict()`"""

    captured_frames: int
    capture_dropped_frames: int
    processed_frames: int
    served_count: int
    dropped_before_infer: int
    postprocess_errors: int
    max_source_queue_depth: int
    avg_scheduling_latency_ms: float
    avg_infer_time_ms: float
    avg_end_to_end_lag_ms: float
    writer_queue_size: int
    runtime_stats: Dict[str, Any] = field(default_factory=dict)
    enabled_detectors: List[str] = field(default_factory=list)
    detector_statuses: List[Dict[str, Any]] = field(default_factory=list)
    detector_schedule: Dict[str, Dict[str, int]] = field(default_factory=dict)
    visual_config: Dict[str, Any] = field(default_factory=dict)
    capture_running: bool = False
    health_status: str = "unknown"
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление статистики источника.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь статистики источника."""
        return asdict(self)


@dataclass(frozen=True)
class RuntimeStatsSnapshot:
    """Класс: RuntimeStatsSnapshot
Назначение: описывает стабильную статистику runtime целиком.
Поля класса:
- `running` (`bool`): признак активного runtime.
- `total_sources` (`int`): количество источников в runtime.
- `total_processed_frames` (`int`): суммарное количество обработанных кадров.
- `infer_queue_size` (`int`): текущая глубина infer queue.
- `postprocess_queue_size` (`int`): текущая глубина postprocess queue.
- `max_infer_queue_depth` (`int`): максимальная глубина infer queue.
- `max_postprocess_queue_depth` (`int`): максимальная глубина postprocess queue.
- `command_queue_size` (`int`): текущая глубина command queue.
- `hub_metrics` (`Dict[str, Any]`): метрики inference hub.
- `sources` (`Dict[str, SourceStatsSnapshot]`): статистика по каждому источнику.
- `last_error` (`Optional[str]`): последнее сообщение об ошибке runtime.
Ключевые методы:
- `to_dict()`"""

    running: bool
    total_sources: int
    total_processed_frames: int
    infer_queue_size: int
    postprocess_queue_size: int
    max_infer_queue_depth: int
    max_postprocess_queue_depth: int
    command_queue_size: int
    hub_metrics: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, SourceStatsSnapshot] = field(default_factory=dict)
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление статистики runtime.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь статистики runtime."""
        payload = asdict(self)
        payload["sources"] = {
            source_id: snapshot.to_dict()
            for source_id, snapshot in self.sources.items()
        }
        return payload


@dataclass(frozen=True)
class PreviewFramePayload:
    """Класс: PreviewFramePayload
Назначение: описывает единый payload кадра preview для UI callback.
Поля класса:
- `source_id` (`str`): идентификатор источника кадра.
- `frame` (`Any`): кадр preview.
- `frame_index` (`int`): индекс кадра внутри источника.
- `video_timestamp` (`float`): timestamp кадра в координатах видео.
- `captured_at` (`float`): wall-clock время захвата кадра.
- `processed_at` (`float`): wall-clock время готовности preview payload.
- `metadata` (`Dict[str, Any]`): служебные данные preview-потока.
Ключевые методы:
- `to_dict()`"""

    source_id: str
    frame: Any
    frame_index: int
    video_timestamp: float
    captured_at: float
    processed_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление preview payload.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь preview payload."""
        return asdict(self)


@dataclass(frozen=True)
class RuntimeEvent:
    """Класс: RuntimeEvent
Назначение: описывает событие runtime для UI callback.
Поля класса:
- `event_type` (`str`): тип события runtime.
- `source_id` (`Optional[str]`): идентификатор источника, если событие относится к источнику.
- `timestamp` (`float`): wall-clock время события.
- `severity` (`str`): уровень важности события.
- `data` (`Dict[str, Any]`): полезная нагрузка события.
Ключевые методы:
- `to_dict()`"""

    event_type: str
    source_id: Optional[str]
    timestamp: float
    severity: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление runtime-события.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь runtime-события."""
        return asdict(self)


@dataclass(frozen=True)
class SourceVisualConfig:
    """Класс: SourceVisualConfig
Назначение: описывает per-source конфигурацию preview и overlay-отрисовки.
Поля класса:
- `preview_width` (`Optional[int]`): ширина preview для конкретного источника.
- `show_fps` (`bool`): флаг отрисовки FPS.
- `show_indicators` (`bool`): флаг отрисовки индикаторов.
- `show_stats_panel` (`bool`): флаг отрисовки панели статистики.
- `show_movement_arrow` (`bool`): флаг отрисовки стрелки движения.
- `show_violation_labels` (`bool`): флаг отрисовки подписей нарушений.
- `show_boxes` (`bool`): флаг отрисовки bbox-ов.
Ключевые методы:
- `to_dict()`"""

    preview_width: Optional[int] = None
    show_fps: bool = False
    show_indicators: bool = True
    show_stats_panel: bool = True
    show_movement_arrow: bool = True
    show_violation_labels: bool = True
    show_boxes: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление visual config.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь visual config."""
        return asdict(self)


@dataclass(frozen=True)
class SourceConfigurationSnapshot:
    """Класс: SourceConfigurationSnapshot
Назначение: описывает текущую конфигурацию одного источника для UI и панели настроек.
Поля класса:
- `source_id` (`str`): идентификатор источника.
- `source_type` (`str`): тип источника (`camera`, `rtsp`, `file`, `unknown`).
- `input_source` (`Optional[str]`): строковое представление input source.
- `camera_id` (`Optional[int]`): идентификатор камеры, если источник камерный.
- `base_priority` (`float`): текущий base priority источника.
- `capture_queue_size` (`int`): размер очереди захвата источника.
- `drop_policy` (`str`): стратегия дропа на source queue.
- `enabled_detectors` (`List[str]`): целевой набор активных детекторов.
- `detector_schedule` (`Dict[str, Dict[str, int]]`): расписание запуска детекторов.
- `visual_config` (`Dict[str, Any]`): текущая visual config источника.
- `output_dir` (`str`): каталог сохранения артефактов конкретного источника.
Ключевые методы:
- `to_dict()`"""

    source_id: str
    source_type: str
    input_source: Optional[str]
    camera_id: Optional[int]
    base_priority: float
    capture_queue_size: int
    drop_policy: str
    enabled_detectors: List[str] = field(default_factory=list)
    detector_schedule: Dict[str, Dict[str, int]] = field(default_factory=dict)
    visual_config: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление конфигурации источника.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь конфигурации источника."""
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfigurationSnapshot:
    """Класс: RuntimeConfigurationSnapshot
Назначение: описывает текущую runtime-конфигурацию, доступную UI без чтения внутренних объектов.
Поля класса:
- `engine` (`str`): тип runtime-движка (`legacy` или `centralized`).
- `running` (`bool`): признак активного runtime.
- `save_dir` (`str`): базовый каталог сохранения артефактов.
- `show_preview` (`bool`): глобальный флаг preview.
- `default_visual_config` (`Dict[str, Any]`): visual config по умолчанию для новых/непереопределенных источников.
- `async_violation_writes` (`bool`): режим асинхронной записи артефактов.
- `writer_queue_max_size` (`int`): лимит writer queue.
- `writer_overflow_strategy` (`str`): стратегия переполнения writer queue.
- `preview_callback_attached` (`bool`): подключен ли внешний preview callback.
- `event_callback_attached` (`bool`): подключен ли внешний event callback.
- `command_timeout_sec` (`float`): timeout публичных UI-команд.
- `infer_workers` (`int`): число infer worker-потоков.
- `postprocess_workers` (`int`): число postprocess worker-потоков.
- `scheduler` (`Dict[str, Any]`): активная конфигурация scheduler.
- `sources` (`Dict[str, SourceConfigurationSnapshot]`): конфигурация источников.
Ключевые методы:
- `to_dict()`"""

    engine: str
    running: bool
    save_dir: str
    show_preview: bool
    default_visual_config: Dict[str, Any] = field(default_factory=dict)
    async_violation_writes: bool = True
    writer_queue_max_size: int = 0
    writer_overflow_strategy: str = "drop_newest"
    preview_callback_attached: bool = False
    event_callback_attached: bool = False
    command_timeout_sec: float = 1.0
    infer_workers: int = 1
    postprocess_workers: int = 1
    scheduler: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, SourceConfigurationSnapshot] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление runtime-конфигурации.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь runtime-конфигурации."""
        payload = asdict(self)
        payload["sources"] = {
            source_id: snapshot.to_dict()
            for source_id, snapshot in self.sources.items()
        }
        return payload


@dataclass(frozen=True)
class RuntimeCapabilitiesSnapshot:
    """Класс: RuntimeCapabilitiesSnapshot
Назначение: описывает поддерживаемые UI-возможности и mutable/restart-required поля runtime.
Поля класса:
- `engine` (`str`): тип runtime-движка.
- `supports_per_source_control` (`bool`): доступны ли per-source управляющие команды.
- `supports_preview_callback` (`bool`): доступен ли preview callback API.
- `supports_event_callback` (`bool`): доступен ли event callback API.
- `supports_hot_topology_updates` (`bool`): доступен ли hot-reload topology.
- `supports_runtime_configuration_snapshot` (`bool`): доступен ли конфигурационный snapshot.
- `supports_detector_schedule_updates` (`bool`): доступен ли hot-update detector schedule.
- `detector_catalog` (`List[str]`): канонический каталог детекторов для UI.
- `visual_config_fields` (`List[str]`): список поддерживаемых полей visual config.
- `source_command_names` (`List[str]`): поддерживаемые публичные команды per-source runtime.
- `mutable_runtime_fields` (`List[str]`): runtime-level поля, изменяемые без рестарта.
- `mutable_source_fields` (`List[str]`): source-level поля, изменяемые без рестарта.
- `restart_required_fields` (`Dict[str, List[str]]`): поля, требующие полного restart.
Ключевые методы:
- `to_dict()`"""

    engine: str
    supports_per_source_control: bool
    supports_preview_callback: bool
    supports_event_callback: bool
    supports_hot_topology_updates: bool
    supports_runtime_configuration_snapshot: bool
    supports_detector_schedule_updates: bool
    detector_catalog: List[str] = field(default_factory=list)
    visual_config_fields: List[str] = field(default_factory=list)
    source_command_names: List[str] = field(default_factory=list)
    mutable_runtime_fields: List[str] = field(default_factory=list)
    mutable_source_fields: List[str] = field(default_factory=list)
    restart_required_fields: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление runtime-capabilities snapshot.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь runtime-capabilities."""
        return asdict(self)


@dataclass(frozen=True)
class TopologyUpdateResult:
    """Класс: TopologyUpdateResult
Назначение: описывает результат ограниченного hot-reload topology/runtime настроек.
Поля класса:
- `success` (`bool`): признак применения хотя бы одной настройки без фатальной ошибки.
- `applied` (`Dict[str, Any]`): успешно примененные изменения.
- `rejected` (`Dict[str, str]`): изменения, отклоненные runtime с причиной.
- `restart_required` (`List[str]`): список параметров, требующих полного restart.
- `message` (`Optional[str]`): человекочитаемое описание результата.
Ключевые методы:
- `to_dict()`"""

    success: bool
    applied: Dict[str, Any] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)
    restart_required: List[str] = field(default_factory=list)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Функция: to_dict()
Назначение: формирует сериализуемое представление результата topology update.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь результата topology update."""
        return asdict(self)
