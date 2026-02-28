"""Файл: src/processing/multi_source_runtime.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- ..utils.detector_aliases: используется для передачи данных или вызова связанной логики.
- ..utils.indicators_config: используется для передачи данных или вызова связанной логики.
- ..utils.log_context: используется для передачи данных или вызова связанной логики.
- ..utils.visualizer: используется для передачи данных или вызова связанной логики.
- .capture_worker: используется для передачи данных или вызова связанной логики.
- .detection_manager: используется для передачи данных или вызова связанной логики."""

from __future__ import annotations

import copy
import inspect
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import cv2

from .capture_worker import CaptureWorker
from .detection_manager import DetectionManager
from ..runtime.contracts import (
    PreviewFramePayload,
    RuntimeEvent,
    SourceCommandResult,
    SourceConfigurationSnapshot,
    SourceStateSnapshot,
    SourceStatsSnapshot,
    SourceVisualConfig,
    TopologyUpdateResult,
)
from ..runtime.topology import CapturedFrame, RuntimeTopology, SchedulerConfig, SourceConfig
from ..runtime.commands import (
    apply_reset_movement_reference,
    apply_resume_source,
    apply_set_source_detectors,
    apply_set_source_visual_config,
    apply_stop_source,
    build_source_command_error,
    dispatch_source_command,
    drain_command_queue,
    execute_source_command,
    get_source_detectors,
    get_source_visual_config,
    reset_movement_reference,
    resume_source,
    set_source_detectors,
    set_source_visual_config,
    stop_source,
)
from .preview_pipeline import (
    build_preview_payload,
    enqueue_preview_packet,
    invoke_preview_callback,
    is_source_processing_drained,
    maybe_close_source_preview,
    on_preview_mouse_event,
    preview_loop,
    render_preview_frame,
    request_stop_source,
    reset_movement_reference_for_source,
    resolve_preview_key_source,
)
from ..runtime.state_serializers import (
    build_source_configuration_snapshot_for_runtime,
    build_source_state_snapshot,
    build_source_stats_snapshot,
    copy_visual_config,
    describe_ui_capabilities,
    extract_detector_schedule,
    extract_detector_statuses,
    extract_enabled_detectors,
    get_runtime_configuration,
    get_source_health_status,
    get_source_last_error,
    get_state,
    get_stats,
)
from ..runtime.topology_updates import (
    apply_scheduler_updates,
    apply_source_topology_update,
    apply_topology_updates,
    apply_topology_updates_command,
    describe_restart_required_updates,
)
from ..runtime.scheduler_policy import SchedulerPolicy, SchedulerSourceState
from .source_manager import SourceManager
from .stats_manager import StatsManager
from .violation_manager import ViolationManager
from ..utils.config.detector_aliases import CANONICAL_DETECTORS
from ..utils.config.indicators_config import IndicatorsLayout
from ..utils.media.visualizer import Visualizer
from ..utils.common.log_context import bind_log_source
from ..utils.common.utils import log_summary_block

@dataclass
class InferenceRequest:
    """Класс: InferenceRequest
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `frame_packet` (`CapturedFrame`): кадр/изображение, которое передается на обработку текущему этапу.
- `scheduled_at` (`float`): время, когда задача была поставлена в расписание.
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    source_id: str
    source_generation: int
    frame_packet: CapturedFrame
    scheduled_at: float


@dataclass
class InferenceResultPacket:
    """Класс: InferenceResultPacket
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `infer_key` (`Optional[str]`): ключ инференса для маршрутизации результатов в `DetectionManager`.
- `infer_time_ms` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `raw_results` (`Any`): сырые выходные данные модели до доменной постобработки.
- `request` (`InferenceRequest`): структура запроса на обработку/инференс, передаваемая между этапами.
- `scheduling_latency_ms` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    request: InferenceRequest
    raw_results: Any = None
    infer_key: Optional[str] = None
    infer_time_ms: float = 0.0
    scheduling_latency_ms: float = 0.0


@dataclass
class PreviewPacket:
    """Класс: PreviewPacket
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `close_window` (`bool`): флаг принудительного закрытия окна предпросмотра.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    source_id: str
    frame: Any = None
    close_window: bool = False
    payload: Optional[PreviewFramePayload] = None


@dataclass
class SourceContext:
    """Класс: SourceContext
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `capture_worker` (`Optional[CaptureWorker]`): рабочий исполнитель или их количество для параллельной обработки.
- `detection_manager` (`Optional[Any]`): менеджер вызова детекторов и агрегации их результатов.
- `dropped_before_infer` (`int`): число кадров, отброшенных до этапа инференса из-за перегрузки.
- `end_to_end_lag_sum_ms` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `frame_queue` (`"queue.Queue[CapturedFrame]"`): очередь для передачи данных между асинхронными этапами пайплайна.
- `infer_runs` (`int`): параметр запуска процесса/скрипта в управляющем контуре runtime.
- `infer_time_sum_ms` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `last_processed_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `last_served_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `latest_frame_for_reset` (`Any`): кадр/изображение, которое передается на обработку текущему этапу.
- `max_source_queue_depth` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `postprocess_errors` (`int`): сообщение или объект ошибки, используемый для диагностики и логирования.
- `preview_closed` (`bool`): флаг, что окно предпросмотра было закрыто пользователем.
- `processed_frames` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `scheduling_latency_sum_ms` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `served_count` (`int`): метрика или счетчик, применяемый для статистики и контроля выполнения.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    source_config: SourceConfig
    frame_queue: "queue.Queue[CapturedFrame]"
    capture_worker: Optional[CaptureWorker] = None
    detection_manager: Optional[Any] = None
    stats_manager: Optional[Any] = None
    violation_manager: Optional[Any] = None

    last_served_at: float = 0.0
    served_count: int = 0
    dropped_before_infer: int = 0
    processed_frames: int = 0
    infer_runs: int = 0
    postprocess_errors: int = 0
    last_processed_at: float = 0.0

    scheduling_latency_sum_ms: float = 0.0
    infer_time_sum_ms: float = 0.0
    end_to_end_lag_sum_ms: float = 0.0
    max_source_queue_depth: int = 0
    latest_frame_for_reset: Any = None
    latest_frame_packet: Optional[CapturedFrame] = None
    preview_closed: bool = False
    stop_requested: bool = False
    visual_config: SourceVisualConfig = None
    generation: int = 0
    last_error: Optional[str] = None


class CentralScheduler:
    """Класс: CentralScheduler
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `infer_queue` (`"queue.Queue[InferenceRequest]"`): очередь для передачи данных между асинхронными этапами пайплайна.
- `max_infer_queue_depth` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `policy` (`Any`): строковый идентификатор политики планирования/дропа кадров.
- `scheduler_config` (`SchedulerConfig`): конфигурация планировщика очередей и dispatch-политики.
- `source_contexts` (`Dict[str, SourceContext]`): словарь контекстов всех активных источников по `source_id`.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `_build_states()`, `select_next_source()`, `dispatch_once()`, `_push_infer_request()`"""

    def __init__(
        self,
        source_contexts: Dict[str, SourceContext],
        infer_queue: "queue.Queue[InferenceRequest]",
        scheduler_config: SchedulerConfig,
        event_callback: Optional[Callable[..., None]] = None,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `source_contexts` (`Dict[str, SourceContext]`): словарь контекстов всех активных источников по `source_id`.
- `infer_queue` (`"queue.Queue[InferenceRequest]"`): очередь для передачи данных между асинхронными этапами пайплайна.
- `scheduler_config` (`SchedulerConfig`): конфигурация планировщика очередей и dispatch-политики.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.source_contexts = source_contexts
        self.infer_queue = infer_queue
        self.scheduler_config = scheduler_config
        self.policy = SchedulerPolicy(config=scheduler_config)
        self.max_infer_queue_depth: int = 0
        self.event_callback = event_callback

    def _build_states(self) -> list[SchedulerSourceState]:
        """Функция: _build_states()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: list[SchedulerSourceState]: результат шага обработки, который используется следующим этапом пайплайна."""
        states: list[SchedulerSourceState] = []
        for source_id, context in self.source_contexts.items():
            if bool(getattr(context, "stop_requested", False)):
                continue
            states.append(
                SchedulerSourceState(
                    source_id=source_id,
                    queue_size=context.frame_queue.qsize(),
                    base_priority=context.source_config.base_priority,
                    last_served_at=context.last_served_at,
                    served_count=context.served_count,
                )
            )
        return states

    def select_next_source(self, now_ts: Optional[float] = None) -> Optional[str]:
        """Функция: select_next_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `now_ts` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: Optional[str]: результат шага обработки, который используется следующим этапом пайплайна."""
        now = time.monotonic() if now_ts is None else float(now_ts)
        return self.policy.select_next(source_states=self._build_states(), now_ts=now)

    def dispatch_once(self, now_ts: Optional[float] = None) -> bool:
        """Функция: dispatch_once()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `now_ts` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        now_monotonic = time.monotonic() if now_ts is None else float(now_ts)
        source_id = self.select_next_source(now_ts=now_monotonic)
        if source_id is None:
            return False

        context = self.source_contexts[source_id]
        context.max_source_queue_depth = max(
            int(context.max_source_queue_depth),
            int(context.frame_queue.qsize()),
        )
        try:
            packet = context.frame_queue.get_nowait()
        except queue.Empty:
            return False

        request = InferenceRequest(
            source_id=source_id,
            source_generation=int(context.generation),
            frame_packet=packet,
            scheduled_at=time.time(),
        )
        if not self._push_infer_request(request=request):
            context.dropped_before_infer += 1

        context.last_served_at = now_monotonic
        context.served_count += 1
        return True

    def _push_infer_request(self, request: InferenceRequest) -> bool:
        """Функция: _push_infer_request()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `request` (`InferenceRequest`): структура запроса на обработку/инференс, передаваемая между этапами.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            self.infer_queue.put_nowait(request)
            self.max_infer_queue_depth = max(
                int(self.max_infer_queue_depth),
                int(self.infer_queue.qsize()),
            )
            return True
        except queue.Full:
            if self.event_callback is not None:
                try:
                    self.event_callback(
                        event_type="infer_queue_overflow",
                        source_id=request.source_id,
                        severity="warning",
                        data={
                            "infer_queue_size": int(self.infer_queue.qsize()),
                            "strategy": self.scheduler_config.infer_overflow_strategy,
                        },
                    )
                except Exception:
                    pass
            if self.scheduler_config.infer_overflow_strategy != "drop_oldest":
                return False

            try:
                self.infer_queue.get_nowait()
                self.infer_queue.put_nowait(request)
                self.max_infer_queue_depth = max(
                    int(self.max_infer_queue_depth),
                    int(self.infer_queue.qsize()),
                )
                return True
            except (queue.Empty, queue.Full):
                return False


class MultiSourceRuntime:
    """Класс: MultiSourceRuntime
Назначение: реализует runtime-логику и синхронизацию этапов обработки.
Поля класса:
- `_active_infer_lock` (`Any`): примитив синхронизации для безопасного доступа к общему состоянию.
- `_active_infer_workers` (`int`): рабочий исполнитель или их количество для параллельной обработки.
- `_active_preview_source_id` (`Optional[str]`): параметр источника/выхода данных, задающий направление потока обработки.
- `_finalized` (`bool`): флаг того, что runtime уже завершил финализацию и cleanup.
- `_infer_threads` (`list[threading.Thread]`): поток выполнения, обрабатывающий часть задач параллельно.
- `_last_error` (`Optional[str]`): сообщение или объект ошибки, используемый для диагностики и логирования.
- `_postprocess_threads` (`list[threading.Thread]`): поток выполнения, обрабатывающий часть задач параллельно.
- `_preview_broken` (`bool`): флаг сбоя подсистемы предпросмотра.
- `_preview_mouse_bound` (`set[str]`): набор окон, где уже привязан обработчик событий мыши.
- `_preview_thread` (`Optional[threading.Thread]`): поток выполнения, обрабатывающий часть задач параллельно.
- `_preview_windows` (`set[str]`): набор имен открытых окон предпросмотра.
- `_running` (`bool`): флаг активного состояния рабочего цикла.
- `_scheduler_finished` (`bool`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `_scheduler_thread` (`Optional[threading.Thread]`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `async_violation_writes` (`Any`): данные детектора нарушений, используемые для итогового решения по кадру.
- `detection_manager_factory` (`Callable[..., Any]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `_build_source_contexts()`, `_build_indicators_config()`, `_create_detection_manager()`, `_should_reconnect_source()`, `start()`, `_preview_loop()`, `_on_preview_mouse_event()`, `_resolve_preview_key_source()`, `_request_stop_source()`"""

    def __init__(
        self,
        topology: RuntimeTopology,
        source_manager_factory: Callable[[], SourceManager] = SourceManager,
        hub: Any = None,
        save_dir: str = "violations",
        async_violation_writes: bool = True,
        writer_queue_max_size: int = 256,
        writer_overflow_strategy: str = "drop_newest",
        detection_manager_factory: Callable[..., Any] = DetectionManager,
        stats_manager_factory: Callable[[], Any] = StatsManager,
        violation_manager_factory: Callable[..., Any] = ViolationManager,
        visualizer_factory: Callable[..., Any] = Visualizer,
        show_preview: bool = False,
        show_fps: bool = False,
        preview_width: Optional[int] = None,
        preview_callback: Optional[Callable[..., None]] = None,
        event_callback: Optional[Callable[[RuntimeEvent], None]] = None,
        command_timeout_sec: float = 1.0,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `topology` (`RuntimeTopology`): описание всех источников и параметров centralized runtime.
- `source_manager_factory` (`Callable[[], SourceManager]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `hub` (`Any`): общий объект инференса, который кэширует модели и выполняет predict.
- `save_dir` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `async_violation_writes` (`bool`): данные детектора нарушений, используемые для итогового решения по кадру.
- `writer_queue_max_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `writer_overflow_strategy` (`str`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `detection_manager_factory` (`Callable[..., Any]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `stats_manager_factory` (`Callable[[], Any]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `violation_manager_factory` (`Callable[..., Any]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `visualizer_factory` (`Callable[..., Any]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `show_preview` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `show_fps` (`bool`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
- `preview_width` (`Optional[int]`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `preview_callback` (`Optional[Callable[[str, Any], None]]`): callback, вызываемый при готовности кадра для предпросмотра.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.topology = topology
        self.source_manager_factory = source_manager_factory
        self.hub = hub
        self.save_dir = save_dir
        self.async_violation_writes = bool(async_violation_writes)
        self.writer_queue_max_size = int(writer_queue_max_size)
        self.writer_overflow_strategy = str(writer_overflow_strategy)

        self.detection_manager_factory = detection_manager_factory
        self.stats_manager_factory = stats_manager_factory
        self.violation_manager_factory = violation_manager_factory
        self.visualizer_factory = visualizer_factory
        self.show_preview = bool(show_preview)
        self.show_fps = bool(show_fps)
        self.preview_width = preview_width
        self.preview_callback = preview_callback
        self.event_callback = event_callback
        self.command_timeout_sec = max(0.05, float(command_timeout_sec))

        self.stop_event = threading.Event()
        self.infer_queue: "queue.Queue[InferenceRequest]" = queue.Queue(
            maxsize=self.topology.scheduler.infer_queue_size
        )
        self.postprocess_queue: "queue.Queue[InferenceResultPacket]" = queue.Queue(
            maxsize=self.topology.scheduler.infer_queue_size
        )
        self._source_visual_config_cls = SourceVisualConfig
        self._preview_packet_cls = PreviewPacket

        self.source_contexts: Dict[str, SourceContext] = {}
        self._build_source_contexts()

        self.scheduler = CentralScheduler(
            source_contexts=self.source_contexts,
            infer_queue=self.infer_queue,
            scheduler_config=self.topology.scheduler,
            event_callback=self._emit_event,
        )

        self._scheduler_thread: Optional[threading.Thread] = None
        self._infer_threads: list[threading.Thread] = []
        self._postprocess_threads: list[threading.Thread] = []
        self._scheduler_finished = False
        self._active_infer_workers = 0
        self._active_infer_lock = threading.Lock()
        self._running = False
        self._finalized = False
        self._last_error: Optional[str] = None
        self.max_postprocess_queue_depth: int = 0
        self.preview_queue: "queue.Queue[PreviewPacket]" = queue.Queue(
            maxsize=max(2, len(self.topology.sources) * 2)
        )
        self._preview_thread: Optional[threading.Thread] = None
        self._preview_windows: set[str] = set()
        self._preview_mouse_bound: set[str] = set()
        self._preview_broken = False
        self._active_preview_source_id: Optional[str] = None
        self._command_queue_max_size = max(8, len(self.topology.sources) * 4)
        self._command_queue: "queue.Queue[tuple[str, tuple[Any, ...], dict[str, Any], queue.Queue[SourceCommandResult]]]" = queue.Queue(
            maxsize=self._command_queue_max_size
        )

    def _build_source_contexts(self) -> None:
        """Функция: _build_source_contexts()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        for source_cfg in self.topology.sources:
            frame_queue: "queue.Queue[CapturedFrame]" = queue.Queue(
                maxsize=source_cfg.capture_queue_size
            )
            worker = self._create_capture_worker(source_cfg=source_cfg, frame_queue=frame_queue)
            detection_manager = self._create_detection_manager(source_cfg)

            stats_manager = self.stats_manager_factory()
            if hasattr(stats_manager, "start_session"):
                stats_manager.start_session()

            source_output_dir = os.path.join(self.save_dir, source_cfg.source_id)
            visual_config = self._build_default_visual_config()
            visualizer = self.visualizer_factory(
                config={
                    "show_fps": visual_config.show_fps,
                    "indicators": self._build_indicators_config(),
                    "show_stats_panel": visual_config.show_stats_panel,
                    "show_movement_arrow": visual_config.show_movement_arrow,
                    "show_violation_labels": visual_config.show_violation_labels,
                    "show_boxes": visual_config.show_boxes,
                }
            )
            violation_manager_kwargs = {
                "save_dir": source_output_dir,
                "visualizer": visualizer,
                "async_writes": self.async_violation_writes,
                "writer_queue_max_size": self.writer_queue_max_size,
                "writer_overflow_strategy": self.writer_overflow_strategy,
                "log_source_id": source_cfg.source_id,
            }
            try:
                signature = inspect.signature(self.violation_manager_factory)
            except (TypeError, ValueError):
                signature = None
            if signature is None or "event_callback" in signature.parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            ):
                violation_manager_kwargs["event_callback"] = self._emit_event

            violation_manager = self.violation_manager_factory(**violation_manager_kwargs)

            self.source_contexts[source_cfg.source_id] = SourceContext(
                source_config=source_cfg,
                frame_queue=frame_queue,
                capture_worker=worker,
                detection_manager=detection_manager,
                stats_manager=stats_manager,
                violation_manager=violation_manager,
                visual_config=visual_config,
            )
            self._apply_visual_config_to_context(self.source_contexts[source_cfg.source_id])

    def _create_capture_worker(
        self,
        source_cfg: SourceConfig,
        frame_queue: "queue.Queue[CapturedFrame]",
    ) -> CaptureWorker:
        """Функция: _create_capture_worker()
Назначение: создает новый capture worker для source context.
Параметры функции:
- `source_cfg` (`SourceConfig`): конфигурация источника.
- `frame_queue` (`queue.Queue[CapturedFrame]`): очередь кадров источника.
Возвращаемое значение: CaptureWorker: новый worker захвата."""
        return CaptureWorker(
            source_config=source_cfg,
            output_queue=frame_queue,
            source_manager=self.source_manager_factory(),
            stop_event=self.stop_event,
            reconnect_on_loss=self._should_reconnect_source(source_cfg),
            event_callback=self._emit_event,
        )

    def _build_default_visual_config(self) -> SourceVisualConfig:
        """Функция: _build_default_visual_config()
Назначение: формирует visual config по умолчанию для нового source context.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: SourceVisualConfig: базовая visual config источника."""
        return SourceVisualConfig(
            preview_width=self.preview_width,
            show_fps=self.show_fps,
            show_indicators=True,
            show_stats_panel=True,
            show_movement_arrow=True,
            show_violation_labels=True,
            show_boxes=True,
        )

    def _apply_visual_config_to_context(self, context: SourceContext) -> None:
        """Функция: _apply_visual_config_to_context()
Назначение: синхронизирует visualizer источника с его текущей visual config.
Параметры функции:
- `context` (`SourceContext`): контекст источника.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        visualizer = (
            getattr(context.violation_manager, "visualizer", None)
            if context.violation_manager is not None
            else None
        )
        visual_config = self._copy_visual_config(context.visual_config)
        context.visual_config = visual_config
        if visualizer is None:
            return
        if hasattr(visualizer, "apply_source_visual_config"):
            visualizer.apply_source_visual_config(visual_config)
            return

        indicators_config = getattr(visualizer, "indicators_config", None)
        if indicators_config is not None:
            indicators_config.show_indicators = bool(visual_config.show_indicators)
        if hasattr(visualizer, "show_fps"):
            visualizer.show_fps = bool(visual_config.show_fps)

    def _build_indicators_config(self) -> IndicatorsLayout:
        """Функция: _build_indicators_config()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: IndicatorsLayout: результат шага обработки, который используется следующим этапом пайплайна."""
        indicators_config = IndicatorsLayout()

        def check_obstruction(result: Dict[str, Any], movement: Dict[str, Any]) -> bool:
            """Функция: check_obstruction()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return bool(result.get("detected", False))

        def check_movement(result: Dict[str, Any], movement: Dict[str, Any]) -> bool:
            """Функция: check_movement()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return bool(movement.get("movement_detected", False))

        def check_forbidden(result: Dict[str, Any], movement: Dict[str, Any]) -> bool:
            """Функция: check_forbidden()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return len(result.get("forbidden_objects", [])) > 0

        def check_dms(result: Dict[str, Any], movement: Dict[str, Any]) -> bool:
            """Функция: check_dms()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return len(result.get("dms_violations", [])) > 0

        def check_cigarette(result: Dict[str, Any], movement: Dict[str, Any]) -> bool:
            """Функция: check_cigarette()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return any(
                obj.get("class", "").lower() == "cigarette"
                for obj in result.get("dms_objects", [])
            )

        def check_phone(result: Dict[str, Any], movement: Dict[str, Any]) -> bool:
            """Функция: check_phone()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return any(
                obj.get("class", "").lower() == "phone"
                for obj in result.get("dms_objects", [])
            )

        indicators_config.indicators["obstruction"].condition = check_obstruction
        indicators_config.indicators["movement"].condition = check_movement
        indicators_config.indicators["forbidden"].condition = check_forbidden
        indicators_config.indicators["dms"].condition = check_dms
        indicators_config.indicators["cigarette"].condition = check_cigarette
        indicators_config.indicators["phone"].condition = check_phone
        return indicators_config

    def _create_detection_manager(self, source_cfg: SourceConfig) -> Any:
        """Функция: _create_detection_manager()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `source_cfg` (`SourceConfig`): параметры конкретного видеоисточника.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        enabled = list(source_cfg.enabled_detectors or ["all"])

        if self.hub is None:
            forbidden_without_hub = {"yolo", "forbidden", "dms"}
            if "all" in enabled:
                enabled = [d for d in CANONICAL_DETECTORS if d not in forbidden_without_hub]
            else:
                enabled = [d for d in enabled if d not in forbidden_without_hub]

        return self.detection_manager_factory(
            hub=self.hub,
            enabled_detectors=enabled,
            detector_schedule=copy.deepcopy(source_cfg.detector_schedule),
        )

    @staticmethod
    def _should_reconnect_source(source_cfg: SourceConfig) -> bool:
        """Функция: _should_reconnect_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `source_cfg` (`SourceConfig`): параметры конкретного видеоисточника.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if source_cfg.camera_id is not None:
            return True
        source_text = str(source_cfg.input_source or "").strip().lower()
        return source_text.startswith(("rtsp://", "rtsps://"))

    def _build_source_command_error(
        self,
        source_id: str,
        error: str,
        message: str,
    ) -> SourceCommandResult:
        """Функция: _build_source_command_error()
Назначение: формирует унифицированный результат неуспешной команды источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
- `error` (`str`): код ошибки.
- `message` (`str`): человекочитаемое сообщение.
Возвращаемое значение: SourceCommandResult: результат команды."""
        return build_source_command_error(source_id=source_id, error=error, message=message)

    def _extract_enabled_detectors(self, context: Optional[SourceContext]) -> list[str]:
        """Функция: _extract_enabled_detectors()
Назначение: возвращает текущий нормализованный набор детекторов source context.
Параметры функции:
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: list[str]: список активных детекторов."""
        return extract_enabled_detectors(self, context)

    def _extract_detector_statuses(self, context: Optional[SourceContext]) -> list[dict[str, Any]]:
        """Функция: _extract_detector_statuses()
Назначение: возвращает сериализуемый runtime-статус детекторов конкретного источника.
Параметры функции:
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: list[dict[str, Any]]: список статусов детекторов."""
        return extract_detector_statuses(self, context)

    def _extract_detector_schedule(self, context: Optional[SourceContext]) -> dict[str, dict[str, int]]:
        """Функция: _extract_detector_schedule()
Назначение: возвращает сериализуемое расписание детекторов источника.
Параметры функции:
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: dict[str, dict[str, int]]: расписание детекторов источника."""
        return extract_detector_schedule(self, context)

    def _get_source_last_error(self, context: Optional[SourceContext]) -> Optional[str]:
        """Функция: _get_source_last_error()
Назначение: возвращает последнее известное сообщение об ошибке для источника.
Параметры функции:
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: Optional[str]: последнее сообщение об ошибке или `None`."""
        return get_source_last_error(self, context)

    def _get_source_health_status(self, context: Optional[SourceContext]) -> str:
        """Функция: _get_source_health_status()
Назначение: вычисляет агрегированный health-статус источника для UI snapshot.
Параметры функции:
- `context` (`Optional[SourceContext]`): контекст источника.
Возвращаемое значение: str: агрегированный health-статус источника."""
        return get_source_health_status(self, context)

    def _build_source_state_snapshot(self, context: SourceContext) -> SourceStateSnapshot:
        """Функция: _build_source_state_snapshot()
Назначение: формирует state snapshot одного источника для UI.
Параметры функции:
- `context` (`SourceContext`): контекст источника.
Возвращаемое значение: SourceStateSnapshot: снимок состояния источника."""
        return build_source_state_snapshot(self, context)

    def _build_source_stats_snapshot(self, context: SourceContext) -> SourceStatsSnapshot:
        """Функция: _build_source_stats_snapshot()
Назначение: формирует stats snapshot одного источника для UI.
Параметры функции:
- `context` (`SourceContext`): контекст источника.
Возвращаемое значение: SourceStatsSnapshot: снимок статистики источника."""
        return build_source_stats_snapshot(self, context)

    def _build_source_configuration_snapshot(self, context: SourceContext) -> SourceConfigurationSnapshot:
        """Функция: _build_source_configuration_snapshot()
Назначение: формирует конфигурационный snapshot одного источника для UI.
Параметры функции:
- `context` (`SourceContext`): контекст источника.
Возвращаемое значение: SourceConfigurationSnapshot: snapshot конфигурации источника."""
        return build_source_configuration_snapshot_for_runtime(self, context)

    def _copy_visual_config(self, visual_config: Optional[SourceVisualConfig]) -> SourceVisualConfig:
        """Функция: _copy_visual_config()
Назначение: создает копию visual config для безопасной передачи наружу.
Параметры функции:
- `visual_config` (`Optional[SourceVisualConfig]`): исходная visual config.
Возвращаемое значение: SourceVisualConfig: копия visual config."""
        return copy_visual_config(self, visual_config)

    def _emit_event(
        self,
        event_type: str,
        source_id: Optional[str] = None,
        severity: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Функция: _emit_event()
Назначение: отправляет типизированное runtime-событие во внешний callback.
Параметры функции:
- `event_type` (`str`): тип события.
- `source_id` (`Optional[str]`): идентификатор источника.
- `severity` (`str`): уровень важности события.
- `data` (`Optional[Dict[str, Any]]`): полезная нагрузка события.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.event_callback is None:
            return
        event = RuntimeEvent(
            event_type=str(event_type),
            source_id=source_id,
            timestamp=time.time(),
            severity=str(severity),
            data=dict(data or {}),
        )
        try:
            self.event_callback(event)
        except Exception:
            return

    def _dispatch_source_command(
        self,
        command_name: str,
        *args: Any,
        allow_when_stopped: bool = False,
        **kwargs: Any,
    ) -> SourceCommandResult:
        """Функция: _dispatch_source_command()
Назначение: выполняет команду источника напрямую или через command queue runtime.
Параметры функции:
- `command_name` (`str`): имя внутренней команды.
- `args` (`Any`): позиционные аргументы команды.
- `allow_when_stopped` (`bool`): флаг разрешения прямого вызова вне запущенного runtime.
- `kwargs` (`Any`): именованные аргументы команды.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return dispatch_source_command(self, command_name, *args, allow_when_stopped=allow_when_stopped, **kwargs)

    def _drain_command_queue(self) -> None:
        """Функция: _drain_command_queue()
Назначение: обрабатывает накопившиеся команды UI перед очередной итерацией scheduler.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        drain_command_queue(self)

    def _execute_source_command(
        self,
        command_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> SourceCommandResult:
        """Функция: _execute_source_command()
Назначение: выполняет внутреннюю команду источника.
Параметры функции:
- `command_name` (`str`): имя внутренней команды.
- `args` (`Any`): позиционные аргументы команды.
- `kwargs` (`Any`): именованные аргументы команды.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return execute_source_command(self, command_name, *args, **kwargs)

    def set_source_detectors(self, source_id: str, detectors: list[str]) -> SourceCommandResult:
        """Функция: set_source_detectors()
Назначение: обновляет набор детекторов для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
- `detectors` (`list[str]`): целевой список активных детекторов.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return set_source_detectors(self, source_id, detectors)

    def get_source_detectors(self, source_id: str) -> Dict[str, Any]:
        """Функция: get_source_detectors()
Назначение: возвращает набор активных детекторов для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: Dict[str, Any]: словарь с текущими детекторами источника."""
        return get_source_detectors(self, source_id)

    def stop_source(self, source_id: str) -> SourceCommandResult:
        """Функция: stop_source()
Назначение: останавливает один источник во время работы runtime.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return stop_source(self, source_id)

    def resume_source(self, source_id: str) -> SourceCommandResult:
        """Функция: resume_source()
Назначение: возобновляет обработку одного источника во время работы runtime.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return resume_source(self, source_id)

    def reset_movement_reference(self, source_id: str) -> SourceCommandResult:
        """Функция: reset_movement_reference()
Назначение: сбрасывает movement reference для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return reset_movement_reference(self, source_id)

    def set_source_visual_config(
        self,
        source_id: str,
        config: Dict[str, Any] | SourceVisualConfig,
    ) -> SourceCommandResult:
        """Функция: set_source_visual_config()
Назначение: обновляет visual config для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
- `config` (`Dict[str, Any] | SourceVisualConfig`): новый visual config источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return set_source_visual_config(self, source_id, config)

    def get_source_visual_config(self, source_id: str) -> Dict[str, Any]:
        """Функция: get_source_visual_config()
Назначение: возвращает текущую visual config источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: Dict[str, Any]: словарь visual config."""
        return get_source_visual_config(self, source_id)

    def get_runtime_configuration(self) -> Dict[str, Any]:
        """Функция: get_runtime_configuration()
Назначение: возвращает активную runtime-конфигурацию в сериализуемом виде для UI.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь runtime-конфигурации."""
        return get_runtime_configuration(self)

    def describe_ui_capabilities(self) -> Dict[str, Any]:
        """Функция: describe_ui_capabilities()
Назначение: возвращает UI-discovery контракт с доступными командами и mutable полями runtime.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь capabilities runtime."""
        return describe_ui_capabilities(self)

    def apply_topology_updates(self, updates: Dict[str, Any]) -> TopologyUpdateResult:
        """Функция: apply_topology_updates()
Назначение: применяет ограниченный hot-reload topology/runtime настроек.
Параметры функции:
- `updates` (`Dict[str, Any]`): патч настроек runtime/topology.
        Возвращаемое значение: TopologyUpdateResult: результат применения обновлений."""
        return apply_topology_updates(self, updates)

    def describe_restart_required_updates(self) -> Dict[str, list[str]]:
        """Функция: describe_restart_required_updates()
Назначение: возвращает список параметров, требующих полного restart runtime.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, list[str]]: словарь с группами restart-required параметров."""
        return describe_restart_required_updates(self)

    def _apply_set_source_detectors(self, source_id: str, detectors: list[str]) -> SourceCommandResult:
        """Функция: _apply_set_source_detectors()
Назначение: применяет обновление детекторов для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
- `detectors` (`list[str]`): целевой список детекторов.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return apply_set_source_detectors(self, source_id, detectors)

    def _apply_stop_source(self, source_id: str) -> SourceCommandResult:
        """Функция: _apply_stop_source()
Назначение: применяет остановку одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return apply_stop_source(self, source_id)

    def _apply_resume_source(self, source_id: str) -> SourceCommandResult:
        """Функция: _apply_resume_source()
Назначение: применяет возобновление одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return apply_resume_source(self, source_id)

    def _apply_reset_movement_reference(self, source_id: str) -> SourceCommandResult:
        """Функция: _apply_reset_movement_reference()
Назначение: применяет сброс movement reference для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return apply_reset_movement_reference(self, source_id)

    def _apply_set_source_visual_config(
        self,
        source_id: str,
        config: Dict[str, Any] | SourceVisualConfig,
    ) -> SourceCommandResult:
        """Функция: _apply_set_source_visual_config()
Назначение: применяет visual config для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
- `config` (`Dict[str, Any] | SourceVisualConfig`): visual config источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        return apply_set_source_visual_config(self, source_id, config)

    def _apply_topology_updates(self, _scope: str, updates: Dict[str, Any]) -> SourceCommandResult:
        """Функция: _apply_topology_updates()
Назначение: применяет ограниченный hot-reload runtime/topology настроек.
Параметры функции:
- `_scope` (`str`): служебный идентификатор области команды.
- `updates` (`Dict[str, Any]`): патч обновлений runtime/topology.
Возвращаемое значение: SourceCommandResult: результат применения topology patch."""
        return apply_topology_updates_command(self, _scope, updates)

    def _apply_scheduler_updates(
        self,
        scheduler_updates: Dict[str, Any],
        applied: Dict[str, Any],
        rejected: Dict[str, str],
        restart_required: list[str],
    ) -> None:
        """Функция: _apply_scheduler_updates()
Назначение: применяет безопасные обновления scheduler-конфига runtime.
Параметры функции:
- `scheduler_updates` (`Dict[str, Any]`): патч scheduler-полей.
- `applied` (`Dict[str, Any]`): накопитель успешно примененных изменений.
- `rejected` (`Dict[str, str]`): накопитель отклоненных изменений.
- `restart_required` (`list[str]`): накопитель restart-required полей.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        apply_scheduler_updates(self, scheduler_updates, applied, rejected, restart_required)

    def _apply_source_topology_update(
        self,
        source_update: Dict[str, Any],
        applied: Dict[str, Any],
        rejected: Dict[str, str],
        restart_required: list[str],
    ) -> None:
        """Функция: _apply_source_topology_update()
Назначение: применяет безопасные topology-обновления одного источника.
Параметры функции:
- `source_update` (`Dict[str, Any]`): патч настроек одного источника.
- `applied` (`Dict[str, Any]`): накопитель успешно примененных изменений.
- `rejected` (`Dict[str, str]`): накопитель отклоненных изменений.
- `restart_required` (`list[str]`): накопитель restart-required полей.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        apply_source_topology_update(self, source_update, applied, rejected, restart_required)

    def start(self) -> bool:
        """Функция: start()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Примечание: функция входит в основной путь выполнения и влияет на производительность/устойчивость обработки.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._running:
            return False

        self._running = True
        self._finalized = False
        self._last_error = None
        self._scheduler_finished = False
        self.max_postprocess_queue_depth = 0
        self.scheduler.max_infer_queue_depth = 0
        self.stop_event.clear()

        for context in self.source_contexts.values():
            context.stop_requested = False
            context.preview_closed = False
            if context.capture_worker is not None:
                context.capture_worker.start()
                self._emit_event(
                    event_type="source_started",
                    source_id=context.source_config.source_id,
                    severity="info",
                    data={},
                )

        with self._active_infer_lock:
            self._active_infer_workers = max(1, int(self.topology.infer_workers))

        self._infer_threads = []
        for idx in range(max(1, int(self.topology.infer_workers))):
            t = threading.Thread(
                target=self._inference_loop,
                name=f"InferenceWorker-{idx}",
                daemon=True,
            )
            t.start()
            self._infer_threads.append(t)

        self._postprocess_threads = []
        for idx in range(max(1, int(self.topology.postprocess_workers))):
            t = threading.Thread(
                target=self._postprocess_loop,
                name=f"PostprocessWorker-{idx}",
                daemon=True,
            )
            t.start()
            self._postprocess_threads.append(t)

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="CentralScheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

        if self.show_preview and self.preview_callback is None:
            self._preview_thread = threading.Thread(
                target=self._preview_loop,
                name="PreviewWorker",
                daemon=True,
            )
            self._preview_thread.start()
        return True

    def _preview_loop(self) -> None:
        """Функция: _preview_loop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        preview_loop(self)

    def _on_preview_mouse_event(self, event: int, _x: int, _y: int, _flags: int, param: Any) -> None:
        """Функция: _on_preview_mouse_event()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `event` (`int`): событие синхронизации или тип события для переключения ветки обработки.
- `_x` (`int`): координата X точки или левого края области.
- `_y` (`int`): координата Y точки или верхнего края области.
- `_flags` (`int`): битовая маска флагов события/вызова в callback OpenCV.
- `param` (`Any`): одиночный параметр конфигурации или тестового сценария.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        on_preview_mouse_event(self, event, _x, _y, _flags, param)

    def _resolve_preview_key_source(self, fallback_source_id: str) -> Optional[str]:
        """Функция: _resolve_preview_key_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `fallback_source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: Optional[str]: результат шага обработки, который используется следующим этапом пайплайна."""
        return resolve_preview_key_source(self, fallback_source_id)

    def _request_stop_source(self, source_id: str) -> bool:
        """Функция: _request_stop_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        return request_stop_source(self, source_id, self._preview_packet_cls)

    def _pipeline_finished(self) -> bool:
        """Функция: _pipeline_finished()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return bool(
            self._scheduler_finished
            and all(not t.is_alive() for t in self._infer_threads)
            and all(not t.is_alive() for t in self._postprocess_threads)
        )

    def _scheduler_loop(self) -> None:
        """Функция: _scheduler_loop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        idle_sleep = self.topology.scheduler.dispatch_sleep_sec
        try:
            while True:
                self._drain_command_queue()
                self._drain_stopped_source_queues()
                has_item = self.scheduler.dispatch_once()
                if has_item:
                    continue
                if self._all_captures_finished() and self._all_source_queues_empty():
                    break
                time.sleep(idle_sleep)
        except Exception as e:
            self._last_error = str(e)
            self._emit_event(
                event_type="runtime_error",
                severity="error",
                data={"stage": "scheduler", "error": str(e)},
            )
            logging.exception(f"[runtime:scheduler_failed] error={e}")
        finally:
            self._scheduler_finished = True

    def _drain_stopped_source_queues(self) -> None:
        """Функция: _drain_stopped_source_queues()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        for context in self.source_contexts.values():
            if not context.stop_requested:
                continue
            while True:
                try:
                    context.frame_queue.get_nowait()
                except queue.Empty:
                    break

    def _inference_loop(self) -> None:
        """Функция: _inference_loop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            while True:
                try:
                    request = self.infer_queue.get(timeout=0.1)
                except queue.Empty:
                    if self._scheduler_finished and self.infer_queue.empty():
                        break
                    continue

                packet = self._build_inference_packet(request)
                self._enqueue_postprocess_packet(packet)
                self.infer_queue.task_done()
        except Exception as e:
            self._last_error = str(e)
            self._emit_event(
                event_type="runtime_error",
                severity="error",
                data={"stage": "inference_worker", "error": str(e)},
            )
            logging.exception(f"[runtime:inference_worker_failed] error={e}")
        finally:
            with self._active_infer_lock:
                self._active_infer_workers = max(0, self._active_infer_workers - 1)

    def _build_inference_packet(self, request: InferenceRequest) -> InferenceResultPacket:
        """Функция: _build_inference_packet()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `request` (`InferenceRequest`): структура запроса на обработку/инференс, передаваемая между этапами.
Возвращаемое значение: InferenceResultPacket: результат шага обработки, который используется следующим этапом пайплайна."""
        context = self.source_contexts[request.source_id]
        detection_manager = context.detection_manager
        frame_packet = request.frame_packet
        scheduling_latency_ms = max(0.0, (request.scheduled_at - frame_packet.captured_at) * 1000.0)

        raw_results = None
        infer_key = None
        infer_time_ms = 0.0

        if int(request.source_generation) != int(context.generation) or context.stop_requested:
            return InferenceResultPacket(
                request=request,
                raw_results=None,
                infer_key=None,
                infer_time_ms=0.0,
                scheduling_latency_ms=scheduling_latency_ms,
            )

        if (
            detection_manager is not None
            and hasattr(detection_manager, "get_shared_yolo_infer_params")
            and self.hub is not None
            and hasattr(self.hub, "predict")
        ):
            params = detection_manager.get_shared_yolo_infer_params()
            if params is not None:
                infer_key = detection_manager.build_shared_yolo_infer_key(params)
                infer_started = time.perf_counter()
                raw_results = self.hub.predict(
                    model_key=params["model_key"],
                    weights_path=params["weights_path"],
                    frame_bgr=frame_packet.frame,
                    frame_id=frame_packet.frame_index,
                    conf=params["conf"],
                    iou=params["iou"],
                    max_det=params["max_det"],
                    imgsz=params["imgsz"],
                    verbose=False,
                    classes=None,
                    cache_tag="shared_full_frame",
                )
                infer_time_ms = (time.perf_counter() - infer_started) * 1000.0

        return InferenceResultPacket(
            request=request,
            raw_results=raw_results,
            infer_key=infer_key,
            infer_time_ms=infer_time_ms,
            scheduling_latency_ms=scheduling_latency_ms,
        )

    def _enqueue_postprocess_packet(self, packet: InferenceResultPacket) -> None:
        """Функция: _enqueue_postprocess_packet()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `packet` (`InferenceResultPacket`): пакет данных в очереди межпоточного обмена (кадр, метаданные, результат).
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            self.postprocess_queue.put_nowait(packet)
            self.max_postprocess_queue_depth = max(
                int(self.max_postprocess_queue_depth),
                int(self.postprocess_queue.qsize()),
            )
        except queue.Full:
            if self.topology.scheduler.infer_overflow_strategy != "drop_oldest":
                context = self.source_contexts[packet.request.source_id]
                context.postprocess_errors += 1
                return
            try:
                self.postprocess_queue.get_nowait()
                self.postprocess_queue.put_nowait(packet)
                self.max_postprocess_queue_depth = max(
                    int(self.max_postprocess_queue_depth),
                    int(self.postprocess_queue.qsize()),
                )
            except (queue.Empty, queue.Full):
                context = self.source_contexts[packet.request.source_id]
                context.postprocess_errors += 1

    def _postprocess_loop(self) -> None:
        """Функция: _postprocess_loop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            while True:
                try:
                    packet = self.postprocess_queue.get(timeout=0.1)
                except queue.Empty:
                    if self._all_infer_workers_finished() and self.postprocess_queue.empty():
                        break
                    continue

                self._postprocess_packet(packet)
                self.postprocess_queue.task_done()
        except Exception as e:
            self._last_error = str(e)
            self._emit_event(
                event_type="runtime_error",
                severity="error",
                data={"stage": "postprocess_worker", "error": str(e)},
            )
            logging.exception(f"[runtime:postprocess_worker_failed] error={e}")

    def _postprocess_packet(self, packet: InferenceResultPacket) -> None:
        """Функция: _postprocess_packet()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `packet` (`InferenceResultPacket`): пакет данных в очереди межпоточного обмена (кадр, метаданные, результат).
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        request = packet.request
        frame_packet = request.frame_packet
        context = self.source_contexts[request.source_id]
        detection_manager = context.detection_manager
        violation_manager = context.violation_manager

        if detection_manager is None:
            return
        if int(request.source_generation) != int(context.generation):
            return
        if context.stop_requested:
            return

        with bind_log_source(request.source_id):
            try:
                context.last_error = None
                if (
                    packet.raw_results is not None
                    and packet.infer_key is not None
                    and hasattr(detection_manager, "seed_shared_yolo_results")
                ):
                    detection_manager.seed_shared_yolo_results(
                        frame_count=frame_packet.frame_index,
                        infer_key=packet.infer_key,
                        raw_results=packet.raw_results,
                    )

                source_fps = 30.0
                if context.capture_worker is not None and context.capture_worker.source_manager is not None:
                    source_fps = float(getattr(context.capture_worker.source_manager, "fps", 30.0) or 30.0)
                if violation_manager is not None and hasattr(violation_manager, "set_source_fps"):
                    violation_manager.set_source_fps(source_fps)

                frame = frame_packet.frame
                frame_count = int(frame_packet.frame_index)
                wall_now = time.time()
                context.latest_frame_for_reset = frame
                context.latest_frame_packet = frame_packet

                obstruction = detection_manager.detect_obstruction(frame, frame_count)
                movement = detection_manager.detect_movement(
                    frame,
                    obstruction.get("detected", False),
                    frame_count=frame_count,
                )
                forbidden = detection_manager.detect_forbidden(frame, frame_count)
                dms = detection_manager.detect_dms(frame, frame_count)

                self._update_stats(
                    context=context,
                    obstruction=obstruction,
                    movement=movement,
                    forbidden=forbidden,
                    dms=dms,
                    now_ts=wall_now,
                )
                self._run_violation_pipeline(
                    context=context,
                    frame=frame,
                    video_timestamp=frame_packet.video_timestamp,
                    processing_time=wall_now,
                    frame_count=frame_count,
                    obstruction=obstruction,
                    movement=movement,
                    forbidden=forbidden,
                    dms=dms,
                )
                total_duration = max(0.0, time.time() - wall_now)
                display_frame = self._build_display_frame(
                    context=context,
                    frame=frame,
                    frame_count=frame_count,
                    video_timestamp=frame_packet.video_timestamp,
                    total_duration=total_duration,
                    obstruction=obstruction,
                    movement=movement,
                    forbidden=forbidden,
                    dms=dms,
                )
                self._render_preview_frame(
                    source_id=request.source_id,
                    frame=display_frame,
                    frame_index=frame_count,
                    video_timestamp=float(frame_packet.video_timestamp),
                    captured_at=float(frame_packet.captured_at),
                    processed_at=float(time.time()),
                    processed_frames=int(context.processed_frames + 1),
                )

                context.processed_frames += 1
                if packet.infer_time_ms > 0:
                    context.infer_runs += 1
                    context.infer_time_sum_ms += packet.infer_time_ms
                context.scheduling_latency_sum_ms += packet.scheduling_latency_ms
                context.end_to_end_lag_sum_ms += max(0.0, (time.time() - frame_packet.captured_at) * 1000.0)
                self._maybe_close_source_preview(context)
            except Exception as exc:
                context.postprocess_errors += 1
                context.last_error = str(exc)
                self._emit_event(
                    event_type="detector_error",
                    source_id=request.source_id,
                    severity="error",
                    data={"error": str(exc), "frame_index": int(frame_packet.frame_index)},
                )
                raise

    def _build_display_frame(
        self,
        context: SourceContext,
        frame: Any,
        frame_count: int,
        video_timestamp: float,
        total_duration: float,
        obstruction: Dict[str, Any],
        movement: Dict[str, Any],
        forbidden: Dict[str, Any],
        dms: Dict[str, Any],
    ) -> Any:
        """Функция: _build_display_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `context` (`SourceContext`): контекст текущего источника с состоянием пайплайна и очередями.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `total_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `obstruction` (`Dict[str, Any]`): результат проверки перекрытия объектива/обструкции камеры.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
- `forbidden` (`Dict[str, Any]`): результат детектора запрещенных предметов.
- `dms` (`Dict[str, Any]`): результат DMS-детектора (глаза, ремень, телефон и сопутствующие нарушения).
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        display_frame = frame
        if frame is None:
            return frame

        stats_manager = context.stats_manager
        violation_manager = context.violation_manager
        visualizer = getattr(violation_manager, "visualizer", None) if violation_manager is not None else None
        visual_config = self._copy_visual_config(context.visual_config)

        if visualizer is not None and hasattr(visualizer, "draw"):
            saved_alerts = 0
            if (
                stats_manager is not None
                and hasattr(stats_manager, "stats")
                and hasattr(stats_manager.stats, "saved_violations")
            ):
                saved_alerts = int(stats_manager.stats.saved_violations)

            viz_stats = {
                "obstruction": {
                    "total": int(getattr(stats_manager.stats, "obstruction_violations", 0)) if stats_manager is not None and hasattr(stats_manager, "stats") else 0,
                    "current_duration": float(getattr(stats_manager.stats, "current_obstruction_duration", 0.0)) if stats_manager is not None and hasattr(stats_manager, "stats") else 0.0,
                },
                "movement": {
                    "total": int(getattr(stats_manager.stats, "camera_movements", 0)) if stats_manager is not None and hasattr(stats_manager, "stats") else 0,
                },
                "forbidden_items": {
                    "total": int(getattr(stats_manager.stats, "forbidden_items_violations", 0)) if stats_manager is not None and hasattr(stats_manager, "stats") else 0,
                    "current_objects": int(len(forbidden.get("objects", []))),
                    "active_cooldowns": forbidden.get("stats", {}).get("active_cooldowns", {}),
                },
                "dms": {
                    "total": int(getattr(stats_manager.stats, "dms_violations", 0)) if stats_manager is not None and hasattr(stats_manager, "stats") else 0,
                },
                "frame_count": int(frame_count),
                "alerts_count": int(saved_alerts),
            }
            viz_result = {
                "yolo_objects": obstruction.get("yolo_objects", []),
                "forbidden_objects": forbidden.get("objects", []),
                "dms_objects": dms.get("objects", []),
                "dms_violations": dms.get("violations", []),
                "detected": obstruction.get("detected", False),
                "detectors_count": obstruction.get("detectors_count", 0),
                "stats": viz_stats,
            }
            fps_value = None
            if visual_config.show_fps and stats_manager is not None:
                fps_value = float(getattr(stats_manager, "current_fps", 0.0) or 0.0)
            display_frame = visualizer.draw(
                frame=frame,
                result=viz_result,
                movement_info=movement,
                video_timestamp=float(video_timestamp),
                total_duration=float(total_duration),
                    fps=fps_value,
            )

        preview_width = visual_config.preview_width
        if preview_width is not None and cv2 is not None and display_frame is not None:
            try:
                original_h, original_w = display_frame.shape[:2]
                if original_w > 0 and int(preview_width) != int(original_w):
                    scale = float(preview_width) / float(original_w)
                    new_h = max(1, int(original_h * scale))
                    display_frame = cv2.resize(
                        display_frame,
                        (int(preview_width), int(new_h)),
                        interpolation=cv2.INTER_AREA,
                    )
            except Exception:
                pass
        return display_frame

    def _enqueue_preview_packet(self, packet: PreviewPacket) -> None:
        """Функция: _enqueue_preview_packet()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `packet` (`PreviewPacket`): пакет данных в очереди межпоточного обмена (кадр, метаданные, результат).
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        enqueue_preview_packet(self, packet)

    def _render_preview_frame(
        self,
        source_id: str,
        frame: Any,
        frame_index: int,
        video_timestamp: float,
        captured_at: float,
        processed_at: float,
        processed_frames: int,
    ) -> None:
        """Функция: _render_preview_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        render_preview_frame(
            self,
            source_id=source_id,
            frame=frame,
            frame_index=frame_index,
            video_timestamp=video_timestamp,
            captured_at=captured_at,
            processed_at=processed_at,
            processed_frames=processed_frames,
            preview_packet_cls=self._preview_packet_cls,
        )

    def _build_preview_payload(
        self,
        source_id: str,
        frame: Any,
        frame_index: int,
        video_timestamp: float,
        captured_at: float,
        processed_at: float,
        processed_frames: int,
    ) -> PreviewFramePayload:
        """Функция: _build_preview_payload()
Назначение: формирует структурированный preview payload для UI callback.
Параметры функции:
- `source_id` (`str`): идентификатор источника.
- `frame` (`Any`): кадр preview.
- `frame_index` (`int`): индекс кадра.
- `video_timestamp` (`float`): timestamp кадра в координатах видео.
- `captured_at` (`float`): wall-clock время захвата.
- `processed_at` (`float`): wall-clock время формирования preview payload.
- `processed_frames` (`int`): число обработанных кадров источника.
Возвращаемое значение: PreviewFramePayload: payload preview callback."""
        return build_preview_payload(
            self,
            source_id=source_id,
            frame=frame,
            frame_index=frame_index,
            video_timestamp=video_timestamp,
            captured_at=captured_at,
            processed_at=processed_at,
            processed_frames=processed_frames,
        )

    def _invoke_preview_callback(self, payload: PreviewFramePayload) -> None:
        """Функция: _invoke_preview_callback()
Назначение: вызывает preview callback с поддержкой legacy и нового payload-контракта.
Параметры функции:
- `payload` (`PreviewFramePayload`): структурированный payload preview-кадра.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        invoke_preview_callback(self, payload)

    def _is_source_processing_drained(self, context: SourceContext) -> bool:
        """Функция: _is_source_processing_drained()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `context` (`SourceContext`): контекст текущего источника с состоянием пайплайна и очередями.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return is_source_processing_drained(self, context)

    def _maybe_close_source_preview(self, context: SourceContext) -> None:
        """Функция: _maybe_close_source_preview()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `context` (`SourceContext`): контекст текущего источника с состоянием пайплайна и очередями.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        maybe_close_source_preview(self, context, self._preview_packet_cls)

    def _reset_movement_reference_for_source(self, source_id: str) -> bool:
        """Функция: _reset_movement_reference_for_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        return reset_movement_reference_for_source(self, source_id)

    def _update_stats(
        self,
        context: SourceContext,
        obstruction: Dict[str, Any],
        movement: Dict[str, Any],
        forbidden: Dict[str, Any],
        dms: Dict[str, Any],
        now_ts: float,
    ) -> None:
        """Функция: _update_stats()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `context` (`SourceContext`): контекст текущего источника с состоянием пайплайна и очередями.
- `obstruction` (`Dict[str, Any]`): результат проверки перекрытия объектива/обструкции камеры.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
- `forbidden` (`Dict[str, Any]`): результат детектора запрещенных предметов.
- `dms` (`Dict[str, Any]`): результат DMS-детектора (глаза, ремень, телефон и сопутствующие нарушения).
- `now_ts` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        stats_manager = context.stats_manager
        if stats_manager is None:
            return

        if hasattr(stats_manager, "increment_frames"):
            stats_manager.increment_frames()

        if context.last_processed_at > 0 and hasattr(stats_manager, "update_fps"):
            frame_time = max(0.0, now_ts - context.last_processed_at)
            stats_manager.update_fps(frame_time)
        context.last_processed_at = now_ts

        if obstruction.get("detected") and hasattr(stats_manager, "record_obstruction"):
            obstruction_duration = 0.0
            violation_manager = context.violation_manager
            if (
                violation_manager is not None
                and getattr(violation_manager, "obstruction_start_time", None) is not None
            ):
                obstruction_duration = max(0.0, now_ts - violation_manager.obstruction_start_time)
            stats_manager.record_obstruction(obstruction_duration)

        if movement.get("movement_detected") and hasattr(stats_manager, "record_movement"):
            stats_manager.record_movement(float(movement.get("movement_duration", 0.0)))

        if forbidden.get("current_violation") and hasattr(stats_manager, "record_forbidden"):
            stats_manager.record_forbidden()

        if dms.get("violations") and hasattr(stats_manager, "record_dms_violation"):
            for violation in dms["violations"]:
                stats_manager.record_dms_violation(violation.get("type", "unknown"))

        dms_detector = getattr(context.detection_manager, "dms_detector", None)
        if dms_detector is not None and hasattr(stats_manager, "update_dms_state"):
            dms_stats = getattr(dms_detector, "stats", {})
            stats_manager.update_dms_state(
                eye_state=dms_stats.get("eye_state", "unknown"),
                seatbelt_state=dms_stats.get("seatbelt_state", "unknown"),
            )

    def _run_violation_pipeline(
        self,
        context: SourceContext,
        frame: Any,
        video_timestamp: float,
        processing_time: float,
        frame_count: int,
        obstruction: Dict[str, Any],
        movement: Dict[str, Any],
        forbidden: Dict[str, Any],
        dms: Dict[str, Any],
    ) -> None:
        """Функция: _run_violation_pipeline()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `context` (`SourceContext`): контекст текущего источника с состоянием пайплайна и очередями.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `processing_time` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
- `obstruction` (`Dict[str, Any]`): результат проверки перекрытия объектива/обструкции камеры.
- `movement` (`Dict[str, Any]`): результат детектора движения/смещения камеры.
- `forbidden` (`Dict[str, Any]`): результат детектора запрещенных предметов.
- `dms` (`Dict[str, Any]`): результат DMS-детектора (глаза, ремень, телефон и сопутствующие нарушения).
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        violation_manager = context.violation_manager
        stats_manager = context.stats_manager
        if violation_manager is None:
            return

        obstruction_violation = violation_manager.process_obstruction(
            obstruction,
            frame,
            video_timestamp,
            processing_time,
        )
        if obstruction_violation is not None and stats_manager is not None and hasattr(stats_manager, "record_obstruction_violation"):
            stats_manager.record_obstruction_violation()

        movement_violation = violation_manager.process_movement(
            movement,
            frame,
            video_timestamp,
        )
        if movement_violation is not None and stats_manager is not None and hasattr(stats_manager, "record_movement_saved"):
            stats_manager.record_movement_saved()

        if forbidden.get("current_violation"):
            forbidden_violation = violation_manager.process_forbidden(
                forbidden,
                frame,
                video_timestamp,
            )
            if forbidden_violation is not None and stats_manager is not None and hasattr(stats_manager, "record_forbidden_saved"):
                stats_manager.record_forbidden_saved()

        if dms.get("violations"):
            dms_violation = violation_manager.process_dms(
                dms,
                frame,
                video_timestamp,
                processing_time,
                frame_count,
            )
            if dms_violation is not None and stats_manager is not None and hasattr(stats_manager, "record_dms_saved"):
                stats_manager.record_dms_saved()

    def _all_captures_finished(self) -> bool:
        """Функция: _all_captures_finished()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        for context in self.source_contexts.values():
            worker = context.capture_worker
            if worker is not None and worker.is_running():
                return False
        return True

    def _all_source_queues_empty(self) -> bool:
        """Функция: _all_source_queues_empty()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return all(ctx.frame_queue.empty() for ctx in self.source_contexts.values())

    def _all_infer_workers_finished(self) -> bool:
        """Функция: _all_infer_workers_finished()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        with self._active_infer_lock:
            return self._active_infer_workers == 0

    def stop(self, timeout: float = 5.0) -> None:
        """Функция: stop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`float`): максимальное время ожидания завершения операции.
Примечание: функция входит в основной путь выполнения и влияет на производительность/устойчивость обработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stop_event.set()
        deadline = time.monotonic() + max(0.0, float(timeout))

        for context in self.source_contexts.values():
            if context.capture_worker is not None:
                context.capture_worker.stop()
        for context in self.source_contexts.values():
            if context.capture_worker is not None:
                remaining = max(0.0, deadline - time.monotonic())
                context.capture_worker.join(timeout=remaining)

        self._join_thread(self._scheduler_thread, deadline)
        for thread in self._infer_threads:
            self._join_thread(thread, deadline)
        for thread in self._postprocess_threads:
            self._join_thread(thread, deadline)
        self._join_thread(self._preview_thread, deadline)

        self._running = False
        self._finalize_sources(timeout=timeout)

    def wait(self, timeout: Optional[float] = None) -> None:
        """Функция: wait()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`Optional[float]`): максимальное время ожидания завершения операции.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        deadline = None if timeout is None else (time.monotonic() + max(0.0, float(timeout)))
        self._join_thread(self._scheduler_thread, deadline)
        for thread in self._infer_threads:
            self._join_thread(thread, deadline)
        for thread in self._postprocess_threads:
            self._join_thread(thread, deadline)
        self._join_thread(self._preview_thread, deadline)

        all_done = (
            self._scheduler_finished
            and all(not t.is_alive() for t in self._infer_threads)
            and all(not t.is_alive() for t in self._postprocess_threads)
        )
        if all_done:
            self._running = False
            self._finalize_sources(timeout=5.0)

    def _join_thread(self, thread: Optional[threading.Thread], deadline: Optional[float]) -> None:
        """Функция: _join_thread()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `thread` (`Optional[threading.Thread]`): рабочий поток, выполняющий часть пайплайна.
- `deadline` (`Optional[float]`): параметр политики планирования/деградации под нагрузкой.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if thread is None:
            return
        if deadline is None:
            thread.join()
            return
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)

    def _finalize_sources(self, timeout: float = 5.0) -> None:
        """Функция: _finalize_sources()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`float`): максимальное время ожидания завершения операции.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._finalized:
            return
        self._finalized = True

        for context in self.source_contexts.values():
            violation_manager = context.violation_manager
            if violation_manager is None:
                continue
            try:
                if hasattr(violation_manager, "flush_writes"):
                    violation_manager.flush_writes(timeout=timeout)
            except Exception:
                pass

        for source_id, context in self.source_contexts.items():
            runtime_stats: Dict[str, Any] = {}
            if context.stats_manager is not None and hasattr(context.stats_manager, "get_current_stats"):
                try:
                    runtime_stats = dict(context.stats_manager.get_current_stats() or {})
                except Exception:
                    runtime_stats = {}

            capture_stats_payload: Dict[str, Any] = {}
            capture_worker_stats = context.capture_worker.get_stats() if context.capture_worker else None
            capture_stats_payload["captured_frames"] = int(
                capture_worker_stats.read_frames if capture_worker_stats is not None else 0
            )
            capture_stats_payload["capture_dropped_frames"] = int(
                capture_worker_stats.dropped_frames if capture_worker_stats is not None else 0
            )
            capture_stats_payload["writer_queue_size"] = int(
                context.violation_manager.get_writer_queue_size()
                if context.violation_manager is not None
                and hasattr(context.violation_manager, "get_writer_queue_size")
                else 0
            )

            if runtime_stats:
                with bind_log_source(source_id):
                    log_summary_block(
                        StatsManager.build_summary_from_stats(
                            stats=runtime_stats,
                            title=f"SOURCE SUMMARY [{source_id}]",
                            capture_stats=capture_stats_payload,
                        )
                    )

        for context in self.source_contexts.values():
            violation_manager = context.violation_manager
            if violation_manager is None:
                continue
            try:
                if hasattr(violation_manager, "cleanup"):
                    violation_manager.cleanup()
            except Exception:
                pass

        if self.show_preview and cv2 is not None and self._preview_thread is None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def is_running(self) -> bool:
        """Функция: is_running()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
        threads_alive = any(t.is_alive() for t in self._infer_threads) or any(
            t.is_alive() for t in self._postprocess_threads
        )
        scheduler_alive = self._scheduler_thread is not None and self._scheduler_thread.is_alive()
        return bool(self._running or scheduler_alive or threads_alive)

    def update_detectors(
        self,
        enabled_detectors: Optional[list[str]] = None,
    ) -> Dict[str, Dict[str, list[str]]]:
        """Функция: update_detectors()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `enabled_detectors` (`Optional[list[str]]`): список активных детекторов, участвующих в обработке кадра.
Возвращаемое значение: Dict[str, Dict[str, list[str]]]: результат шага обработки, который используется следующим этапом пайплайна."""
        result: Dict[str, Dict[str, list[str]]] = {}
        for source_id, context in self.source_contexts.items():
            manager = context.detection_manager
            if manager is None or not hasattr(manager, "set_runtime_detectors"):
                continue
            updated = manager.set_runtime_detectors(
                enabled_detectors=enabled_detectors,
            )
            context.source_config.enabled_detectors = list(updated.get("enabled_detectors", []))
            result[source_id] = {
                "enabled_detectors": list(updated.get("enabled_detectors", [])),
            }
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Функция: get_stats()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        return get_stats(self)

    def get_state(self) -> Dict[str, Any]:
        """Функция: get_state()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
        Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        return get_state(self)
