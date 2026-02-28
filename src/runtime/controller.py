"""Файл: src/runtime/controller.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- .multi_source_runtime: используется для передачи данных или вызова связанной логики.
- .runtime_models: используется для передачи данных или вызова связанной логики.
- .universal_processor: используется для передачи данных или вызова связанной логики."""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

from ..processing.universal_processor import UniversalProcessor
from ..processing.multi_source_runtime import MultiSourceRuntime
from .topology import RuntimeTopology, SourceConfig, SchedulerConfig
from .contracts import (
    SourceCommandResult,
    SourceVisualConfig,
    TopologyUpdateResult,
)
from .snapshot_builders import (
    build_runtime_capabilities_snapshot,
    build_runtime_configuration_snapshot,
    build_source_configuration_snapshot,
)


@dataclass
class RuntimeConfig:
    """Класс: RuntimeConfig
Назначение: реализует runtime-логику и синхронизацию этапов обработки.
Поля класса:
- `async_violation_writes` (`bool`): данные детектора нарушений, используемые для итогового решения по кадру.
- `camera_id` (`Optional[int]`): идентификатор/индекс для адресации и сопоставления сущностей.
- `device` (`str`): вычислительное устройство для инференса (`cpu`, `cuda`, `cuda:N`).
- `enabled_detectors` (`Optional[List[str]]`): список активных детекторов, участвующих в обработке кадра.
- `hub` (`Any`): общий объект инференса, который кэширует модели и выполняет predict.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `infer_workers` (`int`): рабочий исполнитель или их количество для параллельной обработки.
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `max_duration` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `postprocess_workers` (`int`): рабочий исполнитель или их количество для параллельной обработки.
- `preview_width` (`Optional[int]`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `runtime_engine` (`str`): режим выполнения runtime (`legacy` или `centralized`).
- `save_dir` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `scheduler_dispatch_sleep_sec` (`float`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `scheduler_infer_queue_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `show_fps` (`bool`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    input_source: Any
    camera_id: Optional[int] = None
    save_dir: str = "violations"
    enabled_detectors: Optional[List[str]] = None
    hub: Any = None
    device: str = "cuda:0"
    use_half: bool = True
    imgsz: int = 720
    show_fps: bool = False
    preview_width: Optional[int] = None
    async_violation_writes: bool = True
    writer_queue_max_size: int = 256
    writer_overflow_strategy: str = "drop_newest"
    show_preview: bool = True
    max_duration: Optional[float] = None
    runtime_engine: str = "centralized"
    source_queue_max_size: int = 8
    source_drop_policy: str = "drop_oldest"
    scheduler_infer_queue_size: int = 64
    infer_workers: int = 1
    postprocess_workers: int = 1
    scheduler_dispatch_sleep_sec: float = 0.003
    topology: Optional[RuntimeTopology] = None
    preview_callback: Optional[Callable[..., None]] = None
    event_callback: Optional[Callable[..., None]] = None
    command_timeout_sec: float = 1.0


class RuntimeController:
    """Класс: RuntimeController
Назначение: управляет жизненным циклом runtime и внешними командами запуска/остановки.
Поля класса:
- `_last_error` (`Optional[str]`): сообщение или объект ошибки, используемый для диагностики и логирования.
- `_lock` (`Any`): синхронизатор доступа к общему состоянию между потоками.
- `_multi_source_runtime_factory` (`Callable[..., MultiSourceRuntime]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `_processor` (`Optional[Any]`): экземпляр процессора кадра, выполняющий основной pipeline.
- `_processor_factory` (`Callable[..., UniversalProcessor]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `_started_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_thread` (`Optional[threading.Thread]`): рабочий поток, выполняющий часть пайплайна.
- `config` (`RuntimeConfig`): структура конфигурации компонента/подсистемы.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `start()`, `_create_runtime_engine()`, `_resolve_preview_width()`, `_should_use_centralized_runtime()`, `_build_single_source_topology()`, `_build_single_source_id()`, `_run_pipeline()`, `stop()`, `wait()`"""

    def __init__(
        self,
        config: RuntimeConfig,
        processor_factory: Callable[..., UniversalProcessor] = UniversalProcessor,
        multi_source_runtime_factory: Callable[..., MultiSourceRuntime] = MultiSourceRuntime,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `config` (`RuntimeConfig`): структура конфигурации компонента/подсистемы.
- `processor_factory` (`Callable[..., UniversalProcessor]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
- `multi_source_runtime_factory` (`Callable[..., MultiSourceRuntime]`): фабричная функция/объект, создающий экземпляры подсистемы по запросу.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.config = config
        self._processor_factory = processor_factory
        self._multi_source_runtime_factory = multi_source_runtime_factory
        self._processor: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._started_at: float = 0.0

    def start(self) -> bool:
        """Функция: start()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Примечание: функция входит в основной путь выполнения и влияет на производительность/устойчивость обработки.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        with self._lock:
            if self.is_running():
                return False

            self._last_error = None
            self._processor = self._create_runtime_engine()
            self._thread = threading.Thread(
                target=self._run_pipeline,
                name="RuntimeControllerWorker",
                daemon=True,
            )
            self._started_at = time.time()
            self._thread.start()
            return True

    def _create_runtime_engine(self) -> Any:
        """Функция: _create_runtime_engine()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        effective_preview_width = self._resolve_preview_width()
        if self._should_use_centralized_runtime():
            topology = self.config.topology or self._build_single_source_topology()
            return self._multi_source_runtime_factory(
                topology=topology,
                hub=self.config.hub,
                save_dir=self.config.save_dir,
                async_violation_writes=self.config.async_violation_writes,
                writer_queue_max_size=self.config.writer_queue_max_size,
                writer_overflow_strategy=self.config.writer_overflow_strategy,
                show_preview=self.config.show_preview,
                show_fps=self.config.show_fps,
                preview_width=effective_preview_width,
                preview_callback=self.config.preview_callback,
                event_callback=self.config.event_callback,
                command_timeout_sec=self.config.command_timeout_sec,
            )

        return self._processor_factory(
            input_source=self.config.input_source,
            camera_id=self.config.camera_id,
            save_dir=self.config.save_dir,
            enabled_detectors=self.config.enabled_detectors,
            hub=self.config.hub,
            device=self.config.device,
            use_half=self.config.use_half,
            imgsz=self.config.imgsz,
            show_fps=self.config.show_fps,
            preview_width=effective_preview_width,
            async_violation_writes=self.config.async_violation_writes,
            writer_queue_max_size=self.config.writer_queue_max_size,
            writer_overflow_strategy=self.config.writer_overflow_strategy,
        )

    def _resolve_preview_width(self) -> Optional[int]:
        """Функция: _resolve_preview_width()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Optional[int]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.config.preview_width is not None:
            width = int(self.config.preview_width)
            return width if width > 0 else None

        try:
            imgsz = int(self.config.imgsz)
        except Exception:
            return None
        if imgsz <= 0:
            return None
        return max(1, (imgsz * 16) // 9)

    def _should_use_centralized_runtime(self) -> bool:
        """Функция: _should_use_centralized_runtime()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        mode = str(self.config.runtime_engine).strip().lower()
        return mode in {"centralized", "multi_source"}

    def _build_single_source_topology(self) -> RuntimeTopology:
        """Функция: _build_single_source_topology()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: RuntimeTopology: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.config.camera_id is not None:
            source_id = f"camera_{int(self.config.camera_id)}"
        else:
            source_id = self._build_single_source_id(self.config.input_source)

        source_config = SourceConfig(
            source_id=source_id,
            input_source=self.config.input_source,
            camera_id=self.config.camera_id,
            capture_queue_size=self.config.source_queue_max_size,
            drop_policy=self.config.source_drop_policy,
            enabled_detectors=self.config.enabled_detectors or ["all"],
        )
        scheduler_config = SchedulerConfig(
            infer_queue_size=self.config.scheduler_infer_queue_size,
            dispatch_sleep_sec=self.config.scheduler_dispatch_sleep_sec,
        )
        return RuntimeTopology(
            sources=[source_config],
            scheduler=scheduler_config,
            infer_workers=self.config.infer_workers,
            postprocess_workers=self.config.postprocess_workers,
        )

    @staticmethod
    def _build_single_source_id(input_source: Any) -> str:
        """Функция: _build_single_source_id()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        source_text = str(input_source or "").strip()
        if not source_text:
            return "source_0"
        lower = source_text.lower()
        if lower.startswith(("rtsp://", "rtsps://")):
            return "source_rtsp"
        base_name = os.path.splitext(os.path.basename(source_text))[0] or "source"
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in base_name)
        return f"source_{safe_name}"

    def _run_pipeline(self) -> None:
        """Функция: _run_pipeline()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        assert self._processor is not None
        try:
            if self._should_use_centralized_runtime():
                self._processor.start()
                self._processor.wait(timeout=self.config.max_duration)
                if self.config.max_duration is not None:
                    self._processor.stop(timeout=2.0)
            else:
                self._processor.process_source(
                    show_preview=self.config.show_preview,
                    max_duration=self.config.max_duration,
                )
        except Exception as e:
            self._last_error = str(e)
            logging.exception(f"[runtime:controller_failed] error={e}")

    def stop(self, timeout: float = 5.0) -> bool:
        """Функция: stop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`float`): максимальное время ожидания завершения операции.
Примечание: функция входит в основной путь выполнения и влияет на производительность/устойчивость обработки.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        with self._lock:
            processor = self._processor
            thread = self._thread

        if processor is None:
            return False

        if self._should_use_centralized_runtime():
            try:
                processor.stop(timeout=timeout)
            except Exception:
                pass
        else:
            processor.is_running = False
            try:
                if hasattr(processor, "source_manager") and processor.source_manager:
                    processor.source_manager.release()
            except Exception:
                pass

        if thread is not None:
            thread.join(timeout=max(0.0, float(timeout)))
        return True

    def wait(self, timeout: Optional[float] = None) -> None:
        """Функция: wait()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`Optional[float]`): максимальное время ожидания завершения операции.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Функция: is_running()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
        return self._thread is not None and self._thread.is_alive()

    def _get_centralized_runtime(self) -> Optional[MultiSourceRuntime]:
        """Функция: _get_centralized_runtime()
Назначение: возвращает активный centralized runtime, если он используется текущим контроллером.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Optional[MultiSourceRuntime]: экземпляр centralized runtime или `None`."""
        if self._processor is None or not self._should_use_centralized_runtime():
            return None
        if isinstance(self._processor, MultiSourceRuntime):
            return self._processor
        return self._processor if hasattr(self._processor, "get_state") else None

    def update_detectors(
        self,
        enabled_detectors: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """Функция: update_detectors()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `enabled_detectors` (`Optional[List[str]]`): список активных детекторов, участвующих в обработке кадра.
Возвращаемое значение: Dict[str, List[str]]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._processor is None:
            return {"enabled_detectors": []}
        if self._should_use_centralized_runtime():
            updated = self._processor.update_detectors(
                enabled_detectors=enabled_detectors,
            )
            enabled_union = set()
            for source_payload in updated.values():
                enabled_union.update(source_payload.get("enabled_detectors", []))
            if enabled_detectors is not None:
                self.config.enabled_detectors = sorted(enabled_union)
            return {
                "enabled_detectors": list(self.config.enabled_detectors or []),
            }
        return self._processor.detection_manager.set_runtime_detectors(
            enabled_detectors=enabled_detectors,
        )

    def set_source_detectors(
        self,
        source_id: str,
        detectors: List[str],
    ) -> SourceCommandResult:
        """Функция: set_source_detectors()
Назначение: обновляет набор детекторов для одного источника в runtime.
Параметры функции:
- `source_id` (`str`): идентификатор целевого источника.
- `detectors` (`List[str]`): целевой список активных детекторов для источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return SourceCommandResult(
                success=False,
                source_id=source_id,
                error="unsupported_runtime",
                message="Per-source detector updates are supported only in centralized runtime.",
            )
        return runtime.set_source_detectors(source_id=source_id, detectors=detectors)

    def get_source_detectors(self, source_id: str) -> Dict[str, Any]:
        """Функция: get_source_detectors()
Назначение: возвращает текущий набор детекторов для одного источника.
Параметры функции:
- `source_id` (`str`): идентификатор целевого источника.
Возвращаемое значение: Dict[str, Any]: словарь с текущими детекторами источника."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return {
                "source_id": source_id,
                "enabled_detectors": list(self.config.enabled_detectors or []),
                "error": "unsupported_runtime",
            }
        return runtime.get_source_detectors(source_id=source_id)

    def stop_source(self, source_id: str) -> SourceCommandResult:
        """Функция: stop_source()
Назначение: останавливает обработку одного источника в centralized runtime.
Параметры функции:
- `source_id` (`str`): идентификатор целевого источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return SourceCommandResult(
                success=False,
                source_id=source_id,
                error="unsupported_runtime",
                message="Per-source stop is supported only in centralized runtime.",
            )
        return runtime.stop_source(source_id=source_id)

    def resume_source(self, source_id: str) -> SourceCommandResult:
        """Функция: resume_source()
Назначение: возобновляет обработку одного источника в centralized runtime.
Параметры функции:
- `source_id` (`str`): идентификатор целевого источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return SourceCommandResult(
                success=False,
                source_id=source_id,
                error="unsupported_runtime",
                message="Per-source resume is supported only in centralized runtime.",
            )
        return runtime.resume_source(source_id=source_id)

    def reset_movement_reference(self, source_id: str) -> SourceCommandResult:
        """Функция: reset_movement_reference()
Назначение: сбрасывает reference frame для детектора движения конкретного источника.
Параметры функции:
- `source_id` (`str`): идентификатор целевого источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return SourceCommandResult(
                success=False,
                source_id=source_id,
                error="unsupported_runtime",
                message="Movement reset is supported only in centralized runtime.",
            )
        return runtime.reset_movement_reference(source_id=source_id)

    def set_source_visual_config(self, source_id: str, config: Dict[str, Any]) -> SourceCommandResult:
        """Функция: set_source_visual_config()
Назначение: обновляет visual config конкретного источника.
Параметры функции:
- `source_id` (`str`): идентификатор целевого источника.
- `config` (`Dict[str, Any]`): visual config источника.
Возвращаемое значение: SourceCommandResult: результат выполнения команды."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return SourceCommandResult(
                success=False,
                source_id=source_id,
                error="unsupported_runtime",
                message="Per-source visual config is supported only in centralized runtime.",
            )
        return runtime.set_source_visual_config(source_id=source_id, config=config)

    def get_source_visual_config(self, source_id: str) -> Dict[str, Any]:
        """Функция: get_source_visual_config()
Назначение: возвращает visual config конкретного источника.
Параметры функции:
- `source_id` (`str`): идентификатор целевого источника.
Возвращаемое значение: Dict[str, Any]: словарь visual config источника."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return {
                "source_id": source_id,
                "visual_config": {},
                "error": "unsupported_runtime",
            }
        return runtime.get_source_visual_config(source_id=source_id)

    def get_runtime_configuration(self) -> Dict[str, Any]:
        """Функция: get_runtime_configuration()
Назначение: возвращает сериализуемую конфигурацию runtime для UI и редактора настроек.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь runtime-конфигурации."""
        runtime = self._get_centralized_runtime()
        if runtime is not None and hasattr(runtime, "get_runtime_configuration"):
            return runtime.get_runtime_configuration()

        engine = "centralized" if self._should_use_centralized_runtime() else "legacy"
        topology = self.config.topology or self._build_single_source_topology()
        default_visual = SourceVisualConfig(
            preview_width=self._resolve_preview_width(),
            show_fps=bool(self.config.show_fps),
        ).to_dict()
        sources = {}
        for source_cfg in topology.sources:
            sources[source_cfg.source_id] = build_source_configuration_snapshot(
                source_cfg=source_cfg,
                enabled_detectors=list(source_cfg.enabled_detectors or []),
                detector_schedule=dict(source_cfg.detector_schedule or {}),
                visual_config=default_visual,
                save_dir=self.config.save_dir,
            )
        snapshot = build_runtime_configuration_snapshot(
            engine=engine,
            running=self.is_running(),
            save_dir=self.config.save_dir,
            show_preview=bool(self.config.show_preview),
            default_visual_config=default_visual,
            async_violation_writes=bool(self.config.async_violation_writes),
            writer_queue_max_size=int(self.config.writer_queue_max_size),
            writer_overflow_strategy=str(self.config.writer_overflow_strategy),
            preview_callback_attached=bool(self.config.preview_callback is not None),
            event_callback_attached=bool(self.config.event_callback is not None),
            command_timeout_sec=float(self.config.command_timeout_sec),
            topology=topology,
            sources=sources,
        )
        return snapshot.to_dict()

    def describe_ui_capabilities(self) -> Dict[str, Any]:
        """Функция: describe_ui_capabilities()
Назначение: возвращает UI-discovery контракт с доступными runtime-возможностями и mutable полями.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: словарь runtime-capabilities."""
        runtime = self._get_centralized_runtime()
        if runtime is not None and hasattr(runtime, "describe_ui_capabilities"):
            return runtime.describe_ui_capabilities()

        centralized = self._should_use_centralized_runtime()
        snapshot = build_runtime_capabilities_snapshot(
            engine="centralized" if centralized else "legacy",
            supports_per_source_control=centralized,
            supports_preview_callback=centralized,
            supports_event_callback=centralized,
            supports_hot_topology_updates=centralized,
            supports_detector_schedule_updates=centralized,
            restart_required_fields=self.describe_restart_required_updates() if centralized else {"runtime": [], "source": []},
        )
        return snapshot.to_dict()

    def apply_topology_updates(self, updates: Dict[str, Any]) -> TopologyUpdateResult:
        """Функция: apply_topology_updates()
Назначение: применяет ограниченный hot-reload runtime/topology настроек через centralized runtime.
Параметры функции:
- `updates` (`Dict[str, Any]`): патч runtime/topology настроек.
Возвращаемое значение: TopologyUpdateResult: результат применения topology updates."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            return TopologyUpdateResult(
                success=False,
                rejected={"runtime": "unsupported_runtime"},
                message="Topology hot-reload is supported only in centralized runtime.",
            )
        return runtime.apply_topology_updates(updates)

    def describe_restart_required_updates(self) -> Dict[str, list[str]]:
        """Функция: describe_restart_required_updates()
Назначение: возвращает список параметров, требующих полного restart runtime.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, list[str]]: словарь restart-required параметров."""
        runtime = self._get_centralized_runtime()
        if runtime is None:
            if self._should_use_centralized_runtime():
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
            return {"runtime": [], "source": []}
        return runtime.describe_restart_required_updates()

    def get_stats(self) -> Dict[str, Any]:
        """Функция: get_stats()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._processor is None:
            return {}
        if self._should_use_centralized_runtime():
            if hasattr(self._processor, "get_stats"):
                return self._processor.get_stats()
            runtime_state = self._processor.get_state() if hasattr(self._processor, "get_state") else {}
            return runtime_state
        if not hasattr(self._processor, "stats_manager") or self._processor.stats_manager is None:
            return {}
        return self._processor.stats_manager.get_current_stats()

    def get_state(self) -> Dict[str, Any]:
        """Функция: get_state()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        frame_count = 0
        queue_size = 0
        runtime_state: Dict[str, Any] = {}
        if self._processor is not None:
            if self._should_use_centralized_runtime():
                runtime_state = self._processor.get_state() if hasattr(self._processor, "get_state") else {}
                sources = runtime_state.get("sources", {})
                if isinstance(sources, dict):
                    frame_count = int(
                        sum(int(src.get("captured_frames", 0)) for src in sources.values())
                    )
                    queue_size = int(
                        sum(int(src.get("writer_queue_size", 0)) for src in sources.values())
                    )
            else:
                frame_count = int(getattr(self._processor, "frame_count", 0))
                if (
                    hasattr(self._processor, "violation_manager")
                    and self._processor.violation_manager is not None
                    and hasattr(self._processor.violation_manager, "get_writer_queue_size")
                ):
                    queue_size = int(self._processor.violation_manager.get_writer_queue_size())

        state = {
            "running": self.is_running(),
            "frame_count": frame_count,
            "uptime_sec": max(0.0, time.time() - self._started_at) if self._started_at > 0 else 0.0,
            "writer_queue_size": queue_size,
            "last_error": self._last_error or runtime_state.get("last_error"),
        }
        if self._processor is not None and self._should_use_centralized_runtime():
            state.update(
                {
                    "infer_queue_size": int(runtime_state.get("infer_queue_size", 0)),
                    "postprocess_queue_size": int(runtime_state.get("postprocess_queue_size", 0)),
                    "max_infer_queue_depth": int(runtime_state.get("max_infer_queue_depth", 0)),
                    "max_postprocess_queue_depth": int(runtime_state.get("max_postprocess_queue_depth", 0)),
                    "command_queue_size": int(runtime_state.get("command_queue_size", 0)),
                    "sources": dict(runtime_state.get("sources", {})),
                }
            )
        return state
