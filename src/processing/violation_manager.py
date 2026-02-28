"""Файл: src/processing/violation_manager.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- ..utils.file_manager: используется для передачи данных или вызова связанной логики.
- ..utils.log_context: используется для передачи данных или вызова связанной логики.
- ..utils.visualizer: используется для передачи данных или вызова связанной логики."""

import cv2
import time
import logging
import queue
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..utils.io.file_manager import FileManager
from ..utils.media.visualizer import Visualizer
from ..utils.common.log_context import bind_log_source
from .movement_segment_tracker import MovementSegmentResult, MovementSegmentTracker
from .violation_artifact_writer import ViolationArtifactWriter


@dataclass
class ObstructionViolation:
    """Класс: ObstructionViolation
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `detectors_count` (`int`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `frame` (`Optional[Any]`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `metrics` (`Dict[str, Any]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `reasons` (`List[str]`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `timestamp` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `violation_id` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    violation_id: int
    timestamp: float
    video_timestamp: float
    reasons: List[str]
    detectors_count: int
    metrics: Dict[str, Any]
    frame: Optional[Any] = None


@dataclass
class MovementViolation:
    """Класс: MovementViolation
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `frame` (`Optional[Any]`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `movement_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `timestamp` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `violation_id` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    violation_id: int
    timestamp: float
    video_timestamp: float
    movement_info: Dict[str, Any]
    duration: float
    frame: Optional[Any] = None


@dataclass
class ForbiddenViolation:
    """Класс: ForbiddenViolation
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `affected_classes` (`List[str]`): классы объектов, на которые распространяется текущее нарушение.
- `objects` (`List[Dict[str, Any]]`): список обнаруженных объектов в кадре.
- `timestamp` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `violation_id` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    violation_id: int
    timestamp: float
    video_timestamp: float
    objects: List[Dict[str, Any]]
    affected_classes: List[str]


@dataclass
class DMSViolation:
    """Класс: DMSViolation
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `objects` (`List[Dict[str, Any]]`): список обнаруженных объектов в кадре.
- `stats` (`Dict[str, Any]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `timestamp` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `violation_id` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `violations` (`List[Dict[str, Any]]`): список найденных нарушений за кадр/интервал.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    violation_id: int
    timestamp: float
    video_timestamp: float
    violations: List[Dict[str, Any]]
    objects: List[Dict[str, Any]]
    stats: Dict[str, Any]


@dataclass
class WriteTask:
    """Класс: WriteTask
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `created_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
- `run` (`Callable[[], bool]`): параметр запуска процесса/скрипта в управляющем контуре runtime.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    name: str
    run: Callable[[], bool]
    created_at: float
    event_payload: Optional[Dict[str, Any]] = None


class ViolationManager:
    """Класс: ViolationManager
Назначение: координирует подсистему и управляет ее состоянием во время обработки.
Поля класса:
- `_last_dms_saved_at` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_last_dms_signature` (`Optional[str]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `_last_forbidden_saved_at` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_last_forbidden_signature` (`Optional[str]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `_last_forbidden_source_violation_id` (`Optional[Any]`): параметр источника/выхода данных, задающий направление потока обработки.
- `_movement_change_times` (`deque`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_movement_last_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `_movement_last_saved_at` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_movement_last_turning_at` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_movement_last_video_timestamp` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_movement_prebuffer` (`deque`): данные детектора нарушений, используемые для итогового решения по кадру.
- `_movement_prev_vector` (`Optional[tuple[float, float]]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `_movement_segment_active` (`bool`): данные детектора нарушений, используемые для итогового решения по кадру.
- `_movement_segment_frames` (`List[Dict[str, Any]]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `_movement_segment_started_at` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `_state_lock` (`Any`): примитив синхронизации для безопасного доступа к общему состоянию.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `_start_writer()`, `_writer_loop()`, `_enqueue_write_task()`, `_wait_for_queue_drain()`, `flush_writes()`, `get_writer_queue_size()`, `_shutdown_writer()`, `set_source_fps()`, `_get_movement_writer_fps()`"""
    
    def __init__(
        self,
        save_dir: str,
        visualizer: Visualizer,
        async_writes: bool = True,
        writer_queue_max_size: int = 256,
        writer_overflow_strategy: str = "drop_newest",
        forbidden_duplicate_window_sec: float = 1.0,
        dms_duplicate_window_sec: float = 1.0,
        movement_pre_event_sec: float = 1.0,
        movement_post_event_sec: float = 0.8,
        movement_turn_delta_threshold: float = 0.05,
        movement_change_window_sec: float = 0.8,
        movement_confirm_changes: int = 2,
        movement_segment_cooldown_sec: float = 1.5,
        movement_max_clip_sec: float = 12.0,
        movement_min_clip_frames: int = 6,
        log_source_id: Optional[str] = None,
        event_callback: Optional[Callable[..., None]] = None,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `save_dir` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `visualizer` (`Visualizer`): объект, рисующий оверлеи и диагностические метки на кадре.
- `async_writes` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `writer_queue_max_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `writer_overflow_strategy` (`str`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `forbidden_duplicate_window_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `dms_duplicate_window_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `movement_pre_event_sec` (`float`): событие синхронизации или тип события для переключения ветки обработки.
- `movement_post_event_sec` (`float`): событие синхронизации или тип события для переключения ветки обработки.
- `movement_turn_delta_threshold` (`float`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `movement_change_window_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `movement_confirm_changes` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `movement_segment_cooldown_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `movement_max_clip_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `movement_min_clip_frames` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `log_source_id` (`Optional[str]`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.save_dir = save_dir
        self.file_manager = FileManager(save_dir)
        self.artifact_writer = ViolationArtifactWriter(self.file_manager)
        self.visualizer = visualizer
        self.log_source_id = None if log_source_id is None else str(log_source_id)
        self.event_callback = event_callback
        self.async_writes = bool(async_writes)
        self.writer_queue_max_size = max(1, int(writer_queue_max_size))
        self.writer_overflow_strategy = str(writer_overflow_strategy)
        
        # Счетчики нарушений
        self.obstruction_count = 0
        self.movement_count = 0
        self.forbidden_count = 0
        self.dms_count = 0
        self._state_lock = threading.Lock()

        # Анти-дубликаты событий (scheduler cache может возвращать тот же сигнал на соседних кадрах)
        self.forbidden_duplicate_window_sec = max(0.0, float(forbidden_duplicate_window_sec))
        self.dms_duplicate_window_sec = max(0.0, float(dms_duplicate_window_sec))
        self._last_forbidden_saved_at: Optional[float] = None
        self._last_forbidden_source_violation_id: Optional[Any] = None
        self._last_forbidden_signature: Optional[str] = None
        self._last_dms_saved_at: Optional[float] = None
        self._last_dms_signature: Optional[str] = None
        
        # Состояния для отслеживания
        self.is_obstruction_active = False
        self.obstruction_start_time: Optional[float] = None
        self.current_segment_start: Optional[float] = None
        self.obstruction_segments: List[Dict[str, Any]] = []
        
        # Кулдауны
        self.last_obstruction_violation_time: Optional[float] = None
        self.obstruction_cooldown = 5.0
        self.min_obstruction_duration = 5.0
        self.capture_frame_at = 3.0
        
        # Для сохранения
        self.frame_to_save: Optional[Any] = None
        self.saved_for_current_violation = False

        # Параметры сегментного сохранения движения
        self.movement_pre_event_sec = max(0.0, float(movement_pre_event_sec))
        self.movement_post_event_sec = max(0.0, float(movement_post_event_sec))
        self.movement_turn_delta_threshold = max(0.0, float(movement_turn_delta_threshold))
        self.movement_change_window_sec = max(0.1, float(movement_change_window_sec))
        self.movement_confirm_changes = max(1, int(movement_confirm_changes))
        self.movement_segment_cooldown_sec = max(0.0, float(movement_segment_cooldown_sec))
        self.movement_max_clip_sec = max(1.0, float(movement_max_clip_sec))
        self.movement_min_clip_frames = max(1, int(movement_min_clip_frames))
        self.movement_tracker = MovementSegmentTracker(
            movement_pre_event_sec=self.movement_pre_event_sec,
            movement_post_event_sec=self.movement_post_event_sec,
            movement_turn_delta_threshold=self.movement_turn_delta_threshold,
            movement_change_window_sec=self.movement_change_window_sec,
            movement_confirm_changes=self.movement_confirm_changes,
            movement_segment_cooldown_sec=self.movement_segment_cooldown_sec,
            movement_max_clip_sec=self.movement_max_clip_sec,
            movement_min_clip_frames=self.movement_min_clip_frames,
        )

        # Асинхронная запись
        self._write_queue: Optional[queue.Queue] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_stop_event: Optional[threading.Event] = None
        self._writer_started = False
        if self.async_writes:
            self._start_writer()

    def _start_writer(self) -> None:
        """Функция: _start_writer()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self._write_queue = queue.Queue(maxsize=self.writer_queue_max_size)
        self._writer_stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="ViolationWriter",
            daemon=True,
        )
        self._writer_thread.start()
        self._writer_started = True
        logging.info(
            f"[writer:start] queue_max={self.writer_queue_max_size} "
            f"overflow={self.writer_overflow_strategy}"
        )

    def _writer_loop(self) -> None:
        """Функция: _writer_loop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        with bind_log_source(self.log_source_id):
            assert self._write_queue is not None
            assert self._writer_stop_event is not None
            while not self._writer_stop_event.is_set() or not self._write_queue.empty():
                try:
                    task: WriteTask = self._write_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    ok = bool(task.run())
                    if not ok:
                        logging.warning(f"[writer:task_failed] task={task.name}")
                    elif task.event_payload:
                        self._emit_saved_event(task.event_payload)
                except Exception as e:
                    logging.error(f"[writer:task_crashed] task={task.name} error={e}")
                finally:
                    self._write_queue.task_done()

    def _emit_saved_event(self, event_payload: Optional[Dict[str, Any]]) -> None:
        """Функция: _emit_saved_event()
Назначение: безопасно эмитит событие успешной записи нарушения.
Параметры функции:
- `event_payload` (`Optional[Dict[str, Any]]`): полезная нагрузка события записи.
Возвращаемое значение: None: событие передается во внешний callback при наличии."""
        if self.event_callback is None or not event_payload:
            return
        try:
            self.event_callback(
                event_type="violation_saved",
                source_id=self.log_source_id,
                severity="info",
                data=dict(event_payload),
            )
        except Exception as exc:
            logging.warning(f"[writer:event_callback_failed] error={exc}")

    def _enqueue_write_task(
        self,
        task_name: str,
        write_fn: Callable[[], bool],
        event_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Функция: _enqueue_write_task()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `task_name` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `write_fn` (`Callable[[], bool]`): функция записи данных/кадра в целевой буфер или файл.
- `event_payload` (`Optional[Dict[str, Any]]`): полезная нагрузка события успешной записи.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.async_writes:
            try:
                ok = bool(write_fn())
                if ok:
                    self._emit_saved_event(event_payload)
                return ok
            except Exception as e:
                logging.error(f"[writer:sync_task_crashed] task={task_name} error={e}")
                return False

        if self._write_queue is None:
            return False

        task = WriteTask(
            name=task_name,
            run=write_fn,
            created_at=time.time(),
            event_payload=dict(event_payload) if event_payload else None,
        )
        try:
            self._write_queue.put_nowait(task)
            return True
        except queue.Full:
            if self.writer_overflow_strategy == "drop_oldest":
                try:
                    dropped = self._write_queue.get_nowait()
                    self._write_queue.task_done()
                    logging.warning(f"[writer:drop_oldest] dropped_task={dropped.name}")
                    self._write_queue.put_nowait(task)
                    return True
                except queue.Empty:
                    return False
                except queue.Full:
                    return False

            logging.warning(f"[writer:drop_new] task={task_name} reason=queue_full")
            return False

    def _wait_for_queue_drain(self, timeout: Optional[float] = None) -> bool:
        """Функция: _wait_for_queue_drain()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`Optional[float]`): максимальное время ожидания завершения операции.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.async_writes or self._write_queue is None:
            return True

        deadline = None if timeout is None else (time.monotonic() + max(0.0, float(timeout)))
        while True:
            if self._write_queue.unfinished_tasks == 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def flush_writes(self, timeout: Optional[float] = None) -> bool:
        """Функция: flush_writes()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`Optional[float]`): максимальное время ожидания завершения операции.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return self._wait_for_queue_drain(timeout=timeout)

    def get_writer_queue_size(self) -> int:
        """Функция: get_writer_queue_size()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._write_queue is None:
            return 0
        return int(self._write_queue.qsize())

    def _shutdown_writer(self, timeout: float = 5.0) -> None:
        """Функция: _shutdown_writer()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `timeout` (`float`): максимальное время ожидания завершения операции.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.async_writes or not self._writer_started:
            return
        drained = self._wait_for_queue_drain(timeout=timeout)
        if not drained:
            logging.warning("[writer:drain_timeout] shutdown continues")
        if self._writer_stop_event is not None:
            self._writer_stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)
            if self._writer_thread.is_alive():
                logging.warning("[writer:join_timeout] worker still alive")
        self._writer_started = False

    def set_source_fps(self, source_fps: Optional[float]) -> None:
        """Функция: set_source_fps()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `source_fps` (`Optional[float]`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.movement_tracker.set_source_fps(source_fps)

    def _get_movement_writer_fps(self) -> float:
        """Функция: _get_movement_writer_fps()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: float: результат шага обработки, который используется следующим этапом пайплайна."""
        return self.movement_tracker.get_writer_fps()

    def _build_forbidden_signature(self, violation_info: Dict[str, Any]) -> str:
        """Функция: _build_forbidden_signature()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_info` (`Dict[str, Any]`): детали зафиксированного нарушения (тип, время, метаданные).
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        classes = sorted([str(c) for c in violation_info.get("affected_classes", [])])
        objects = []
        for obj in violation_info.get("objects", []):
            objects.append(
                (
                    str(obj.get("class", "")),
                    str(obj.get("object_id", "")),
                    tuple(obj.get("bbox", [])[:4]) if isinstance(obj.get("bbox"), (list, tuple)) else (),
                )
            )
        objects = sorted(objects)
        return f"classes={classes}|objects={objects}"

    def _is_forbidden_duplicate(self, violation_info: Dict[str, Any], now_ts: float) -> bool:
        """Функция: _is_forbidden_duplicate()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_info` (`Dict[str, Any]`): детали зафиксированного нарушения (тип, время, метаданные).
- `now_ts` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        source_violation_id = violation_info.get("violation_id")
        if source_violation_id is not None:
            return source_violation_id == self._last_forbidden_source_violation_id

        signature = self._build_forbidden_signature(violation_info)
        if (
            self._last_forbidden_signature is not None
            and signature == self._last_forbidden_signature
            and self._last_forbidden_saved_at is not None
            and (now_ts - self._last_forbidden_saved_at) < self.forbidden_duplicate_window_sec
        ):
            return True
        return False

    def _mark_forbidden_saved(self, violation_info: Dict[str, Any], now_ts: float) -> None:
        """Функция: _mark_forbidden_saved()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_info` (`Dict[str, Any]`): детали зафиксированного нарушения (тип, время, метаданные).
- `now_ts` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self._last_forbidden_saved_at = float(now_ts)
        self._last_forbidden_source_violation_id = violation_info.get("violation_id")
        self._last_forbidden_signature = self._build_forbidden_signature(violation_info)

    def _build_dms_signature(self, violations: List[Dict[str, Any]]) -> str:
        """Функция: _build_dms_signature()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violations` (`List[Dict[str, Any]]`): список найденных нарушений за кадр/интервал.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        tokens = []
        for item in violations:
            try:
                start_time = float(item.get("start_time", 0.0) or 0.0)
            except (TypeError, ValueError):
                start_time = 0.0
            tokens.append(
                (
                    str(item.get("type", "")),
                    str(item.get("class", "")),
                    str(item.get("severity", "")),
                    round(start_time, 2),
                )
            )
        tokens.sort()
        return repr(tokens)

    def _is_dms_duplicate(self, violations: List[Dict[str, Any]], now_ts: float) -> bool:
        """Функция: _is_dms_duplicate()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violations` (`List[Dict[str, Any]]`): список найденных нарушений за кадр/интервал.
- `now_ts` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        signature = self._build_dms_signature(violations)
        if (
            self._last_dms_signature is None
            or self._last_dms_saved_at is None
            or signature != self._last_dms_signature
        ):
            return False
        return (now_ts - self._last_dms_saved_at) < self.dms_duplicate_window_sec

    def _mark_dms_saved(self, violations: List[Dict[str, Any]], now_ts: float) -> None:
        """Функция: _mark_dms_saved()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violations` (`List[Dict[str, Any]]`): список найденных нарушений за кадр/интервал.
- `now_ts` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self._last_dms_saved_at = float(now_ts)
        self._last_dms_signature = self._build_dms_signature(violations)
        
    def process_obstruction(
        self,
        result: Dict[str, Any],
        frame: Any,
        video_timestamp: float,
        processing_time: float
    ) -> Optional[ObstructionViolation]:
        """Функция: process_obstruction()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `processing_time` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: Optional[ObstructionViolation]: результат шага обработки, который используется следующим этапом пайплайна."""
        current_time = time.time()
        
        if result.get("detected", False):
            return self._handle_obstruction_detected(
                result, frame, video_timestamp, processing_time, current_time
            )
        else:
            self._handle_obstruction_cleared(current_time)
            return None
    
    def _handle_obstruction_detected(
        self,
        result: Dict[str, Any],
        frame: Any,
        video_timestamp: float,
        processing_time: float,
        current_time: float
    ) -> Optional[ObstructionViolation]:
        """Функция: _handle_obstruction_detected()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `processing_time` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: Optional[ObstructionViolation]: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.is_obstruction_active:
            self.is_obstruction_active = True
            self.obstruction_start_time = current_time
            self.current_segment_start = video_timestamp
            logging.info("[obstruction:start] tracking_started=true")
        
        duration = current_time - self.obstruction_start_time
        
        # Захват кадра для сохранения
        if duration >= self.capture_frame_at and not self.saved_for_current_violation:
            self.frame_to_save = frame.copy()
            self.saved_for_current_violation = True
        
        # Фиксация нарушения
        if duration >= self.min_obstruction_duration:
            if self._check_cooldown(self.last_obstruction_violation_time, current_time):
                violation = self._save_obstruction_violation(
                    frame, result, video_timestamp, processing_time
                )
                if violation:
                    self.last_obstruction_violation_time = current_time
                    return violation
        
        return None
    
    def _handle_obstruction_cleared(self, current_time: float) -> None:
        """Функция: _handle_obstruction_cleared()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.is_obstruction_active:
            duration = current_time - self.obstruction_start_time
            if duration >= self.min_obstruction_duration:
                self.obstruction_segments.append({
                    "start": self.current_segment_start,
                    "duration": duration
                })
            self.is_obstruction_active = False
            self.saved_for_current_violation = False
            self.frame_to_save = None
            logging.info("[obstruction:end] tracking_started=false")
    
    def _save_obstruction_violation(
        self,
        frame: Any,
        result: Dict[str, Any],
        video_timestamp: float,
        processing_time: float
    ) -> Optional[ObstructionViolation]:
        """Функция: _save_obstruction_violation()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `processing_time` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: Optional[ObstructionViolation]: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            next_id = self.obstruction_count + 1
            
            # Создание аннотированного кадра
            annotated_frame = self._annotate_obstruction_frame(frame, result)
            
            # Сохранение изображения
            img_filename = self.file_manager.generate_filename("camera_obstruction", "jpg")
            img_path = self.file_manager.get_full_path(img_filename, event_type="obstruction")
            
            # Сохранение отчета
            violation_info = {
                "violation_type": "obstruction",
                "violation_id": next_id,
                "timestamp": processing_time,
                "video_timestamp": video_timestamp,
                "media_file": img_filename,
                "summary": "Camera obstruction detected",
                "details": {
                    "reasons": result.get("reasons", []),
                    "detectors_count": result.get("detectors_count", 0),
                    "metrics": result.get("metrics", {})
                }
            }
            write_ok = self._enqueue_write_task(
                task_name="obstruction",
                write_fn=lambda: self._write_obstruction_artifacts(
                    annotated_frame=annotated_frame,
                    img_path=img_path,
                    violation_info=violation_info,
                    img_filename=img_filename,
                ),
                event_payload={
                    "violation_kind": "obstruction",
                    "violation_id": next_id,
                    "media_file": img_filename,
                },
            )
            if not write_ok:
                logging.warning("[obstruction:write_dropped] reason=writer_queue_overflow")
                return None
            self.obstruction_count = next_id
            
            return ObstructionViolation(
                violation_id=next_id,
                timestamp=processing_time,
                video_timestamp=video_timestamp,
                reasons=result.get("reasons", []),
                detectors_count=result.get("detectors_count", 0),
                metrics=result.get("metrics", {}),
                frame=annotated_frame
            )
            
        except Exception as e:
            logging.error(f"[obstruction:save_failed] error={e}")
            import traceback
            traceback.print_exc()
            return None

    def _write_obstruction_artifacts(
        self,
        annotated_frame: Any,
        img_path: str,
        violation_info: Dict[str, Any],
        img_filename: str,
    ) -> bool:
        """Функция: _write_obstruction_artifacts()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `annotated_frame` (`Any`): кадр с нанесенной визуализацией детекций и служебных меток.
- `img_path` (`str`): кадр/изображение, которое передается на обработку текущему этапу.
- `violation_info` (`Dict[str, Any]`): детали зафиксированного нарушения (тип, время, метаданные).
- `img_filename` (`str`): кадр/изображение, которое передается на обработку текущему этапу.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return self.artifact_writer.write_obstruction_artifacts(
            annotated_frame=annotated_frame,
            img_path=img_path,
            violation_info=violation_info,
            img_filename=img_filename,
        )
    
    def _annotate_obstruction_frame(self, frame: Any, result: Dict[str, Any]) -> Any:
        """Функция: _annotate_obstruction_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        annotated = frame.copy()
        
        # Текст нарушения
        cv2.putText(
            annotated, "CAMERA OBSTRUCTION!", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.visualizer.colors["red"], 3
        )
        
        # Reasons
        y_offset = 90
        for i, reason in enumerate(result.get("reasons", [])[:3]):
            cv2.putText(
                annotated, f"{i+1}. {reason}", (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.visualizer.colors["yellow"], 2
            )
            y_offset += 40
        
        # Bounding boxes от YOLO
        for obj in result.get("yolo_objects", []):
            self.visualizer._draw_yolo_object(annotated, obj)
        
        return annotated
    
    def process_movement(
        self,
        movement_info: Dict[str, Any],
        frame: Any,
        video_timestamp: float
    ) -> Optional[MovementViolation]:
        """Функция: process_movement()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `movement_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
Возвращаемое значение: Optional[MovementViolation]: результат шага обработки, который используется следующим этапом пайплайна."""
        now_ts = time.time()
        segment = self.movement_tracker.process_frame(
            frame=frame,
            movement_info=movement_info,
            video_timestamp=video_timestamp,
            captured_at=now_ts,
        )
        if segment is None:
            return None
        return self._save_movement_segment(segment=segment, now_ts=now_ts)

    def _write_movement_artifacts(
        self,
        frames_data: List[Dict[str, Any]],
        video_path: str,
        video_filename: str,
        movement_info: Dict[str, Any],
        unified_info: Dict[str, Any],
    ) -> bool:
        """Функция: _write_movement_artifacts()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frames_data` (`List[Dict[str, Any]]`): кадр/изображение, которое передается на обработку текущему этапу.
- `video_path` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `video_filename` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `movement_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `unified_info` (`Dict[str, Any]`): объединенная структура метаданных о нарушении перед сохранением.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return self.artifact_writer.write_movement_artifacts(
            frames_data=frames_data,
            video_path=video_path,
            video_filename=video_filename,
            movement_info=movement_info,
            unified_info=unified_info,
            writer_fps=self._get_movement_writer_fps(),
        )

    def _save_movement_segment(
        self,
        segment: MovementSegmentResult,
        now_ts: float,
    ) -> Optional[MovementViolation]:
        """Функция: _save_movement_segment()
Назначение: ставит готовый movement segment на запись и формирует доменный результат.
Параметры функции:
- `segment` (`MovementSegmentResult`): готовый сегмент движения из tracker-а.
- `now_ts` (`float`): текущее время фиксации нарушения.
Возвращаемое значение: Optional[MovementViolation]: сохраненное нарушение или `None` при ошибке."""
        next_id = self.movement_count + 1
        try:
            video_filename = self.file_manager.generate_filename("camera_movement", "mp4")
            video_path = self.file_manager.get_full_path(video_filename, event_type="movement")
            unified_info = {
                "violation_type": "movement",
                "violation_id": next_id,
                "timestamp": now_ts,
                "video_timestamp": segment.clip_video_timestamp,
                "media_file": video_filename,
                "summary": "Camera movement segment detected",
                "details": {
                    "duration": segment.clip_duration,
                    "movement_info": segment.movement_info,
                },
            }
            write_ok = self._enqueue_write_task(
                task_name="movement",
                write_fn=lambda: self._write_movement_artifacts(
                    frames_data=segment.frames_data,
                    video_path=video_path,
                    video_filename=video_filename,
                    movement_info=segment.movement_info,
                    unified_info=unified_info,
                ),
                event_payload={
                    "violation_kind": "movement",
                    "violation_id": next_id,
                    "media_file": video_filename,
                },
            )
            if not write_ok:
                logging.warning("[movement:write_dropped] reason=writer_queue_overflow")
                return None

            self.movement_count = next_id
            self.movement_tracker.mark_saved(now_ts)
            logging.info(
                f"[movement:segment_queued] frames={len(segment.frames_data)} duration={segment.clip_duration:.2f}s"
            )
            return MovementViolation(
                violation_id=next_id,
                timestamp=now_ts,
                video_timestamp=segment.clip_video_timestamp,
                movement_info=segment.movement_info,
                duration=segment.clip_duration,
                frame=segment.frames_data[-1]["frame"],
            )
        except Exception as e:
            logging.error(f"[movement:save_failed] error={e}")
            return None
    
    def process_forbidden(
        self,
        forbidden_result: Dict[str, Any],
        frame: Any,
        video_timestamp: float
    ) -> Optional[ForbiddenViolation]:
        """Функция: process_forbidden()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `forbidden_result` (`Dict[str, Any]`): структурированный результат детекции запрещенных предметов.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
Возвращаемое значение: Optional[ForbiddenViolation]: результат шага обработки, который используется следующим этапом пайплайна."""
        if not forbidden_result.get("current_violation", False):
            return None
        
        violation_info = forbidden_result.get("violation_info")
        if not violation_info:
            return None
        
        try:
            # Аннотирование
            annotated_frame = self._annotate_forbidden_frame(
                frame, violation_info.get("objects", [])
            )
            
            # Сохранение
            img_filename = self.file_manager.generate_filename("forbidden_items", "jpg")
            img_path = self.file_manager.get_full_path(img_filename, event_type="forbidden_items")
            
            # Отчет
            report_filename = img_filename.replace(".jpg", ".txt")
            with self._state_lock:
                now_ts = time.time()
                if self._is_forbidden_duplicate(violation_info, now_ts):
                    logging.debug("[forbidden:duplicate_suppressed]")
                    return None

                next_id = self.forbidden_count + 1
                report_info = {
                    "violation_type": "forbidden",
                    "violation_id": next_id,
                    "timestamp": now_ts,
                    "video_timestamp": float(video_timestamp),
                    "media_file": img_filename,
                    "summary": "Forbidden items detected",
                    "details": {
                        "objects": violation_info.get("objects", []),
                        "affected_classes": violation_info.get("affected_classes", []),
                        "source_violation_id": violation_info.get("violation_id"),
                        "cooldown_remaining": violation_info.get("cooldown_remaining", 0.0),
                        "is_class_specific": violation_info.get("is_class_specific", False),
                    },
                }
                write_ok = self._enqueue_write_task(
                    task_name="forbidden",
                    write_fn=lambda: self._write_forbidden_artifacts(
                        annotated_frame=annotated_frame,
                        img_path=img_path,
                        img_filename=img_filename,
                        report_info=report_info,
                        report_filename=report_filename,
                    ),
                    event_payload={
                        "violation_kind": "forbidden",
                        "violation_id": next_id,
                        "media_file": img_filename,
                    },
                )
                if not write_ok:
                    logging.warning("[forbidden:write_dropped] reason=writer_queue_overflow")
                    return None
                self.forbidden_count = next_id
                self._mark_forbidden_saved(violation_info, now_ts)

            return ForbiddenViolation(
                violation_id=next_id,
                timestamp=now_ts,
                video_timestamp=video_timestamp,
                objects=violation_info.get("objects", []),
                affected_classes=violation_info.get("affected_classes", [])
            )
            
        except Exception as e:
            logging.error(f"[forbidden:save_failed] error={e}")
            return None

    def _write_forbidden_artifacts(
        self,
        annotated_frame: Any,
        img_path: str,
        img_filename: str,
        report_info: Dict[str, Any],
        report_filename: str,
    ) -> bool:
        """Функция: _write_forbidden_artifacts()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `annotated_frame` (`Any`): кадр с нанесенной визуализацией детекций и служебных меток.
- `img_path` (`str`): кадр/изображение, которое передается на обработку текущему этапу.
- `img_filename` (`str`): кадр/изображение, которое передается на обработку текущему этапу.
- `report_info` (`Dict[str, Any]`): метаданные сформированного отчета (пути, имена, счетчики).
- `report_filename` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return self.artifact_writer.write_forbidden_artifacts(
            annotated_frame=annotated_frame,
            img_path=img_path,
            img_filename=img_filename,
            report_info=report_info,
            report_filename=report_filename,
        )
    
    def _annotate_forbidden_frame(self, frame: Any, objects: List[Dict]) -> Any:
        """Функция: _annotate_forbidden_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `objects` (`List[Dict]`): список обнаруженных объектов в кадре.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        annotated = frame.copy()
        
        cv2.putText(
            annotated, "FORBIDDEN ITEM VIOLATION!", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.visualizer.colors["purple"], 3
        )
        
        y_offset = 90
        for i, obj in enumerate(objects[:3]):
            class_name = obj.get("class", "Unknown")
            duration = obj.get("duration", 0)
            cv2.putText(
                annotated, f"{i+1}. {class_name} ({duration:.1f}s)", (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.visualizer.colors["yellow"], 2
            )
            y_offset += 40
        
        for obj in objects:
            self.visualizer._draw_forbidden_object(annotated, obj)
        
        return annotated
    
    def process_dms(
        self,
        dms_result: Dict[str, Any],
        frame: Any,
        video_timestamp: float,
        processing_time: float,
        frame_count: int
    ) -> Optional[DMSViolation]:
        """Функция: process_dms()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `dms_result` (`Dict[str, Any]`): структурированный результат DMS-детекции на текущем кадре.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `processing_time` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Optional[DMSViolation]: результат шага обработки, который используется следующим этапом пайплайна."""
        violations = dms_result.get("violations", [])
        if not violations:
            return None
        
        try:
            # Аннотирование
            annotated_frame = self._annotate_dms_frame(frame, violations, dms_result)
            
            # Сохранение
            img_filename = self.file_manager.generate_filename("dms_violation", "jpg")
            img_path = self.file_manager.get_full_path(img_filename, event_type="dms")
            
            report_filename = img_filename.replace(".jpg", ".txt")
            with self._state_lock:
                if self._is_dms_duplicate(violations, processing_time):
                    logging.debug("[dms:duplicate_suppressed]")
                    return None

                next_id = self.dms_count + 1
                # Отчет
                violation_info = {
                    "violation_type": "dms",
                    "violation_id": next_id,
                    "timestamp": processing_time,
                    "video_timestamp": float(video_timestamp),
                    "frame_number": frame_count,
                    "media_file": img_filename,
                    "summary": "DMS violation detected",
                    "details": {
                        "violations": violations,
                        "dms_stats": dms_result.get("stats", {}),
                        "objects": dms_result.get("objects", []),
                    },
                }

                write_ok = self._enqueue_write_task(
                    task_name="dms",
                    write_fn=lambda: self._write_dms_artifacts(
                        annotated_frame=annotated_frame,
                        img_path=img_path,
                        img_filename=img_filename,
                        violation_info=violation_info,
                        report_filename=report_filename,
                    ),
                    event_payload={
                        "violation_kind": "dms",
                        "violation_id": next_id,
                        "media_file": img_filename,
                    },
                )
                if not write_ok:
                    logging.warning("[dms:write_dropped] reason=writer_queue_overflow")
                    return None
                self.dms_count = next_id
                self._mark_dms_saved(violations, processing_time)
            
            return DMSViolation(
                violation_id=next_id,
                timestamp=processing_time,
                video_timestamp=video_timestamp,
                violations=violations,
                objects=dms_result.get("objects", []),
                stats=dms_result.get("stats", {})
            )
            
        except Exception as e:
            logging.error(f"[dms:save_failed] error={e}")
            import traceback
            traceback.print_exc()
            return None

    def _write_dms_artifacts(
        self,
        annotated_frame: Any,
        img_path: str,
        img_filename: str,
        violation_info: Dict[str, Any],
        report_filename: str,
    ) -> bool:
        """Функция: _write_dms_artifacts()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `annotated_frame` (`Any`): кадр с нанесенной визуализацией детекций и служебных меток.
- `img_path` (`str`): кадр/изображение, которое передается на обработку текущему этапу.
- `img_filename` (`str`): кадр/изображение, которое передается на обработку текущему этапу.
- `violation_info` (`Dict[str, Any]`): детали зафиксированного нарушения (тип, время, метаданные).
- `report_filename` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return self.artifact_writer.write_dms_artifacts(
            annotated_frame=annotated_frame,
            img_path=img_path,
            img_filename=img_filename,
            violation_info=violation_info,
            report_filename=report_filename,
        )
    
    def _annotate_dms_frame(self, frame: Any, violations: List[Dict], dms_result: Dict) -> Any:
        """Функция: _annotate_dms_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `violations` (`List[Dict]`): список найденных нарушений за кадр/интервал.
- `dms_result` (`Dict`): структурированный результат DMS-детекции на текущем кадре.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        annotated = frame.copy()
        
        cv2.putText(
            annotated, "DMS VIOLATION!", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.visualizer.colors["red"], 3
        )
        
        y_offset = 90
        for i, violation in enumerate(violations[:3]):
            message = violation.get("message", "Unknown")
            cv2.putText(
                annotated, f"{i+1}. {message}", (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.visualizer.colors["yellow"], 2
            )
            y_offset += 40
        
        for obj in dms_result.get("objects", []):
            self.visualizer._draw_dms_object(annotated, obj)
        
        return annotated
    
    def _check_cooldown(
        self,
        last_violation_time: Optional[float],
        current_time: float,
        cooldown: Optional[float] = None
    ) -> bool:
        """Функция: _check_cooldown()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `last_violation_time` (`Optional[float]`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `current_time` (`float`): текущее время для расчета длительностей, кулдаунов и интервалов.
- `cooldown` (`Optional[float]`): интервал блокировки повторного события после срабатывания.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if last_violation_time is None:
            return True
        
        cooldown = cooldown or self.obstruction_cooldown
        return (current_time - last_violation_time) >= cooldown
    
    def reset(self) -> None:
        """Функция: reset()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.is_obstruction_active = False
        self.obstruction_start_time = None
        self.current_segment_start = None
        self.obstruction_segments = []
        self.last_obstruction_violation_time = None
        self.frame_to_save = None
        self.saved_for_current_violation = False
        self.movement_tracker.reset()
        self._last_forbidden_saved_at = None
        self._last_forbidden_source_violation_id = None
        self._last_forbidden_signature = None
        self._last_dms_saved_at = None
        self._last_dms_signature = None
    
    def cleanup(self) -> None:
        """Функция: cleanup()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        segment = self.movement_tracker.force_finalize(now_ts=time.time())
        if segment is not None:
            try:
                self._save_movement_segment(segment=segment, now_ts=time.time())
            except Exception:
                pass
        self._shutdown_writer(timeout=5.0)
        self.reset()
