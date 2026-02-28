"""Файл: src/processing/capture_worker.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Импортируемые внутренние модули:
- ..utils.log_context: используется для передачи данных или вызова связанной логики.
- .runtime_models: используется для передачи данных или вызова связанной логики.
- .source_manager: используется для передачи данных или вызова связанной логики."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .source_manager import SourceManager
from ..runtime.topology import CapturedFrame, SourceConfig
from ..utils.common.log_context import bind_log_source


@dataclass
class CaptureWorkerStats:
    """Класс: CaptureWorkerStats
Назначение: выполняет фоновую работу в отдельном потоке/исполнителе.
Поля класса:
- `dropped_frames` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `finished_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `queued_frames` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `read_frames` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `started_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    read_frames: int = 0
    queued_frames: int = 0
    dropped_frames: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0


class CaptureWorker:
    """Класс: CaptureWorker
Назначение: выполняет фоновую работу в отдельном потоке/исполнителе.
Поля класса:
- `_is_running` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `_lock` (`Any`): синхронизатор доступа к общему состоянию между потоками.
- `_thread` (`Optional[threading.Thread]`): рабочий поток, выполняющий часть пайплайна.
- `last_error` (`Optional[str]`): сообщение или объект ошибки, используемый для диагностики и логирования.
- `output_queue` (`"queue.Queue[CapturedFrame]"`): очередь для передачи данных между асинхронными этапами пайплайна.
- `realtime_file_playback` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `reconnect_backoff_sec` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `reconnect_on_loss` (`Optional[bool]`): логический флаг, включающий/отключающий соответствующее поведение.
- `source_config` (`SourceConfig`): параметры конкретного видеоисточника.
- `source_manager` (`Any`): объект захвата кадров из камеры/файла/RTSP.
- `stats` (`Any`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `stop_event` (`Any`): событие остановки для корректного завершения потоков.
Ключевые методы:
- `__init__()`, `start()`, `stop()`, `join()`, `is_running()`, `get_stats()`, `_enqueue_frame()`, `_run()`, `_is_reconnectable_source()`, `_sleep_reconnect_backoff()`"""

    def __init__(
        self,
        source_config: SourceConfig,
        output_queue: "queue.Queue[CapturedFrame]",
        source_manager: Optional[SourceManager] = None,
        stop_event: Optional[threading.Event] = None,
        reconnect_backoff_sec: float = 2.0,
        reconnect_on_loss: Optional[bool] = False,
        realtime_file_playback: bool = True,
        event_callback: Optional[Callable[..., None]] = None,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `source_config` (`SourceConfig`): параметры конкретного видеоисточника.
- `output_queue` (`"queue.Queue[CapturedFrame]"`): очередь для передачи данных между асинхронными этапами пайплайна.
- `source_manager` (`Optional[SourceManager]`): объект захвата кадров из камеры/файла/RTSP.
- `stop_event` (`Optional[threading.Event]`): событие остановки для корректного завершения потоков.
- `reconnect_backoff_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `reconnect_on_loss` (`Optional[bool]`): логический флаг, включающий/отключающий соответствующее поведение.
- `realtime_file_playback` (`bool`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.source_config = source_config
        self.output_queue = output_queue
        self.source_manager = source_manager or SourceManager()
        self.stop_event = threading.Event()
        self._shared_stop_event = stop_event or threading.Event()
        self.reconnect_backoff_sec = max(0.0, float(reconnect_backoff_sec))
        self.reconnect_on_loss = reconnect_on_loss
        self.realtime_file_playback = bool(realtime_file_playback)
        self.event_callback = event_callback

        self.stats = CaptureWorkerStats()
        self.last_error: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_running = False

    def start(self) -> bool:
        """Функция: start()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self.last_error = None
            self.stats = CaptureWorkerStats(started_at=time.time())
            self.stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"CaptureWorker-{self.source_config.source_id}",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        """Функция: stop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        """Функция: join()
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
        thread = self._thread
        return bool(thread and thread.is_alive()) or self._is_running

    def get_stats(self) -> CaptureWorkerStats:
        """Функция: get_stats()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: CaptureWorkerStats: результат шага обработки, который используется следующим этапом пайплайна."""
        return self.stats

    def _enqueue_frame(self, packet: CapturedFrame) -> tuple[bool, int]:
        """Функция: _enqueue_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `packet` (`CapturedFrame`): пакет данных в очереди межпоточного обмена (кадр, метаданные, результат).
Возвращаемое значение: tuple[bool, int]: результат шага обработки, который используется следующим этапом пайплайна."""
        try:
            self.output_queue.put_nowait(packet)
            return True, 0
        except queue.Full:
            if self.source_config.drop_policy == "drop_oldest":
                try:
                    self.output_queue.get_nowait()
                    self.output_queue.put_nowait(packet)
                    self._emit_event(
                        event_type="source_queue_drop",
                        severity="warning",
                        data={
                            "drop_policy": self.source_config.drop_policy,
                            "dropped_frames": 1,
                        },
                    )
                    return True, 1
                except queue.Empty:
                    return False, 0
                except queue.Full:
                    return False, 1
            self._emit_event(
                event_type="source_queue_drop",
                severity="warning",
                data={
                    "drop_policy": self.source_config.drop_policy,
                    "dropped_frames": 1,
                },
            )
            return False, 1

    def _run(self) -> None:
        """Функция: _run()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        with bind_log_source(self.source_config.source_id):
            self._is_running = True
            frame_index = 0
            start_time = time.time()
            cfg = self.source_config
            reconnectable = self._is_reconnectable_source()
            try:
                while not self._should_stop():
                    opened = self.source_manager.open_source(
                        input_source=cfg.input_source,
                        camera_id=cfg.camera_id,
                    )
                    if not opened:
                        self.last_error = f"Failed to open source: {cfg.source_id}"
                        if reconnectable and not self._should_stop():
                            self._emit_event(
                                event_type="source_reconnect_open",
                                severity="warning",
                                data={"backoff_sec": float(self.reconnect_backoff_sec)},
                            )
                            logging.warning(
                                f"[capture:reconnect_open] source={cfg.source_id} backoff_sec={self.reconnect_backoff_sec:.1f}"
                            )
                            self._sleep_reconnect_backoff()
                            continue
                        return

                    self.last_error = None
                    source_started_at = time.monotonic()
                    while not self._should_stop():
                        if self.realtime_file_playback and bool(getattr(self.source_manager, "is_file", False)):
                            self._sync_file_playback(
                                frame_index=frame_index,
                                source_started_at=source_started_at,
                                source_fps=float(getattr(self.source_manager, "fps", 0.0) or 0.0),
                            )

                        ret, frame = self.source_manager.read_frame()
                        if not ret:
                            if reconnectable and not self._should_stop():
                                self._emit_event(
                                    event_type="source_reconnect_read",
                                    severity="warning",
                                    data={"backoff_sec": float(self.reconnect_backoff_sec)},
                                )
                                logging.warning(
                                    f"[capture:reconnect_read] source={cfg.source_id} backoff_sec={self.reconnect_backoff_sec:.1f}"
                                )
                                try:
                                    self.source_manager.release()
                                except Exception:
                                    pass
                                self._sleep_reconnect_backoff()
                                break
                            return

                        packet = CapturedFrame(
                            source_id=cfg.source_id,
                            frame_index=frame_index,
                            frame=frame,
                            video_timestamp=self.source_manager.get_timestamp(frame_index, start_time),
                            captured_at=time.time(),
                        )
                        accepted, dropped = self._enqueue_frame(packet)

                        self.stats.read_frames += 1
                        self.stats.dropped_frames += int(dropped)
                        if accepted:
                            self.stats.queued_frames += 1
                        frame_index += 1
            except Exception as e:
                self.last_error = str(e)
                logging.error(f"[capture:worker_failed] source={cfg.source_id} error={e}")
            finally:
                try:
                    self.source_manager.release()
                except Exception:
                    pass
                self.stats.finished_at = time.time()
                self._is_running = False

    def _should_stop(self) -> bool:
        """Функция: _should_stop()
Назначение: проверяет локальный и общий сигнал остановки worker-а.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: признак необходимости остановки worker-а."""
        return self.stop_event.is_set() or self._shared_stop_event.is_set()

    def _emit_event(self, event_type: str, severity: str, data: Optional[dict] = None) -> None:
        """Функция: _emit_event()
Назначение: отправляет runtime-событие через callback, если он задан.
Параметры функции:
- `event_type` (`str`): тип события.
- `severity` (`str`): уровень важности события.
- `data` (`Optional[dict]`): полезная нагрузка события.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.event_callback is None:
            return
        try:
            self.event_callback(
                event_type=event_type,
                source_id=self.source_config.source_id,
                severity=severity,
                data=dict(data or {}),
            )
        except Exception:
            return

    def _is_reconnectable_source(self) -> bool:
        """Функция: _is_reconnectable_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.reconnect_on_loss is not None:
            return bool(self.reconnect_on_loss)

        if self.source_config.camera_id is not None:
            return True

        input_source = self.source_config.input_source
        source_text = str(input_source or "").strip().lower()
        return source_text.startswith(("rtsp://", "rtsps://"))

    def _sleep_reconnect_backoff(self) -> None:
        """Функция: _sleep_reconnect_backoff()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.reconnect_backoff_sec <= 0:
            return
        self._wait_for_stop(self.reconnect_backoff_sec)

    def _sync_file_playback(
        self,
        frame_index: int,
        source_started_at: float,
        source_fps: float,
    ) -> None:
        """Функция: _sync_file_playback()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame_index` (`int`): порядковый номер кадра внутри источника.
- `source_started_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `source_fps` (`float`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if frame_index <= 0 or source_fps <= 0:
            return
        elapsed = time.monotonic() - source_started_at
        target_elapsed = float(frame_index) / float(source_fps)
        sleep_for = target_elapsed - elapsed
        if sleep_for > 0:
            self._wait_for_stop(sleep_for)

    def _wait_for_stop(self, timeout: float) -> None:
        """Функция: _wait_for_stop()
Назначение: ожидает таймаут с возможностью раннего выхода по локальному или общему stop-сигналу.
Параметры функции:
- `timeout` (`float`): длительность ожидания.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        if not isinstance(self._shared_stop_event, threading.Event):
            try:
                self._shared_stop_event.wait(max(0.0, float(timeout)))
            except Exception:
                return
            return
        while not self._should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self.stop_event.wait(min(0.05, remaining))
