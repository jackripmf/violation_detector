"""Файл: src/processing/preview_pipeline.py
Тип: helper-модуль preview pipeline centralized runtime.
Назначение: содержит preview loop, callback payload и управление preview-окнами.
Связи: используется MultiSourceRuntime и OpenCV preview path."""

from __future__ import annotations

import inspect
import logging
import queue
from typing import TYPE_CHECKING, Any, Optional

import cv2

from ..runtime.contracts import PreviewFramePayload

if TYPE_CHECKING:
    from .multi_source_runtime import MultiSourceRuntime, SourceContext


def preview_loop(runtime: "MultiSourceRuntime") -> None:
    """Функция: preview_loop()
Назначение: обрабатывает preview queue и lifecycle preview-окон.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
Возвращаемое значение: None: helper обслуживает preview loop до завершения pipeline."""
    if cv2 is None:
        return
    try:
        while True:
            try:
                packet = runtime.preview_queue.get(timeout=0.05)
            except queue.Empty:
                if runtime._pipeline_finished() and runtime.preview_queue.empty():
                    break
                continue

            if runtime._preview_broken:
                runtime.preview_queue.task_done()
                continue

            window_name = f"Violation Detector [{packet.source_id}]"
            try:
                if packet.close_window:
                    if window_name in runtime._preview_windows:
                        try:
                            cv2.destroyWindow(window_name)
                        except Exception:
                            pass
                        runtime._preview_windows.discard(window_name)
                        runtime._preview_mouse_bound.discard(window_name)
                        if runtime._active_preview_source_id == packet.source_id:
                            runtime._active_preview_source_id = None
                else:
                    cv2.imshow(window_name, packet.frame)
                    runtime._preview_windows.add(window_name)
                    if window_name not in runtime._preview_mouse_bound:
                        try:
                            cv2.setMouseCallback(window_name, runtime._on_preview_mouse_event, packet.source_id)
                            runtime._preview_mouse_bound.add(window_name)
                        except Exception:
                            pass
                    if runtime._active_preview_source_id is None:
                        runtime._active_preview_source_id = packet.source_id
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("r")):
                    target_source = resolve_preview_key_source(runtime, fallback_source_id=packet.source_id)
                    if target_source is None:
                        logging.debug(f"[preview:key_ignored] key={chr(key)} reason=no_active_window")
                    elif key == ord("q"):
                        request_stop_source(runtime, target_source, runtime._preview_packet_cls)
                    elif key == ord("r"):
                        reset_movement_reference_for_source(runtime, target_source)
            except cv2.error:
                runtime._preview_broken = True
            finally:
                runtime.preview_queue.task_done()
    finally:
        try:
            for window_name in list(runtime._preview_windows):
                try:
                    cv2.destroyWindow(window_name)
                except Exception:
                    pass
            runtime._preview_windows.clear()
            runtime._preview_mouse_bound.clear()
            cv2.destroyAllWindows()
        except Exception:
            pass


def on_preview_mouse_event(runtime: "MultiSourceRuntime", event: int, _x: int, _y: int, _flags: int, param: Any) -> None:
    """Функция: on_preview_mouse_event()
Назначение: помечает активный preview source по пользовательскому взаимодействию с окном.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- параметры callback-а OpenCV: служебные значения события мыши.
Возвращаемое значение: None: helper обновляет active preview source."""
    del _x, _y, _flags
    if cv2 is None:
        return
    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
        return
    source_id = str(param)
    if source_id not in runtime.source_contexts:
        return
    runtime._active_preview_source_id = source_id


def resolve_preview_key_source(runtime: "MultiSourceRuntime", fallback_source_id: str) -> Optional[str]:
    """Функция: resolve_preview_key_source()
Назначение: определяет, к какому source относить клавиатурную команду preview.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `fallback_source_id` (`str`): запасной идентификатор источника.
Возвращаемое значение: Optional[str]: целевой source_id или `None`."""
    del fallback_source_id
    active_id = runtime._active_preview_source_id
    if active_id and active_id in runtime.source_contexts:
        context = runtime.source_contexts[active_id]
        if not context.preview_closed and not context.stop_requested:
            return active_id

    alive_sources = [sid for sid, context in runtime.source_contexts.items() if not context.preview_closed and not context.stop_requested]
    if len(alive_sources) == 1:
        return alive_sources[0]
    return None


def request_stop_source(runtime: "MultiSourceRuntime", source_id: str, preview_packet_cls: type[Any]) -> bool:
    """Функция: request_stop_source()
Назначение: помечает source как остановленный пользователем и закрывает его preview.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
- `preview_packet_cls` (`type[Any]`): класс packet-а preview.
Возвращаемое значение: bool: признак успешной обработки запроса."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return False
    if context.stop_requested:
        return True

    context.stop_requested = True
    context.preview_closed = True
    if context.capture_worker is not None:
        context.capture_worker.stop()

    while True:
        try:
            context.frame_queue.get_nowait()
        except queue.Empty:
            break

    if runtime._active_preview_source_id == source_id:
        runtime._active_preview_source_id = None
    enqueue_preview_packet(runtime, preview_packet_cls(source_id=source_id, close_window=True))
    logging.info(f"[preview:source_stopped] source={source_id} reason=user_key_q")
    return True


def enqueue_preview_packet(runtime: "MultiSourceRuntime", packet: Any) -> None:
    """Функция: enqueue_preview_packet()
Назначение: помещает preview packet в очередь с drop-oldest поведением.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `packet` (`Any`): preview packet.
Возвращаемое значение: None: helper обновляет preview queue."""
    try:
        runtime.preview_queue.put_nowait(packet)
    except queue.Full:
        try:
            runtime.preview_queue.get_nowait()
            runtime.preview_queue.put_nowait(packet)
        except (queue.Empty, queue.Full):
            pass


def render_preview_frame(
    runtime: "MultiSourceRuntime",
    source_id: str,
    frame: Any,
    frame_index: int,
    video_timestamp: float,
    captured_at: float,
    processed_at: float,
    processed_frames: int,
    preview_packet_cls: type[Any],
) -> None:
    """Функция: render_preview_frame()
Назначение: отправляет кадр либо в callback, либо в preview queue OpenCV.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- параметры preview payload: данные кадра и метаданные обработки.
- `preview_packet_cls` (`type[Any]`): класс packet-а preview.
Возвращаемое значение: None: helper отправляет кадр по нужному каналу preview."""
    if not runtime.show_preview or frame is None:
        return
    context = runtime.source_contexts.get(source_id)
    if context is not None and (context.preview_closed or context.stop_requested):
        return

    if runtime.preview_callback is not None:
        try:
            payload = build_preview_payload(
                runtime=runtime,
                source_id=source_id,
                frame=frame,
                frame_index=frame_index,
                video_timestamp=video_timestamp,
                captured_at=captured_at,
                processed_at=processed_at,
                processed_frames=processed_frames,
            )
            invoke_preview_callback(runtime, payload)
        except Exception as exc:
            runtime._emit_event(
                event_type="runtime_warning",
                source_id=source_id,
                severity="warning",
                data={"stage": "preview_callback", "error": str(exc)},
            )
        return

    if cv2 is None or runtime._preview_broken:
        return
    enqueue_preview_packet(runtime, preview_packet_cls(source_id=source_id, frame=frame))


def build_preview_payload(
    runtime: "MultiSourceRuntime",
    source_id: str,
    frame: Any,
    frame_index: int,
    video_timestamp: float,
    captured_at: float,
    processed_at: float,
    processed_frames: int,
) -> PreviewFramePayload:
    """Функция: build_preview_payload()
Назначение: формирует структурированный preview payload для UI callback.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- параметры preview payload: данные кадра и метаданные обработки.
Возвращаемое значение: PreviewFramePayload: payload preview callback."""
    context = runtime.source_contexts.get(source_id)
    visual_config = runtime._copy_visual_config(context.visual_config if context is not None else None)
    writer_queue_size = 0
    if context is not None and context.violation_manager is not None and hasattr(context.violation_manager, "get_writer_queue_size"):
        writer_queue_size = int(context.violation_manager.get_writer_queue_size())
    return PreviewFramePayload(
        source_id=source_id,
        frame=frame,
        frame_index=int(frame_index),
        video_timestamp=float(video_timestamp),
        captured_at=float(captured_at),
        processed_at=float(processed_at),
        metadata={
            "processed_frames": int(processed_frames),
            "writer_queue_size": int(writer_queue_size),
            "show_fps": bool(visual_config.show_fps),
            "preview_width": visual_config.preview_width,
            "overlays_enabled": bool(
                visual_config.show_indicators
                or visual_config.show_stats_panel
                or visual_config.show_movement_arrow
                or visual_config.show_violation_labels
                or visual_config.show_boxes
            ),
        },
    )


def invoke_preview_callback(runtime: "MultiSourceRuntime", payload: PreviewFramePayload) -> None:
    """Функция: invoke_preview_callback()
Назначение: вызывает preview callback с поддержкой legacy и нового payload-контракта.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `payload` (`PreviewFramePayload`): структурированный payload preview-кадра.
Возвращаемое значение: None: helper вызывает внешний callback."""
    callback = runtime.preview_callback
    if callback is None:
        return
    try:
        signature = inspect.signature(callback)
        positional_params = [
            param for param in signature.parameters.values()
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_params) <= 1:
            callback(payload)
            return
    except (TypeError, ValueError):
        pass
    callback(payload.source_id, payload.frame)


def is_source_processing_drained(runtime: "MultiSourceRuntime", context: "SourceContext") -> bool:
    """Функция: is_source_processing_drained()
Назначение: проверяет, полностью ли дренирован pipeline для source.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`SourceContext`): контекст текущего источника.
Возвращаемое значение: bool: pipeline source полностью дренирован."""
    del runtime
    worker = context.capture_worker
    worker_done = worker is None or (not worker.is_running())
    return bool(worker_done and context.frame_queue.empty() and int(context.processed_frames) >= int(context.served_count))


def maybe_close_source_preview(runtime: "MultiSourceRuntime", context: "SourceContext", preview_packet_cls: type[Any]) -> None:
    """Функция: maybe_close_source_preview()
Назначение: закрывает preview-окно источника после полного дренирования pipeline.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `context` (`SourceContext`): контекст текущего источника.
- `preview_packet_cls` (`type[Any]`): класс packet-а preview.
Возвращаемое значение: None: helper при необходимости ставит close_window packet."""
    if not runtime.show_preview or runtime.preview_callback is not None or cv2 is None:
        return
    if context.preview_closed or context.processed_frames <= 0:
        return
    if not is_source_processing_drained(runtime, context):
        return
    context.preview_closed = True
    enqueue_preview_packet(runtime, preview_packet_cls(source_id=context.source_config.source_id, close_window=True))


def reset_movement_reference_for_source(runtime: "MultiSourceRuntime", source_id: str) -> bool:
    """Функция: reset_movement_reference_for_source()
Назначение: сбрасывает movement reference для конкретного source через detector manager.
Параметры функции:
- `runtime` (`MultiSourceRuntime`): экземпляр centralized runtime.
- `source_id` (`str`): идентификатор источника.
Возвращаемое значение: bool: признак успешного сброса reference frame."""
    context = runtime.source_contexts.get(source_id)
    if context is None:
        return False
    frame = context.latest_frame_for_reset
    detection_manager = context.detection_manager
    if frame is None or detection_manager is None:
        return False
    movement_detector = getattr(detection_manager, "movement_detector", None)
    if movement_detector is None or not hasattr(movement_detector, "set_reference_frame"):
        return False
    try:
        movement_detector.set_reference_frame(frame)
        logging.info(f"[runtime:movement_reference_reset] source={source_id}")
        return True
    except Exception:
        return False
