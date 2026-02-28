"""Файл: src/processing/movement_segment_tracker.py
Тип: вспомогательный модуль сегментации движения камеры.
Назначение: инкапсулирует prebuffer, детекцию reliable turning и сборку готового movement segment.
Связи: используется ViolationManager как state machine для movement-нарушений."""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional


@dataclass
class MovementSegmentResult:
    """Класс: MovementSegmentResult
Назначение: описывает готовый сегмент движения для последующей записи и формирования нарушения.
Поля класса:
- `frames_data` (`List[Dict[str, Any]]`): кадры сегмента.
- `clip_duration` (`float`): длительность сегмента в секундах.
- `clip_video_timestamp` (`float`): video timestamp первого кадра сегмента.
- `movement_info` (`Dict[str, Any]`): нормализованные метаданные сегмента.
Ключевые методы:
- Методы не объявлены явно в теле класса."""

    frames_data: List[Dict[str, Any]]
    clip_duration: float
    clip_video_timestamp: float
    movement_info: Dict[str, Any]


class MovementSegmentTracker:
    """Класс: MovementSegmentTracker
Назначение: управляет состоянием сегментов движения камеры и prebuffer перед сохранением.
Поля класса:
- `source_fps` (`float`): FPS источника для расчета prebuffer и записи видео.
- параметры сегментации (`movement_*`): настройки окна, cooldown и порогов.
Ключевые методы:
- `set_source_fps()`, `get_writer_fps()`, `process_frame()`, `force_finalize()`, `reset()`"""

    def __init__(
        self,
        movement_pre_event_sec: float,
        movement_post_event_sec: float,
        movement_turn_delta_threshold: float,
        movement_change_window_sec: float,
        movement_confirm_changes: int,
        movement_segment_cooldown_sec: float,
        movement_max_clip_sec: float,
        movement_min_clip_frames: int,
    ):
        """Функция: __init__()
Назначение: инициализирует tracker сегментов движения.
Параметры функции:
- параметры `movement_*`: конфигурация сегментации движения.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.source_fps: float = 30.0
        self.movement_pre_event_sec = max(0.0, float(movement_pre_event_sec))
        self.movement_post_event_sec = max(0.0, float(movement_post_event_sec))
        self.movement_turn_delta_threshold = max(0.0, float(movement_turn_delta_threshold))
        self.movement_change_window_sec = max(0.1, float(movement_change_window_sec))
        self.movement_confirm_changes = max(1, int(movement_confirm_changes))
        self.movement_segment_cooldown_sec = max(0.0, float(movement_segment_cooldown_sec))
        self.movement_max_clip_sec = max(1.0, float(movement_max_clip_sec))
        self.movement_min_clip_frames = max(1, int(movement_min_clip_frames))

        self._movement_prebuffer: Deque[Dict[str, Any]] = deque()
        self._movement_segment_active = False
        self._movement_segment_frames: List[Dict[str, Any]] = []
        self._movement_segment_started_at: Optional[float] = None
        self._movement_last_turning_at: Optional[float] = None
        self._movement_last_saved_at: Optional[float] = None
        self._movement_prev_vector: Optional[tuple[float, float]] = None
        self._movement_change_times: Deque[float] = deque()
        self._movement_last_info: Dict[str, Any] = {}
        self._movement_last_video_timestamp: float = 0.0
        self._reconfigure_prebuffer()

    def set_source_fps(self, source_fps: Optional[float]) -> None:
        """Функция: set_source_fps()
Назначение: обновляет FPS источника и пересчитывает размер prebuffer.
Параметры функции:
- `source_fps` (`Optional[float]`): FPS источника.
Возвращаемое значение: None: tracker обновляет внутреннее состояние."""
        try:
            value = float(source_fps)
            self.source_fps = value if value > 0 else 30.0
        except (TypeError, ValueError):
            self.source_fps = 30.0
        self._reconfigure_prebuffer()

    def get_writer_fps(self) -> float:
        """Функция: get_writer_fps()
Назначение: возвращает FPS, который нужно использовать при записи movement-видео.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: float: валидный FPS для writer-а."""
        return self.source_fps if self.source_fps > 0 else 30.0

    def process_frame(
        self,
        frame: Any,
        movement_info: Dict[str, Any],
        video_timestamp: float,
        captured_at: float,
    ) -> Optional[MovementSegmentResult]:
        """Функция: process_frame()
Назначение: принимает новый кадр движения и при необходимости завершает сегмент.
Параметры функции:
- `frame` (`Any`): кадр текущего источника.
- `movement_info` (`Dict[str, Any]`): данные движения.
- `video_timestamp` (`float`): video timestamp кадра.
- `captured_at` (`float`): wall-clock время захвата.
Возвращаемое значение: Optional[MovementSegmentResult]: готовый сегмент или `None`."""
        now_ts = float(captured_at)
        self._movement_last_video_timestamp = float(video_timestamp)
        self._movement_last_info = copy.deepcopy(movement_info or {})
        self._append_frame(
            frame=frame,
            movement_info=movement_info,
            video_timestamp=video_timestamp,
            captured_at=now_ts,
        )

        turning_now = self._is_reliable_turning_now(movement_info=movement_info, now_ts=now_ts)
        if turning_now:
            self._movement_last_turning_at = now_ts
            if not self._movement_segment_active and self._can_start_segment(now_ts):
                self._start_segment(now_ts=now_ts)

        if not self._movement_segment_active:
            return None

        segment_age = max(0.0, now_ts - float(self._movement_segment_started_at or now_ts))
        quiet_for = max(0.0, now_ts - float(self._movement_last_turning_at or now_ts))
        should_finish = quiet_for >= self.movement_post_event_sec or segment_age >= self.movement_max_clip_sec
        if not should_finish:
            return None
        return self._finalize_segment(now_ts=now_ts)

    def force_finalize(self, now_ts: float) -> Optional[MovementSegmentResult]:
        """Функция: force_finalize()
Назначение: принудительно завершает активный сегмент, например во время cleanup.
Параметры функции:
- `now_ts` (`float`): текущее время.
Возвращаемое значение: Optional[MovementSegmentResult]: готовый сегмент или `None`."""
        if not self._movement_segment_active:
            return None
        return self._finalize_segment(now_ts=float(now_ts))

    def mark_saved(self, now_ts: float) -> None:
        """Функция: mark_saved()
Назначение: фиксирует время успешного сохранения сегмента для cooldown-логики.
Параметры функции:
- `now_ts` (`float`): время успешной постановки записи.
Возвращаемое значение: None: tracker обновляет внутреннее состояние cooldown."""
        self._movement_last_saved_at = float(now_ts)

    def reset(self) -> None:
        """Функция: reset()
Назначение: сбрасывает внутреннее состояние сегментации движения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: tracker очищает накопленное состояние."""
        self._movement_segment_active = False
        self._movement_segment_frames = []
        self._movement_segment_started_at = None
        self._movement_last_turning_at = None
        self._movement_last_saved_at = None
        self._movement_prev_vector = None
        self._movement_change_times.clear()
        self._movement_last_info = {}
        self._movement_last_video_timestamp = 0.0
        self._movement_prebuffer.clear()

    def _reconfigure_prebuffer(self) -> None:
        """Функция: _reconfigure_prebuffer()
Назначение: пересчитывает размер movement prebuffer по FPS источника.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: tracker обновляет deque prebuffer."""
        max_frames = int(round(max(1.0, self.source_fps) * max(0.0, self.movement_pre_event_sec)))
        max_frames = max(1, min(120, max_frames))
        if getattr(self._movement_prebuffer, "maxlen", None) == max_frames:
            return
        self._movement_prebuffer = deque(self._movement_prebuffer, maxlen=max_frames)

    def _append_frame(
        self,
        frame: Any,
        movement_info: Dict[str, Any],
        video_timestamp: float,
        captured_at: float,
    ) -> None:
        """Функция: _append_frame()
Назначение: добавляет кадр в prebuffer и активный сегмент.
Параметры функции:
- `frame` (`Any`): кадр для накопления.
- `movement_info` (`Dict[str, Any]`): данные движения для кадра.
- `video_timestamp` (`float`): video timestamp кадра.
- `captured_at` (`float`): wall-clock время захвата.
Возвращаемое значение: None: tracker обновляет буферы кадра."""
        frame_entry = {
            "frame": frame.copy(),
            "movement_info": copy.deepcopy(movement_info or {}),
            "video_timestamp": float(video_timestamp),
            "captured_at": float(captured_at),
        }
        self._movement_prebuffer.append(frame_entry)
        if self._movement_segment_active:
            self._movement_segment_frames.append(frame_entry)

    def _can_start_segment(self, now_ts: float) -> bool:
        """Функция: _can_start_segment()
Назначение: проверяет cooldown перед стартом нового сегмента.
Параметры функции:
- `now_ts` (`float`): текущее время.
Возвращаемое значение: bool: можно ли запускать новый сегмент."""
        if self._movement_last_saved_at is None:
            return True
        return (now_ts - self._movement_last_saved_at) >= self.movement_segment_cooldown_sec

    def _start_segment(self, now_ts: float) -> None:
        """Функция: _start_segment()
Назначение: переводит tracker в состояние активного segment capture.
Параметры функции:
- `now_ts` (`float`): текущее время.
Возвращаемое значение: None: tracker обновляет внутреннее состояние."""
        self._movement_segment_active = True
        self._movement_segment_started_at = now_ts
        self._movement_segment_frames = list(self._movement_prebuffer)
        self._movement_last_turning_at = now_ts
        logging.info("[movement:segment_started]")

    def _is_reliable_turning_now(self, movement_info: Dict[str, Any], now_ts: float) -> bool:
        """Функция: _is_reliable_turning_now()
Назначение: определяет, достаточно ли устойчиво изменилось движение камеры.
Параметры функции:
- `movement_info` (`Dict[str, Any]`): данные движения.
- `now_ts` (`float`): текущее время.
Возвращаемое значение: bool: устойчивое изменение движения подтверждено."""
        if not movement_info.get("movement_detected", False):
            self._movement_prev_vector = None
            self._movement_change_times.clear()
            return False
        if not movement_info.get("filtered", False):
            return False

        tx = float(movement_info.get("translation_x", 0.0) or 0.0)
        ty = float(movement_info.get("translation_y", 0.0) or 0.0)
        translation = float(movement_info.get("translation", 0.0) or 0.0)
        abs_rotation = abs(float(movement_info.get("rotation", 0.0) or 0.0))

        deviation_above_noise = bool(translation >= 0.12 or abs_rotation >= 8.0)
        if not deviation_above_noise:
            return False

        if self._movement_prev_vector is None:
            delta_mag = math.hypot(tx, ty)
        else:
            prev_tx, prev_ty = self._movement_prev_vector
            delta_mag = math.hypot(tx - prev_tx, ty - prev_ty)
        self._movement_prev_vector = (tx, ty)

        if delta_mag >= self.movement_turn_delta_threshold:
            self._movement_change_times.append(now_ts)

        while self._movement_change_times and (now_ts - self._movement_change_times[0]) > self.movement_change_window_sec:
            self._movement_change_times.popleft()

        return len(self._movement_change_times) >= self.movement_confirm_changes

    def _finalize_segment(self, now_ts: float) -> Optional[MovementSegmentResult]:
        """Функция: _finalize_segment()
Назначение: завершает активный сегмент и возвращает готовый payload для ViolationManager.
Параметры функции:
- `now_ts` (`float`): текущее время.
Возвращаемое значение: Optional[MovementSegmentResult]: готовый сегмент или `None`."""
        frames_data = list(self._movement_segment_frames)
        self._movement_segment_active = False
        self._movement_segment_frames = []
        self._movement_segment_started_at = None
        self._movement_last_turning_at = None
        self._movement_change_times.clear()

        if len(frames_data) < self.movement_min_clip_frames:
            logging.info("[movement:segment_dropped] reason=too_short")
            return None

        first_ts = float(frames_data[0]["captured_at"])
        last_ts = float(frames_data[-1]["captured_at"])
        clip_duration = max(0.0, last_ts - first_ts)
        clip_video_timestamp = float(frames_data[0]["video_timestamp"])
        movement_info_copy = copy.deepcopy(self._movement_last_info or {})
        movement_info_copy["is_consolidated"] = False
        movement_info_copy["segment_frames"] = int(len(frames_data))
        movement_info_copy["segment_duration"] = float(clip_duration)
        logging.info(f"[movement:segment_ready] frames={len(frames_data)} duration={clip_duration:.2f}s")
        return MovementSegmentResult(
            frames_data=frames_data,
            clip_duration=clip_duration,
            clip_video_timestamp=clip_video_timestamp,
            movement_info=movement_info_copy,
        )
