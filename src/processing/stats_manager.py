"""Файл: src/processing/stats_manager.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

import time
from typing import Dict, Any, List, DefaultDict, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SessionStats:
    """Класс: SessionStats
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `camera_movements` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `current_movement_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `current_obstruction_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `dms_violation_types` (`DefaultDict[str, int]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `dms_violations` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `dms_violations_saved` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `eye_state` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `forbidden_items_saved` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `forbidden_items_violations` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `max_movement_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `max_obstruction_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `movement_violations` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `movement_violations_saved` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `obstruction_violations` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `saved_violations` (`int`): данные детектора нарушений, используемые для итогового решения по кадру.
- `seatbelt_state` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    # Общее
    total_detections: int = 0
    total_frames: int = 0
    saved_violations: int = 0
    
    # Обструкция
    obstruction_violations: int = 0
    current_obstruction_duration: float = 0.0
    max_obstruction_duration: float = 0.0
    
    # Движение
    camera_movements: int = 0
    movement_violations: int = 0
    movement_violations_saved: int = 0
    current_movement_duration: float = 0.0
    max_movement_duration: float = 0.0
    
    # Forbidden items
    forbidden_items_violations: int = 0
    forbidden_items_saved: int = 0
    
    # DMS
    dms_violations: int = 0
    dms_violations_saved: int = 0
    eye_state: str = "unknown"
    seatbelt_state: str = "unknown"
    
    # Детализация по классам DMS
    dms_violation_types: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))


class StatsManager:
    """Класс: StatsManager
Назначение: координирует подсистему и управляет ее состоянием во время обработки.
Поля класса:
- `current_fps` (`float`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
- `fps_buffer` (`List[float]`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
- `start_time` (`float`): время начала текущей операции/сессии.
- `stats` (`Any`): метрика или счетчик, применяемый для статистики и контроля выполнения.
Ключевые методы:
- `__init__()`, `start_session()`, `increment_frames()`, `update_fps()`, `record_obstruction()`, `record_obstruction_violation()`, `record_movement()`, `record_movement_saved()`, `record_forbidden()`, `record_forbidden_saved()`"""
    
    def __init__(self):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.stats = SessionStats()
        self.start_time: float = 0.0
        self.fps_buffer: List[float] = []
        self.current_fps: float = 0.0
        
    def start_session(self) -> None:
        """Функция: start_session()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.start_time = time.time()
        self.stats = SessionStats()
        
    def increment_frames(self, count: int = 1) -> None:
        """Функция: increment_frames()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `count` (`int`): метрика или счетчик, применяемый для статистики и контроля выполнения.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.total_frames += count
        
    def update_fps(self, frame_time: float) -> None:
        """Функция: update_fps()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `frame_time` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if frame_time > 0:
            self.fps_buffer.append(1.0 / frame_time)
            if len(self.fps_buffer) > 30:
                self.fps_buffer.pop(0)
            self.current_fps = sum(self.fps_buffer) / len(self.fps_buffer)
    
    def record_obstruction(self, duration: float) -> None:
        """Функция: record_obstruction()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.total_detections += 1
        self.stats.current_obstruction_duration = duration
        if duration > self.stats.max_obstruction_duration:
            self.stats.max_obstruction_duration = duration
    
    def record_obstruction_violation(self) -> None:
        """Функция: record_obstruction_violation()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.obstruction_violations += 1
        self.stats.saved_violations += 1
    
    def record_movement(self, duration: float) -> None:
        """Функция: record_movement()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.camera_movements += 1
        self.stats.movement_violations += 1
        self.stats.current_movement_duration = duration
        if duration > self.stats.max_movement_duration:
            self.stats.max_movement_duration = duration
    
    def record_movement_saved(self) -> None:
        """Функция: record_movement_saved()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.movement_violations_saved += 1
        self.stats.saved_violations += 1
    
    def record_forbidden(self) -> None:
        """Функция: record_forbidden()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.forbidden_items_violations += 1
    
    def record_forbidden_saved(self) -> None:
        """Функция: record_forbidden_saved()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.forbidden_items_saved += 1
        self.stats.saved_violations += 1
    
    def record_dms_violation(self, violation_type: str = "unknown") -> None:
        """Функция: record_dms_violation()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `violation_type` (`str`): строковый тип нарушения для маршрутизации логики и логирования.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.dms_violations += 1
        self.stats.dms_violation_types[violation_type] += 1
    
    def record_dms_saved(self) -> None:
        """Функция: record_dms_saved()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.dms_violations_saved += 1
        self.stats.saved_violations += 1
    
    def update_dms_state(self, eye_state: str, seatbelt_state: str) -> None:
        """Функция: update_dms_state()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `eye_state` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `seatbelt_state` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats.eye_state = eye_state
        self.stats.seatbelt_state = seatbelt_state
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Функция: get_current_stats()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0.0
        avg_fps = self.stats.total_frames / elapsed if elapsed > 0 else 0.0

        return {
            "total_frames": self.stats.total_frames,
            "total_detections": self.stats.total_detections,
            "saved_violations": self.stats.saved_violations,
            "obstruction_violations": self.stats.obstruction_violations,
            "camera_movements": self.stats.camera_movements,
            "movement_violations": self.stats.movement_violations,
            "movement_violations_saved": self.stats.movement_violations_saved,
            "forbidden_items_violations": self.stats.forbidden_items_violations,
            "forbidden_items_saved": self.stats.forbidden_items_saved,
            "dms_violations": self.stats.dms_violations,
            "dms_violations_saved": self.stats.dms_violations_saved,
            "dms_violation_types": dict(self.stats.dms_violation_types),
            "eye_state": self.stats.eye_state,
            "seatbelt_state": self.stats.seatbelt_state,
            "current_obstruction_duration": self.stats.current_obstruction_duration,
            "max_obstruction_duration": self.stats.max_obstruction_duration,
            "current_movement_duration": self.stats.current_movement_duration,
            "max_movement_duration": self.stats.max_movement_duration,
            "current_fps": self.current_fps,
            "avg_fps": avg_fps,
            "elapsed_time": elapsed,
        }
    
    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Функция: _format_elapsed()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `seconds` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        total = max(0, int(seconds))
        hours = total // 3600
        minutes = (total % 3600) // 60
        secs = total % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def build_summary_from_stats(
        stats: Dict[str, Any],
        title: str = "SESSION SUMMARY",
        capture_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Функция: build_summary_from_stats()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `stats` (`Dict[str, Any]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `title` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `capture_stats` (`Optional[Dict[str, Any]]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        width = 98
        border = "+" + ("-" * (width - 2)) + "+"

        def row(label: str, value: str) -> str:
            """Функция: row()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `label` (`str`): подпись объекта/нарушения для отображения на кадре.
- `value` (`str`): обрабатываемое значение параметра до валидации/преобразования.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
            body = f"{label:<16}: {value}"
            return f"| {body:<{width - 4}} |"

        dms_types = dict(stats.get("dms_violation_types", {}) or {})
        if dms_types:
            dms_breakdown = ", ".join(
                f"{key}:{int(value)}" for key, value in sorted(dms_types.items())
            )
        else:
            dms_breakdown = "none"

        lines = [
            border,
            f"| {title:^{width - 4}} |",
            border,
            row(
                "Frames",
                (
                    f"total={int(stats.get('total_frames', 0))}  "
                    f"elapsed={float(stats.get('elapsed_time', 0.0)):.2f}s "
                    f"({StatsManager._format_elapsed(float(stats.get('elapsed_time', 0.0)))})  "
                    f"avg_fps={float(stats.get('avg_fps', 0.0)):.2f}  "
                    f"current_fps={float(stats.get('current_fps', 0.0)):.2f}"
                ),
            ),
            row(
                "Observed",
                (
                    f"obstruction={int(stats.get('total_detections', 0))}  "
                    f"movement={int(stats.get('movement_violations', 0))}  "
                    f"forbidden={int(stats.get('forbidden_items_violations', 0))}  "
                    f"dms={int(stats.get('dms_violations', 0))}"
                ),
            ),
            row(
                "Saved",
                (
                    f"total={int(stats.get('saved_violations', 0))}  "
                    f"obstruction={int(stats.get('obstruction_violations', 0))}  "
                    f"movement={int(stats.get('movement_violations_saved', 0))}  "
                    f"forbidden={int(stats.get('forbidden_items_saved', 0))}  "
                    f"dms={int(stats.get('dms_violations_saved', 0))}"
                ),
            ),
            row(
                "Current states",
                f"eye={stats.get('eye_state', 'unknown')}  seatbelt={stats.get('seatbelt_state', 'unknown')}",
            ),
            row(
                "Max durations",
                (
                    f"obstruction={float(stats.get('max_obstruction_duration', 0.0)):.2f}s  "
                    f"movement={float(stats.get('max_movement_duration', 0.0)):.2f}s"
                ),
            ),
            row("DMS breakdown", dms_breakdown),
        ]

        if capture_stats:
            lines.append(
                row(
                    "Capture",
                    (
                        f"read={int(capture_stats.get('captured_frames', 0))}  "
                        f"dropped={int(capture_stats.get('capture_dropped_frames', 0))}"
                    ),
                )
            )
            queue_size = capture_stats.get("writer_queue_size")
            if queue_size is not None:
                lines.append(row("Writer queue", f"size={int(queue_size)}"))

        lines.append(border)
        return "\n".join(lines)

    def get_summary(
        self,
        source_id: Optional[str] = None,
        capture_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Функция: get_summary()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `source_id` (`Optional[str]`): параметр источника/выхода данных, задающий направление потока обработки.
- `capture_stats` (`Optional[Dict[str, Any]]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        stats = self.get_current_stats()
        title = f"SOURCE SUMMARY [{source_id}]" if source_id else "SESSION SUMMARY"
        return self.build_summary_from_stats(stats=stats, title=title, capture_stats=capture_stats)
    
    def reset(self) -> None:
        """Функция: reset()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.stats = SessionStats()
        self.fps_buffer = []
        self.current_fps = 0.0
