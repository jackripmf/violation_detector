"""Файл: src/processing/scheduler_policy.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Импортируемые внутренние модули:
- .runtime_models: используется для передачи данных или вызова связанной логики."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .topology import SchedulerConfig


@dataclass
class SchedulerSourceState:
    """Класс: SchedulerSourceState
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `base_priority` (`float`): идентификатор/индекс для адресации и сопоставления сущностей.
- `last_served_at` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `queue_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `served_count` (`int`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    source_id: str
    queue_size: int
    base_priority: float
    last_served_at: float
    served_count: int


class SchedulerPolicy:
    """Класс: SchedulerPolicy
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `config` (`SchedulerConfig`): структура конфигурации компонента/подсистемы.
Ключевые методы:
- `__init__()`, `select_next()`, `score_source()`"""

    def __init__(self, config: SchedulerConfig):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `config` (`SchedulerConfig`): структура конфигурации компонента/подсистемы.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.config = config

    def select_next(
        self,
        source_states: Iterable[SchedulerSourceState],
        now_ts: float,
    ) -> Optional[str]:
        """Функция: select_next()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `source_states` (`Iterable[SchedulerSourceState]`): набор состояний источников для принятия решений планировщиком.
- `now_ts` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: Optional[str]: результат шага обработки, который используется следующим этапом пайплайна."""
        best_source_id: Optional[str] = None
        best_score: Optional[float] = None
        best_qsize = -1

        for state in source_states:
            if state.queue_size <= 0:
                continue
            score = self.score_source(state=state, now_ts=now_ts)
            if (
                best_score is None
                or score > best_score
                or (score == best_score and state.queue_size > best_qsize)
                or (
                    score == best_score
                    and state.queue_size == best_qsize
                    and best_source_id is not None
                    and state.source_id < best_source_id
                )
            ):
                best_source_id = state.source_id
                best_score = score
                best_qsize = state.queue_size

        return best_source_id

    def score_source(self, state: SchedulerSourceState, now_ts: float) -> float:
        """Функция: score_source()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `state` (`SchedulerSourceState`): текущее состояние компонента или автомата обработки.
- `now_ts` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: float: результат шага обработки, который используется следующим этапом пайплайна."""
        wait_sec = max(0.0, float(now_ts) - float(state.last_served_at))

        score = float(state.base_priority)
        score += self.config.aging_factor * wait_sec
        score += self.config.backlog_factor * float(state.queue_size)

        if wait_sec >= self.config.starvation_threshold_sec:
            score += self.config.starvation_boost

        score -= 0.001 * float(state.served_count)
        return score
