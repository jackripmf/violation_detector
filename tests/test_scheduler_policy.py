"""Файл: tests/test_scheduler_policy.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- src.processing.multi_source_runtime: используется для передачи данных или вызова связанной логики.
- src.runtime.topology: используется для передачи данных или вызова связанной логики."""

import queue

from src.processing.multi_source_runtime import CentralScheduler, SourceContext
from src.runtime.topology import CapturedFrame, SchedulerConfig, SourceConfig


def _make_context(source_id: str, priority: float, frames: int) -> SourceContext:
    frame_queue = queue.Queue(maxsize=max(1, frames + 2))
    for idx in range(frames):
        frame_queue.put_nowait(
            CapturedFrame(
                source_id=source_id,
                frame_index=idx,
                frame=None,
                video_timestamp=float(idx),
                captured_at=0.0,
            )
        )
    return SourceContext(
        source_config=SourceConfig(
            source_id=source_id,
            camera_id=0,
            base_priority=priority,
            capture_queue_size=frame_queue.maxsize,
        ),
        frame_queue=frame_queue,
    )


def _drain_sources(infer_queue):
    out = []
    while not infer_queue.empty():
        item = infer_queue.get_nowait()
        out.append(item.source_id)
    return out


def test_scheduler_fairness_across_equal_sources():
    contexts = {
        "a": _make_context("a", priority=1.0, frames=8),
        "b": _make_context("b", priority=1.0, frames=8),
        "c": _make_context("c", priority=1.0, frames=8),
    }
    infer_q = queue.Queue(maxsize=64)
    scheduler = CentralScheduler(
        source_contexts=contexts,
        infer_queue=infer_q,
        scheduler_config=SchedulerConfig(),
    )

    for step in range(9):
        assert scheduler.dispatch_once(now_ts=1.0 + step * 0.2) is True

    picked = _drain_sources(infer_q)
    assert picked.count("a") >= 2
    assert picked.count("b") >= 2
    assert picked.count("c") >= 2


def test_scheduler_priority_prefers_higher_base_priority():
    contexts = {
        "high": _make_context("high", priority=5.0, frames=30),
        "low": _make_context("low", priority=1.0, frames=30),
    }
    infer_q = queue.Queue(maxsize=128)
    scheduler = CentralScheduler(
        source_contexts=contexts,
        infer_queue=infer_q,
        scheduler_config=SchedulerConfig(
            aging_factor=0.1,
            backlog_factor=0.0,
            starvation_threshold_sec=100.0,
            starvation_boost=0.0,
        ),
    )

    for step in range(20):
        assert scheduler.dispatch_once(now_ts=1.0 + step * 0.1) is True

    picked = _drain_sources(infer_q)
    assert picked.count("high") > picked.count("low")


def test_scheduler_anti_starvation_eventually_serves_low_priority_source():
    contexts = {
        "high": _make_context("high", priority=10.0, frames=50),
        "low": _make_context("low", priority=1.0, frames=50),
    }
    infer_q = queue.Queue(maxsize=256)
    scheduler = CentralScheduler(
        source_contexts=contexts,
        infer_queue=infer_q,
        scheduler_config=SchedulerConfig(
            aging_factor=0.0,
            backlog_factor=0.0,
            starvation_threshold_sec=0.4,
            starvation_boost=20.0,
        ),
    )

    for step in range(12):
        assert scheduler.dispatch_once(now_ts=0.1 + step * 0.1) is True

    picked = _drain_sources(infer_q)
    assert "low" in picked


def test_scheduler_infer_queue_drop_oldest_keeps_fresh_requests():
    contexts = {"a": _make_context("a", priority=1.0, frames=5)}
    infer_q = queue.Queue(maxsize=2)
    scheduler = CentralScheduler(
        source_contexts=contexts,
        infer_queue=infer_q,
        scheduler_config=SchedulerConfig(
            infer_queue_size=2,
            infer_overflow_strategy="drop_oldest",
            starvation_threshold_sec=100.0,
        ),
    )

    for step in range(5):
        assert scheduler.dispatch_once(now_ts=1.0 + step) is True

    left = []
    while not infer_q.empty():
        req = infer_q.get_nowait()
        left.append(req.frame_packet.frame_index)

    assert left == [3, 4]
