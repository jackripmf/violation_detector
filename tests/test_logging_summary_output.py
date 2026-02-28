"""Файл: tests/test_logging_summary_output.py
Тип: файл автотестов.
Назначение: проверяет специальный вывод финальной summary-таблицы без formatter-префикса.
Связи: взаимодействует с инфраструктурой логирования через публичные helper-функции из utils."""

import io
import logging

from src.utils.common.utils import LOG_DATEFMT, LOG_FORMAT, log_summary_block


def test_log_summary_block_writes_end_marker_then_plain_table():
    stream = io.StringIO()
    logger = logging.getLogger("tests.summary_output")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT))
    logger.addHandler(handler)

    try:
        log_summary_block(
            "+----+\n| ok |\n+----+",
            logger=logger,
        )
    finally:
        logger.handlers.clear()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 4
    assert lines[0].endswith("|INFO| end log")
    assert lines[1] == "+----+"
    assert lines[2] == "| ok |"
    assert lines[3] == "+----+"
