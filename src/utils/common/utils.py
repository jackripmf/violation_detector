"""Файл: src/utils/common/utils.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- .log_context: используется для передачи данных или вызова связанной логики."""
import cv2
import os
import logging
from datetime import datetime
from typing import Iterable, Optional


LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s.%(msecs)03d |%(levelname)s| %(message)s"


class _SourceLogFilter(logging.Filter):
    """Класс: _SourceLogFilter
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `source_id` (`Any`): параметр источника/выхода данных, задающий направление потока обработки.
Ключевые методы:
- `__init__()`, `filter()`"""

    def __init__(self, source_id: str):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        super().__init__()
        self.source_id = str(source_id)

    def filter(self, record: logging.LogRecord) -> bool:
        """Функция: filter()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `record` (`logging.LogRecord`): объект записи лога (`LogRecord`), обрабатываемый форматтером.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        from .log_context import current_log_source

        return current_log_source() == self.source_id


def _build_formatter() -> logging.Formatter:
    """Функция: _build_formatter()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: logging.Formatter: результат шага обработки, который используется следующим этапом пайплайна."""
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)


def _iter_effective_handlers(logger: logging.Logger):
    """Функция: _iter_effective_handlers()
Назначение: возвращает все handler-ы, которые реально участвуют в выводе логов с учетом propagate-цепочки.
Параметры функции:
- `logger` (`logging.Logger`): экземпляр логгера, для которого нужно получить итоговые handler-ы.
Возвращаемое значение: Iterable[logging.Handler]: последовательность уникальных handler-ов."""
    current: Optional[logging.Logger] = logger
    seen_handler_ids: set[int] = set()
    while current is not None:
        for handler in current.handlers:
            handler_id = id(handler)
            if handler_id in seen_handler_ids:
                continue
            seen_handler_ids.add(handler_id)
            yield handler
        if not current.propagate:
            break
        current = current.parent


def emit_plain_log_block(
    message: str,
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Функция: emit_plain_log_block()
Назначение: печатает текстовый блок напрямую в handler-ы без formatter-префикса, сохраняя handler-level и фильтры.
Параметры функции:
- `message` (`str`): подготовленный текстовый блок, который нужно вывести как есть.
- `level` (`int`): уровень логирования для проверки handler-level и фильтров.
- `logger` (`Optional[logging.Logger]`): целевой логгер; по умолчанию используется root logger.
Возвращаемое значение: None: текстовый блок отправляется в stream/file handler-ы."""
    text = str(message or "")
    if not text:
        return

    active_logger = logger or logging.getLogger()
    if not active_logger.isEnabledFor(level):
        return

    probe_record = active_logger.makeRecord(
        name=active_logger.name or "root",
        level=level,
        fn="",
        lno=0,
        msg="plain_log_block",
        args=(),
        exc_info=None,
    )

    for handler in _iter_effective_handlers(active_logger):
        if level < handler.level:
            continue
        if not handler.filter(probe_record):
            continue
        stream = getattr(handler, "stream", None)
        if stream is None:
            continue

        terminator = getattr(handler, "terminator", "\n")
        handler.acquire()
        try:
            stream.write(text)
            if not text.endswith(terminator):
                stream.write(terminator)
            handler.flush()
        finally:
            handler.release()


def log_summary_block(
    summary: str,
    end_message: str = "end log",
    level: int = logging.INFO,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Функция: log_summary_block()
Назначение: пишет завершающую строку лога в обычном формате и сразу после нее печатает summary без formatter-префикса.
Параметры функции:
- `summary` (`str`): итоговая summary-таблица, которую нужно вывести отдельным текстовым блоком.
- `end_message` (`str`): короткая завершающая строка, остающаяся в стандартном формате логов.
- `level` (`int`): уровень логирования для завершающей строки и summary-блока.
- `logger` (`Optional[logging.Logger]`): целевой логгер; по умолчанию используется root logger.
Возвращаемое значение: None: завершающая строка и summary отправляются в configured handler-ы."""
    active_logger = logger or logging.getLogger()
    active_logger.log(level, str(end_message))
    emit_plain_log_block(summary, level=level, logger=active_logger)


def _remove_source_handlers(logger: logging.Logger) -> None:
    """Функция: _remove_source_handlers()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `logger` (`logging.Logger`): экземпляр логгера для вывода сообщений текущего компонента.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
    for handler in list(logger.handlers):
        if getattr(handler, "_is_source_file_handler", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def setup_logging(log_level=logging.INFO, log_file=None):
    """Функция: setup_logging()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `log_level` (`Any`): уровень журналирования (`DEBUG/INFO/WARNING/...`).
- `log_file` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    formatter = _build_formatter()
    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )


def clear_source_file_logging() -> None:
    """Функция: clear_source_file_logging()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
    logger = logging.getLogger()
    _remove_source_handlers(logger)


def setup_source_file_logging(log_dir: str, source_ids: Iterable[str]) -> None:
    """Функция: setup_source_file_logging()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `log_dir` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `source_ids` (`Iterable[str]`): параметр источника/выхода данных, задающий направление потока обработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    _remove_source_handlers(logger)

    formatter = _build_formatter()
    added: set[str] = set()
    for source_id in source_ids:
        sid = str(source_id).strip()
        if not sid or sid in added:
            continue
        added.add(sid)

        safe_sid = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sid) or "source"
        file_path = os.path.join(log_dir, f"{safe_sid}.log")
        handler = logging.FileHandler(file_path, encoding="utf-8")
        handler.setLevel(logger.level or logging.INFO)
        handler.setFormatter(formatter)
        handler.addFilter(_SourceLogFilter(sid))
        setattr(handler, "_is_source_file_handler", True)
        logger.addHandler(handler)


def create_timestamp():
    """Функция: create_timestamp()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def format_video_time(seconds):
    """Функция: format_video_time()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `seconds` (`Any`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds * 1000) % 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def add_timestamp_to_frame(frame, video_timestamp):
    """Функция: add_timestamp_to_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`Any`): временная метка кадра в координатах видеоисточника.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    display_frame = frame.copy()
    timestamp_str = format_video_time(video_timestamp)
    cv2.putText(
        display_frame,
        timestamp_str,
        (frame.shape[1] - 200, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return display_frame


def get_video_info(video_path):
    """Функция: get_video_info()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `video_path` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration": total_frames / fps if fps > 0 else 0,
    }


def safe_release_video_writer(writer):
    """Функция: safe_release_video_writer()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `writer` (`Any`): объект подсистемы, через который вызывается профильная логика этого этапа.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    if writer is not None:
        try:
            writer.release()
        except Exception as e:
            logging.error(f"Error releasing video writer: {e}")


def check_file_size(file_path, min_size_kb=1):
    """Функция: check_file_size()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `file_path` (`Any`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `min_size_kb` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
Возвращаемое значение: Any: логический/статусный результат проверки условия."""
    if not os.path.exists(file_path):
        return False
    file_size = os.path.getsize(file_path)
    return file_size >= min_size_kb * 1024
