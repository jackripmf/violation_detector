"""Файл: src/inference/model_path_resolver.py
Тип: вспомогательный модуль.
Назначение: централизует безопасное разрешение путей к весам моделей внутри разрешенного каталога.
Связи: используется детекторами и инференс-слоем для единообразной загрузки моделей.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from pathlib import Path
from typing import Optional


def _format_path_for_error(path: str | Path) -> str:
    """Функция: _format_path_for_error()
Назначение: сокращает путь для безопасного текста ошибок без полного раскрытия структуры каталогов.
Параметры функции:
- `path` (`str | Path`): исходный путь для форматирования.
Возвращаемое значение: str: сокращенное строковое представление пути."""
    text = str(path).replace("\\", "/").rstrip("/")
    if not text:
        return "<empty>"
    file_name = Path(text).name
    if not file_name or file_name == text:
        return text
    return f".../{file_name}"


def get_default_model_root() -> Path:
    """Функция: get_default_model_root()
Назначение: возвращает канонический каталог хранения production-весов моделей.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Path: абсолютный путь к каталогу моделей."""
    return Path(__file__).resolve().parent.parent / "utils" / "models"


def resolve_model_path(
    model_name: Optional[str] = None,
    *,
    default_model_filename: str = "best_auto.pt",
    model_root: Optional[str | Path] = None,
) -> str:
    """Функция: resolve_model_path()
Назначение: преобразует имя или путь модели в абсолютный безопасный путь внутри разрешенного каталога.
Параметры функции:
- `model_name` (`Optional[str]`): имя файла модели или путь, переданный вызывающим кодом.
- `default_model_filename` (`str`): имя модели по умолчанию, используемое при пустом значении `model_name`.
- `model_root` (`Optional[str | Path]`): корневой каталог, внутри которого разрешено искать модели.
Возвращаемое значение: str: абсолютный путь к существующему файлу модели."""
    root_path = Path(model_root).resolve() if model_root is not None else get_default_model_root().resolve()
    model_text = str(model_name).strip() if model_name is not None else ""
    requested_name = model_text or str(default_model_filename).strip()
    if not requested_name:
        raise ValueError("Model filename is empty.")

    candidate_path = Path(requested_name)
    if candidate_path.is_absolute():
        resolved_path = candidate_path.resolve()
    else:
        resolved_path = (root_path / candidate_path).resolve()

    try:
        resolved_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            f"Model path '{_format_path_for_error(requested_name)}' is outside allowed model root "
            f"'{_format_path_for_error(root_path)}'."
        ) from exc

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Model file '{_format_path_for_error(requested_name)}' was not found in "
            f"'{_format_path_for_error(root_path)}'."
        )

    return str(resolved_path)
