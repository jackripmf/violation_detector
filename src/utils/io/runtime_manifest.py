"""Файл: src/utils/io/runtime_manifest.py
Тип: вспомогательный модуль.
Назначение: собирает и сохраняет resolved runtime configuration для воспроизводимых запусков.
Связи: используется launcher-скриптами перед стартом runtime.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_package_version(package_name: str) -> Optional[str]:
    """Функция: _safe_package_version()
Назначение: возвращает версию пакета, если он установлен в текущем окружении.
Параметры функции:
- `package_name` (`str`): имя Python-пакета для поиска версии.
Возвращаемое значение: Optional[str]: версия пакета или `None`, если пакет не найден."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def summarize_topology(topology: Any, input_source: Any, camera_id: Optional[int]) -> Dict[str, Any]:
    """Функция: summarize_topology()
Назначение: формирует краткую summary источников для runtime manifest.
Параметры функции:
- `topology` (`Any`): runtime topology после нормализации или `None` для single-source режима.
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `camera_id` (`Optional[int]`): идентификатор локальной камеры, если запуск идёт в camera mode.
Возвращаемое значение: Dict[str, Any]: сериализуемая summary topology для manifest."""
    if topology is None:
        source_kind = "camera" if camera_id is not None else "input"
        source_id = f"camera_{int(camera_id)}" if camera_id is not None else "source_0"
        return {
            "source_count": 1,
            "source_ids": [source_id],
            "source_kinds": [source_kind],
            "mode": "single_source",
            "input_source_provided": input_source is not None,
        }

    source_ids: list[str] = []
    source_kinds: list[str] = []
    for source in getattr(topology, "sources", []):
        source_ids.append(str(getattr(source, "source_id", "source")))
        source_kinds.append("camera" if getattr(source, "camera_id", None) is not None else "input")

    return {
        "source_count": len(source_ids),
        "source_ids": source_ids,
        "source_kinds": source_kinds,
        "mode": "topology",
    }


def build_runtime_manifest(
    *,
    runtime_engine: str,
    runtime_profile_name: str,
    runtime_profile_mode: str,
    requested_runtime_profile: str,
    device: str,
    use_half: bool,
    imgsz: int,
    source_queue_size: int,
    scheduler_infer_queue_size: int,
    infer_workers: int,
    postprocess_workers: int,
    scheduler_dispatch_sleep_sec: float,
    output_dir: str,
    save_dir: str,
    log_file: Optional[str],
    topology_summary: Dict[str, Any],
    source_drop_policy: str,
    show_preview: bool,
    show_fps: bool,
    profile_decision: Any,
    shared_model_path: Optional[str],
) -> Dict[str, Any]:
    """Функция: build_runtime_manifest()
Назначение: формирует сериализуемый resolved runtime manifest для сохранения на диск.
Параметры функции:
- `runtime_engine` (`str`): активный runtime engine запуска.
- `runtime_profile_name` (`str`): имя выбранного runtime profile.
- `runtime_profile_mode` (`str`): режим выбора профиля (`auto` или `manual`).
- `requested_runtime_profile` (`str`): исходное значение `--runtime-profile`.
- `device` (`str`): устройство инференса для YOLO.
- `use_half` (`bool`): флаг использования FP16.
- `imgsz` (`int`): итоговый размер входа модели.
- `source_queue_size` (`int`): итоговый размер очереди source.
- `scheduler_infer_queue_size` (`int`): итоговый размер infer queue.
- `infer_workers` (`int`): итоговое количество infer workers.
- `postprocess_workers` (`int`): итоговое количество postprocess workers.
- `scheduler_dispatch_sleep_sec` (`float`): итоговая пауза scheduler.
- `output_dir` (`str`): launcher output directory.
- `save_dir` (`str`): фактический save dir runtime.
- `log_file` (`Optional[str]`): путь к лог-файлу запуска, если он используется.
- `topology_summary` (`Dict[str, Any]`): краткая summary topology.
- `source_drop_policy` (`str`): политика переполнения source queue.
- `show_preview` (`bool`): включено ли preview.
- `show_fps` (`bool`): включён ли FPS overlay.
- `profile_decision` (`Any`): решение выбора runtime profile.
- `shared_model_path` (`Optional[str]`): канонический путь к общим весам модели.
Возвращаемое значение: Dict[str, Any]: runtime manifest для записи в JSON."""
    return {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runtime": {
            "engine": str(runtime_engine),
            "device": str(device),
            "use_half": bool(use_half),
            "imgsz": int(imgsz),
            "source_queue_size": int(source_queue_size),
            "scheduler_infer_queue_size": int(scheduler_infer_queue_size),
            "infer_workers": int(infer_workers),
            "postprocess_workers": int(postprocess_workers),
            "scheduler_dispatch_sleep_sec": float(scheduler_dispatch_sleep_sec),
            "source_drop_policy": str(source_drop_policy),
            "show_preview": bool(show_preview),
            "show_fps": bool(show_fps),
        },
        "runtime_profile": {
            "requested": str(requested_runtime_profile),
            "selection_mode": str(runtime_profile_mode),
            "selected": str(runtime_profile_name),
            "requested_device": str(getattr(profile_decision, "requested_device", device)),
            "cuda_available": bool(getattr(profile_decision, "cuda_available", False)),
            "cpu_count": int(getattr(profile_decision, "cpu_count", 0) or 0),
            "total_ram_gb": getattr(profile_decision, "total_ram_gb", None),
        },
        "paths": {
            "output_dir": str(output_dir),
            "save_dir": str(save_dir),
            "log_file": (str(log_file) if log_file else None),
            "shared_model_path": (str(shared_model_path) if shared_model_path else None),
        },
        "topology": dict(topology_summary),
        "package_versions": {
            "torch": _safe_package_version("torch"),
            "ultralytics": _safe_package_version("ultralytics"),
            "opencv-python": _safe_package_version("opencv-python"),
            "numpy": _safe_package_version("numpy"),
        },
    }


def write_runtime_manifest(manifest: Dict[str, Any], output_dir: str, file_name: str = "runtime_manifest.json") -> str:
    """Функция: write_runtime_manifest()
Назначение: сохраняет runtime manifest в JSON-файл рядом с runtime output.
Параметры функции:
- `manifest` (`Dict[str, Any]`): сериализуемый manifest запуска.
- `output_dir` (`str`): директория, куда нужно записать manifest.
- `file_name` (`str`): имя JSON-файла manifest.
Возвращаемое значение: str: абсолютный путь к сохранённому manifest-файлу."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / file_name
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(manifest_path)
