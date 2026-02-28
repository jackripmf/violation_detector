"""Файл: src/processing/violation_artifact_writer.py
Тип: вспомогательный модуль записи артефактов нарушений.
Назначение: инкапсулирует запись изображений, видео и отчетов для различных типов нарушений.
Связи: используется ViolationManager и FileManager как низкоуровневый слой persistence."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2

from ..utils.io.file_manager import FileManager


class ViolationArtifactWriter:
    """Класс: ViolationArtifactWriter
Назначение: записывает артефакты нарушений в файловую систему через FileManager.
Поля класса:
- `file_manager` (`FileManager`): файловый менеджер для генерации путей и сохранения отчетов.
Ключевые методы:
- `write_obstruction_artifacts()`, `write_movement_artifacts()`, `write_forbidden_artifacts()`, `write_dms_artifacts()`"""

    def __init__(self, file_manager: FileManager):
        """Функция: __init__()
Назначение: сохраняет зависимости writer-а.
Параметры функции:
- `file_manager` (`FileManager`): файловый менеджер проекта.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.file_manager = file_manager

    def write_obstruction_artifacts(
        self,
        annotated_frame: Any,
        img_path: str,
        violation_info: Dict[str, Any],
        img_filename: str,
    ) -> bool:
        """Функция: write_obstruction_artifacts()
Назначение: сохраняет изображение и отчет по перекрытию камеры.
Параметры функции:
- `annotated_frame` (`Any`): аннотированный кадр.
- `img_path` (`str`): целевой путь изображения.
- `violation_info` (`Dict[str, Any]`): данные нарушения для отчета.
- `img_filename` (`str`): имя сохраненного изображения.
Возвращаемое значение: bool: признак успешной записи всех артефактов."""
        if not cv2.imwrite(img_path, annotated_frame):
            logging.error(f"[obstruction:image_write_failed] path={img_path}")
            return False
        if not self.file_manager.save_violation_report(violation_info, event_type="obstruction"):
            return False
        logging.info(f"[obstruction:saved] file={img_filename}")
        return True

    def write_movement_artifacts(
        self,
        frames_data: List[Dict[str, Any]],
        video_path: str,
        video_filename: str,
        movement_info: Dict[str, Any],
        unified_info: Dict[str, Any],
        writer_fps: float,
    ) -> bool:
        """Функция: write_movement_artifacts()
Назначение: сохраняет видеофрагмент и отчеты по движению камеры.
Параметры функции:
- `frames_data` (`List[Dict[str, Any]]`): кадры сегмента движения.
- `video_path` (`str`): целевой путь видеофайла.
- `video_filename` (`str`): имя сохраненного видеофайла.
- `movement_info` (`Dict[str, Any]`): данные движения для movement-report.
- `unified_info` (`Dict[str, Any]`): унифицированный отчет нарушения.
- `writer_fps` (`float`): FPS для записи видео.
Возвращаемое значение: bool: признак успешной записи всех артефактов."""
        if not frames_data:
            return False

        first_frame = frames_data[0]["frame"]
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(writer_fps),
            (width, height),
        )
        for item in frames_data:
            writer.write(item["frame"])
        writer.release()
        if not self.file_manager.save_movement_report(
            movement_info,
            video_filename,
            movement_info.get("movement_duration", 0),
            event_type="movement",
        ):
            return False
        if not self.file_manager.save_violation_report(unified_info, event_type="movement"):
            return False
        logging.info(f"[movement:saved] file={video_filename}")
        return True

    def write_forbidden_artifacts(
        self,
        annotated_frame: Any,
        img_path: str,
        img_filename: str,
        report_info: Dict[str, Any],
        report_filename: str,
    ) -> bool:
        """Функция: write_forbidden_artifacts()
Назначение: сохраняет изображение и отчет по запрещенным предметам.
Параметры функции:
- `annotated_frame` (`Any`): аннотированный кадр.
- `img_path` (`str`): целевой путь изображения.
- `img_filename` (`str`): имя сохраненного изображения.
- `report_info` (`Dict[str, Any]`): данные нарушения для отчета.
- `report_filename` (`str`): имя отчета.
Возвращаемое значение: bool: признак успешной записи всех артефактов."""
        if not cv2.imwrite(img_path, annotated_frame):
            return False
        if not self.file_manager.save_violation_report(
            report_info,
            report_filename=report_filename,
            event_type="forbidden_items",
        ):
            return False
        logging.info(f"[forbidden:saved] file={img_filename}")
        return True

    def write_dms_artifacts(
        self,
        annotated_frame: Any,
        img_path: str,
        img_filename: str,
        violation_info: Dict[str, Any],
        report_filename: str,
    ) -> bool:
        """Функция: write_dms_artifacts()
Назначение: сохраняет изображение и отчет по DMS-нарушению.
Параметры функции:
- `annotated_frame` (`Any`): аннотированный кадр.
- `img_path` (`str`): целевой путь изображения.
- `img_filename` (`str`): имя сохраненного изображения.
- `violation_info` (`Dict[str, Any]`): данные нарушения для отчета.
- `report_filename` (`str`): имя отчета.
Возвращаемое значение: bool: признак успешной записи всех артефактов."""
        if not cv2.imwrite(img_path, annotated_frame):
            return False
        if not self.file_manager.save_violation_report(
            violation_info,
            report_filename=report_filename,
            event_type="dms",
        ):
            return False
        logging.info(f"[dms:saved] file={img_filename}")
        return True
