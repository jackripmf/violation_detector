"""Файл: src/utils/media/visualizer.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- .indicators_config: используется для передачи данных или вызова связанной логики."""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

from ..config.indicators_config import IndicatorsLayout
from ...runtime.contracts import SourceVisualConfig


class Visualizer:
    """Класс: Visualizer
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `base_height` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `base_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `colors` (`dict`): словарь/палитра цветов для разных типов нарушений или объектов.
- `indicator_field_width` (`int`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `indicators_config` (`Any`): параметры конфигурации или аргументы вызова для настройки поведения.
- `scale_factor` (`float`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `show_fps` (`Any`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
Ключевые методы:
- `__init__()`, `_scale_value()`, `_draw_indicators()`, `_get_text_start_x()`, `_draw_movement_info()`, `_draw_stats_panel()`, `_draw_stat_line()`, `_draw_obstruction_info()`, `_draw_video_time()`, `_draw_fps()`"""

    def __init__(self, config: Optional[Dict] = None):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `config` (`Optional[Dict]`): структура конфигурации компонента/подсистемы.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "white": (255, 255, 255),
            "blue": (255, 0, 0),
            "purple": (255, 0, 255),
            "cyan": (255, 255, 0),
            "dark_green": (0, 100, 0),
            "dark_blue": (139, 0, 0),
            "dark_red": (0, 0, 139),
        }
        
        # Конфигурация
        self.show_fps = config.get("show_fps", False) if config else False
        self.base_width = config.get("base_width", 1280) if config else 1280
        self.base_height = config.get("base_height", 720) if config else 720
        self.show_stats_panel = config.get("show_stats_panel", True) if config else True
        self.show_movement_arrow = config.get("show_movement_arrow", True) if config else True
        self.show_violation_labels = config.get("show_violation_labels", True) if config else True
        self.show_boxes = config.get("show_boxes", True) if config else True
        
        # Конфигурация индикаторов
        self.indicators_config = config.get("indicators", IndicatorsLayout()) if config else IndicatorsLayout()
        
        # Для масштабирования
        self.scale_factor = 1.0
        
        # Вычисляемая ширина поля индикаторов
        self.indicator_field_width = 0

    def apply_source_visual_config(self, visual_config: SourceVisualConfig) -> None:
        """Функция: apply_source_visual_config()
Назначение: применяет per-source visual config к visualizer.
Параметры функции:
- `visual_config` (`SourceVisualConfig`): конфигурация отрисовки конкретного источника.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.show_fps = bool(visual_config.show_fps)
        self.show_stats_panel = bool(visual_config.show_stats_panel)
        self.show_movement_arrow = bool(visual_config.show_movement_arrow)
        self.show_violation_labels = bool(visual_config.show_violation_labels)
        self.show_boxes = bool(visual_config.show_boxes)
        self.indicators_config.show_indicators = bool(visual_config.show_indicators)
        
    def _scale_value(self, value: float) -> float:
        """Функция: _scale_value()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `value` (`float`): обрабатываемое значение параметра до валидации/преобразования.
Возвращаемое значение: float: результат шага обработки, который используется следующим этапом пайплайна."""
        return value * self.scale_factor
        
    def _draw_indicators(self, frame: np.ndarray, result: Dict[str, Any], movement_info: Dict[str, Any]) -> int:
        """Функция: _draw_indicators()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.indicators_config.show_indicators:
            return 0
        
        visible_indicators = self.indicators_config.get_visible_indicators()
        if not visible_indicators:
            return 0
        
        # Масштабированные размеры
        indicator_size = int(self.indicators_config.indicator_size * self.scale_factor)
        spacing = int(self.indicators_config.indicator_spacing * self.scale_factor)
        left_margin = int(self.indicators_config.left_margin * self.scale_factor)
        top_margin = int(self.indicators_config.top_margin * self.scale_factor)
        
        # Вычисляем ширину поля
        field_width = left_margin + indicator_size + spacing
        self.indicator_field_width = field_width
        
        # Позиция Y
        y = top_margin
        
        for ind in visible_indicators:
            # Получаем цвет в зависимости от состояния
            is_active = False
            if ind.condition:
                is_active = ind.condition(result, movement_info)
            
            color = ind.color_active if is_active else ind.color_inactive
            
            # Рисуем квадрат
            cv2.rectangle(
                frame,
                (left_margin, y),
                (left_margin + indicator_size, y + indicator_size),
                color,
                -1
            )
            
            # Рисуем букву
            font_scale = self._scale_value(0.4)
            thickness = max(1, int(self._scale_value(1)))
            
            # Центрируем букву в квадрате
            text_size = cv2.getTextSize(ind.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = left_margin + (indicator_size - text_size[0]) // 2
            text_y = y + (indicator_size + text_size[1]) // 2
            
            cv2.putText(
                frame,
                ind.label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.colors["white"],
                thickness,
            )
            
            # Следующий индикатор ниже
            y += indicator_size + spacing
        
        return field_width
    
    def _get_text_start_x(self) -> int:
        """Функция: _get_text_start_x()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.indicator_field_width > 0:
            return self.indicator_field_width + int(10 * self.scale_factor)
        return 10  # Базовый отступ если нет индикаторов
    
    def _draw_movement_info(self, frame: np.ndarray, movement_info: Dict[str, Any]) -> None:
        """Функция: _draw_movement_info()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `movement_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_movement_arrow and not self.show_violation_labels:
            return
        rotation = movement_info.get("rotation", 0)
        translation = movement_info.get("translation", 0)
        consecutive_frames = movement_info.get("consecutive_frames", 0)
        movement_duration = movement_info.get("movement_duration", 0)

        # Определение цвета и статуса движения
        if movement_info.get("filtered", False):
            if movement_info.get("movement_detected", False):
                move_color = self.colors["red"]
                move_status = "CAMERA MOVED!"
            else:
                if consecutive_frames > 0:
                    move_color = self.colors["orange"]
                    move_status = f"CAMERA MOVING...({consecutive_frames} frames)"
                else:
                    move_color = self.colors["green"]
                    move_status = "CAMERA STABLE"
        else:
            move_color = self.colors["yellow"]
            move_status = "MOVEMENT DETECTION..."

        # Позиция X с учётом поля индикаторов
        start_x = self._get_text_start_x()
        
        # Статус движения
        font_scale = self._scale_value(0.8)
        thickness = max(1, int(self._scale_value(2)))
        y_offset = int(30 * self.scale_factor)
        
        if self.show_violation_labels:
            cv2.putText(
                frame,
                f"MOVEMENT: {move_status}",
                (start_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                move_color,
                thickness,
            )

        # Детальная информация о движении
        y_offset += int(30 * self.scale_factor)
        info_lines = [
            f"Rotation: {rotation:+.1f}°",
            f"Translation: {translation:.3f}",
            f"Frames: {consecutive_frames}",
            f"Duration: {movement_duration:.1f}s",
        ]

        font_scale_detail = self._scale_value(0.6)
        line_spacing = int(25 * self.scale_factor)
        
        if self.show_violation_labels:
            for line in info_lines:
                cv2.putText(
                    frame,
                    line,
                    (start_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale_detail,
                    move_color,
                    max(1, int(self._scale_value(2))),
                )
                y_offset += line_spacing

        # Визуализация смещения
        if not self.show_movement_arrow:
            return
        if movement_info.get("movement_detected", False):
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            frame_h, frame_w = frame.shape[:2]
            translation_x = float(movement_info.get("translation_x", 0) or 0.0)
            translation_y = float(movement_info.get("translation_y", 0) or 0.0)
            vector_norm = float(np.hypot(translation_x, translation_y))
            if vector_norm <= 1e-6:
                return

            # Направление стрелки (нормированный вектор)
            dir_x = translation_x / vector_norm
            dir_y = translation_y / vector_norm

            # Базовая длина + рост по величине смещения, но без бесконтрольного разлета.
            base_len = max(20.0, 36.0 * self.scale_factor)
            gain_len = 260.0 * self.scale_factor
            desired_len = base_len + min(vector_norm, 2.0) * gain_len

            # Ограничение длины по границам кадра (стрелка не выходит за экран).
            edge_margin = max(8.0, 12.0 * self.scale_factor)
            max_len = float("inf")
            if abs(dir_x) > 1e-6:
                if dir_x > 0:
                    max_len = min(max_len, (frame_w - 1 - edge_margin - center_x) / dir_x)
                else:
                    max_len = min(max_len, (center_x - edge_margin) / abs(dir_x))
            if abs(dir_y) > 1e-6:
                if dir_y > 0:
                    max_len = min(max_len, (frame_h - 1 - edge_margin - center_y) / dir_y)
                else:
                    max_len = min(max_len, (center_y - edge_margin) / abs(dir_y))

            arrow_len = max(6.0, min(desired_len, max_len))
            end_x = int(center_x + dir_x * arrow_len)
            end_y = int(center_y + dir_y * arrow_len)
            end_x = int(np.clip(end_x, 0, frame_w - 1))
            end_y = int(np.clip(end_y, 0, frame_h - 1))

            # Фиксированный размер наконечника в пикселях (не растет вместе со всей стрелкой).
            head_len_px = max(8.0, 12.0 * self.scale_factor)
            tip_ratio = float(np.clip(head_len_px / max(arrow_len, 1.0), 0.02, 0.45))

            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (end_x, end_y),
                self.colors["yellow"],
                max(2, int(self._scale_value(3))),
                tipLength=tip_ratio,
            )

    def _draw_stats_panel(self, frame: np.ndarray, stats: Dict[str, Any], movement_info: Dict[str, Any]) -> None:
        """Функция: _draw_stats_panel()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `stats` (`Dict[str, Any]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `movement_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_stats_panel:
            return
        right_x = frame.shape[1] - int(320 * self.scale_factor)
        y_offset = int(30 * self.scale_factor)
        line_height = int(25 * self.scale_factor)

        # Увеличиваем высоту панели
        panel_width = int(310 * self.scale_factor)
        panel_height = int(200 * self.scale_factor)
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (right_x - 10, y_offset - 10),
            (right_x + panel_width, y_offset + panel_height),
            (0, 0, 0),
            -1
        )

        # Наложение с прозрачностью
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Заголовок панели
        font_scale_title = self._scale_value(0.7)
        cv2.putText(
            frame,
            "VIOLATION STATISTICS",
            (right_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_title,
            self.colors["cyan"],
            max(1, int(self._scale_value(2))),
        )

        y_offset += line_height

        # 1. Статистика перекрытий
        obstruction_stats = stats.get("obstruction", {})
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Obstructions:",
            f"{obstruction_stats.get('total', 0)}",
            self.colors["orange"]
        )
        y_offset += line_height

        # Длительность текущего перекрытия
        if obstruction_stats.get("current_duration", 0) > 0:
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Current duration:",
                f"{obstruction_stats.get('current_duration', 0):.1f}s",
                self.colors["yellow"]
            )
            y_offset += line_height

        # 2. Статистика движений камеры
        movement_stats = stats.get("movement", {})
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Camera movements:",
            f"{movement_stats.get('total', 0)}",
            self.colors["purple"]
        )
        y_offset += line_height

        # Длительность текущего движения
        if movement_info.get("movement_detected", False):
            duration = movement_info.get("movement_duration", 0)
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Moving for:",
                f"{duration:.1f}s",
                self.colors["red"]
            )
            y_offset += line_height

        # 3. Статистика запрещенных объектов
        forbidden_stats = stats.get("forbidden_items", {})
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Forbidden items:",
            f"{forbidden_stats.get('total', 0)}",
            self.colors["red"]
        )
        y_offset += line_height

        # Текущие запрещенные объекты
        current_objects = forbidden_stats.get("current_objects", 0)
        if current_objects > 0:
            color = self.colors["red"] if current_objects > 0 else self.colors["green"]
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Current objects:",
                f"{current_objects}",
                color
            )
            y_offset += line_height

        # 4. Статистика DMS
        dms_stats = stats.get("dms", {})
        dms_total = dms_stats.get("total", 0)
        dms_color = self.colors["purple"] if dms_total > 0 else self.colors["white"]
        self._draw_stat_line(
            frame, right_x, y_offset,
            "DMS violations:",
            f"{dms_total}",
            dms_color
        )
        y_offset += line_height

        # Информация о кулдаунах
        active_cooldowns = forbidden_stats.get("active_cooldowns", {})
        if active_cooldowns:
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Cooldowns active:",
                f"{len(active_cooldowns)}",
                self.colors["orange"]
            )
            y_offset += line_height

            # Показываем первые 2 кулдауна
            for i, (class_name, info) in enumerate(list(active_cooldowns.items())[:2]):
                cooldown_left = info.get("cooldown_left", 0)
                self._draw_stat_line(
                    frame, right_x + 10, y_offset,
                    f"{class_name}:",
                    f"{cooldown_left:.0f}s",
                    self.colors["yellow"]
                )
                y_offset += line_height - 5

        # 5. Общая статистика
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Total frames:",
            f"{stats.get('frame_count', 0)}",
            self.colors["white"]
        )
        y_offset += line_height

        self._draw_stat_line(
            frame, right_x, y_offset,
            "Total alerts:",
            f"{stats.get('alerts_count', 0)}",
            self.colors["yellow"]
        )

    def _draw_stat_line(self, frame: np.ndarray, x: int, y: int, label: str, value: str, color: Tuple[int, int, int]) -> None:
        """Функция: _draw_stat_line()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `x` (`int`): координата X точки или левого края области.
- `y` (`int`): координата Y точки или верхнего края области.
- `label` (`str`): подпись объекта/нарушения для отображения на кадре.
- `value` (`str`): обрабатываемое значение параметра до валидации/преобразования.
- `color` (`Tuple[int, int, int]`): цвет визуализации элемента на кадре.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        font_scale_label = self._scale_value(0.5)
        font_scale_value = self._scale_value(0.6)
        value_x = x + int(180 * self.scale_factor)
        
        # Метка
        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_label,
            self.colors["white"],
            max(1, int(self._scale_value(1))),
        )

        # Значение
        cv2.putText(
            frame,
            value,
            (value_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_value,
            color,
            max(1, int(self._scale_value(2))),
        )

    def _draw_obstruction_info(self, frame: np.ndarray, result: Dict[str, Any], total_duration: float) -> None:
        """Функция: _draw_obstruction_info()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `total_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_violation_labels:
            return
        y_offset = int(180 * self.scale_factor)

        # Статус перекрытия
        obstructed = result.get("detected", False)
        status = "BLOCKED!" if obstructed else "NORMAL"
        color = self.colors["red"] if obstructed else self.colors["green"]
        
        # Позиция X с учётом поля индикаторов
        start_x = self._get_text_start_x()
        
        font_scale = self._scale_value(0.8)
        cv2.putText(
            frame,
            f"OBSTRUCTION: {status}",
            (start_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, int(self._scale_value(2))),
        )

        # Детали
        detectors_count = result.get("detectors_count", 0)
        confidence = (detectors_count / 3.0) * 100 if detectors_count > 0 else 0
        
        font_scale_detail = self._scale_value(0.6)
        line_spacing = int(25 * self.scale_factor)
        
        info_lines = [
            f"Confidence: {confidence:.0f}%",
            f"Detectors: {detectors_count}/3",
            f"Duration: {total_duration:.1f}s",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (start_x, y_offset + line_spacing + i * line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_detail,
                self.colors["white"],
                max(1, int(self._scale_value(1))),
            )

    def _draw_video_time(self, frame: np.ndarray, video_timestamp: float) -> None:
        """Функция: _draw_video_time()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_violation_labels:
            return
        # Форматируем время: MM:SS.ms
        minutes = int(video_timestamp // 60)
        seconds = int(video_timestamp % 60)
        milliseconds = int((video_timestamp * 1000) % 1000)

        time_text = f"Time: {minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        # Позиция: ПРАВЫЙ НИЖНИЙ УГОЛ, в самом низу
        font_scale = self._scale_value(0.7)
        thickness = max(1, int(self._scale_value(2)))
        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        # Время всегда в самом низу (в 10 пикселях от края)
        x = frame.shape[1] - text_size[0] - 20
        y = frame.shape[0] - 10  # В самом низу, 10px от края

        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - text_size[1] - 10),
            (x + text_size[0] + 10, y + 10),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Текст времени
        cv2.putText(
            frame,
            time_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.colors["cyan"],
            thickness,
        )

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Функция: _draw_fps()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `fps` (`float`): частота кадров (кадров/с), используемая для таймингов и видео-вывода.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_fps or fps is None:
            return

        fps_text = f"FPS: {fps:.1f}"
        font_scale = self._scale_value(0.7)
        thickness = max(1, int(self._scale_value(2)))
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        x = frame.shape[1] - text_size[0] - 20
        # FPS рисуется над временем с отступом 35 пикселей
        # Время на y=frame.shape[0]-10, высота текста ~20px, значит FPS на y=frame.shape[0]-10-35-20
        y = frame.shape[0] - 45

        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - text_size[1] - 10),
            (x + text_size[0] + 10, y + 10),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Текст FPS
        cv2.putText(
            frame,
            fps_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.colors["green"],
            thickness,
        )

    def _draw_yolo_object(self, frame: np.ndarray, obj: Dict[str, Any]) -> None:
        """Функция: _draw_yolo_object()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `obj` (`Dict[str, Any]`): тестовый или рабочий объект, над которым выполняется проверка/операция.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_boxes:
            return
        if "bbox" not in obj or len(obj["bbox"]) != 4:
            return
            
        x1, y1, x2, y2 = obj["bbox"]
        class_name = obj.get("class", "Object")
        confidence = obj.get("confidence", 0.0)
        area_ratio = obj.get("area_ratio", 0.0)

        color = self.colors["red"]
        thickness = max(1, int(self._scale_value(2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {confidence:.2f} ({area_ratio:.1%})"
        font_scale = self._scale_value(0.5)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        if self.show_violation_labels:
            cv2.rectangle(
                frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1),
                color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.colors["white"],
                max(1, int(self._scale_value(1))),
                cv2.LINE_AA,
            )

    def _draw_forbidden_object(self, frame: np.ndarray, obj: Dict[str, Any]) -> None:
        """Функция: _draw_forbidden_object()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `obj` (`Dict[str, Any]`): тестовый или рабочий объект, над которым выполняется проверка/операция.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_boxes:
            return
        if "bbox" not in obj or len(obj["bbox"]) != 4:
            return

        x1, y1, x2, y2 = obj["bbox"]
        class_name = obj.get("class", "Forbidden")

        # Цвет для запрещенных объектов
        color = self.colors["purple"]
        thickness = max(1, int(self._scale_value(2)))

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Подпись
        label = f"{class_name}"
        font_scale = self._scale_value(0.5)
        label_y = y1 - 10 if y1 > 20 else y1 + 20
        if self.show_violation_labels:
            cv2.putText(
                frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    def _draw_dms_object(self, frame: np.ndarray, obj: Dict[str, Any]) -> None:
        """Функция: _draw_dms_object()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `obj` (`Dict[str, Any]`): тестовый или рабочий объект, над которым выполняется проверка/операция.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_boxes:
            return
        if "bbox" not in obj or len(obj["bbox"]) != 4:
            return

        x1, y1, x2, y2 = obj["bbox"]
        conf = float(obj.get("confidence", 0.0))
        kind = str(obj.get("dms_kind", obj.get("class", "other"))).lower()

        # Цвет и толщина в зависимости от типа
        if kind == "open_eye":
            color = self.colors["cyan"]
            thickness = max(1, int(self._scale_value(2)))
            label = f"OPEN_EYE {conf:.2f}"
        elif kind == "closed_eye":
            color = self.colors["red"]
            thickness = max(1, int(self._scale_value(2)))
            label = f"CLOSED_EYE {conf:.2f}"
        elif kind == "seatbelt":
            color = self.colors["green"]
            thickness = max(1, int(self._scale_value(2)))
            label = f"SEATBELT {conf:.2f}"
        elif kind == "phone":
            color = self.colors["purple"]
            thickness = max(1, int(self._scale_value(3)))
            label = f"PHONE {conf:.2f}"
        elif kind == "cigarette":
            color = self.colors["orange"]
            thickness = max(1, int(self._scale_value(3)))
            label = f"CIGARETTE {conf:.2f}"
        else:
            color = self.colors["yellow"]
            thickness = max(1, int(self._scale_value(2)))
            label = f"OTHER {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Подпись
        font_scale = self._scale_value(0.5)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, int(self._scale_value(1))))[0]
        if self.show_violation_labels:
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.colors["white"],
                max(1, int(self._scale_value(1))),
                cv2.LINE_AA,
            )

    def draw(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        movement_info: Dict[str, Any],
        video_timestamp: float,
        total_duration: float = 0.0,
        fps: Optional[float] = None
    ) -> np.ndarray:
        """Функция: draw()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`np.ndarray`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `result` (`Dict[str, Any]`): структурированный результат обработки текущего шага.
- `movement_info` (`Dict[str, Any]`): данные детектора нарушений, используемые для итогового решения по кадру.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `total_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `fps` (`Optional[float]`): частота кадров (кадров/с), используемая для таймингов и видео-вывода.
Возвращаемое значение: np.ndarray: результат шага обработки, который используется следующим этапом пайплайна."""
        # Масштабирование под размер кадра
        self.scale_factor = frame.shape[1] / self.base_width
        self.scale_factor = max(self.scale_factor, frame.shape[0] / self.base_height)
        
        display_frame = frame.copy()
        
        # Сброс ширины поля индикаторов
        self.indicator_field_width = 0

        # 1. Индикаторы статуса (ЛЕВЫЙ ВЕРХНИЙ УГОЛ) - рисуем ПЕРВЫМИ
        #    Вычисляет self.indicator_field_width для выравнивания текста
        self._draw_indicators(display_frame, result, movement_info)

        # 2. Движение камеры (ВЕРХНИЙ ЛЕВЫЙ УГОЛ, справа от индикаторов)
        self._draw_movement_info(display_frame, movement_info)

        # 3. Статистика (ПРАВЫЙ ВЕРХНИЙ УГОЛ)
        stats = result.get("stats", {})
        self._draw_stats_panel(display_frame, stats, movement_info)

        # 4. Обструкция (ЦЕНТР СЛЕВА, справа от индикаторов)
        self._draw_obstruction_info(display_frame, result, total_duration)

        # 5. FPS (ПРАВЫЙ НИЖНИЙ УГОЛ, над временем)
        self._draw_fps(display_frame, fps)

        # 6. Время видео (ПРАВЫЙ НИЖНИЙ УГОЛ, в самом низу)
        self._draw_video_time(display_frame, video_timestamp)

        # 7. Запрещенные объекты
        for obj in result.get("forbidden_objects", []):
            self._draw_forbidden_object(display_frame, obj)

        # 8. DMS объекты
        for obj in result.get("dms_objects", []):
            self._draw_dms_object(display_frame, obj)

        # 9. YOLO объекты (обструкция)
        for obj in result.get("yolo_objects", []):
            self._draw_yolo_object(display_frame, obj)

        return display_frame
