"""Файл: src/detectors/dark_area_detector.py
Тип: детектор нарушений.
Назначение: получает кадр, применяет алгоритм детекции и возвращает структурированный результат.
Связи: детекторы вызываются менеджером детекции и передают данные в слой сохранения/визуализации.
Импортируемые внутренние модули:
- .base_detector: используется для передачи данных или вызова связанной логики."""
import cv2

from .base_detector import BaseDetector


class DarkAreaDetector(BaseDetector):
    """Класс: DarkAreaDetector
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `dark_area_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `dark_pixel_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `processing_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
Ключевые методы:
- `__init__()`, `detect()`"""

    def __init__(self, dark_area_threshold=0.8, processing_width=640, dark_pixel_threshold=50):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `dark_area_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `processing_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `dark_pixel_threshold` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        super().__init__("DarkAreaDetector")
        self.dark_area_threshold = dark_area_threshold
        self.processing_width = int(processing_width)
        self.dark_pixel_threshold = int(dark_pixel_threshold)

    def detect(self, frame):
        """Функция: detect()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if self.processing_width > 0 and w > self.processing_width:
            scale = self.processing_width / float(w)
            new_h = max(1, int(h * scale))
            gray = cv2.resize(gray, (self.processing_width, new_h), interpolation=cv2.INTER_AREA)

        _, dark_mask = cv2.threshold(
            gray, self.dark_pixel_threshold, 255, cv2.THRESH_BINARY_INV
        )

        dark_pixels = cv2.countNonZero(dark_mask)
        total_pixels = dark_mask.shape[0] * dark_mask.shape[1]
        dark_ratio = dark_pixels / total_pixels

        dark_detected = dark_ratio > self.dark_area_threshold

        return {"detected": dark_detected, "metrics": {"dark_ratio": dark_ratio}}
