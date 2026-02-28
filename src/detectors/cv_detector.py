"""Файл: src/detectors/cv_detector.py
Тип: детектор нарушений.
Назначение: получает кадр, применяет алгоритм детекции и возвращает структурированный результат.
Связи: детекторы вызываются менеджером детекции и передают данные в слой сохранения/визуализации.
Импортируемые внутренние модули:
- .base_detector: используется для передачи данных или вызова связанной логики."""
import cv2

from .base_detector import BaseDetector


class CVDetector(BaseDetector):
    """Класс: CVDetector
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `brightness_thresh` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `contrast_thresh` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `processing_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `sharpness_thresh` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
Ключевые методы:
- `__init__()`, `detect()`, `_get_metrics()`"""

    def __init__(
        self,
        brightness_thresh=25,
        contrast_thresh=10,
        sharpness_thresh=20,
        processing_width=640,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `brightness_thresh` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `contrast_thresh` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `sharpness_thresh` (`Any`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `processing_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        super().__init__("CVDetector")
        self.brightness_thresh = brightness_thresh
        self.contrast_thresh = contrast_thresh
        self.sharpness_thresh = sharpness_thresh
        self.processing_width = int(processing_width)

    def detect(self, frame):
        """Функция: detect()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        brightness, contrast, sharpness = self._get_metrics(frame)
        is_dark = brightness < self.brightness_thresh
        is_low_contrast = contrast < self.contrast_thresh
        is_blurry = sharpness < self.sharpness_thresh

        cv_detected = is_dark or is_blurry or is_low_contrast

        return {
            "detected": cv_detected,
            "metrics": {
                "Brightness": brightness,
                "Contrast": contrast,
                "Sharpness": sharpness,
            },
            "reasons": {
                "Dark": is_dark,
                "Low_contrast": is_low_contrast,
                "Blurry": is_blurry,
            },
        }

    def _get_metrics(self, frame):
        """Функция: _get_metrics()
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

        mean, stddev = cv2.meanStdDev(gray)
        brightness = float(mean[0, 0])
        contrast = float(stddev[0, 0])

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness = float(lap.var())
        return brightness, contrast, sharpness
