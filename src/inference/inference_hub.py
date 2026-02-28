"""Файл: src/inference/inference_hub.py
Тип: инференс-слой.
Назначение: управляет загрузкой моделей и выполнением предиктов с едиными параметрами.
Связи: используется детекторами и runtime-компонентами как единая точка инференса.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""
import threading
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
from ultralytics import YOLO

from .model_path_resolver import resolve_model_path
from ..utils.io.source_redaction import format_path_for_logging


@dataclass(frozen=True)
class YoloRequest:
    """Класс: YoloRequest
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `cache_tag` (`str`): строковая метка типа запроса в кэше (full-frame, ROI и т.п.).
- `classes` (`Tuple[int, ...]`): список class-id, по которым фильтруются результаты модели.
- `conf` (`float`): порог confidence для фильтрации детекций модели.
- `frame_id` (`int`): уникальный индекс кадра, используется для кэша и синхронизации этапов.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `input_shape` (`Tuple[int, int]`): форма входного изображения в виде `(height, width)` для кэш-ключа.
- `iou` (`float`): порог IoU для NMS при постобработке детекций.
- `max_det` (`int`): максимальное число детекций, возвращаемых моделью на кадр.
- `model_id` (`Tuple[str, str]`): идентификатор/индекс для адресации и сопоставления сущностей.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    model_id: Tuple[str, str]
    frame_id: int
    conf: float
    iou: float
    max_det: int
    imgsz: int
    cache_tag: str
    input_shape: Tuple[int, int]
    classes: Tuple[int, ...]

class InferenceHub:
    """Класс: InferenceHub
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `_cache` (`Dict[YoloRequest, Any]`): кэш результатов инференса по ключу запроса, чтобы не считать повторно.
- `_cache_frame_ids` (`Dict[Tuple[str, str], int]`): карта последнего `frame_id` по каждой модели для очистки кэша.
- `_cache_hits` (`int`): счетчик попаданий в кэш инференса для метрик эффективности.
- `_cache_lock` (`Any`): lock для потокобезопасного доступа к структурам кэша.
- `_global_lock` (`Any`): примитив синхронизации для безопасного доступа к общему состоянию.
- `_locks` (`Dict[str, threading.Lock]`): примитив синхронизации для безопасного доступа к общему состоянию.
- `_models` (`Dict[str, YOLO]`): кэш загруженных экземпляров моделей, доступных для повторного использования.
- `_predict_calls` (`int`): количество обращений к инференсу для метрик производительности.
- `cache_max_frames` (`Any`): глубина кэша по кадрам: сколько последних `frame_id` хранить.
- `device` (`Optional[str]`): вычислительное устройство для инференса (`cpu`, `cuda`, `cuda:N`).
- `enable_cache` (`bool`): флаг включения кэширования результатов инференса.
- `imgsz` (`Any`): размер входного изображения для инференса модели.
- `use_half` (`bool`): флаг включения FP16 на CUDA для ускорения инференса.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `get_model()`, `clear_cache()`, `reset_metrics()`, `get_metrics()`, `_prune_cache_locked()`, `predict()`"""

    def __init__(
        self,
        device: Optional[str] = None,     # "cuda:0" / "cpu" / None (авто)
        use_half: bool = True,
        imgsz: int = 640,
        enable_cache: bool = True,
        cache_max_frames: int = 4,        # держим кэш на последние N frame_id
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `device` (`Optional[str]`): вычислительное устройство для инференса (`cpu`, `cuda`, `cuda:N`).
- `use_half` (`bool`): флаг включения FP16 на CUDA для ускорения инференса.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `enable_cache` (`bool`): флаг включения кэширования результатов инференса.
- `cache_max_frames` (`int`): глубина кэша по кадрам: сколько последних `frame_id` хранить.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.device = device
        self.use_half = use_half
        self.imgsz = int(imgsz)
        self.enable_cache = enable_cache
        self.cache_max_frames = int(cache_max_frames)

        self._models: Dict[str, YOLO] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        self._cache: Dict[YoloRequest, Any] = {}
        self._cache_frame_ids: Dict[Tuple[str, str], int] = {}
        self._cache_lock = threading.Lock()
        self._predict_calls = 0
        self._cache_hits = 0

        # Лог + базовая проверка CUDA
        if self.device and "cuda" in str(self.device).lower():
            if torch.cuda.is_available():
                logging.info(f"[InferenceHub] device={self.device} (CUDA available: True)")
                try:
                    idx = int(str(self.device).split(":")[1])
                    logging.info(f"[InferenceHub] GPU name: {torch.cuda.get_device_name(idx)}")
                except Exception:
                    logging.info("[InferenceHub] GPU name: <unknown>")
            else:
                logging.warning(f"[InferenceHub] device={self.device} but CUDA available: False (will likely fall back)")

        logging.info(f"[InferenceHub] default_imgsz={self.imgsz}, use_half={self.use_half}")

    def get_model(self, model_key: str, weights_path: str) -> YOLO:
        """Функция: get_model()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `model_key` (`str`): логический ключ модели в InferenceHub для повторного использования экземпляра.
- `weights_path` (`str`): путь к весам, передаваемый в слой инференса.
Возвращаемое значение: YOLO: результат шага обработки, который используется следующим этапом пайплайна."""
        resolved_weights_path = resolve_model_path(weights_path)
        model_id = (str(model_key), resolved_weights_path)

        if model_id in self._models:
            return self._models[model_id]

        with self._global_lock:
            if model_id in self._models:
                return self._models[model_id]

            logging.info(
                f"[InferenceHub] Loading model id={model_id} from: {format_path_for_logging(resolved_weights_path)}"
            )
            model = YOLO(resolved_weights_path)

            try:
                model.model.eval()
            except Exception:
                pass

            self._models[model_id] = model
            self._locks[model_id] = threading.Lock()
            return model

    def clear_cache(self) -> None:
        """Функция: clear_cache()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_frame_ids.clear()

    def reset_metrics(self) -> None:
        """Функция: reset_metrics()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        with self._cache_lock:
            self._predict_calls = 0
            self._cache_hits = 0

    def get_metrics(self) -> Dict[str, int]:
        """Функция: get_metrics()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, int]: результат шага обработки, который используется следующим этапом пайплайна."""
        with self._cache_lock:
            return {
                "predict_calls": int(self._predict_calls),
                "cache_hits": int(self._cache_hits),
            }

    def _prune_cache_locked(self) -> None:
        """Функция: _prune_cache_locked()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self._cache:
            return

        keep_min_frame: Dict[Tuple[str, str], int] = {}
        for model_id, last_fid in self._cache_frame_ids.items():
            keep_min_frame[model_id] = last_fid - self.cache_max_frames

        to_delete = []
        for req in list(self._cache.keys()):
            min_fid = keep_min_frame.get(req.model_id, -10**18)
            if req.frame_id < min_fid:
                to_delete.append(req)

        for req in to_delete:
            self._cache.pop(req, None)

    def predict(
        self,
        *,
        model_key: str,
        weights_path: str,
        frame_bgr,
        conf: float,
        frame_id: Optional[int] = None,
        iou: float = 0.5,
        max_det: int = 50,
        imgsz: Optional[int] = None,
        verbose: bool = False,
        classes: Optional[list] = None,
        cache_tag: str = "full",
    ):
        """Функция: predict()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `model_key` (`str`): логический ключ модели в InferenceHub для повторного использования экземпляра.
- `weights_path` (`str`): путь к весам, передаваемый в слой инференса.
- `frame_bgr` (`Any`): кадр в формате BGR для передачи в модель инференса.
- `conf` (`float`): порог confidence для фильтрации детекций модели.
- `frame_id` (`Optional[int]`): уникальный индекс кадра, используется для кэша и синхронизации этапов.
- `iou` (`float`): порог IoU для NMS при постобработке детекций.
- `max_det` (`int`): максимальное число детекций, возвращаемых моделью на кадр.
- `imgsz` (`Optional[int]`): размер входного изображения для инференса модели.
- `verbose` (`bool`): флаг подробного логирования и вывода отладочной информации.
- `classes` (`Optional[list]`): список class-id, по которым фильтруются результаты модели.
- `cache_tag` (`str`): строковая метка типа запроса в кэше (full-frame, ROI и т.п.).
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        resolved_weights_path = resolve_model_path(weights_path)
        model_id = (str(model_key), resolved_weights_path)
        model = self.get_model(model_key, resolved_weights_path)

        half = bool(self.use_half and self.device is not None and "cuda" in str(self.device).lower())
        final_imgsz = int(imgsz) if imgsz is not None else int(self.imgsz)

        classes_tuple: Tuple[int, ...] = ()
        if classes is not None:
            classes_tuple = tuple(sorted({int(x) for x in classes}))

        try:
            h, w = frame_bgr.shape[:2]
            input_shape = (int(h), int(w))
        except Exception:
            input_shape = (-1, -1)

        req = None
        if self.enable_cache and frame_id is not None:
            req = YoloRequest(
                model_id=model_id,
                frame_id=int(frame_id),
                conf=float(conf),
                iou=float(iou),
                max_det=int(max_det),
                imgsz=int(final_imgsz),
                cache_tag=str(cache_tag),
                input_shape=input_shape,
                classes=classes_tuple,
            )
            with self._cache_lock:
                if req in self._cache:
                    self._cache_hits += 1
                    return self._cache[req]

        with self._cache_lock:
            self._predict_calls += 1
        lock = self._locks[model_id]
        with lock:
            with torch.inference_mode():
                results = model.predict(
                    source=frame_bgr,
                    conf=float(conf),
                    iou=float(iou),
                    max_det=int(max_det),
                    device=self.device,
                    half=half,
                    imgsz=int(final_imgsz),
                    verbose=verbose,
                    classes=list(classes_tuple) if classes is not None else None,
                )

        if self.enable_cache and frame_id is not None and req is not None:
            with self._cache_lock:
                self._cache[req] = results
                self._cache_frame_ids[model_id] = int(frame_id)
                self._prune_cache_locked()

        return results
