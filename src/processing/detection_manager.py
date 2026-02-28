"""Файл: src/processing/detection_manager.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- ..detectors.cv_detector: используется для передачи данных или вызова связанной логики.
- ..detectors.dark_area_detector: используется для передачи данных или вызова связанной логики.
- ..detectors.dms_detector: используется для передачи данных или вызова связанной логики.
- ..detectors.forbidden_items_detector: используется для передачи данных или вызова связанной логики.
- ..detectors.movement_detector: используется для передачи данных или вызова связанной логики.
- ..detectors.yolo_detector: используется для передачи данных или вызова связанной логики."""

import logging
import copy
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..utils.config.detector_aliases import CANONICAL_DETECTORS, normalize_detector_list
from ..inference.inference_hub import InferenceHub
from ..detectors.cv_detector import CVDetector
from ..detectors.dark_area_detector import DarkAreaDetector
from ..detectors.movement_detector import CameraMovementDetector
from ..detectors.yolo_detector import YOLODetector
from ..detectors.forbidden_items_detector import ForbiddenItemsDetector
from ..detectors.dms_detector import DMSDetector
from ..utils.config.detector_schedule import DetectorScheduleConfig, DEFAULT_DETECTOR_SCHEDULE


@dataclass
class DetectorStatus:
    """Класс: DetectorStatus
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `active` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `enabled` (`bool`): флаг включения соответствующей функции/подсистемы.
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    name: str
    enabled: bool
    active: bool


@dataclass
class DetectorRuntimeState:
    """Класс: DetectorRuntimeState
Назначение: инкапсулирует алгоритм детекции и формирует стандартный результат для пайплайна.
Поля класса:
- `last_result` (`Optional[Dict[str, Any]]`): последний вычисленный результат, сохраненный между итерациями.
- `last_result_frame` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `last_run_frame` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `last_run_ms` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    last_run_frame: int = -1
    last_run_ms: float = 0.0
    last_result: Optional[Dict[str, Any]] = None
    last_result_frame: int = -1


@dataclass
class SharedYoloFrameState:
    """Класс: SharedYoloFrameState
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
- `infer_key` (`Optional[str]`): ключ инференса для маршрутизации результатов в `DetectionManager`.
- `raw_results` (`Any`): сырые выходные данные модели до доменной постобработки.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    frame_count: int = -1
    infer_key: Optional[str] = None
    raw_results: Any = None


class DetectionManager:
    """Класс: DetectionManager
Назначение: координирует подсистему и управляет ее состоянием во время обработки.
Поля класса:
- `_schedule_logged_once` (`bool`): флаг одноразового логирования расписания детекторов.
- `_schedule_state` (`Dict[str, DetectorRuntimeState]`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `_shared_yolo_state` (`Any`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `cv_detector` (`Optional[CVDetector]`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `dark_detector` (`Optional[DarkAreaDetector]`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `detector_schedule` (`Any`): расписание запуска детекторов по кадрам/интервалам.
- `dms_detector` (`Optional[DMSDetector]`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `enabled_detectors` (`Any`): список активных детекторов, участвующих в обработке кадра.
- `forbidden_detector` (`Optional[ForbiddenItemsDetector]`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `hub` (`InferenceHub`): общий объект инференса, который кэширует модели и выполняет predict.
- `movement_detector` (`Optional[CameraMovementDetector]`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `yolo_detector` (`Optional[YOLODetector]`): объект подсистемы, через который вызывается профильная логика этого этапа.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `_build_detector_schedule()`, `_log_detector_schedule()`, `_should_run_detector()`, `_get_cached_result()`, `_run_scheduled()`, `_initialize_detectors()`, `_ensure_detector_instance()`, `_get_detector_instance()`, `_is_runtime_detector_enabled()`"""
    _schedule_logged_once = False
    
    def __init__(
        self,
        hub: InferenceHub,
        enabled_detectors: List[str] = None,
        detector_schedule: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `hub` (`InferenceHub`): общий объект инференса, который кэширует модели и выполняет predict.
- `enabled_detectors` (`List[str]`): список активных детекторов, участвующих в обработке кадра.
- `detector_schedule` (`Optional[Dict[str, Dict[str, int]]]`): расписание запуска детекторов по кадрам/интервалам.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        self.hub = hub
        self.enabled_detectors = normalize_detector_list(enabled_detectors or ["all"])
        logging.info(f"[detectors:init] enabled={self.enabled_detectors}")
        self.detector_schedule = self._build_detector_schedule(detector_schedule or {})
        self._schedule_state: Dict[str, DetectorRuntimeState] = {
            name: DetectorRuntimeState() for name in self.detector_schedule
        }
        self._shared_yolo_state = SharedYoloFrameState()
        
        # Инициализация детекторов
        self.cv_detector: Optional[CVDetector] = None
        self.dark_detector: Optional[DarkAreaDetector] = None
        self.movement_detector: Optional[CameraMovementDetector] = None
        self.yolo_detector: Optional[YOLODetector] = None
        self.forbidden_detector: Optional[ForbiddenItemsDetector] = None
        self.dms_detector: Optional[DMSDetector] = None
        
        self._initialize_detectors()
        self._log_detector_schedule()

    def _build_detector_schedule(
        self,
        schedule_overrides: Dict[str, Dict[str, int]],
    ) -> Dict[str, DetectorScheduleConfig]:
        """Функция: _build_detector_schedule()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `schedule_overrides` (`Dict[str, Dict[str, int]]`): переопределения расписания детекторов относительно базовой политики.
Возвращаемое значение: Dict[str, DetectorScheduleConfig]: результат шага обработки, который используется следующим этапом пайплайна."""
        schedule: Dict[str, DetectorScheduleConfig] = {
            name: cfg for name, cfg in DEFAULT_DETECTOR_SCHEDULE.items()
        }

        for name, override in schedule_overrides.items():
            if name not in schedule:
                logging.warning(f"[detectors:schedule_override_ignored] unknown_detector={name}")
                continue
            base = schedule[name]
            schedule[name] = DetectorScheduleConfig(
                every_n_frames=max(1, int(override.get("every_n_frames", base.every_n_frames))),
                min_interval_ms=max(0, int(override.get("min_interval_ms", base.min_interval_ms))),
                priority=int(override.get("priority", base.priority)),
                result_ttl_frames=max(0, int(override.get("result_ttl_frames", base.result_ttl_frames))),
            )
        return schedule

    def _log_detector_schedule(self) -> None:
        """Функция: _log_detector_schedule()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if DetectionManager._schedule_logged_once:
            return
        schedule_chunks = []
        for name, cfg in self.detector_schedule.items():
            schedule_chunks.append(
                f"{name}(n={cfg.every_n_frames},min_ms={cfg.min_interval_ms},prio={cfg.priority},ttl={cfg.result_ttl_frames})"
            )
        logging.info(f"[detectors:schedule] {'; '.join(schedule_chunks)}")
        DetectionManager._schedule_logged_once = True

    def _should_run_detector(self, detector_name: str, frame_count: int, now_ms: float) -> bool:
        """Функция: _should_run_detector()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
- `now_ms` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        cfg = self.detector_schedule[detector_name]
        state = self._schedule_state[detector_name]

        if state.last_run_frame < 0:
            return True
        if frame_count % cfg.every_n_frames != 0:
            return False
        if cfg.min_interval_ms > 0 and (now_ms - state.last_run_ms) < cfg.min_interval_ms:
            return False
        return True

    def _get_cached_result(self, detector_name: str, frame_count: int) -> Optional[Dict[str, Any]]:
        """Функция: _get_cached_result()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Optional[Dict[str, Any]]: результат шага обработки, который используется следующим этапом пайплайна."""
        cfg = self.detector_schedule[detector_name]
        state = self._schedule_state[detector_name]

        if state.last_result is None:
            return None
        if cfg.result_ttl_frames <= 0:
            return None
        age = frame_count - state.last_result_frame
        if age <= cfg.result_ttl_frames:
            return copy.deepcopy(state.last_result)
        return None

    def _run_scheduled(
        self,
        detector_name: str,
        frame_count: int,
        run_fn: Callable[[], Dict[str, Any]],
        default_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Функция: _run_scheduled()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
- `run_fn` (`Callable[[], Dict[str, Any]]`): функция запуска рабочего этапа, передаваемая как callback.
- `default_result` (`Dict[str, Any]`): результат по умолчанию, возвращаемый при ошибке или пропуске шага.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        now_ms = time.monotonic() * 1000.0
        state = self._schedule_state[detector_name]

        if self._should_run_detector(detector_name, frame_count, now_ms):
            result = run_fn()
            state.last_run_frame = frame_count
            state.last_run_ms = now_ms
            state.last_result = copy.deepcopy(result)
            state.last_result_frame = frame_count
            return result

        cached = self._get_cached_result(detector_name, frame_count)
        if cached is not None:
            return cached
        return copy.deepcopy(default_result)
    
    def _initialize_detectors(self) -> None:
        """Функция: _initialize_detectors()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        config = self._parse_detector_config()
        for detector_name, enabled in config.items():
            if enabled:
                self._ensure_detector_instance(detector_name)

    def _ensure_detector_instance(self, detector_name: str) -> None:
        """Функция: _ensure_detector_instance()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if detector_name == "cv" and self.cv_detector is None:
            self.cv_detector = CVDetector()
            logging.info("[detectors:enabled] detector=cv")
            return
        if detector_name == "dark" and self.dark_detector is None:
            self.dark_detector = DarkAreaDetector()
            logging.info("[detectors:enabled] detector=dark")
            return
        if detector_name == "movement" and self.movement_detector is None:
            self.movement_detector = CameraMovementDetector()
            logging.info("[detectors:enabled] detector=movement")
            return
        if detector_name == "yolo" and self.yolo_detector is None:
            self.yolo_detector = YOLODetector(hub=self.hub)
            logging.info("[detectors:enabled] detector=yolo")
            return
        if detector_name == "forbidden" and self.forbidden_detector is None:
            self.forbidden_detector = ForbiddenItemsDetector(
                hub=self.hub,
                confidence_threshold=0.4,
                min_detection_duration=3.0,
                violation_cooldown=30.0,
                min_object_area_ratio=0.005,
                class_specific_cooldown=True
            )
            logging.info("[detectors:enabled] detector=forbidden")
            return
        if detector_name == "dms" and self.dms_detector is None:
            self.dms_detector = DMSDetector(
                hub=self.hub,
                confidence_threshold=0.05,
                iou_threshold=0.5
            )
            logging.info("[detectors:enabled] detector=dms")
            return

    def _get_detector_instance(self, detector_name: str) -> Optional[Any]:
        """Функция: _get_detector_instance()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
Возвращаемое значение: Optional[Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        mapping = {
            "cv": self.cv_detector,
            "dark": self.dark_detector,
            "movement": self.movement_detector,
            "yolo": self.yolo_detector,
            "forbidden": self.forbidden_detector,
            "dms": self.dms_detector,
        }
        return mapping.get(detector_name)

    def _is_runtime_detector_enabled(self, detector_name: str) -> bool:
        """Функция: _is_runtime_detector_enabled()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `detector_name` (`str`): имя детектора, к которому применяется операция управления.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        detector = self._get_detector_instance(detector_name)
        if detector is None:
            return False
        return self.is_detector_enabled(detector_name) and bool(getattr(detector, "is_active", True))

    def _shared_yolo_available(self) -> bool:
        """Функция: _shared_yolo_available()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
        return bool(
            self.hub is not None and
            hasattr(self.hub, "predict") and
            (
                self.yolo_detector is not None or
                self.forbidden_detector is not None or
                self.dms_detector is not None
            )
        )

    def _build_shared_yolo_infer_params(self) -> Optional[Dict[str, Any]]:
        """Функция: _build_shared_yolo_infer_params()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Optional[Dict[str, Any]]: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self._shared_yolo_available():
            return None

        configs = []
        for detector in (self.yolo_detector, self.forbidden_detector, self.dms_detector):
            if detector is None:
                continue
            model_key = getattr(detector, "model_key", None)
            model_path = getattr(detector, "model_path", None)
            if not model_key or not model_path:
                continue
            configs.append(
                {
                    "model_key": model_key,
                    "model_path": model_path,
                    "conf": float(getattr(detector, "conf_threshold", getattr(detector, "confidence_threshold", 0.5))),
                    "iou": float(getattr(detector, "iou_threshold", 0.5)),
                    "max_det": int(getattr(detector, "max_det", 50)),
                    "imgsz": int(getattr(detector, "imgsz", 640)),
                }
            )

        if not configs:
            return None

        model_key = configs[0]["model_key"]
        model_path = configs[0]["model_path"]
        for cfg in configs[1:]:
            if cfg["model_key"] != model_key or cfg["model_path"] != model_path:
                logging.warning(
                    "[detectors:shared_yolo_disabled] reason=model_mismatch"
                )
                return None

        return {
            "model_key": model_key,
            "weights_path": model_path,
            "conf": min(cfg["conf"] for cfg in configs),
            "iou": max(cfg["iou"] for cfg in configs),
            "max_det": max(cfg["max_det"] for cfg in configs),
            "imgsz": max(cfg["imgsz"] for cfg in configs),
        }

    def _build_shared_yolo_infer_key(self, params: Dict[str, Any]) -> str:
        """Функция: _build_shared_yolo_infer_key()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `params` (`Dict[str, Any]`): словарь параметров текущего шага обработки.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        return (
            f"{params['model_key']}|{params['weights_path']}|"
            f"{params['conf']:.4f}|{params['iou']:.4f}|"
            f"{params['max_det']}|{params['imgsz']}"
        )

    def get_shared_yolo_infer_params(self) -> Optional[Dict[str, Any]]:
        """Функция: get_shared_yolo_infer_params()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Optional[Dict[str, Any]]: результат шага обработки, который используется следующим этапом пайплайна."""
        params = self._build_shared_yolo_infer_params()
        if params is None:
            return None
        return copy.deepcopy(params)

    def build_shared_yolo_infer_key(self, params: Dict[str, Any]) -> str:
        """Функция: build_shared_yolo_infer_key()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `params` (`Dict[str, Any]`): словарь параметров текущего шага обработки.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        return self._build_shared_yolo_infer_key(params)

    def seed_shared_yolo_results(self, frame_count: int, infer_key: str, raw_results: Any) -> None:
        """Функция: seed_shared_yolo_results()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
- `infer_key` (`str`): ключ инференса для маршрутизации результатов в `DetectionManager`.
- `raw_results` (`Any`): сырые выходные данные модели до доменной постобработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self._shared_yolo_state = SharedYoloFrameState(
            frame_count=int(frame_count),
            infer_key=str(infer_key),
            raw_results=raw_results,
        )

    def _get_shared_yolo_raw_results(self, frame: Any, frame_count: int) -> Optional[Any]:
        """Функция: _get_shared_yolo_raw_results()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Optional[Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        params = self._build_shared_yolo_infer_params()
        if params is None:
            return None

        infer_key = self._build_shared_yolo_infer_key(params)
        state = self._shared_yolo_state
        if state.frame_count == frame_count and state.infer_key == infer_key:
            return state.raw_results

        raw_results = self.hub.predict(
            model_key=params["model_key"],
            weights_path=params["weights_path"],
            frame_bgr=frame,
            frame_id=frame_count,
            conf=params["conf"],
            iou=params["iou"],
            max_det=params["max_det"],
            imgsz=params["imgsz"],
            verbose=False,
            classes=None,
            cache_tag="shared_full_frame",
        )
        self._shared_yolo_state = SharedYoloFrameState(
            frame_count=frame_count,
            infer_key=infer_key,
            raw_results=raw_results,
        )
        return raw_results

    def _run_yolo_detector(self, frame: Any, frame_count: int) -> Dict[str, Any]:
        """Функция: _run_yolo_detector()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.yolo_detector is None:
            return {"detected": False, "objects": [], "metrics": {}}
        raw = self._get_shared_yolo_raw_results(frame, frame_count)
        if raw is not None and hasattr(self.yolo_detector, "postprocess_shared_results"):
            return self.yolo_detector.postprocess_shared_results(frame, raw)
        return self.yolo_detector.detect(frame, frame_id=frame_count)

    def _run_forbidden_detector(self, frame: Any, frame_count: int) -> Dict[str, Any]:
        """Функция: _run_forbidden_detector()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.forbidden_detector is None:
            return {
                "detected": False,
                "objects": [],
                "current_violation": False,
                "violation_info": None,
                "stats": {},
            }
        raw = self._get_shared_yolo_raw_results(frame, frame_count)
        if raw is not None and hasattr(self.forbidden_detector, "postprocess_shared_results"):
            return self.forbidden_detector.postprocess_shared_results(frame, raw)
        return self.forbidden_detector.detect(frame, frame_id=frame_count)

    def _run_dms_detector(self, frame: Any, frame_count: int) -> Dict[str, Any]:
        """Функция: _run_dms_detector()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self.dms_detector is None:
            return {
                "detected": False,
                "objects": [],
                "violations": [],
                "current_violations": [],
                "stats": {},
            }
        raw = self._get_shared_yolo_raw_results(frame, frame_count)
        if raw is not None and hasattr(self.dms_detector, "postprocess_shared_results"):
            return self.dms_detector.postprocess_shared_results(frame, raw, frame_id=frame_count)
        return self.dms_detector.detect(frame, frame_id=frame_count)
    
    def _parse_detector_config(self) -> Dict[str, bool]:
        """Функция: _parse_detector_config()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, bool]: результат шага обработки, который используется следующим этапом пайплайна."""
        all_detectors = list(CANONICAL_DETECTORS)
        
        if "all" in self.enabled_detectors:
            config = {d: True for d in all_detectors}
        else:
            config = {d: d in self.enabled_detectors for d in all_detectors}
        return config
    
    def detect_obstruction(self, frame: Any, frame_count: int) -> Dict[str, Any]:
        """Функция: detect_obstruction()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        result = {
            "detected": False,
            "reasons": [],
            "metrics": {},
            "detectors_count": 0,
            "yolo_objects": []
        }
        
        # CV детектор
        if self._is_runtime_detector_enabled("cv"):
            cv_result = self._run_scheduled(
                detector_name="cv",
                frame_count=frame_count,
                run_fn=lambda: self.cv_detector.detect(frame),
                default_result={"detected": False, "reasons": {}, "metrics": {}},
            )
            if cv_result.get("detected", False):
                result["detected"] = True
                result["reasons"].extend([
                    k for k, v in cv_result.get("reasons", {}).items() if v
                ])
                result["metrics"].update(cv_result.get("metrics", {}))
                result["detectors_count"] += 1
        
        # Dark детектор
        if self._is_runtime_detector_enabled("dark"):
            dark_result = self._run_scheduled(
                detector_name="dark",
                frame_count=frame_count,
                run_fn=lambda: self.dark_detector.detect(frame),
                default_result={"detected": False, "metrics": {}},
            )
            if dark_result.get("detected", False):
                result["detected"] = True
                result["reasons"].append("Dark area")
                result["metrics"].update(dark_result.get("metrics", {}))
                result["detectors_count"] += 1
        
        # YOLO детектор
        if self._is_runtime_detector_enabled("yolo"):
            yolo_result = self._run_scheduled(
                detector_name="yolo",
                frame_count=frame_count,
                run_fn=lambda: self._run_yolo_detector(frame, frame_count),
                default_result={"detected": False, "objects": [], "metrics": {}},
            )
            result["yolo_objects"] = yolo_result.get("objects", [])
            
            if yolo_result.get("detected", False):
                result["detected"] = True
                result["reasons"].append("Large object")
                result["metrics"].update(yolo_result.get("metrics", {}))
                result["detectors_count"] += 1
        
        return result
    
    def detect_movement(self, frame: Any, is_obstructed: bool, frame_count: int = 0) -> Dict[str, Any]:
        """Функция: detect_movement()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `is_obstructed` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._is_runtime_detector_enabled("movement"):
            default_result = {
                "movement_detected": False,
                "filtered": True,
                "reason": "Movement skipped by scheduler",
                "consecutive_frames": 0,
                "movement_duration": 0,
            }
            if self.movement_detector is not None and hasattr(self.movement_detector, "get_last_movement_info"):
                try:
                    default_result = self.movement_detector.get_last_movement_info()
                except Exception:
                    pass
            return self._run_scheduled(
                detector_name="movement",
                frame_count=frame_count,
                run_fn=lambda: self.movement_detector.detect(frame, is_obstructed),
                default_result=default_result,
            )
        
        return {
            "movement_detected": False,
            "filtered": True,
            "reason": "No movement (detector disabled)",
            "consecutive_frames": 0,
            "movement_duration": 0
        }
    
    def detect_forbidden(self, frame: Any, frame_count: int) -> Dict[str, Any]:
        """Функция: detect_forbidden()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._is_runtime_detector_enabled("forbidden"):
            return self._run_scheduled(
                detector_name="forbidden",
                frame_count=frame_count,
                run_fn=lambda: self._run_forbidden_detector(frame, frame_count),
                default_result={
                    "detected": False,
                    "objects": [],
                    "current_violation": False,
                    "violation_info": None,
                    "stats": {},
                },
            )
        
        return {
            "detected": False,
            "objects": [],
            "current_violation": False,
            "violation_info": None,
            "stats": {}
        }
    
    def detect_dms(self, frame: Any, frame_count: int) -> Dict[str, Any]:
        """Функция: detect_dms()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_count` (`int`): счетчик обработанных кадров для метрик и контроля цикла.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
        if self._is_runtime_detector_enabled("dms"):
            return self._run_scheduled(
                detector_name="dms",
                frame_count=frame_count,
                run_fn=lambda: self._run_dms_detector(frame, frame_count),
                default_result={
                    "detected": False,
                    "objects": [],
                    "violations": [],
                    "current_violations": [],
                    "stats": {},
                },
            )
        
        return {
            "detected": False,
            "objects": [],
            "violations": [],
            "current_violations": [],
            "stats": {}
        }
    
    def get_detector_status(self) -> List[DetectorStatus]:
        """Функция: get_detector_status()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: List[DetectorStatus]: результат шага обработки, который используется следующим этапом пайплайна."""
        status_list = []
        
        detectors = [
            ("CVDetector", self.cv_detector),
            ("DarkAreaDetector", self.dark_detector),
            ("CameraMovementDetector", self.movement_detector),
            ("YOLODetector", self.yolo_detector),
            ("ForbiddenItemsDetector", self.forbidden_detector),
            ("DMSDetector", self.dms_detector),
        ]
        
        for name, detector in detectors:
            status_list.append(DetectorStatus(
                name=name,
                enabled=detector is not None,
                active=detector.is_active if detector else False
            ))
        
        return status_list

    def get_detector_schedule_snapshot(self) -> Dict[str, Dict[str, int]]:
        """Функция: get_detector_schedule_snapshot()
Назначение: возвращает текущее runtime-расписание детекторов в сериализуемом виде.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Dict[str, Dict[str, int]]: словарь конфигурации расписания детекторов."""
        snapshot: Dict[str, Dict[str, int]] = {}
        for name, cfg in self.detector_schedule.items():
            snapshot[str(name)] = {
                "every_n_frames": int(cfg.every_n_frames),
                "min_interval_ms": int(cfg.min_interval_ms),
                "priority": int(cfg.priority),
                "result_ttl_frames": int(cfg.result_ttl_frames),
            }
        return snapshot

    def set_detector_schedule(
        self,
        detector_schedule: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Функция: set_detector_schedule()
Назначение: обновляет runtime-расписание детекторов без пересоздания manager.
Параметры функции:
- `detector_schedule` (`Optional[Dict[str, Dict[str, int]]]`): переопределения расписания детекторов.
Возвращаемое значение: Dict[str, Dict[str, Dict[str, int]]]: словарь с примененным расписанием."""
        rebuilt_schedule = self._build_detector_schedule(detector_schedule or {})
        previous_state = dict(self._schedule_state)
        self.detector_schedule = rebuilt_schedule
        self._schedule_state = {
            name: previous_state.get(name, DetectorRuntimeState())
            for name in rebuilt_schedule
        }
        return {
            "detector_schedule": self.get_detector_schedule_snapshot(),
        }
    
    def enable_detector(self, name: str) -> None:
        """Функция: enable_detector()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        canonical = normalize_detector_list([name])[0] if name else name
        if canonical in CANONICAL_DETECTORS:
            self._ensure_detector_instance(canonical)
            detector = self._get_detector_instance(canonical)
            if detector and hasattr(detector, "enable"):
                detector.enable()
        if canonical not in self.enabled_detectors:
            self.enabled_detectors.append(canonical)
        logging.info(f"[detectors:runtime_enable] detector={canonical}")
    
    def disable_detector(self, name: str) -> None:
        """Функция: disable_detector()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        canonical = normalize_detector_list([name])[0] if name else name
        detector = self._get_detector_instance(canonical) if canonical in CANONICAL_DETECTORS else None
        if detector and hasattr(detector, "disable"):
            detector.disable()
        if "all" in self.enabled_detectors and canonical in CANONICAL_DETECTORS:
            self.enabled_detectors = [d for d in CANONICAL_DETECTORS if d != canonical]
        if canonical in self.enabled_detectors:
            self.enabled_detectors.remove(canonical)
        logging.info(f"[detectors:runtime_disable] detector={canonical}")

    def set_runtime_detectors(
        self,
        enabled_detectors: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """Функция: set_runtime_detectors()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `enabled_detectors` (`Optional[List[str]]`): список активных детекторов, участвующих в обработке кадра.
Возвращаемое значение: Dict[str, List[str]]: результат шага обработки, который используется следующим этапом пайплайна."""
        if enabled_detectors is not None:
            normalized_enabled = normalize_detector_list(enabled_detectors)
            self.enabled_detectors = normalized_enabled
            if "all" in normalized_enabled:
                target_enabled = set(CANONICAL_DETECTORS)
            else:
                target_enabled = set([d for d in normalized_enabled if d in CANONICAL_DETECTORS])
            for detector_name in target_enabled:
                self._ensure_detector_instance(detector_name)
                detector = self._get_detector_instance(detector_name)
                if detector and hasattr(detector, "enable"):
                    detector.enable()

        return {
            "enabled_detectors": list(self.enabled_detectors),
        }
    
    def is_detector_enabled(self, name: str) -> bool:
        """Функция: is_detector_enabled()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
        if "all" in self.enabled_detectors:
            return True
        return name in self.enabled_detectors
