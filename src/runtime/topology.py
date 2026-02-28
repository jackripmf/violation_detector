"""Файл: src/runtime/topology.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_VALID_DROP_POLICIES = {"drop_oldest", "drop_newest"}
_VALID_SCHEDULER_POLICIES = {"weighted_round_robin", "lag_aware"}
_VALID_RUNTIME_PROFILES = {"balanced", "laptop", "server"}
_VALID_RUNTIME_PROFILE_REQUESTS = {"auto", *sorted(_VALID_RUNTIME_PROFILES)}


def _normalize_source_id(raw_value: str, fallback: str) -> str:
    """Функция: _normalize_source_id()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `raw_value` (`str`): исходное строковое значение до нормализации/валидации.
- `fallback` (`str`): резервное значение, используемое при ошибке разбора/валидации.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
    text = str(raw_value).strip() if raw_value is not None else ""
    if not text:
        text = fallback
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def _build_default_source_id(idx: int, camera_id: Optional[int], input_source: Optional[Any]) -> str:
    """Функция: _build_default_source_id()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `idx` (`int`): короткая форма индекса элемента в цикле.
- `camera_id` (`Optional[int]`): идентификатор/индекс для адресации и сопоставления сущностей.
- `input_source` (`Optional[Any]`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
    if camera_id is not None:
        return f"camera_{int(camera_id)}"

    source_text = str(input_source or "").strip()
    if not source_text:
        return f"source_{idx}"

    lower = source_text.lower()
    if lower.startswith(("rtsp://", "rtsps://")):
        return f"source_rtsp_{idx}"

    base_name = os.path.splitext(os.path.basename(source_text))[0] or f"source_{idx}"
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in base_name)
    return f"source_{safe_name}"


@dataclass(frozen=True)
class RuntimeProfile:
    """Класс: RuntimeProfile
Назначение: реализует runtime-логику и синхронизацию этапов обработки.
Поля класса:
- `dispatch_sleep_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `infer_queue_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `infer_workers` (`int`): рабочий исполнитель или их количество для параллельной обработки.
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
- `postprocess_workers` (`int`): рабочий исполнитель или их количество для параллельной обработки.
- `source_queue_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""

    name: str
    imgsz: int
    source_queue_size: int
    infer_queue_size: int
    infer_workers: int
    postprocess_workers: int
    dispatch_sleep_sec: float


@dataclass(frozen=True)
class RuntimeProfileDecision:
    """Класс: RuntimeProfileDecision
Назначение: реализует runtime-логику и синхронизацию этапов обработки.
Поля класса:
- `cpu_count` (`int`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `cuda_available` (`bool`): флаг доступности CUDA на текущей машине.
- `profile` (`RuntimeProfile`): профиль runtime/производительности, влияющий на выбор параметров.
- `requested_device` (`str`): устройство, запрошенное пользователем через CLI/конфиг.
- `total_ram_gb` (`Optional[float]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""

    profile: RuntimeProfile
    cpu_count: int
    total_ram_gb: Optional[float]
    cuda_available: bool
    requested_device: str


RUNTIME_PROFILES: Dict[str, RuntimeProfile] = {
    "balanced": RuntimeProfile(
        name="balanced",
        imgsz=720,
        source_queue_size=8,
        infer_queue_size=64,
        infer_workers=1,
        postprocess_workers=1,
        dispatch_sleep_sec=0.003,
    ),
    "laptop": RuntimeProfile(
        name="laptop",
        imgsz=640,
        source_queue_size=4,
        infer_queue_size=24,
        infer_workers=1,
        postprocess_workers=1,
        dispatch_sleep_sec=0.005,
    ),
    "server": RuntimeProfile(
        name="server",
        imgsz=960,
        source_queue_size=16,
        infer_queue_size=256,
        infer_workers=1,
        postprocess_workers=2,
        dispatch_sleep_sec=0.0015,
    ),
}


def get_runtime_profile(profile_name: str) -> RuntimeProfile:
    """Функция: get_runtime_profile()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `profile_name` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
Возвращаемое значение: RuntimeProfile: результат шага обработки, который используется следующим этапом пайплайна."""
    name = str(profile_name or "balanced").strip().lower()
    if name not in _VALID_RUNTIME_PROFILES:
        raise ValueError(f"runtime profile must be one of {sorted(_VALID_RUNTIME_PROFILES)}")
    return RUNTIME_PROFILES[name]


def get_runtime_profile_request(profile_name: str) -> str:
    """Функция: get_runtime_profile_request()
Назначение: нормализует пользовательский запрос runtime profile, включая режим `auto`.
Параметры функции:
- `profile_name` (`str`): исходное значение runtime profile из CLI или конфига.
Возвращаемое значение: str: нормализованное имя профиля или `auto`."""
    name = str(profile_name or "auto").strip().lower()
    if name not in _VALID_RUNTIME_PROFILE_REQUESTS:
        raise ValueError(f"runtime profile must be one of {sorted(_VALID_RUNTIME_PROFILE_REQUESTS)}")
    return name


def _detect_total_ram_gb() -> Optional[float]:
    """Функция: _detect_total_ram_gb()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Optional[float]: результат шага обработки, который используется следующим этапом пайплайна."""
    # POSIX путь (Linux)
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if int(pages) > 0 and int(page_size) > 0:
                return (int(pages) * int(page_size)) / (1024.0 ** 3)
        except Exception:
            pass

    # Windows путь
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            """Класс: MEMORYSTATUSEX
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `_fields_` (`list`): служебный перечень полей именованного кортежа/структуры.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):  # type: ignore[attr-defined]
            return float(memory_status.ullTotalPhys) / (1024.0 ** 3)
    except Exception:
        pass

    return None


def _detect_cuda_available(requested_device: Optional[str]) -> bool:
    """Функция: _detect_cuda_available()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `requested_device` (`Optional[str]`): устройство, запрошенное пользователем через CLI/конфиг.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
    device_text = str(requested_device or "cuda:0").strip().lower()
    if device_text == "cpu":
        return False
    if "cuda" not in device_text:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def auto_select_runtime_profile(
    requested_device: Optional[str],
    cpu_count: Optional[int] = None,
    total_ram_gb: Optional[float] = None,
    cuda_available: Optional[bool] = None,
) -> RuntimeProfileDecision:
    """Функция: auto_select_runtime_profile()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `requested_device` (`Optional[str]`): устройство, запрошенное пользователем через CLI/конфиг.
- `cpu_count` (`Optional[int]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `total_ram_gb` (`Optional[float]`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `cuda_available` (`Optional[bool]`): флаг доступности CUDA на текущей машине.
Возвращаемое значение: RuntimeProfileDecision: результат шага обработки, который используется следующим этапом пайплайна."""
    cpu = int(cpu_count) if cpu_count is not None else int(os.cpu_count() or 4)
    ram_gb = float(total_ram_gb) if total_ram_gb is not None else _detect_total_ram_gb()
    cuda = bool(cuda_available) if cuda_available is not None else _detect_cuda_available(requested_device)

    # Эвристика:
    # - server: CUDA + много CPU + достаточный RAM
    # - laptop: мало CPU/RAM или CPU-only
    # - balanced: промежуточный вариант
    if cuda and cpu >= 12 and (ram_gb is None or ram_gb >= 24.0):
        profile = RUNTIME_PROFILES["server"]
    elif cpu <= 6 or (ram_gb is not None and ram_gb <= 10.0):
        profile = RUNTIME_PROFILES["laptop"]
    else:
        profile = RUNTIME_PROFILES["balanced"]

    return RuntimeProfileDecision(
        profile=profile,
        cpu_count=cpu,
        total_ram_gb=ram_gb,
        cuda_available=cuda,
        requested_device=str(requested_device or "cuda:0"),
    )


def resolve_runtime_profile(
    requested_profile: Optional[str],
    requested_device: Optional[str],
    cpu_count: Optional[int] = None,
    total_ram_gb: Optional[float] = None,
    cuda_available: Optional[bool] = None,
) -> RuntimeProfileDecision:
    """Функция: resolve_runtime_profile()
Назначение: выбирает runtime profile либо явно, либо через автодетект по ресурсам машины.
Параметры функции:
- `requested_profile` (`Optional[str]`): запрос пользователя на профиль (`auto|balanced|laptop|server`).
- `requested_device` (`Optional[str]`): устройство, запрошенное пользователем через CLI/конфиг.
- `cpu_count` (`Optional[int]`): число CPU для тестов или ручной подстановки.
- `total_ram_gb` (`Optional[float]`): объём RAM в GB для тестов или ручной подстановки.
- `cuda_available` (`Optional[bool]`): флаг CUDA для тестов или ручной подстановки.
Возвращаемое значение: RuntimeProfileDecision: итоговое решение по runtime profile."""
    profile_request = get_runtime_profile_request(str(requested_profile or "auto"))
    if profile_request == "auto":
        return auto_select_runtime_profile(
            requested_device=requested_device,
            cpu_count=cpu_count,
            total_ram_gb=total_ram_gb,
            cuda_available=cuda_available,
        )

    cpu = int(cpu_count) if cpu_count is not None else int(os.cpu_count() or 4)
    ram_gb = float(total_ram_gb) if total_ram_gb is not None else _detect_total_ram_gb()
    cuda = bool(cuda_available) if cuda_available is not None else _detect_cuda_available(requested_device)
    return RuntimeProfileDecision(
        profile=get_runtime_profile(profile_request),
        cpu_count=cpu,
        total_ram_gb=ram_gb,
        cuda_available=cuda,
        requested_device=str(requested_device or "cuda:0"),
    )


def apply_profile_defaults_to_topology_payload(payload: Dict[str, Any], profile: RuntimeProfile) -> Dict[str, Any]:
    """Функция: apply_profile_defaults_to_topology_payload()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `payload` (`Dict[str, Any]`): словарь полезной нагрузки для сериализации/обмена между слоями.
- `profile` (`RuntimeProfile`): профиль runtime/производительности, влияющий на выбор параметров.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
    out = dict(payload or {})
    scheduler_raw = dict(out.get("scheduler", {}))
    scheduler_raw.setdefault("infer_queue_size", profile.infer_queue_size)
    scheduler_raw.setdefault("dispatch_sleep_sec", profile.dispatch_sleep_sec)
    out["scheduler"] = scheduler_raw

    out.setdefault("infer_workers", profile.infer_workers)
    out.setdefault("postprocess_workers", profile.postprocess_workers)

    raw_sources = out.get("sources")
    if raw_sources is None:
        raw_sources = out.get("cameras", [])
        key = "cameras"
    else:
        key = "sources"
    patched_sources = []
    for raw in raw_sources:
        source = dict(raw)
        source.setdefault("capture_queue_size", profile.source_queue_size)
        patched_sources.append(source)
    out[key] = patched_sources
    return out


@dataclass
class SourceConfig:
    """Класс: SourceConfig
Назначение: описывает структуру конфигурации и типизированные данные компонента.
Поля класса:
- `base_priority` (`float`): идентификатор/индекс для адресации и сопоставления сущностей.
- `camera_id` (`Optional[int]`): идентификатор/индекс для адресации и сопоставления сущностей.
- `capture_queue_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `drop_policy` (`str`): параметр политики планирования/деградации под нагрузкой.
- `detector_schedule` (`Dict[str, Dict[str, int]]`): runtime-расписание отдельных детекторов для source.
- `enabled_detectors` (`List[str]`): список активных детекторов, участвующих в обработке кадра.
- `input_source` (`Optional[Any]`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__post_init__()`"""

    source_id: str
    input_source: Optional[Any] = None
    camera_id: Optional[int] = None
    base_priority: float = 1.0
    capture_queue_size: int = 8
    drop_policy: str = "drop_oldest"
    detector_schedule: Dict[str, Dict[str, int]] = field(default_factory=dict)
    enabled_detectors: List[str] = field(default_factory=lambda: ["all"])

    def __post_init__(self) -> None:
        """Функция: __post_init__()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.source_id = _normalize_source_id(self.source_id, "source")

        if self.camera_id is None:
            text = str(self.input_source).strip() if self.input_source is not None else ""
            if not text:
                raise ValueError("SourceConfig requires either camera_id or input_source")
        else:
            self.camera_id = int(self.camera_id)

        self.base_priority = float(self.base_priority)
        if self.base_priority <= 0:
            raise ValueError("base_priority must be > 0")

        self.capture_queue_size = int(self.capture_queue_size)
        if self.capture_queue_size <= 0:
            raise ValueError("capture_queue_size must be > 0")

        self.drop_policy = str(self.drop_policy).strip().lower()
        if self.drop_policy not in _VALID_DROP_POLICIES:
            raise ValueError(
                f"drop_policy must be one of {sorted(_VALID_DROP_POLICIES)}"
            )

        normalized_schedule: Dict[str, Dict[str, int]] = {}
        if isinstance(self.detector_schedule, dict):
            for name, cfg in self.detector_schedule.items():
                if not isinstance(cfg, dict):
                    continue
                normalized_schedule[str(name).strip().lower()] = dict(cfg)
        self.detector_schedule = normalized_schedule
        self.enabled_detectors = [str(x) for x in (self.enabled_detectors or ["all"])]


@dataclass
class SchedulerConfig:
    """Класс: SchedulerConfig
Назначение: описывает структуру конфигурации и типизированные данные компонента.
Поля класса:
- `aging_factor` (`float`): параметр политики планирования/деградации под нагрузкой.
- `backlog_factor` (`float`): параметр политики планирования/деградации под нагрузкой.
- `dispatch_sleep_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `infer_overflow_strategy` (`str`): параметр политики планирования/деградации под нагрузкой.
- `infer_queue_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `policy` (`str`): строковый идентификатор политики планирования/дропа кадров.
- `starvation_boost` (`float`): параметр политики планирования/деградации под нагрузкой.
- `starvation_threshold_sec` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__post_init__()`"""

    policy: str = "weighted_round_robin"
    aging_factor: float = 2.0
    backlog_factor: float = 0.7
    starvation_threshold_sec: float = 1.0
    starvation_boost: float = 10.0
    dispatch_sleep_sec: float = 0.003
    infer_queue_size: int = 64
    infer_overflow_strategy: str = "drop_oldest"

    def __post_init__(self) -> None:
        """Функция: __post_init__()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.policy = str(self.policy).strip().lower()
        if self.policy not in _VALID_SCHEDULER_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(_VALID_SCHEDULER_POLICIES)}"
            )

        self.aging_factor = float(self.aging_factor)
        self.backlog_factor = float(self.backlog_factor)
        self.starvation_threshold_sec = max(0.0, float(self.starvation_threshold_sec))
        self.starvation_boost = max(0.0, float(self.starvation_boost))
        self.dispatch_sleep_sec = max(0.0005, float(self.dispatch_sleep_sec))
        self.infer_queue_size = max(1, int(self.infer_queue_size))

        self.infer_overflow_strategy = str(self.infer_overflow_strategy).strip().lower()
        if self.infer_overflow_strategy not in _VALID_DROP_POLICIES:
            raise ValueError(
                "infer_overflow_strategy must be one of "
                f"{sorted(_VALID_DROP_POLICIES)}"
            )


@dataclass
class RuntimeTopology:
    """Класс: RuntimeTopology
Назначение: реализует runtime-логику и синхронизацию этапов обработки.
Поля класса:
- `infer_workers` (`int`): рабочий исполнитель или их количество для параллельной обработки.
- `postprocess_workers` (`int`): рабочий исполнитель или их количество для параллельной обработки.
- `scheduler` (`SchedulerConfig`): объект планировщика, определяющий порядок вызова детекторов.
- `sources` (`List[SourceConfig]`): параметр источника/выхода данных, задающий направление потока обработки.
- `use_centralized_runtime` (`bool`): объект подсистемы, через который вызывается профильная логика этого этапа.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__post_init__()`, `from_dict()`"""

    sources: List[SourceConfig]
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    infer_workers: int = 1
    postprocess_workers: int = 1
    use_centralized_runtime: bool = True

    def __post_init__(self) -> None:
        """Функция: __post_init__()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.sources:
            raise ValueError("RuntimeTopology requires at least one source")

        seen_ids = set()
        for source in self.sources:
            if source.source_id in seen_ids:
                raise ValueError(f"Duplicate source_id: {source.source_id}")
            seen_ids.add(source.source_id)

        self.infer_workers = max(1, int(self.infer_workers))
        self.postprocess_workers = max(1, int(self.postprocess_workers))
        self.use_centralized_runtime = bool(self.use_centralized_runtime)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RuntimeTopology":
        """Функция: from_dict()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `payload` (`Dict[str, Any]`): словарь полезной нагрузки для сериализации/обмена между слоями.
Возвращаемое значение: "RuntimeTopology": результат шага обработки, который используется следующим этапом пайплайна."""
        if not isinstance(payload, dict):
            raise ValueError("RuntimeTopology payload must be a dict")

        raw_sources = payload.get("sources")
        if raw_sources is None:
            raw_sources = payload.get("cameras")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("Topology requires non-empty 'sources' (or 'cameras') list")

        sources: List[SourceConfig] = []
        for idx, raw in enumerate(raw_sources):
            if not isinstance(raw, dict):
                raise ValueError(f"sources[{idx}] must be an object")

            camera_id = raw.get("camera_id")
            source_id_default = _build_default_source_id(
                idx=idx,
                camera_id=camera_id,
                input_source=raw.get("input"),
            )
            source_id = _normalize_source_id(raw.get("source_id"), source_id_default)

            sources.append(
                SourceConfig(
                    source_id=source_id,
                    input_source=raw.get("input"),
                    camera_id=camera_id,
                    base_priority=raw.get("base_priority", 1.0),
                    capture_queue_size=raw.get("capture_queue_size", 8),
                    drop_policy=raw.get("drop_policy", "drop_oldest"),
                    detector_schedule=raw.get("detector_schedule", {}),
                    enabled_detectors=raw.get("detectors", raw.get("enabled_detectors", ["all"])),
                )
            )

        scheduler_raw = payload.get("scheduler", {})
        scheduler = SchedulerConfig(
            policy=scheduler_raw.get("policy", "weighted_round_robin"),
            aging_factor=scheduler_raw.get("aging_factor", 2.0),
            backlog_factor=scheduler_raw.get("backlog_factor", 0.7),
            starvation_threshold_sec=scheduler_raw.get("starvation_threshold_sec", 1.0),
            starvation_boost=scheduler_raw.get("starvation_boost", 10.0),
            dispatch_sleep_sec=scheduler_raw.get("dispatch_sleep_sec", 0.003),
            infer_queue_size=scheduler_raw.get("infer_queue_size", 64),
            infer_overflow_strategy=scheduler_raw.get("infer_overflow_strategy", "drop_oldest"),
        )

        return cls(
            sources=sources,
            scheduler=scheduler,
            infer_workers=payload.get("infer_workers", 1),
            postprocess_workers=payload.get("postprocess_workers", 1),
            use_centralized_runtime=payload.get("use_centralized_runtime", True),
        )


@dataclass
class CapturedFrame:
    """Класс: CapturedFrame
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `captured_at` (`float`): временная метка момента захвата кадра.
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `frame_index` (`int`): порядковый номер кадра внутри источника.
- `source_id` (`str`): параметр источника/выхода данных, задающий направление потока обработки.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- Методы не объявлены явно в теле класса."""

    source_id: str
    frame_index: int
    frame: Any
    video_timestamp: float
    captured_at: float
