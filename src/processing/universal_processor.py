"""Файл: src/processing/universal_processor.py
Тип: слой оркестрации обработки видеопотока.
Назначение: связывает источники кадров, детекторы, статистику, сохранение нарушений и runtime-управление.
Связи: взаимодействует с detector/inference/utils модулями через менеджеры и runtime-контракты.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- ..inference.inference_hub: используется для передачи данных или вызова связанной логики.
- ..utils.indicators_config: используется для передачи данных или вызова связанной логики.
- ..utils.utils: используется для передачи данных или вызова связанной логики.
- ..utils.visualizer: используется для передачи данных или вызова связанной логики.
- .base_processor: используется для передачи данных или вызова связанной логики.
- .detection_manager: используется для передачи данных или вызова связанной логики."""

import os
import sys
import time
import logging
from typing import Any, Optional, Tuple, Dict
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

from .base_processor import BaseProcessor, ProcessorConfig, ViolationResult
from .source_manager import SourceManager
from .detection_manager import DetectionManager
from .violation_manager import ViolationManager
from .preview_manager import PreviewManager
from .stats_manager import StatsManager

from ..inference.inference_hub import InferenceHub
from ..utils.media.visualizer import Visualizer
from ..utils.config.indicators_config import IndicatorsLayout
from ..utils.common.utils import log_summary_block


class UniversalProcessor(BaseProcessor):
    """Класс: UniversalProcessor
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `_profile_accum` (`Any`): параметр политики планирования/деградации под нагрузкой.
- `_profile_counts` (`Any`): метрика или счетчик, применяемый для статистики и контроля выполнения.
- `_profile_every_n` (`int`): параметр политики планирования/деградации под нагрузкой.
- `_profile_frame_idx` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `async_violation_writes` (`Any`): данные детектора нарушений, используемые для итогового решения по кадру.
- `camera_id` (`Optional[int]`): идентификатор/индекс для адресации и сопоставления сущностей.
- `fps_enabled` (`bool`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `preview_width` (`Any`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `show_fps` (`bool`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
- `writer_overflow_strategy` (`Any`): объект подсистемы, через который вызывается профильная логика этого этапа.
- `writer_queue_max_size` (`Any`): очередь для передачи данных между асинхронными этапами пайплайна.
Примечание: класс используется в критичном контуре обработки, поэтому его поля состояния влияют на стабильность runtime.
Ключевые методы:
- `__init__()`, `_init_managers()`, `process_source()`, `process_frame()`, `_update_stats()`, `_visualize_frame()`, `_sync_file_playback()`, `_show_frame()`, `_print_profile()`, `cleanup()`"""

    def __init__(
        self,
        input_source: Any,
        camera_id: Optional[int] = None,
        save_dir: str = "violations",
        enabled_detectors: list = None,
        hub: InferenceHub = None,
        device: str = "cuda:0",
        use_half: bool = True,
        imgsz: int = 720,
        show_fps: bool = False,
        preview_width: Optional[int] = None,
        async_violation_writes: bool = True,
        writer_queue_max_size: int = 256,
        writer_overflow_strategy: str = "drop_newest",
    ):
        """Функция: __init__()
Назначение: инициализирует объект, подготавливает поля и стартовое состояние.
Параметры функции:
- `input_source` (`Any`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
- `camera_id` (`Optional[int]`): идентификатор/индекс для адресации и сопоставления сущностей.
- `save_dir` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `enabled_detectors` (`list`): список активных детекторов, участвующих в обработке кадра.
- `hub` (`InferenceHub`): общий объект инференса, который кэширует модели и выполняет predict.
- `device` (`str`): вычислительное устройство для инференса (`cpu`, `cuda`, `cuda:N`).
- `use_half` (`bool`): флаг включения FP16 на CUDA для ускорения инференса.
- `imgsz` (`int`): размер входного изображения для инференса модели.
- `show_fps` (`bool`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
- `preview_width` (`Optional[int]`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `async_violation_writes` (`bool`): данные детектора нарушений, используемые для итогового решения по кадру.
- `writer_queue_max_size` (`int`): очередь для передачи данных между асинхронными этапами пайплайна.
- `writer_overflow_strategy` (`str`): объект подсистемы, через который вызывается профильная логика этого этапа.
Возвращаемое значение: None: конструктор выполняет только инициализацию объекта."""
        # Конфигурация
        config = ProcessorConfig(
            enabled_detectors=enabled_detectors or ["all"],
            save_dir=save_dir,
            device=device,
            use_half=use_half,
            imgsz=imgsz,
        )

        super().__init__(config)

        self.input_source = input_source
        self.camera_id = camera_id
        self.show_fps = show_fps
        self.fps_enabled = show_fps
        
        self.preview_width = preview_width if preview_width is not None else (imgsz * 16 // 9)
        self.async_violation_writes = bool(async_violation_writes)
        self.writer_queue_max_size = int(writer_queue_max_size)
        self.writer_overflow_strategy = str(writer_overflow_strategy)

        # Инициализация менеджеров
        self._init_managers(hub)

        # Профилирование
        self._profile_every_n = 30
        self._profile_frame_idx = 0
        self._profile_accum = defaultdict(float)
        self._profile_counts = defaultdict(int)

        logging.info("=" * 40)
        logging.info("[runtime:start] Violation Detection System")
        logging.info("=" * 40)
        logging.info(f"[runtime:detectors] enabled={config.enabled_detectors}")
        logging.info(f"[runtime:preview] show_fps={self.show_fps} width={self.preview_width}")

    def _init_managers(self, hub: Optional[InferenceHub]) -> None:
        """Функция: _init_managers()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `hub` (`Optional[InferenceHub]`): общий объект инференса, который кэширует модели и выполняет predict.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        # InferenceHub
        if hub is None:
            hub = InferenceHub(
                device=self.config.device,
                use_half=self.config.use_half,
                imgsz=self.config.imgsz,
            )
        self.hub = hub

        # Source Manager
        self.source_manager = SourceManager()

        # Detection Manager
        self.detection_manager = DetectionManager(
            hub=self.hub,
            enabled_detectors=self.config.enabled_detectors,
        )

        # Конфигурация индикаторов с функциями проверки состояния
        indicators_config = IndicatorsLayout()
        
        # Функции проверки состояния для каждого индикатора
        def check_obstruction(result: Dict, movement: Dict) -> bool:
            """Функция: check_obstruction()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict`): структурированный результат обработки текущего шага.
- `movement` (`Dict`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return result.get("detected", False)
        
        def check_movement(result: Dict, movement: Dict) -> bool:
            """Функция: check_movement()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict`): структурированный результат обработки текущего шага.
- `movement` (`Dict`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return movement.get("movement_detected", False)
        
        def check_forbidden(result: Dict, movement: Dict) -> bool:
            """Функция: check_forbidden()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict`): структурированный результат обработки текущего шага.
- `movement` (`Dict`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return len(result.get("forbidden_objects", [])) > 0
        
        def check_dms(result: Dict, movement: Dict) -> bool:
            """Функция: check_dms()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict`): структурированный результат обработки текущего шага.
- `movement` (`Dict`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return len(result.get("dms_violations", [])) > 0
        
        def check_cigarette(result: Dict, movement: Dict) -> bool:
            """Функция: check_cigarette()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict`): структурированный результат обработки текущего шага.
- `movement` (`Dict`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return any(obj.get("class", "").lower() == "cigarette" for obj in result.get("dms_objects", []))
        
        def check_phone(result: Dict, movement: Dict) -> bool:
            """Функция: check_phone()
Назначение: выполняет проверку условия и возвращает ее результат.
Параметры функции:
- `result` (`Dict`): структурированный результат обработки текущего шага.
- `movement` (`Dict`): результат детектора движения/смещения камеры.
Возвращаемое значение: bool: логический/статусный результат проверки условия."""
            return any(obj.get("class", "").lower() == "phone" for obj in result.get("dms_objects", []))
        
        # Устанавливаем функции проверки
        indicators_config.indicators["obstruction"].condition = check_obstruction
        indicators_config.indicators["movement"].condition = check_movement
        indicators_config.indicators["forbidden"].condition = check_forbidden
        indicators_config.indicators["dms"].condition = check_dms
        indicators_config.indicators["cigarette"].condition = check_cigarette
        indicators_config.indicators["phone"].condition = check_phone

        # Visualizer с конфигурацией
        # base_width/base_height фиксированные (1280x720) для правильного масштабирования текста
        # Визуализатор будет рисовать в "базовом" размере, потом кадр ресайзится до preview_width
        viz_config = {
            "show_fps": self.show_fps,
            "base_width": 1280,
            "base_height": 720,
            "indicators": indicators_config,
        }
        self.visualizer = Visualizer(config=viz_config)

        # Violation Manager
        self.violation_manager = ViolationManager(
            save_dir=self.config.save_dir,
            visualizer=self.visualizer,
            async_writes=self.async_violation_writes,
            writer_queue_max_size=self.writer_queue_max_size,
            writer_overflow_strategy=self.writer_overflow_strategy,
        )

        # Stats Manager
        self.stats_manager = StatsManager()

        # Preview Manager (будет инициализирован после открытия источника)
        self.preview_manager: Optional[PreviewManager] = None
    
    def process_source(self, show_preview: bool = True, max_duration: float = None) -> None:
        """Функция: process_source()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `show_preview` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `max_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Примечание: функция входит в основной путь выполнения и влияет на производительность/устойчивость обработки.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        # Открытие источника
        if not self.source_manager.open_source(self.input_source, camera_id=self.camera_id):
            logging.error("[runtime:source_open_failed]")
            return
        
        # Инициализация Preview Manager
        self.preview_manager = PreviewManager(
            file_manager=self.violation_manager.file_manager,
            fps=self.source_manager.fps,
        )
        self.violation_manager.set_source_fps(self.source_manager.fps)
        
        # Начало сессии
        self.stats_manager.start_session()
        self.is_running = True
        
        start_time_wall = time.time()
        start_time_perf = time.perf_counter()
        file_playback_started_monotonic = time.monotonic()
        
        prev_frame_time = start_time_wall
        fps_buffer = []
        current_fps = self.source_manager.fps
        
        try:
            while self.is_running:
                # Проверка максимальной длительности
                if max_duration and (time.time() - start_time_wall) > max_duration:
                    logging.info(f"[runtime:max_duration_reached] seconds={max_duration}")
                    break

                # Для видеофайла держим скорость предпросмотра/обработки в нативном FPS.
                self._sync_file_playback(file_playback_started_monotonic)
                
                # Захват кадра
                t_capture = time.perf_counter()
                ret, frame = self.source_manager.read_frame()
                self._profile_accum["capture"] += time.perf_counter() - t_capture
                self._profile_counts["capture"] += 1
                
                if not ret:
                    logging.info("[runtime:source_finished]")
                    break
                
                current_time = time.time()
                video_timestamp = self.source_manager.get_timestamp(
                    self.frame_count, start_time_wall
                )
                
                # Расчет FPS (только если включено)
                current_fps = None
                if self.fps_enabled:
                    frame_time = current_time - prev_frame_time
                    prev_frame_time = current_time
                    if frame_time > 0:
                        fps_buffer.append(1.0 / frame_time)

                    # Обновление FPS
                    if fps_buffer and (current_time - start_time_wall) % 0.5 < 0.02:
                        current_fps = sum(fps_buffer[-30:]) / min(len(fps_buffer), 30)
                        fps_buffer = fps_buffer[-30:]
                    
                    # Если FPS ещё не вычислен, но есть данные в буфере
                    if current_fps is None and fps_buffer:
                        current_fps = sum(fps_buffer[-5:]) / min(len(fps_buffer), 5)
                else:
                    # Просто обновляем prev_frame_time для следующего кадра
                    prev_frame_time = current_time
                
                # Обработка кадра
                t_process = time.perf_counter()
                result, total_duration, _ = self.process_frame(frame, video_timestamp, current_time)
                self._profile_accum["process"] += time.perf_counter() - t_process
                self._profile_counts["process"] += 1

                # Визуализация
                if show_preview:
                    t_visualize = time.perf_counter()
                    
                    # Сначала визуализация в полном размере
                    display_frame = self._visualize_frame(
                        frame, result, video_timestamp, total_duration, current_fps
                    )
                    
                    # Потом ресайз до preview_width (текст масштабируется вместе с кадром)
                    if self.preview_width is not None and self.preview_width != frame.shape[1]:
                        original_h, original_w = display_frame.shape[:2]
                        scale = self.preview_width / original_w
                        new_h = int(original_h * scale)
                        display_frame = cv2.resize(
                            display_frame,
                            (self.preview_width, new_h),
                            interpolation=cv2.INTER_AREA
                        )
                        logging.debug(
                            f"[preview:resize] from={original_w}x{original_h} "
                            f"to={display_frame.shape[1]}x{display_frame.shape[0]}"
                        )
                    
                    self._profile_accum["visualize"] += time.perf_counter() - t_visualize
                    self._profile_counts["visualize"] += 1

                    # Запись превью
                    t_preview = time.perf_counter()
                    self.preview_manager.write_frame(display_frame)
                    self._profile_accum["preview_write"] += time.perf_counter() - t_preview
                    self._profile_counts["preview_write"] += 1

                    # Показ окна
                    t_show = time.perf_counter()
                    self._show_frame(display_frame)
                    self._profile_accum["show"] += time.perf_counter() - t_show
                    self._profile_counts["show"] += 1
                    
                    # Меняем размер окна ПОСЛЕ первого кадра
                    if self.preview_width is not None and self.preview_width != frame.shape[1]:
                        scale = self.preview_width / frame.shape[1]
                        new_h = int(frame.shape[0] * scale)
                        cv2.resizeWindow("Violation Detector", self.preview_width, new_h)
                
                # Профилирование
                self._profile_frame_idx += 1
                if self._profile_frame_idx % self._profile_every_n == 0:
                    self._print_profile()
                
                # Инкремент счетчиков
                self.frame_count += 1
                self.stats_manager.increment_frames()
                
                # Лог прогресса
                if self.frame_count % 100 == 0:
                    elapsed = time.perf_counter() - start_time_perf
                    avg_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
                    logging.info(
                        f"[runtime:progress] processed_frames={self.frame_count} avg_fps={avg_fps:.2f}"
                    )
        
        except KeyboardInterrupt:
            logging.info("[runtime:interrupted] reason=keyboard_interrupt")
        except Exception as e:
            logging.error(f"[runtime:processing_failed] error={e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def process_frame(
        self,
        frame: Any,
        video_timestamp: float,
        processing_time: float
    ) -> Tuple[ViolationResult, float, Dict[str, Any]]:
        """Функция: process_frame()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `processing_time` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Примечание: функция входит в основной путь выполнения и влияет на производительность/устойчивость обработки.
Возвращаемое значение: Tuple[ViolationResult, float, Dict[str, Any]]: результат шага обработки, который используется следующим этапом пайплайна."""
        start_time = time.time()
        
        result = ViolationResult()

        # Детекция обструкции
        obstruction_result = self.detection_manager.detect_obstruction(
            frame, self.frame_count
        )
        result.obstruction = obstruction_result
        result.yolo_objects = obstruction_result.get("yolo_objects", [])

        # Детекция движения
        movement_info = self.detection_manager.detect_movement(
            frame, obstruction_result.get("detected", False), frame_count=self.frame_count
        )
        result.movement = movement_info

        # Детекция forbidden items
        forbidden_result = self.detection_manager.detect_forbidden(
            frame, self.frame_count
        )
        result.forbidden = forbidden_result
        result.forbidden_objects = forbidden_result.get("objects", [])

        # Детекция DMS
        dms_result = self.detection_manager.detect_dms(frame, self.frame_count)
        result.dms = dms_result
        result.dms_objects = dms_result.get("objects", [])

        # Обновление статистики
        self._update_stats(result)

        total_duration = time.time() - start_time

        # Обработка нарушений
        obstruction_violation = self.violation_manager.process_obstruction(
            obstruction_result, frame, video_timestamp, processing_time
        )
        if obstruction_violation is not None:
            self.stats_manager.record_obstruction_violation()

        movement_violation = self.violation_manager.process_movement(
            movement_info, frame, video_timestamp
        )
        if movement_violation is not None:
            self.stats_manager.record_movement_saved()

        if forbidden_result.get("current_violation"):
            forbidden_violation = self.violation_manager.process_forbidden(
                forbidden_result, frame, video_timestamp
            )
            if forbidden_violation is not None:
                self.stats_manager.record_forbidden_saved()

        if dms_result.get("violations"):
            dms_violation = self.violation_manager.process_dms(
                dms_result, frame, video_timestamp, processing_time, self.frame_count
            )
            if dms_violation is not None:
                self.stats_manager.record_dms_saved()

        # Проверка детекции
        result.detected = (
            obstruction_result.get("detected", False) or
            movement_info.get("movement_detected", False) or
            forbidden_result.get("current_violation", False) or
            len(dms_result.get("violations", [])) > 0
        )

        return result, total_duration, movement_info
    
    def _update_stats(self, result: ViolationResult) -> None:
        """Функция: _update_stats()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `result` (`ViolationResult`): структурированный результат обработки текущего шага.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if result.obstruction.get("detected"):
            # Вычисляем длительность обструкции
            obstruction_duration = 0.0
            if self.violation_manager.obstruction_start_time is not None:
                import time
                obstruction_duration = time.time() - self.violation_manager.obstruction_start_time
            self.stats_manager.record_obstruction(obstruction_duration)

        if result.movement.get("movement_detected"):
            self.stats_manager.record_movement(
                result.movement.get("movement_duration", 0)
            )

        if result.forbidden.get("current_violation"):
            self.stats_manager.record_forbidden()

        if result.dms.get("violations"):
            for v in result.dms["violations"]:
                self.stats_manager.record_dms_violation(v.get("type", "unknown"))

        # Обновление состояния DMS
        if self.detection_manager.dms_detector:
            dms_stats = self.detection_manager.dms_detector.stats
            self.stats_manager.update_dms_state(
                eye_state=dms_stats.get("eye_state", "unknown"),
                seatbelt_state=dms_stats.get("seatbelt_state", "unknown"),
            )
    
    def _visualize_frame(
        self,
        frame: Any,
        result: ViolationResult,
        video_timestamp: float,
        total_duration: float,
        current_fps: Optional[float] = None
    ) -> Any:
        """Функция: _visualize_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
- `result` (`ViolationResult`): структурированный результат обработки текущего шага.
- `video_timestamp` (`float`): временная метка кадра в координатах видеоисточника.
- `total_duration` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `current_fps` (`Optional[float]`): частота кадров (кадров/с), применяемая в таймингах и видео-пайплайне.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
        # Формируем статистику для визуализатора
        stats = {
            "obstruction": {
                "total": self.stats_manager.stats.obstruction_violations,
                "current_duration": self.stats_manager.stats.current_obstruction_duration,
            },
            "movement": {
                "total": self.stats_manager.stats.camera_movements,
            },
            "forbidden_items": {
                "total": self.stats_manager.stats.forbidden_items_violations,
                "current_objects": len(result.forbidden_objects),
                "active_cooldowns": {},
            },
            "dms": {
                "total": self.stats_manager.stats.dms_violations,
            },
            "frame_count": self.frame_count,
            "alerts_count": self.stats_manager.stats.saved_violations,
        }
        
        # Формируем результат для визуализатора
        viz_result = {
            "yolo_objects": result.yolo_objects,
            "forbidden_objects": result.forbidden_objects,
            "dms_objects": result.dms_objects,
            "dms_violations": result.dms.get("violations", []),
            "detected": result.obstruction.get("detected", False),
            "detectors_count": result.obstruction.get("detectors_count", 0),
            "stats": stats,
        }
        
        return self.visualizer.draw(
            frame=frame,
            result=viz_result,
            movement_info=result.movement,
            video_timestamp=video_timestamp,
            total_duration=total_duration,
            fps=current_fps
        )

    def _sync_file_playback(self, started_monotonic: float) -> None:
        """Функция: _sync_file_playback()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `started_monotonic` (`float`): временной параметр, определяющий интервалы, задержки или длительность этапа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if not bool(getattr(self.source_manager, "is_file", False)):
            return
        source_fps = float(getattr(self.source_manager, "fps", 0.0) or 0.0)
        if source_fps <= 0 or self.frame_count <= 0:
            return
        elapsed = time.monotonic() - started_monotonic
        target_elapsed = float(self.frame_count) / float(source_fps)
        sleep_for = target_elapsed - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
    
    def _show_frame(self, frame: Any) -> None:
        """Функция: _show_frame()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        import cv2

        try:
            cv2.imshow("Violation Detector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.is_running = False
                logging.info("[preview:exit_requested] key=q")
            elif key == ord("r"):
                if self.detection_manager.movement_detector:
                    self.detection_manager.movement_detector.set_reference_frame(frame)
                    logging.info("[preview:movement_reference_reset] key=r")
            elif key == ord("p"):
                cv2.waitKey(0)  # Пауза
        except cv2.error as e:
            # GUI не доступен (headless режим) или ошибка отображения
            error_msg = str(e)
            if "The function is not implemented" in error_msg or "size.width>0" in error_msg:
                # Тихо игнорируем, если GUI не доступен или кадр некорректен
                logging.debug(f"[preview:headless_or_invalid_frame] error={error_msg}")
            else:
                raise
    
    def _print_profile(self) -> None:
        """Функция: _print_profile()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        def avg_ms(name: str) -> float:
            """Функция: avg_ms()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `name` (`str`): имя сущности (класс, источник, ключ), используемое в логике маршрутизации.
Возвращаемое значение: float: результат шага обработки, который используется следующим этапом пайплайна."""
            count = self._profile_counts.get(name, 0)
            if count <= 0:
                return 0.0
            return (self._profile_accum.get(name, 0.0) / count) * 1000.0
        
        msg = (
            f"[profile:timings] avg_over_frames={self._profile_every_n} "
            f"capture={avg_ms('capture'):.2f}ms "
            f"process={avg_ms('process'):.2f}ms "
            f"visualize={avg_ms('visualize'):.2f}ms "
            f"preview_write={avg_ms('preview_write'):.2f}ms "
            f"show={avg_ms('show'):.2f}ms"
        )
        logging.info(msg)
        
        # Сброс
        self._profile_accum.clear()
        self._profile_counts.clear()
    
    def cleanup(self) -> None:
        """Функция: cleanup()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        logging.info("[runtime:cleanup_start]")
        
        # Остановка превью
        if self.preview_manager:
            self.preview_manager.cleanup()
        
        # Освобождение источника
        self.source_manager.release()
        
        # Очистка менеджеров
        if hasattr(self.violation_manager, "flush_writes"):
            try:
                self.violation_manager.flush_writes(timeout=5.0)
            except Exception:
                pass
        self.violation_manager.cleanup()
        
        # Закрытие окон (с обработкой headless режима)
        import cv2
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # GUI не доступен
            pass
        
        # Печать статистики
        log_summary_block(self.stats_manager.get_summary())
        logging.info("[runtime:cleanup_done]")
        
        self.is_running = False


# Импорты для совместимости
import cv2
