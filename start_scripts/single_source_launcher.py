"""Файл: start_scripts/single_source_launcher.py
Тип: скрипт запуска.
Назначение: разбирает параметры запуска и инициирует нужный режим работы системы.
Связи: связывает CLI и runtime-контроллер, передавая нормализованную конфигурацию запуска.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- src.inference.inference_hub: используется для передачи данных или вызова связанной логики.
- src.runtime.controller: используется для передачи данных или вызова связанной логики.
- src.runtime.topology: используется для передачи данных или вызова связанной логики.
- src.utils.common.log_context: используется для передачи данных или вызова связанной логики.
- src.utils.common.utils: используется для передачи данных или вызова связанной логики."""
import os
import sys
import argparse
import logging
import importlib.util
import json
import signal
import threading


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

detector_aliases_path = os.path.join(project_root, "src", "utils", "config", "detector_aliases.py")
detector_aliases_spec = importlib.util.spec_from_file_location(
    "detector_aliases",
    detector_aliases_path
)
detector_aliases_module = importlib.util.module_from_spec(detector_aliases_spec)
detector_aliases_spec.loader.exec_module(detector_aliases_module)
get_cli_detector_choices = detector_aliases_module.get_cli_detector_choices
normalize_detector_list = detector_aliases_module.normalize_detector_list

from src.utils.io.source_redaction import format_path_for_logging, format_source_for_logging


def ensure_qt_fontdir() -> None:
    """Функция: ensure_qt_fontdir()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
    if os.environ.get("QT_QPA_FONTDIR"):
        return

    candidates = [
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/dejavu",
        "/usr/share/fonts/truetype/freefont",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            os.environ["QT_QPA_FONTDIR"] = candidate
            return


def resolve_runtime_paths(
    output_root: str,
    camera_id: int | None = None,
    input_source: str | None = None,
 ) -> tuple[str, str]:
    """Функция: resolve_runtime_paths()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `output_root` (`str`): корневая директория, куда сохраняются результаты запуска.
- `camera_id` (`int | None`): идентификатор/индекс для адресации и сопоставления сущностей.
- `input_source` (`str | None`): источник входного видео: путь к файлу, URL RTSP или индекс камеры.
Возвращаемое значение: tuple[str, str]: результат шага обработки, который используется следующим этапом пайплайна."""
    normalized_output_root = os.path.abspath(output_root)
    effective_log_dir = os.path.join(normalized_output_root, "logs")

    if camera_id is not None:
        camera_name = f"camera_{int(camera_id)}"
        output_dir = os.path.join(normalized_output_root, camera_name)
        log_file = os.path.join(effective_log_dir, f"{camera_name}.log")
        return output_dir, log_file

    output_dir = normalized_output_root
    source_name = "source"
    if input_source:
        source_text = str(input_source).strip()
        if source_text:
            source_name = os.path.splitext(os.path.basename(source_text))[0] or "source"
            source_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source_name)
            output_dir = os.path.join(normalized_output_root, f"source_{source_name}")
    log_file = os.path.join(effective_log_dir, f"{source_name}.log")
    return output_dir, log_file


def resolve_runtime_manifest_output_dir(
    output_root: str,
    runtime_engine: str,
    resolved_output_dir: str,
) -> str:
    """Функция: resolve_runtime_manifest_output_dir()
Назначение: определяет директорию для сохранения runtime manifest рядом с итоговым output запуска.
Параметры функции:
- `output_root` (`str`): исходная корневая директория результатов запуска.
- `runtime_engine` (`str`): выбранный runtime engine.
- `resolved_output_dir` (`str`): вычисленная директория вывода для текущего запуска.
Возвращаемое значение: str: директория, куда нужно записать runtime manifest."""
    engine_name = str(runtime_engine or "").strip().lower()
    if engine_name == "centralized":
        return os.path.abspath(output_root)
    return os.path.abspath(resolved_output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    """Функция: build_arg_parser()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: argparse.ArgumentParser: результат шага обработки, который используется следующим этапом пайплайна."""
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Violation Detection System\n"
            "Запуск детектора нарушений для одного источника или topology-конфига."
        ),
        epilog=(
            "Примеры запуска:\n"
            "  1) Камера:\n"
            "     python start_scripts/single_source_launcher.py --camera-id 0 --detectors all\n"
            "  2) Видео-файл (CPU, без preview):\n"
            "     python start_scripts/single_source_launcher.py --input /data/video.mp4 --device cpu --no-preview\n"
        ),
        formatter_class=HelpFormatter,
    )

    # Источник
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        metavar="PATH_OR_RTSP",
        help=(
            "Источник видео: путь к файлу или RTSP URL.\n"
            "Примеры: /data/cam.mp4, rtsp://user:pass@host:554/stream"
        ),
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Индекс локальной камеры OpenCV.\n"
            "Используйте вместо --input для веб-камеры. Пример: 0"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="violations",
        metavar="DIR",
        help=(
            "Корневая папка для результатов и логов.\n"
            "В centralized-режиме внутри создаются подпапки по source_id."
        ),
    )
    parser.add_argument(
        "--topology-config",
        type=str,
        default=None,
        metavar="PATH_JSON",
        help=(
            "Путь к JSON topology-конфигу для centralized runtime.\n"
            "Когда задан, источники берутся из этого файла."
        ),
    )

    # Детекторы
    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",
        default=["all"],
        choices=get_cli_detector_choices(include_all=True),
        metavar="DETECTOR",
        help=(
            "Список включенных детекторов через пробел.\n"
            "Можно использовать алиасы. Примеры:\n"
            "  --detectors all\n"
            "  --detectors cv yolo movement"
        ),
    )

    # Превью
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Отключить окно предпросмотра (удобно для серверов/headless).",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Показывать FPS в окне предпросмотра.",
    )
    parser.add_argument(
        "--runtime-engine",
        type=str,
        default="centralized",
        choices=["legacy", "centralized"],
        help=(
            "Режим выполнения:\n"
            "  legacy      - классический цикл для одного источника\n"
            "  centralized - scheduler + очереди + worker-потоки"
        ),
    )

    # Устройство и параметры инференса
    parser.add_argument(
        "--device",
        default="cuda:0",
        metavar="DEVICE",
        help=(
            "Устройство инференса для YOLO: cpu | cuda | cuda:0 | cuda:1 ...\n"
            "Для машин без GPU используйте: --device cpu"
        ),
    )
    parser.add_argument(
        "--runtime-profile",
        type=str,
        default="auto",
        choices=["auto", "balanced", "laptop", "server"],
        help=(
            "Профиль runtime-параметров.\n"
            "  auto     - выбрать по ресурсам машины\n"
            "  balanced - средний профиль\n"
            "  laptop   - профиль для CPU/слабых машин\n"
            "  server   - профиль для мощных машин"
        ),
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Отключить FP16 (half precision) на CUDA.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Размер входа модели (квадрат), например 640/704/720.\n"
            "Также влияет на размер окна preview."
        ),
    )
    parser.add_argument(
        "--source-queue-size",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Максимальный размер очереди кадров на каждый source (только centralized).\n"
            "Больше значение = меньше дропов, но выше задержка."
        ),
    )
    parser.add_argument(
        "--scheduler-infer-queue-size",
        type=int,
        default=None,
        metavar="INT",
        help="Максимальный размер общей infer-очереди scheduler (только centralized).",
    )
    parser.add_argument(
        "--source-drop-policy",
        type=str,
        default="drop_oldest",
        choices=["drop_oldest", "drop_newest"],
        help=(
            "Политика переполнения source queue (centralized):\n"
            "  drop_oldest - удалить самый старый кадр\n"
            "  drop_newest - отбросить новый кадр"
        ),
    )
    parser.add_argument(
        "--infer-workers",
        type=int,
        default=None,
        metavar="INT",
        help="Количество worker-потоков инференса (только centralized).",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=None,
        metavar="INT",
        help="Количество worker-потоков постобработки (только centralized).",
    )
    parser.add_argument(
        "--scheduler-dispatch-sleep",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Пауза scheduler при простое в секундах (float).\n"
            "Пример: 0.005"
        ),
    )

    return parser


def parse_cli_args() -> argparse.Namespace:
    """Функция: parse_cli_args()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: argparse.Namespace: результат шага обработки, который используется следующим этапом пайплайна."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.topology_config is None and args.camera_id is None:
        if args.input is None or not args.input.strip():
            parser.error(
                "Укажите источник: --input <path_or_rtsp> или --camera-id <id> "
                "(либо задайте --topology-config <path.json>)."
            )

    args.input = args.input.strip() if isinstance(args.input, str) else args.input
    args.detectors = normalize_detector_list(args.detectors)
    return args


def run_from_cli(args: argparse.Namespace) -> int:
    """Функция: run_from_cli()
Назначение: выполняет ключевой этап обработки и управляет рабочим циклом.
Параметры функции:
- `args` (`argparse.Namespace`): позиционные аргументы CLI или вызова функции до дополнительной обработки.
Примечание: функция входит в основной путь выполнения и влияет на производительность/устойчивость обработки.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
    ensure_qt_fontdir()

    from src.utils.common.utils import (
        setup_logging,
        clear_source_file_logging,
        setup_source_file_logging,
    )
    from src.utils.common.log_context import bind_log_source
    from src.utils.io.runtime_manifest import (
        build_runtime_manifest,
        summarize_topology,
        write_runtime_manifest,
    )
    from src.inference.inference_hub import InferenceHub
    from src.inference.model_path_resolver import resolve_model_path
    from src.runtime.controller import RuntimeController, RuntimeConfig
    from src.runtime.topology import (
        RuntimeTopology,
        apply_profile_defaults_to_topology_payload,
        resolve_runtime_profile,
    )

    profile_decision = resolve_runtime_profile(
        requested_profile=args.runtime_profile,
        requested_device=args.device,
    )
    runtime_profile = profile_decision.profile
    runtime_profile_mode = "auto" if str(args.runtime_profile).strip().lower() == "auto" else "manual"
    effective_imgsz = int(args.imgsz) if args.imgsz is not None else int(runtime_profile.imgsz)
    effective_source_queue_size = (
        int(args.source_queue_size)
        if args.source_queue_size is not None
        else int(runtime_profile.source_queue_size)
    )
    effective_scheduler_queue_size = (
        int(args.scheduler_infer_queue_size)
        if args.scheduler_infer_queue_size is not None
        else int(runtime_profile.infer_queue_size)
    )
    effective_infer_workers = (
        int(args.infer_workers) if args.infer_workers is not None else int(runtime_profile.infer_workers)
    )
    effective_postprocess_workers = (
        int(args.postprocess_workers)
        if args.postprocess_workers is not None
        else int(runtime_profile.postprocess_workers)
    )
    effective_dispatch_sleep = (
        float(args.scheduler_dispatch_sleep)
        if args.scheduler_dispatch_sleep is not None
        else float(runtime_profile.dispatch_sleep_sec)
    )

    resolved_output_dir, resolved_log_file = resolve_runtime_paths(
        output_root=args.output,
        camera_id=args.camera_id,
        input_source=args.input,
    )
    use_source_split_logs = bool(args.topology_config)
    setup_logging(logging.INFO, log_file=(None if use_source_split_logs else resolved_log_file))

    logging.info("=" * 40)
    logging.info("Violation Detection System")
    logging.info("=" * 40)
    logging.info(f"Normalized detectors: {args.detectors}")
    logging.info(f"Input: {format_source_for_logging(args.input)}")
    if args.camera_id is not None:
        logging.info(f"Camera mode enabled, camera id: {args.camera_id}")
    logging.info(f"Output dir: {resolved_output_dir}")
    if resolved_log_file and not use_source_split_logs:
        logging.info(f"Log file: {resolved_log_file}")
    logging.info(f"Device: {args.device}")
    ram_text = (
        f"{profile_decision.total_ram_gb:.1f}GB"
        if profile_decision.total_ram_gb is not None
        else "unknown"
    )
    logging.info(
        f"Runtime profile ({runtime_profile_mode}): {runtime_profile.name} "
        f"[device={profile_decision.requested_device}, cuda={profile_decision.cuda_available}, "
        f"cpu={profile_decision.cpu_count}, ram={ram_text}]"
    )
    logging.info(
        f"Image size: {effective_imgsz} (preview: {effective_imgsz * 16 // 9}x{effective_imgsz})"
    )
    logging.info(f"Show FPS: {args.show_fps}")
    logging.info(f"Preview: {'disabled' if args.no_preview else 'enabled'}")
    logging.info(
        f"Runtime queues/workers: source_queue={effective_source_queue_size}, "
        f"infer_queue={effective_scheduler_queue_size}, infer_workers={effective_infer_workers}, "
        f"postprocess_workers={effective_postprocess_workers}, dispatch_sleep={effective_dispatch_sleep:.4f}s"
    )

    topology = None
    source_ids: list[str] = []
    runtime_engine = args.runtime_engine
    if args.topology_config:
        with open(args.topology_config, "r", encoding="utf-8") as f:
            topology_payload = json.load(f)
        topology_payload = apply_profile_defaults_to_topology_payload(
            topology_payload,
            runtime_profile,
        )

        source_key = "sources" if "sources" in topology_payload else "cameras"
        if source_key in topology_payload:
            for source_cfg in topology_payload[source_key]:
                source_cfg["capture_queue_size"] = effective_source_queue_size
                source_cfg["drop_policy"] = args.source_drop_policy

        scheduler_raw = dict(topology_payload.get("scheduler", {}))
        scheduler_raw["infer_queue_size"] = effective_scheduler_queue_size
        scheduler_raw["dispatch_sleep_sec"] = effective_dispatch_sleep
        topology_payload["scheduler"] = scheduler_raw
        topology_payload["infer_workers"] = effective_infer_workers
        topology_payload["postprocess_workers"] = effective_postprocess_workers

        topology = RuntimeTopology.from_dict(topology_payload)
        runtime_engine = "centralized"
        logging.info(f"Topology config loaded: {format_path_for_logging(args.topology_config)}")
        logging.info(f"Topology sources: {len(topology.sources)}")
        source_ids = [str(source.source_id) for source in topology.sources]
        split_log_dir = os.path.join(os.path.abspath(args.output), "logs")
        setup_source_file_logging(log_dir=split_log_dir, source_ids=source_ids)
        logging.info(
            f"[logging:split_by_source] enabled=true log_dir={split_log_dir} sources={source_ids}"
        )
    logging.info(f"Runtime engine: {runtime_engine}")

    if use_source_split_logs and source_ids:
        topology_path_abs = os.path.abspath(args.topology_config) if args.topology_config else ""
        for source_id in source_ids:
            with bind_log_source(source_id):
                logging.info("=" * 40)
                logging.info("Violation Detection System")
                logging.info("=" * 40)
                logging.info(f"Source id: {source_id}")
                logging.info(f"Normalized detectors: {args.detectors}")
                logging.info(f"Input: {format_source_for_logging(args.input)}")
                logging.info(f"Output dir: {resolved_output_dir}")
                logging.info(f"Device: {args.device}")
                logging.info(
                    f"Runtime profile ({runtime_profile_mode}): {runtime_profile.name} "
                    f"[mode={runtime_profile_mode}, "
                    f"device={profile_decision.requested_device}, cuda={profile_decision.cuda_available}, "
                    f"cpu={profile_decision.cpu_count}, ram={ram_text}]"
                )
                logging.info(
                    f"Image size: {effective_imgsz} (preview: {effective_imgsz * 16 // 9}x{effective_imgsz})"
                )
                logging.info(f"Show FPS: {args.show_fps}")
                logging.info(f"Preview: {'disabled' if args.no_preview else 'enabled'}")
                logging.info(
                    f"Runtime queues/workers: source_queue={effective_source_queue_size}, "
                    f"infer_queue={effective_scheduler_queue_size}, infer_workers={effective_infer_workers}, "
                    f"postprocess_workers={effective_postprocess_workers}, dispatch_sleep={effective_dispatch_sleep:.4f}s"
                )
                logging.info(f"Topology config loaded: {format_path_for_logging(topology_path_abs)}")
                logging.info(f"Topology sources: {len(source_ids)}")
                logging.info(f"Runtime engine: {runtime_engine}")

    hub = InferenceHub(
        device=args.device,
        use_half=(not args.no_half),
        imgsz=effective_imgsz,
    )
    effective_save_dir = resolved_output_dir
    if runtime_engine == "centralized":
        effective_save_dir = os.path.abspath(args.output)

    manifest_output_dir = resolve_runtime_manifest_output_dir(
        output_root=args.output,
        runtime_engine=runtime_engine,
        resolved_output_dir=resolved_output_dir,
    )
    topology_summary = summarize_topology(
        topology=topology,
        input_source=args.input,
        camera_id=args.camera_id,
    )
    try:
        shared_model_path = resolve_model_path()
    except Exception:
        shared_model_path = None

    manifest = build_runtime_manifest(
        runtime_engine=runtime_engine,
        runtime_profile_name=runtime_profile.name,
        runtime_profile_mode=runtime_profile_mode,
        requested_runtime_profile=str(args.runtime_profile),
        device=args.device,
        use_half=(not args.no_half),
        imgsz=effective_imgsz,
        source_queue_size=effective_source_queue_size,
        scheduler_infer_queue_size=effective_scheduler_queue_size,
        infer_workers=effective_infer_workers,
        postprocess_workers=effective_postprocess_workers,
        scheduler_dispatch_sleep_sec=effective_dispatch_sleep,
        output_dir=os.path.abspath(args.output),
        save_dir=effective_save_dir,
        log_file=(None if use_source_split_logs else resolved_log_file),
        topology_summary=topology_summary,
        source_drop_policy=args.source_drop_policy,
        show_preview=(not args.no_preview),
        show_fps=args.show_fps,
        profile_decision=profile_decision,
        shared_model_path=shared_model_path,
    )
    manifest_path = write_runtime_manifest(manifest, manifest_output_dir)
    logging.info(f"Runtime manifest: {manifest_path}")

    config = RuntimeConfig(
        input_source=args.input,
        camera_id=args.camera_id,
        save_dir=effective_save_dir,
        enabled_detectors=args.detectors,
        hub=hub,
        device=args.device,
        use_half=(not args.no_half),
        imgsz=effective_imgsz,
        show_fps=args.show_fps,
        show_preview=(not args.no_preview),
        runtime_engine=runtime_engine,
        source_queue_max_size=effective_source_queue_size,
        source_drop_policy=args.source_drop_policy,
        scheduler_infer_queue_size=effective_scheduler_queue_size,
        infer_workers=effective_infer_workers,
        postprocess_workers=effective_postprocess_workers,
        scheduler_dispatch_sleep_sec=effective_dispatch_sleep,
        topology=topology,
    )

    controller = RuntimeController(config=config)
    stop_requested = threading.Event()
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _request_stop(signum, _frame) -> None:
        """Функция: _request_stop()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `signum` (`Any`): номер UNIX-сигнала, переданный в обработчик завершения.
- `_frame` (`Any`): текущий кадр видеопотока в формате BGR (обычно `numpy.ndarray`).
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if stop_requested.is_set():
            return
        stop_requested.set()
        sig_name = signal.Signals(signum).name if signum in {signal.SIGINT, signal.SIGTERM} else str(signum)
        logging.warning(f"[runtime:signal_received] signal={sig_name} action=graceful_stop")
        try:
            controller.stop(timeout=5.0)
        except Exception as e:
            logging.error(f"[runtime:graceful_stop_failed] signal={sig_name} error={e}")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        started = controller.start()
        if not started:
            logging.error("Failed to start runtime controller")
            return 1
        while controller.is_running():
            controller.wait(timeout=0.2)
            if stop_requested.is_set():
                break
        if stop_requested.is_set() and controller.is_running():
            controller.stop(timeout=5.0)
            controller.wait(timeout=5.0)
        state = controller.get_state()
        if state.get("last_error"):
            logging.error(f"Runtime finished with error: {state['last_error']}")
            return 1
        return 0
    except KeyboardInterrupt:
        _request_stop(signal.SIGINT, None)
        controller.wait(timeout=5.0)
        return 130
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        clear_source_file_logging()


def main() -> int:
    """Функция: main()
Назначение: служит точкой входа и запускает основной сценарий выполнения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
    args = parse_cli_args()
    return run_from_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
