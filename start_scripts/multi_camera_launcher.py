"""Файл: start_scripts/multi_camera_launcher.py
Тип: скрипт запуска.
Назначение: разбирает параметры запуска и инициирует нужный режим работы системы.
Связи: связывает CLI и runtime-контроллер, передавая нормализованную конфигурацию запуска.
Критичность: файл входит в ключевой runtime-контур, поэтому изменения нужно проверять тестами и запуском сценариев.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.io.source_redaction import format_command_for_logging


def _as_list(value: Any) -> List[str]:
    """Функция: _as_list()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `value` (`Any`): обрабатываемое значение параметра до валидации/преобразования.
Возвращаемое значение: List[str]: результат шага обработки, который используется следующим этапом пайплайна."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def load_config(path: str) -> Dict[str, Any]:
    """Функция: load_config()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `path` (`str`): путь к файлу или директории, участвующей в текущей операции.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object")
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Config must contain non-empty 'cameras' list")
    return data


def build_camera_command(
    python_executable: str,
    main_script: str,
    camera_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> List[str]:
    """Функция: build_camera_command()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `python_executable` (`str`): путь к интерпретатору Python для запуска дочерних процессов.
- `main_script` (`str`): путь к основному скрипту, который запускается дочерним процессом.
- `camera_cfg` (`Dict[str, Any]`): конфигурация отдельной камеры/источника в multi-camera запуске.
- `global_cfg` (`Dict[str, Any]`): глобальная часть конфигурации, общая для всех источников.
Возвращаемое значение: List[str]: результат шага обработки, который используется следующим этапом пайплайна."""
    cfg = dict(global_cfg)
    cfg.update(camera_cfg or {})

    cmd = [python_executable, main_script]
    input_source = cfg.get("input")
    camera_id = cfg.get("camera_id")

    if camera_id is None and (input_source is None or str(input_source).strip() == ""):
        raise ValueError("Each camera config must include either 'camera_id' or 'input'")

    if input_source is not None:
        cmd.extend(["--input", str(input_source)])
    if camera_id is not None:
        cmd.extend(["--camera-id", str(int(camera_id))])

    if "output" in cfg:
        cmd.extend(["--output", str(cfg["output"])])
    if "runtime_engine" in cfg:
        cmd.extend(["--runtime-engine", str(cfg["runtime_engine"])])

    if cfg.get("no_preview", False):
        cmd.append("--no-preview")
    if cfg.get("show_fps", False):
        cmd.append("--show-fps")
    if cfg.get("no_half", False):
        cmd.append("--no-half")

    if "device" in cfg:
        cmd.extend(["--device", str(cfg["device"])])
    if "imgsz" in cfg:
        cmd.extend(["--imgsz", str(int(cfg["imgsz"]))])
    if "source_queue_size" in cfg:
        cmd.extend(["--source-queue-size", str(int(cfg["source_queue_size"]))])
    if "scheduler_infer_queue_size" in cfg:
        cmd.extend(["--scheduler-infer-queue-size", str(int(cfg["scheduler_infer_queue_size"]))])
    if "source_drop_policy" in cfg:
        cmd.extend(["--source-drop-policy", str(cfg["source_drop_policy"])])
    if "infer_workers" in cfg:
        cmd.extend(["--infer-workers", str(int(cfg["infer_workers"]))])
    if "postprocess_workers" in cfg:
        cmd.extend(["--postprocess-workers", str(int(cfg["postprocess_workers"]))])
    if "scheduler_dispatch_sleep" in cfg:
        cmd.extend(["--scheduler-dispatch-sleep", str(float(cfg["scheduler_dispatch_sleep"]))])

    detectors = _as_list(cfg.get("detectors"))
    if detectors:
        cmd.append("--detectors")
        cmd.extend(detectors)

    extra_args = _as_list(cfg.get("extra_args"))
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _resolve_source_kind(cfg: Dict[str, Any]) -> str:
    """Функция: _resolve_source_kind()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `cfg` (`Dict[str, Any]`): словарь конфигурации компонента.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
    camera_id = cfg.get("camera_id")
    if camera_id is not None:
        return "camera"
    input_source = str(cfg.get("input") or "").strip().lower()
    if input_source.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    return "file"


def build_process_meta(camera_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Функция: build_process_meta()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `camera_cfg` (`Dict[str, Any]`): конфигурация отдельной камеры/источника в multi-camera запуске.
- `global_cfg` (`Dict[str, Any]`): глобальная часть конфигурации, общая для всех источников.
Возвращаемое значение: Dict[str, Any]: результат шага обработки, который используется следующим этапом пайплайна."""
    cfg = dict(global_cfg)
    cfg.update(camera_cfg or {})
    source_kind = _resolve_source_kind(cfg)
    restart_default = source_kind in {"camera", "rtsp"}
    restart_on_failure = bool(cfg.get("restart_on_failure", restart_default))
    restart_delay_sec = float(cfg.get("restart_delay_sec", 2.0))
    max_restarts_raw = cfg.get("max_restarts")
    max_restarts = None if max_restarts_raw is None else int(max_restarts_raw)
    return {
        "source_kind": source_kind,
        "restart_on_failure": restart_on_failure,
        "restart_delay_sec": max(0.0, restart_delay_sec),
        "max_restarts": max_restarts,
    }


def build_centralized_command(
    python_executable: str,
    main_script: str,
    config_path: str,
    global_cfg: Dict[str, Any],
) -> List[str]:
    """Функция: build_centralized_command()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `python_executable` (`str`): путь к интерпретатору Python для запуска дочерних процессов.
- `main_script` (`str`): путь к основному скрипту, который запускается дочерним процессом.
- `config_path` (`str`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `global_cfg` (`Dict[str, Any]`): глобальная часть конфигурации, общая для всех источников.
Возвращаемое значение: List[str]: результат шага обработки, который используется следующим этапом пайплайна."""
    runtime_engine = str(global_cfg.get("runtime_engine", "centralized"))
    cmd = [python_executable, main_script, "--runtime-engine", runtime_engine, "--topology-config", config_path]

    if "output" in global_cfg:
        cmd.extend(["--output", str(global_cfg["output"])])
    if global_cfg.get("no_preview", False):
        cmd.append("--no-preview")
    if global_cfg.get("show_fps", False):
        cmd.append("--show-fps")
    if global_cfg.get("no_half", False):
        cmd.append("--no-half")
    if "device" in global_cfg:
        cmd.extend(["--device", str(global_cfg["device"])])
    if "imgsz" in global_cfg:
        cmd.extend(["--imgsz", str(int(global_cfg["imgsz"]))])

    detectors = _as_list(global_cfg.get("detectors"))
    if detectors:
        cmd.append("--detectors")
        cmd.extend(detectors)

    if "source_queue_size" in global_cfg:
        cmd.extend(["--source-queue-size", str(int(global_cfg["source_queue_size"]))])
    if "scheduler_infer_queue_size" in global_cfg:
        cmd.extend(
            [
                "--scheduler-infer-queue-size",
                str(int(global_cfg["scheduler_infer_queue_size"])),
            ]
        )
    if "source_drop_policy" in global_cfg:
        cmd.extend(["--source-drop-policy", str(global_cfg["source_drop_policy"])])
    if "infer_workers" in global_cfg:
        cmd.extend(["--infer-workers", str(int(global_cfg["infer_workers"]))])
    if "postprocess_workers" in global_cfg:
        cmd.extend(["--postprocess-workers", str(int(global_cfg["postprocess_workers"]))])
    if "scheduler_dispatch_sleep" in global_cfg:
        cmd.extend(["--scheduler-dispatch-sleep", str(float(global_cfg["scheduler_dispatch_sleep"]))])

    extra_args = _as_list(global_cfg.get("extra_args"))
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def launch_processes(
    commands: List[List[str]],
    dry_run: bool = False,
    process_meta: List[Dict[str, Any]] | None = None,
) -> int:
    """Функция: launch_processes()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `commands` (`List[List[str]]`): список команд запуска, подготовленных для исполнения.
- `dry_run` (`bool`): параметр запуска процесса/скрипта в управляющем контуре runtime.
- `process_meta` (`List[Dict[str, Any]] | None`): метаданные процессов, используемые для запуска и мониторинга.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
    if dry_run:
        for idx, cmd in enumerate(commands):
            print(f"[DRY-RUN #{idx}] {format_command_for_logging(cmd)}")
        return 0

    meta = list(process_meta or [])
    if len(meta) < len(commands):
        meta.extend({} for _ in range(len(commands) - len(meta)))

    slots: List[Dict[str, Any]] = []
    for idx, cmd in enumerate(commands):
        slots.append(
            {
                "index": idx,
                "cmd": cmd,
                "meta": dict(meta[idx] or {}),
                "proc": None,
                "done": False,
                "restarts": 0,
                "next_start_at": 0.0,
                "last_exit": 0,
            }
        )

    def _start_slot(slot: Dict[str, Any]) -> None:
        """Функция: _start_slot()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `slot` (`Dict[str, Any]`): идентификатор/индекс для адресации и сопоставления сущностей.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        proc = subprocess.Popen(slot["cmd"])
        slot["proc"] = proc
        print(
            f"[STARTED #{slot['index']}] pid={proc.pid} cmd={format_command_for_logging(slot['cmd'])}"
        )

    def _terminate_all() -> None:
        """Функция: _terminate_all()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        # 1) Мягко просим завершиться (SIGINT), чтобы child успел сделать cleanup/log summary.
        for slot in slots:
            proc = slot.get("proc")
            if proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    pass

        interrupt_deadline = time.monotonic() + 3.0
        for slot in slots:
            proc = slot.get("proc")
            if proc is None or proc.poll() is not None:
                continue
            remaining = max(0.0, interrupt_deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        # 2) Если не завершились, даем SIGTERM.
        for slot in slots:
            proc = slot.get("proc")
            if proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        for slot in slots:
            proc = slot.get("proc")
            if proc is None:
                continue
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        for slot in slots:
            _start_slot(slot)

        while True:
            now = time.time()
            for slot in slots:
                if slot["done"]:
                    continue

                proc = slot.get("proc")
                if proc is not None and proc.poll() is None:
                    continue

                if proc is not None and proc.poll() is not None:
                    exit_code = int(proc.returncode or 0)
                    slot["last_exit"] = exit_code
                    slot["proc"] = None
                    source_kind = str(slot["meta"].get("source_kind", "source"))
                    restart_on_failure = bool(slot["meta"].get("restart_on_failure", False))
                    max_restarts = slot["meta"].get("max_restarts")
                    restart_delay_sec = float(slot["meta"].get("restart_delay_sec", 2.0))
                    can_restart = restart_on_failure and (max_restarts is None or int(slot["restarts"]) < int(max_restarts))

                    if exit_code == 0 and source_kind == "file":
                        slot["done"] = True
                        print(f"[FINISHED #{slot['index']}] file source completed normally")
                    elif can_restart:
                        slot["restarts"] += 1
                        slot["next_start_at"] = now + max(0.0, restart_delay_sec)
                        print(
                            f"[RESTART #{slot['index']}] exit={exit_code}, "
                            f"restart #{slot['restarts']} in {restart_delay_sec:.1f}s"
                        )
                    else:
                        slot["done"] = True
                        print(f"[EXITED #{slot['index']}] exit={exit_code} (no restart)")

                if slot["done"]:
                    continue
                if slot.get("proc") is None and now >= float(slot.get("next_start_at", 0.0)):
                    _start_slot(slot)

            if all(bool(slot["done"]) for slot in slots):
                non_zero = [int(slot.get("last_exit", 0)) for slot in slots if int(slot.get("last_exit", 0)) != 0]
                return non_zero[0] if non_zero else 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt, terminating all processes...")
        _terminate_all()
        return 130
    finally:
        _terminate_all()


def build_arg_parser() -> argparse.ArgumentParser:
    """Функция: build_arg_parser()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: argparse.ArgumentParser: результат шага обработки, который используется следующим этапом пайплайна."""
    parser = argparse.ArgumentParser(description="Multi-camera launcher (1 process = 1 camera)")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument(
        "--main-script",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_source_launcher.py"),
        help="Path to single-source launcher script",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting processes")
    return parser


def main() -> int:
    """Функция: main()
Назначение: служит точкой входа и запускает основной сценарий выполнения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    global_cfg = dict(config.get("global", {}))
    camera_cfgs = config["cameras"]
    main_script = os.path.abspath(args.main_script)

    mode = str(config.get("mode", global_cfg.get("mode", "legacy"))).strip().lower()
    commands: List[List[str]] = []
    process_meta: List[Dict[str, Any]] = []
    if mode in {"centralized", "single_process"}:
        commands.append(
            build_centralized_command(
                python_executable=args.python,
                main_script=main_script,
                config_path=os.path.abspath(args.config),
                global_cfg=global_cfg,
            )
        )
        process_meta.append({"source_kind": "topology", "restart_on_failure": False})
    else:
        for cam in camera_cfgs:
            if not isinstance(cam, dict):
                raise ValueError("Each item in 'cameras' must be an object")
            commands.append(
                build_camera_command(
                    python_executable=args.python,
                    main_script=main_script,
                    camera_cfg=cam,
                    global_cfg=global_cfg,
                )
            )
            process_meta.append(build_process_meta(camera_cfg=cam, global_cfg=global_cfg))

    return launch_processes(commands, dry_run=args.dry_run, process_meta=process_meta)


if __name__ == "__main__":
    raise SystemExit(main())
