"""Файл: debug/list_connected_cameras.py
Тип: отладочный скрипт.
Назначение: используется для локальной диагностики и проверки отдельных подсистем.
Связи: обычно запускается вручную и читает данные из основных модулей проекта.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""
import glob
import platform
import re

try:
    import cv2
except ImportError as exc:
    raise SystemExit("Не найден модуль cv2. Установите: pip install opencv-python") from exc


def backend_for_os() -> int | None:
    """Функция: backend_for_os()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: int | None: результат шага обработки, который используется следующим этапом пайплайна."""
    os_name = platform.system()
    if os_name == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    if os_name == "Linux" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    return None


def probe_camera(cam_id: int, backend: int | None, tries: int = 4) -> bool:
    """Функция: probe_camera()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `cam_id` (`int`): идентификатор/индекс для адресации и сопоставления сущностей.
- `backend` (`int | None`): рабочие данные `backend`, используемые на текущем этапе обработки.
- `tries` (`int`): рабочие данные `tries`, используемые на текущем этапе обработки.
Возвращаемое значение: bool: результат шага обработки, который используется следующим этапом пайплайна."""
    cap = cv2.VideoCapture(cam_id) if backend is None else cv2.VideoCapture(cam_id, backend)
    if not cap.isOpened():
        cap.release()
        return False

    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            cap.release()
            return True

    cap.release()
    return False


def linux_video_ids() -> list[int]:
    """Функция: linux_video_ids()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: list[int]: результат шага обработки, который используется следующим этапом пайплайна."""
    ids = set()
    for path in glob.glob("/dev/video*"):
        m = re.search(r"(\d+)$", path)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def find_camera_ids() -> list[int]:
    """Функция: find_camera_ids()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: list[int]: результат шага обработки, который используется следующим этапом пайплайна."""
    backend = backend_for_os()
    found = []

    if platform.system() == "Linux":
        for cam_id in linux_video_ids():
            if probe_camera(cam_id, backend):
                found.append(cam_id)
        return found

    # Windows/другие ОС: авто-остановка после серии "пустых" id
    cam_id = 0
    consecutive_miss = 0
    while consecutive_miss < 8 and cam_id < 64:
        if probe_camera(cam_id, backend):
            found.append(cam_id)
            consecutive_miss = 0
        else:
            consecutive_miss += 1
        cam_id += 1

    return found


def main() -> None:
    """Функция: main()
Назначение: служит точкой входа и запускает основной сценарий выполнения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
    camera_ids = find_camera_ids()
    print(f"Количество подключенных камер: {len(camera_ids)}")
    print(f"ID камер: {camera_ids}")


if __name__ == "__main__":
    main()
