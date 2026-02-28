<a id="top"></a>

# Violation Detector

Этот репозиторий реализует систему детекции нарушений по видеопотокам и видеофайлам. Проект умеет работать как с одной камерой, так и с несколькими источниками, запускать классические CV-детекторы и YOLO/PyTorch-модели, показывать preview, сохранять артефакты нарушений и вести логи по каждому источнику.

Подробная архитектура, блок-схемы и путь выполнения описаны в [ARCHITECTURE.md](./ARCHITECTURE.md). Программный интерфейс и взаимодействие с ядром описаны в [API.md](./API.md).

## Оглавление

- [1. Что это за репозиторий](#what-is-this)
- [2. Что умеет проект](#capabilities)
- [3. Актуальные точки входа](#entrypoints)
- [4. Структура проекта](#project-structure)
- [5. Требования к окружению](#requirements)
- [6. Установка](#installation)
- [7. Как выбрать правильный PyTorch](#choose-pytorch)
- [8. Проверка установки](#verification)
- [9. Быстрый старт](#quick-start)
- [10. Запуск одного источника](#single-source)
- [11. Параметры single_source_launcher.py](#single-source-params)
- [12. Запуск нескольких источников](#multi-source)
- [13. Параметры multi_camera_launcher.py](#multi-source-params)
- [14. Структура сохранения данных](#output-structure)
- [15. Детекторы и их базовые настройки](#detectors)
- [16. Как добавить новый детектор](#add-detector)
- [17. Как добавить новый YOLO-детектор](#add-yolo-detector)
- [18. Как встроить детектор в интерфейс и визуализацию](#ui-integration)
- [19. Типичные сценарии и советы](#tips)

<a id="what-is-this"></a>
## 1. Что это за репозиторий

Проект предназначен для анализа видеопотока и фиксации событий нескольких типов:

- перекрытие камеры;
- затемнение кадра;
- смещение камеры;
- наличие запрещённых предметов;
- DMS-события: телефон, сигарета, закрытые глаза, отсутствие ремня.

Система строится вокруг двух режимов выполнения:

- `legacy`:
  один источник обрабатывается классическим циклом;
- `centralized`:
  один процесс управляет несколькими источниками, очередями, scheduler-ом и worker-потоками.

На практике это значит следующее:

- для одной камеры можно запускать и `legacy`, и `centralized`;
- для нескольких источников удобнее использовать topology JSON и `centralized`;
- все YOLO-зависимые детекторы используют общий слой инференса `InferenceHub`.

<a id="capabilities"></a>
## 2. Что умеет проект

- принимать локальную камеру через `--camera-id`;
- принимать видеофайл через `--input /path/to/file.mp4`;
- принимать RTSP URL через `--input rtsp://...`;
- запускать набор детекторов через `--detectors ...`;
- работать на `cpu` или `cuda`;
- сохранять изображения, видеофрагменты, текстовые отчёты и runtime manifest;
- вести раздельные логи для разных источников;
- запускать несколько камер либо как несколько процессов, либо как один централизованный runtime.

<a id="entrypoints"></a>
## 3. Актуальные точки входа

Важный момент: актуальный запуск идёт не через `main.py`, а через скрипты в `start_scripts/`.

Основные точки входа:

- `start_scripts/single_source_launcher.py`
  основной запуск одного источника или topology-конфига;
- `start_scripts/multi_camera_launcher.py`
  управляющий launcher для нескольких источников;
- `src/utils/config/multi_camera.example.json`
  пример конфигурации для multi-camera запуска.

Когда использовать что:

- если у вас одна камера, один RTSP или один видеофайл:
  используйте `single_source_launcher.py`;
- если у вас несколько источников:
  используйте `multi_camera_launcher.py`;
- если хотите один процесс на все источники:
  используйте `mode: centralized` в JSON-конфиге;
- если хотите по процессу на каждый источник:
  используйте `mode: legacy`.

<a id="project-structure"></a>
## 4. Структура проекта

Ключевые директории:

- `start_scripts/`
  CLI-точки входа;
- `src/detectors/`
  реализации детекторов;
- `src/inference/`
  общий слой загрузки моделей и инференса;
- `src/processing/`
  основной runtime, обработка кадров, менеджеры пайплайна, сохранение нарушений;
- `src/runtime/`
  topology, runtime profiles, controller и orchestration;
- `src/utils/config/`
  алиасы детекторов, schedule, настройки индикаторов и визуализации;
- `src/utils/models/`
  каноническое место для production-весов моделей;
- `violations/`
  результаты работы, логи, runtime manifest и артефакты нарушений.

Мини-карта по логике:

- `single_source_launcher.py` разбирает CLI и строит `RuntimeConfig`;
- `RuntimeController` поднимает нужный runtime;
- `DetectionManager` инициализирует детекторы и управляет их расписанием;
- `InferenceHub` загружает YOLO-модели и кэширует результаты инференса;
- `ViolationManager` сохраняет изображения, видеофрагменты и отчёты;
- `Visualizer` рисует overlay, bbox и индикаторы.

Подробно все связи, блок-схемы и последовательности запуска разобраны в [ARCHITECTURE.md](./ARCHITECTURE.md).

<a id="requirements"></a>
## 5. Требования к окружению

Базовые требования:

- Python `3.10+`;
- `pip`;
- для GPU-режима:
  корректно установленный NVIDIA driver и совместимая сборка PyTorch;
- для Linux preview с GUI:
  системные библиотеки OpenCV/Qt и доступ к display;
- для headless-сервера:
  рекомендуется запускать с `--no-preview`.

Проект использует:

- `ultralytics`;
- `torch`;
- `torchvision`;
- `torchaudio`;
- `opencv-python`;
- `numpy`;
- тестовые и quality-инструменты из `requirements.txt`.

<a id="installation"></a>
## 6. Установка

Ниже рекомендуемый порядок установки для Linux и Windows.

### 6.1. Клонирование и виртуальное окружение

```bash
git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
cd violation_detector-main
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Windows CMD:

```bat
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

### 6.2. Установка зависимостей проекта

Сначала ставим всё из `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 6.3. Переустановка PyTorch под вашу систему

В этом проекте `torch` уже указан в `requirements.txt`, но для реальной работы лучше после общей установки удалить PyTorch-пакеты и поставить их заново именно под вашу машину через официальный селектор PyTorch.

Удаление:

```bash
pip uninstall -y torch torchvision torchaudio
```

После этого перейдите на официальный сайт:

- https://pytorch.org/get-started/locally/

И выберите параметры под свою систему. Подробно это разобрано в следующем разделе.

<a id="choose-pytorch"></a>
## 7. Как выбрать правильный PyTorch

Официальный источник:

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/)

На странице нужно выбрать:

- `PyTorch Build`
  обычно `Stable`;
- `Your OS`
  `Linux` или `Windows`;
- `Package`
  `Pip`;
- `Language`
  `Python`;
- `Compute Platform`
  `CPU`, `CUDA ...` или `ROCm ...`.

По состоянию на 2026-02-28 на странице `Start Locally` доступны варианты `CUDA 11.8`, `CUDA 12.6`, `CUDA 12.8`, `ROCm 6.3` и `CPU`. Актуальный набор на сайте может меняться, поэтому ориентируйтесь на селектор, а не на старые команды из чужих инструкций.

### 7.1. Как понять, что выбирать на Linux

Проверьте систему:

```bash
uname -a
cat /etc/os-release
python3 --version
which python3
lspci | grep -Ei 'nvidia|amd|vga|3d'
nvidia-smi
nvcc --version
```

Как интерпретировать:

- если `nvidia-smi` не найден и у вас нет NVIDIA GPU:
  выбирайте `CPU`;
- если у вас NVIDIA GPU и `nvidia-smi` работает:
  смотрите версию CUDA, видимую драйвером, и выбирайте совместимый вариант на сайте;
- если у вас AMD GPU и вы осознанно используете ROCm:
  выбирайте `ROCm`, но только если ваша система реально настроена под ROCm;
- если вы не уверены:
  начните с `CPU`, затем переходите на GPU.

Практическое правило для NVIDIA:

- есть рабочий `nvidia-smi` и современный драйвер:
  выбирайте один из CUDA-вариантов, доступных в селекторе;
- если сомневаетесь между несколькими CUDA-вариантами:
  выбирайте тот, который точно поддерживается вашим драйвером и который предлагает сайт;
- если на машине нет GPU или это сервер без настроенного CUDA:
  выбирайте `CPU`.

### 7.2. Как понять, что выбирать на Windows

Проверьте систему:

```powershell
python --version
py --version
winver
Get-CimInstance Win32_VideoController | Select-Object Name
nvidia-smi
```

Если нужно посмотреть доступные интерпретаторы:

```powershell
py -0p
```

Как интерпретировать:

- если `nvidia-smi` не работает:
  выбирайте `CPU`;
- если есть NVIDIA GPU и `nvidia-smi` показывает корректную информацию:
  выбирайте подходящий `CUDA` вариант в селекторе;
- если у вас только встроенная графика Intel/AMD без ROCm:
  выбирайте `CPU`.

### 7.3. Какая команда установки нужна

Команду не нужно угадывать вручную. После выбора параметров сайт сам покажет точную команду для вашей платформы.

Примеры того, как это выглядит:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

или

```bash
pip3 install torch torchvision torchaudio
```

Но в проекте лучше использовать именно ту команду, которую прямо сейчас сгенерировал сайт под вашу систему.

### 7.4. Что выбирать чаще всего

Самые частые случаи:

- Linux/Windows без NVIDIA GPU:
  `CPU`;
- Linux/Windows с NVIDIA GPU и рабочим `nvidia-smi`:
  `CUDA`;
- Linux с AMD GPU и заранее подготовленным ROCm-окружением:
  `ROCm`.

<a id="verification"></a>
## 8. Проверка установки

После установки зависимостей и корректной сборки PyTorch проверьте окружение:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Проверьте, что проект вообще стартует:

```bash
python start_scripts/single_source_launcher.py --help
python start_scripts/multi_camera_launcher.py --help
```

Если `--help` отрабатывает без ошибок, значит базовый слой зависимостей и точки входа доступны.

<a id="quick-start"></a>
## 9. Быстрый старт

Самые типовые команды:

Локальная камера:

```bash
python start_scripts/single_source_launcher.py --camera-id 0 --detectors all
```

Видеофайл на CPU без preview:

```bash
python start_scripts/single_source_launcher.py \
  --input /data/video.mp4 \
  --device cpu \
  --no-preview \
  --detectors all
```

RTSP-источник:

```bash
python start_scripts/single_source_launcher.py \
  --input "rtsp://user:pass@host:554/stream" \
  --device cuda:0 \
  --detectors cv dark yolo movement forbidden dms
```

Несколько источников по JSON:

```bash
python start_scripts/multi_camera_launcher.py \
  --config src/utils/config/multi_camera.example.json
```

Dry run для multi-camera:

```bash
python start_scripts/multi_camera_launcher.py \
  --config src/utils/config/multi_camera.example.json \
  --dry-run
```

<a id="single-source"></a>
## 10. Запуск одного источника

Скрипт:

- `start_scripts/single_source_launcher.py`

Он умеет:

- запускаться по `--camera-id`;
- запускаться по `--input`;
- поднимать `legacy` или `centralized` runtime;
- брать topology JSON через `--topology-config`;
- создавать output-структуру, логи и runtime manifest;
- инициализировать `InferenceHub`, `RuntimeController` и выбранные детекторы.

Примеры:

### 10.1. Локальная камера

```bash
python start_scripts/single_source_launcher.py \
  --camera-id 0 \
  --detectors all
```

### 10.2. Видео с CPU

```bash
python start_scripts/single_source_launcher.py \
  --input /data/video.mp4 \
  --device cpu \
  --runtime-engine legacy \
  --no-preview \
  --detectors cv dark movement
```

### 10.3. Один источник в centralized runtime

```bash
python start_scripts/single_source_launcher.py \
  --input /data/video.mp4 \
  --runtime-engine centralized \
  --runtime-profile balanced \
  --device cuda:0 \
  --imgsz 720
```

### 10.4. Topology JSON

```bash
python start_scripts/single_source_launcher.py \
  --topology-config /path/to/topology.json \
  --runtime-engine centralized \
  --output violations
```

<a id="single-source-params"></a>
## 11. Параметры single_source_launcher.py

Ниже перечислены реальные параметры запуска по текущему коду.

| Параметр | Что делает | Пример |
|---|---|---|
| `--input PATH_OR_RTSP` | Путь к видеофайлу или RTSP URL | `--input /data/cam.mp4` |
| `--camera-id ID` | Индекс локальной камеры OpenCV | `--camera-id 0` |
| `--output DIR` | Корневая папка результатов | `--output violations` |
| `--topology-config PATH_JSON` | JSON topology-конфиг для centralized runtime | `--topology-config cfg.json` |
| `--detectors DETECTOR [DETECTOR ...]` | Список включённых детекторов | `--detectors cv yolo movement` |
| `--no-preview` | Отключает окно preview | `--no-preview` |
| `--show-fps` | Показывает FPS в preview | `--show-fps` |
| `--runtime-engine {legacy,centralized}` | Выбор runtime-режима | `--runtime-engine centralized` |
| `--device DEVICE` | Устройство инференса | `--device cpu` или `--device cuda:0` |
| `--runtime-profile {auto,balanced,laptop,server}` | Профиль runtime-параметров | `--runtime-profile balanced` |
| `--no-half` | Выключает FP16 на CUDA | `--no-half` |
| `--imgsz INT` | Размер входа модели | `--imgsz 640` |
| `--source-queue-size INT` | Размер source queue в centralized режиме | `--source-queue-size 8` |
| `--scheduler-infer-queue-size INT` | Размер infer queue scheduler-а | `--scheduler-infer-queue-size 64` |
| `--source-drop-policy {drop_oldest,drop_newest}` | Политика переполнения очереди источника | `--source-drop-policy drop_oldest` |
| `--infer-workers INT` | Количество infer workers | `--infer-workers 1` |
| `--postprocess-workers INT` | Количество postprocess workers | `--postprocess-workers 1` |
| `--scheduler-dispatch-sleep SECONDS` | Пауза scheduler при простое | `--scheduler-dispatch-sleep 0.003` |

Дополнительно:

- если не указан `--topology-config`, то нужно указать либо `--camera-id`, либо `--input`;
- `--detectors all` включает все канонические детекторы;
- список доступных канонических детекторов:
  `cv`, `dark`, `yolo`, `movement`, `forbidden`, `dms`;
- поддерживаются алиасы, но в документации лучше использовать канонические имена.

Что означают `runtime-profile`:

- `auto`
  проект сам выбирает профиль по CPU/RAM/CUDA;
- `balanced`
  средний профиль;
- `laptop`
  более осторожный профиль под слабую машину;
- `server`
  профиль для более мощного окружения.

<a id="multi-source"></a>
## 12. Запуск нескольких источников

Скрипт:

- `start_scripts/multi_camera_launcher.py`

Он читает JSON-конфиг и запускает:

- либо один процесс на каждый источник;
- либо один централизованный процесс на все источники.

Это определяется полем `mode` в JSON.

### 12.1. Основные режимы

- `mode: legacy`
  один процесс на источник;
- `mode: centralized`
  один процесс на все источники через `--topology-config`;
- `mode: single_process`
  фактически идёт через тот же централизованный путь.

### 12.2. Минимальный пример конфига

```json
{
  "mode": "centralized",
  "global": {
    "output": "violations",
    "runtime_engine": "centralized",
    "detectors": ["all"],
    "device": "cuda:0",
    "show_fps": true,
    "no_preview": false,
    "imgsz": 640,
    "source_queue_size": 8,
    "scheduler_infer_queue_size": 64,
    "source_drop_policy": "drop_oldest",
    "infer_workers": 1,
    "postprocess_workers": 1,
    "scheduler_dispatch_sleep": 0.003
  },
  "cameras": [
    {
      "camera_id": 0
    },
    {
      "input": "rtsp://user:pass@host:554/stream"
    },
    {
      "input": "/data/archive/video.mp4"
    }
  ]
}
```

### 12.3. Как работает приоритет параметров

Правило такое:

- значения из `global` применяются ко всем источникам;
- значения из конкретного объекта в `cameras[]` переопределяют `global`.

### 12.4. Запуск

```bash
python start_scripts/multi_camera_launcher.py \
  --config /path/to/config.json
```

Если хотите увидеть, какие команды реально будут собраны:

```bash
python start_scripts/multi_camera_launcher.py \
  --config /path/to/config.json \
  --dry-run
```

<a id="multi-source-params"></a>
## 13. Параметры multi_camera_launcher.py

| Параметр | Что делает | Пример |
|---|---|---|
| `--config CONFIG` | Путь к JSON-конфигу | `--config cfg.json` |
| `--python PYTHON` | Какой Python использовать для запуска дочерних процессов | `--python .venv/bin/python` |
| `--main-script MAIN_SCRIPT` | Какой single-source launcher запускать | `--main-script start_scripts/single_source_launcher.py` |
| `--dry-run` | Только печатает команды без запуска | `--dry-run` |

Что важно знать:

- в `legacy` launcher сам перезапускает процессы для `camera` и `rtsp`, если это разрешено конфигом;
- файловые источники, завершившиеся успешно, не перезапускаются;
- `extra_args` из JSON пробрасываются в `single_source_launcher.py` как есть.

<a id="output-structure"></a>
## 14. Структура сохранения данных

По умолчанию всё пишется в директорию `violations/`.

Типовая структура выглядит так:

```text
violations/
├── logs/
│   ├── camera_0.log
│   ├── camera_4.log
│   └── ...
├── runtime_manifest.json
├── camera_0/
│   ├── obstruction/
│   ├── movement/
│   ├── forbidden_items/
│   ├── dms/
│   └── ...
├── camera_4/
│   └── ...
└── source_<name>/
    └── ...
```

Что именно сохраняется:

- `logs/`
  логи запуска и обработки;
- `runtime_manifest.json`
  зафиксированные параметры запуска;
- `obstruction/`
  изображения и отчёты по перекрытию камеры;
- `movement/`
  видеофрагменты движения камеры и текстовые отчёты;
- `forbidden_items/`
  изображения и отчёты по запрещённым предметам;
- `dms/`
  изображения и отчёты по DMS-нарушениям.

Типы артефактов:

- `.jpg`
  аннотированные кадры;
- `.mp4`
  видеофрагменты движения;
- `.txt`
  текстовые отчёты о нарушениях;
- `.log`
  runtime-логи.

Как формируются каталоги:

- если есть `camera_id`, создаётся каталог вида `camera_<id>`;
- если источник задан через `--input`, может создаваться каталог `source_<имя>`;
- в centralized-режиме верхним корнем обычно остаётся общий `output`.

<a id="detectors"></a>
## 15. Детекторы и их базовые настройки

Канонические детекторы:

| Детектор | Назначение | Базовый принцип |
|---|---|---|
| `cv` | проверка перекрытия/деградации кадра | яркость, контраст, резкость |
| `dark` | затемнение кадра | доля тёмной области |
| `yolo` | крупный объект, перекрывающий кадр | full-frame YOLO |
| `movement` | смещение камеры | сравнение кадров и трекинг движения |
| `forbidden` | запрещённые предметы | full-frame YOLO + доменная логика |
| `dms` | DMS-события | full-frame YOLO + фильтрация по DMS-классам |

### 15.1. Runtime schedule детекторов

По умолчанию у каждого детектора своё расписание:

| Детектор | `every_n_frames` | `priority` | `result_ttl_frames` |
|---|---|---:|---:|
| `cv` | `1` | `95` | `0` |
| `dark` | `1` | `95` | `0` |
| `yolo` | `2` | `90` | `1` |
| `movement` | `2` | `95` | `0` |
| `forbidden` | `2` | `100` | `1` |
| `dms` | `3` | `100` | `1` |

Это означает:

- не все детекторы запускаются на каждом кадре;
- часть результатов кэшируется на несколько кадров;
- частота запуска влияет на задержку и нагрузку.

### 15.2. Базовые настройки существующих детекторов

`cv`:

- `brightness_thresh=25`
- `contrast_thresh=10`
- `sharpness_thresh=20`

`dark`:

- `dark_area_threshold=0.8`
- `processing_width=640`
- `dark_pixel_threshold=50`

`yolo`:

- `area_threshold=0.7`
- `conf_threshold=0.5`
- `iou_threshold=0.5`
- `imgsz=640`
- `max_det=50`

`movement`:

- `min_matches=20`
- `min_movement_duration=2.0`
- `confirmation_frames=8`

`forbidden`:

- в классе по умолчанию:
  `confidence_threshold=0.5`,
  `min_detection_duration=3.0`,
  `violation_cooldown=60.0`,
  `min_object_area_ratio=0.01`,
  `max_object_age=2.0`,
  `class_specific_cooldown=True`,
  `iou_threshold=0.5`,
  `imgsz=640`,
  `max_det=50`;
- в текущем `DetectionManager` создаётся с runtime-настройками:
  `confidence_threshold=0.4`,
  `min_detection_duration=3.0`,
  `violation_cooldown=30.0`,
  `min_object_area_ratio=0.005`,
  `class_specific_cooldown=True`.

`dms`:

- `confidence_threshold=0.05` при создании из `DetectionManager`;
- `eye_closed_threshold=5.0`
- `seatbelt_check_interval=15.0`
- `phone_threshold=4.0`
- `cigarette_threshold=3.0`
- `violation_cooldown=60.0`
- `min_object_area_ratio=0.005`
- `iou_threshold=0.5`

### 15.3. Какие веса используются

Production-резолвер моделей ожидает веса в:

- `src/utils/models/`

Если имя модели не указано явно, по умолчанию ожидается:

- `src/utils/models/best_auto.pt`

Для YOLO-зависимых детекторов это важно:

- `YOLODetector`, `ForbiddenItemsDetector` и `DMSDetector` опираются на общий `InferenceHub`;
- если у них совпадают `model_key` и путь к весам, runtime может переиспользовать общий инференс одного кадра.

<a id="add-detector"></a>
## 16. Как добавить новый детектор

Ниже описан путь для обычного детектора, который не обязан быть YOLO-моделью.

### 16.1. Шаг 1. Создать класс детектора

Создайте новый модуль в `src/detectors/`, например:

- `src/detectors/my_detector.py`

Лучше наследоваться от `BaseDetector` и реализовать метод `detect(frame)`.

Минимальный шаблон:

```python
from .base_detector import BaseDetector


class MyDetector(BaseDetector):
    def __init__(self):
        super().__init__("MyDetector")

    def detect(self, frame):
        return {
            "detected": False,
            "metrics": {},
        }
```

Важно:

- возвращаемая структура должна быть стабильной;
- `DetectionManager` должен понимать, как использовать результат;
- если детектор может быть отключён на runtime, используйте `self.is_active`.

### 16.2. Шаг 2. Подключить детектор в DetectionManager

Нужно обновить:

- импорт в `src/processing/detection_manager.py`;
- поле экземпляра в `__init__`;
- создание в `_ensure_detector_instance()`;
- возврат в `_get_detector_instance()`;
- включение в `_parse_detector_config()` через каноническое имя;
- отдельный runtime-метод детекции, если для него нужен свой путь обработки.

Минимальная идея подключения:

```python
from ..detectors.my_detector import MyDetector
```

И затем в `_ensure_detector_instance()`:

```python
if detector_name == "mydet" and self.my_detector is None:
    self.my_detector = MyDetector()
    logging.info("[detectors:enabled] detector=mydet")
    return
```

### 16.3. Шаг 3. Добавить каноническое имя и алиасы

Обновите:

- `src/utils/config/detector_aliases.py`

Нужно:

- добавить имя в `CANONICAL_DETECTORS`;
- при необходимости добавить алиасы в `DETECTOR_ALIASES`.

Пример:

```python
CANONICAL_DETECTORS = ["cv", "dark", "yolo", "movement", "forbidden", "dms", "mydet"]

DETECTOR_ALIASES = {
    ...
    "mydet": "mydet",
    "mydetector": "mydet",
}
```

### 16.4. Шаг 4. Добавить runtime schedule

Обновите:

- `src/utils/config/detector_schedule.py`

Добавьте новый `DetectorScheduleConfig` для детектора.

Это нужно, чтобы runtime понимал:

- как часто запускать детектор;
- нужно ли кэшировать результат;
- какой у него приоритет.

### 16.5. Шаг 5. Встроить в основной поток

Дальше зависит от типа результата.

Если детектор:

- просто даёт флаг и метрики:
  достаточно встроить его в `DetectionManager` и в то место, где формируется итоговый `viz_result`;
- создаёт отдельный тип нарушения:
  потребуется также обновить `ViolationManager`;
- влияет на логику существующего нарушения:
  нужно встроить его в aggregation-логику текущего пайплайна.

Обычно дополнительно приходится смотреть:

- `src/processing/universal_processor.py`
- `src/processing/multi_source_runtime.py`
- `src/processing/violation_manager.py`

### 16.6. Шаг 6. Добавить тесты

Минимум стоит покрыть:

- создание детектора;
- тип возвращаемой структуры;
- поведение на пустом кадре;
- поведение на ошибочном входе;
- интеграцию через `DetectionManager`.

<a id="add-yolo-detector"></a>
## 17. Как добавить новый YOLO-детектор

Если новый детектор тоже работает через YOLO/PyTorch, путь немного шире.

### 17.1. Создать детектор в `src/detectors/`

Лучше ориентироваться на:

- `src/detectors/yolo_detector.py`
- `src/detectors/forbidden_items_detector.py`
- `src/detectors/dms_detector.py`

Что обычно есть в таком детекторе:

- `hub`
  ссылка на `InferenceHub`;
- `model_key`
  логическое имя модели;
- `model_path`
  путь к весам, полученный через `resolve_model_path(...)`;
- пороги:
  `confidence_threshold`, `iou_threshold`, `imgsz`, `max_det`;
- доменный `postprocess_shared_results(...)`.

### 17.2. Использовать `resolve_model_path(...)`

Для новых YOLO-детекторов не нужно вручную искать `.pt` по репозиторию. Используйте:

```python
from ..inference.model_path_resolver import resolve_model_path
```

И загружайте модель так, чтобы вес лежал в:

- `src/utils/models/`

### 17.3. Использовать `InferenceHub`

`InferenceHub`:

- кэширует загруженные модели;
- умеет переиспользовать инференс одного и того же кадра;
- унифицирует вызов `predict(...)`.

Шаблон вызова:

```python
results = self.hub.predict(
    model_key=self.model_key,
    weights_path=self.model_path,
    frame_bgr=frame,
    frame_id=frame_id,
    conf=self.confidence_threshold,
    iou=self.iou_threshold,
    max_det=self.max_det,
    imgsz=self.imgsz,
    verbose=False,
)
```

### 17.4. Если хотите разделять один инференс между несколькими детекторами

Это уже поддержано в текущем `DetectionManager`.

Чтобы детектор мог участвовать в shared full-frame YOLO path, обычно нужно:

- использовать тот же `model_key`;
- использовать тот же путь к весам;
- иметь совместимый формат `postprocess_shared_results(...)`.

Если новый детектор использует другую модель или другой набор входов, shared path работать не будет, и это нормально.

### 17.5. Если детектор зависит от class id

Это особенно важно для DMS-подобных сценариев.

Нужно явно зафиксировать:

- какие `class_id` ожидаются;
- какие имена классов должны быть в модели;
- какие пороги применяются;
- что делать при несовпадении label space.

Практически это значит:

- не полагаться на неявные допущения;
- явно валидировать классы при инициализации;
- документировать зависимость между весами и логикой детектора.

<a id="ui-integration"></a>
## 18. Как встроить детектор в интерфейс и визуализацию

Если новый детектор должен быть виден в preview и статусных индикаторах, одного `DetectionManager` недостаточно.

Обычно затрагиваются следующие места:

- `src/utils/config/indicators_config.py`
  если нужен новый буквенный индикатор;
- `src/utils/config/visualizer_config.py`
  если нужен новый bbox-стиль или отдельный визуальный блок;
- `src/utils/media/visualizer.py`
  если нужно рисовать новый overlay;
- `src/processing/violation_manager.py`
  если нужен новый тип артефактов и сохранения;
- `src/utils/io/file_manager.py`
  если нужен новый каталог событий;
- `src/processing/detection_manager.py`
  если новый результат должен попадать в общий `viz_result`.

### 18.1. Новый статусный индикатор

В `IndicatorsLayout` уже есть метод:

- `add_indicator(...)`

Если хотите новый индикатор, добавьте его конфигурацию:

- новый `id`;
- краткую `label`;
- цвета активного и неактивного состояния.

### 18.2. Новый bbox или цвет

Если новый детектор рисует рамки:

- добавьте конфиг в `VisualizerConfig`;
- определите толщину, шрифт, флаги `show_label`, `show_confidence`, `show_area`;
- при необходимости добавьте новый цвет в палитру `colors`.

### 18.3. Новый тип сохранения

Если детектор должен писать отдельные файлы:

- добавьте новый `event_type` в `FileManager`;
- заведите отдельную ветку записи в `ViolationArtifactWriter`;
- встроите вызов в `ViolationManager`.

Иначе новый детектор будет вычисляться, но его артефакты не будут сохраняться так, как ожидает пользователь.

<a id="tips"></a>
## 19. Типичные сценарии и советы

### 19.1. Если у вас сервер без GUI

Запускайте так:

```bash
python start_scripts/single_source_launcher.py \
  --input /data/video.mp4 \
  --device cpu \
  --no-preview
```

### 19.2. Если RTSP тормозит

Пробуйте:

- `--runtime-profile laptop` на слабой машине;
- меньший `--imgsz`;
- `--source-drop-policy drop_oldest`;
- отключить preview через `--no-preview`.

### 19.3. Если CUDA не поднялась

Проверьте:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Если вывод `False`, то обычно проблема в одном из трёх мест:

- установлен CPU-only PyTorch;
- драйвер NVIDIA не установлен или не виден;
- выбрана несовместимая CUDA-сборка PyTorch.

### 19.4. Если не находятся веса

Проверьте наличие файла:

- `src/utils/models/best_auto.pt`

Если у вашего детектора другой вес:

- положите его в `src/utils/models/`;
- передавайте корректное имя модели в коде детектора;
- не рассчитывайте на случайные `.pt` в корне репозитория.

### 19.5. Если хотите понять, как всё работает внутри

Смотрите:

- [ARCHITECTURE.md](./ARCHITECTURE.md)
  полная архитектура, блок-схемы и сценарии выполнения;
- [API.md](./API.md)
  программное взаимодействие с ядром.

---

Если вы только начинаете работать с проектом, рекомендуемая последовательность такая:

1. установить зависимости и корректный PyTorch под свою машину;
2. проверить `--help` у launcher-скриптов;
3. запустить один видеофайл без preview;
4. проверить структуру `violations/`;
5. после этого переходить к RTSP и multi-camera конфигам;
6. только потом добавлять новые детекторы и изменять runtime.

[Наверх](#top)
