# API ядра для UI и внешних интеграций

Документ описывает публичный runtime API проекта с точки зрения разработчика интерфейса.

Основная цель этого API: дать UI полный и стабильный контракт для:
- запуска и остановки runtime;
- управления камерами по `source_id`;
- получения кадров, событий, состояния и статистики;
- чтения активной конфигурации runtime;
- безопасного hot-reload части настроек без перезапуска процесса.

Если вы впервые открыли этот документ и не знаете внутреннее устройство ядра, ориентируйтесь так:
- для запуска и управления используйте только `RuntimeController`;
- для построения интерфейса сначала вызовите `describe_ui_capabilities()` и `get_runtime_configuration()`;
- для живого экрана используйте `preview_callback`, `event_callback`, `get_state()` и `get_stats()`;
- все действия над отдельной камерой выполняйте через `source_id`.

## Оглавление

- [0. Если вы видите API впервые](#0-если-вы-видите-api-впервые)
- [1. Рекомендуемая точка входа](#1-рекомендуемая-точка-входа)
- [2. Поддерживаемые режимы runtime](#2-поддерживаемые-режимы-runtime)
- [3. Быстрый сценарий интеграции UI](#3-быстрый-сценарий-интеграции-ui)
- [4. Основные объекты API](#4-основные-объекты-api)
- [5. RuntimeController](#5-runtimecontroller)
- [6. Конфигурация запуска RuntimeConfig](#6-конфигурация-запуска-runtimeconfig)
- [7. Topology-модели](#7-topology-модели)
- [8. Методы управления жизненным циклом](#8-методы-управления-жизненным-циклом)
- [9. Методы per-source управления](#9-методы-per-source-управления)
- [10. Методы discovery и introspection](#10-методы-discovery-и-introspection)
- [11. Снимки состояния и статистики](#11-снимки-состояния-и-статистики)
- [12. Callback preview кадров](#12-callback-preview-кадров)
- [13. Callback runtime-событий](#13-callback-runtime-событий)
- [14. Hot-reload topology и runtime-настроек](#14-hot-reload-topology-и-runtime-настроек)
- [15. Структуры ответов и dataclass-контракты](#15-структуры-ответов-и-dataclass-контракты)
- [16. Коды ошибок и причины отказов](#16-коды-ошибок-и-причины-отказов)
- [17. Потоковая модель и гарантии безопасности](#17-потоковая-модель-и-гарантии-безопасности)
- [18. Что UI может менять без рестарта](#18-что-ui-может-менять-без-рестарта)
- [19. Что требует полного рестарта](#19-что-требует-полного-рестарта)
- [20. Ограничения текущей реализации](#20-ограничения-текущей-реализации)
- [21. Практические рекомендации для интерфейса](#21-практические-рекомендации-для-интерфейса)
- [22. Примеры](#22-примеры)

## 0. Если вы видите API впервые

Это раздел для разработчика, который не знает ядро и хочет быстро понять, как с ним жить.

### Самая короткая ментальная модель

Думайте о системе как о четырех слоях:

1. `RuntimeController`
   Это единственная точка входа, через которую UI должен работать с ядром.

2. `preview_callback`
   Это поток живых кадров для отображения.

3. `event_callback`
   Это поток реактивных событий, которые не нужно вылавливать polling-ом.

4. `get_state()/get_stats()/get_runtime_configuration()`
   Это снимки текущего состояния, статистики и конфигурации.

### Что нужно изучить в первую очередь

Если не хотите читать документ целиком, начните именно с этих разделов:

1. [3. Быстрый сценарий интеграции UI](#3-быстрый-сценарий-интеграции-ui)
2. [9. Методы per-source управления](#9-методы-per-source-управления)
3. [10. Методы discovery и introspection](#10-методы-discovery-и-introspection)
4. [11. Снимки состояния и статистики](#11-снимки-состояния-и-статистики)
5. [12. Callback preview кадров](#12-callback-preview-кадров)
6. [13. Callback runtime-событий](#13-callback-runtime-событий)

### Какие 6 вещей нужны почти любому UI

Практически любой интерфейс использует именно это:

- `start()`
- `stop()`
- `get_state()`
- `get_stats()`
- `get_runtime_configuration()`
- `describe_ui_capabilities()`

И почти всегда еще:

- `preview_callback`
- `event_callback`
- `stop_source()`
- `resume_source()`
- `set_source_visual_config()`
- `set_source_detectors()`

### Чем отличаются похожие методы

Это один из самых частых источников путаницы.

#### `get_state()`

Нужно для текущего “живого” состояния:
- работает ли runtime;
- идет ли камера;
- сколько очередей;
- есть ли ошибка;
- активен ли source.

Используйте для:
- карточек камер;
- индикаторов online/offline;
- кнопок stop/resume;
- статусов очередей и health.

#### `get_stats()`

Нужно для накопительных чисел и производительности:
- средние задержки;
- число ошибок;
- число обработанных кадров;
- hub metrics.

Используйте для:
- панелей метрик;
- отладочных экранов;
- графиков нагрузки;
- профилирования pipeline.

#### `get_runtime_configuration()`

Нужно для ответа на вопрос:
"какая конфигурация сейчас реально активна?"

Используйте для:
- окна настроек;
- инспектора topology;
- формы редактирования visual config;
- отображения активного detector schedule.

#### `describe_ui_capabilities()`

Нужно для ответа на вопрос:
"что вообще умеет этот runtime и какие поля можно менять?"

Используйте для:
- условного показа секций интерфейса;
- отключения unsupported controls;
- маркировки полей “можно менять без рестарта”;
- построения одного UI под разные режимы runtime.

### Откуда брать `source_id`

`source_id` не нужно придумывать вручную после запуска.

Надежные источники:
- из `RuntimeTopology`;
- из `get_runtime_configuration()["sources"]`;
- из `get_state()["sources"]`;
- из `PreviewFramePayload.source_id`;
- из `RuntimeEvent.source_id`.

Не используйте как основной идентификатор:
- индекс камеры в списке;
- `camera_id`;
- имя файла;
- позицию source в массиве.

### Когда UI должен использовать callbacks, а когда polling

Используйте `preview_callback`, если нужен живой видеопоток.

Используйте `event_callback`, если нужна реакция на единичные события:
- reconnect;
- сохранение нарушения;
- ошибка;
- stop/resume.

Используйте polling `get_state()` и `get_stats()`, если нужно:
- обновлять цифры и индикаторы;
- периодически перерисовывать статус;
- иметь источник истины даже если отдельное событие было пропущено.

Рекомендуемая практика:
- `preview_callback`: для кадров;
- `event_callback`: для реактивных уведомлений;
- `get_state()`: раз в 100-500 мс;
- `get_stats()`: раз в 500-2000 мс.

### Самая важная техническая оговорка

`preview_callback` и `event_callback` не следует считать вызовами из UI-потока.

Это означает:
- не нужно напрямую менять `Qt`-виджеты внутри callback;
- лучше передавать данные в `Qt` через signal/slot или thread-safe очередь;
- callback должен быть быстрым и не должен блокировать runtime.

## 1. Рекомендуемая точка входа

Для UI нужно работать через `RuntimeController`.

Почему именно так:
- это единая фасадная точка над `legacy` и `centralized` runtime;
- здесь собраны публичные методы управления;
- здесь уже есть совместимость с будущими изменениями внутренней реализации;
- UI не должен напрямую опираться на приватную структуру `MultiSourceRuntime`.

Рекомендуемый импорт:

```python
from src.runtime.controller import RuntimeController, RuntimeConfig
from src.runtime.topology import RuntimeTopology
```

## 2. Поддерживаемые режимы runtime

### `centralized`

Основной режим для нового UI.

Дает:
- per-source команды;
- structured preview callback;
- structured event callback;
- полный snapshot состояния и статистики;
- hot-reload части topology/runtime настроек;
- introspection через `get_runtime_configuration()` и `describe_ui_capabilities()`.

### `legacy`

Режим совместимости для старого однокамерного пайплайна.

Ограничения:
- нет полноценного per-source API;
- нет multi-source command queue;
- часть discovery-функций возвращает fallback-данные;
- hot-topology update недоступен.

Если вы делаете новый интерфейс, используйте `centralized`.

## 3. Быстрый сценарий интеграции UI

Рекомендуемый порядок:

1. Собрать `RuntimeConfig`.
2. Создать `RuntimeController`.
3. Передать `preview_callback` и `event_callback`.
4. Вызвать `describe_ui_capabilities()` и `get_runtime_configuration()` для построения интерфейса.
5. Запустить runtime через `start()`.
6. Периодически читать `get_state()` и `get_stats()`.
7. На пользовательские действия вызывать per-source команды.

Минимальный скелет:

```python
from src.runtime.controller import RuntimeController, RuntimeConfig
from src.runtime.topology import RuntimeTopology


def on_preview(payload):
    # payload.source_id
    # payload.frame
    # payload.metadata
    pass


def on_event(event):
    # event.event_type
    # event.source_id
    # event.data
    pass


topology = RuntimeTopology.from_dict({
    "sources": [
        {
            "source_id": "camera_0",
            "camera_id": 0,
            "detectors": ["all"],
        }
    ]
})

config = RuntimeConfig(
    input_source=None,
    runtime_engine="centralized",
    topology=topology,
    show_preview=True,
    preview_callback=on_preview,
    event_callback=on_event,
)

controller = RuntimeController(config=config)
controller.start()
```

### Что делать сразу после создания контроллера

Сразу после создания контроллера, но до `start()`, обычно имеет смысл:

1. вызвать `describe_ui_capabilities()`;
2. вызвать `get_runtime_configuration()`;
3. по `sources` построить экран камер;
4. по `visual_config_fields` и `mutable_*_fields` построить редактор настроек.

### Что делать сразу после `start()`

После запуска runtime типичный UI делает следующее:

1. запускает таймер polling для `get_state()`;
2. запускает более редкий таймер polling для `get_stats()`;
3. начинает принимать кадры через `preview_callback`;
4. начинает принимать события через `event_callback`;
5. по `source_id` связывает callbacks с конкретными виджетами камер.

## 4. Основные объекты API

Основные публичные уровни:

- `RuntimeController`: главный фасад для UI.
- `RuntimeConfig`: конфигурация запуска.
- `RuntimeTopology`: multi-source topology.
- `SourceConfig`: описание отдельного источника.
- `SchedulerConfig`: scheduler-конфигурация.
- `PreviewFramePayload`: payload кадра для UI.
- `RuntimeEvent`: payload событий для UI.
- `SourceCommandResult`: результат команд.
- `TopologyUpdateResult`: результат hot-reload.

## 5. RuntimeController

`RuntimeController` находится в `src/runtime/controller.py`.

Это основной публичный класс для UI.

Он отвечает за:
- запуск runtime;
- остановку runtime;
- выбор `legacy` или `centralized` движка;
- проброс per-source команд;
- возврат snapshot-состояния;
- возврат snapshot-конфигурации;
- описание возможностей текущего runtime.

### Базовый пример создания

```python
config = RuntimeConfig(
    input_source=None,
    runtime_engine="centralized",
    topology=my_topology,
    show_preview=True,
    preview_callback=on_preview,
    event_callback=on_event,
)

controller = RuntimeController(config=config)
```

## 6. Конфигурация запуска RuntimeConfig

`RuntimeConfig` находится в `src/runtime/controller.py`.

### Основные поля

#### Источник и режим

- `input_source: Any`
  Назначение: одиночный источник для простого запуска.
  В `centralized` режиме обычно лучше передавать `topology`.

- `camera_id: Optional[int] = None`
  Назначение: индекс камеры для single-source сценария.

- `runtime_engine: str = "centralized"`
  Допустимые значения:
  - `"centralized"`
  - `"multi_source"`
  - `"legacy"`

- `topology: Optional[RuntimeTopology] = None`
  Если задана, runtime запускается по topology.

#### Сохранение и запись

- `save_dir: str = "violations"`
- `async_violation_writes: bool = True`
- `writer_queue_max_size: int = 256`
- `writer_overflow_strategy: str = "drop_newest"`

#### Детекторы и инференс

- `enabled_detectors: Optional[List[str]] = None`
- `hub: Any = None`
- `device: str = "cuda:0"`
- `use_half: bool = True`
- `imgsz: int = 720`

#### Preview

- `show_preview: bool = True`
  Важно: чтобы `preview_callback` реально вызывался, этот флаг должен быть `True`.

- `show_fps: bool = False`
- `preview_width: Optional[int] = None`
- `preview_callback: Optional[Callable[..., None]] = None`

#### Runtime tuning

- `source_queue_max_size: int = 8`
- `source_drop_policy: str = "drop_oldest"`
- `scheduler_infer_queue_size: int = 64`
- `infer_workers: int = 1`
- `postprocess_workers: int = 1`
- `scheduler_dispatch_sleep_sec: float = 0.003`
- `command_timeout_sec: float = 1.0`

#### События

- `event_callback: Optional[Callable[..., None]] = None`

#### Ограничение по времени

- `max_duration: Optional[float] = None`

## 7. Topology-модели

Topology-модели находятся в `src/runtime/topology.py`.

### SourceConfig

Поля:

- `source_id: str`
  Уникальный идентификатор source. Это ключ, с которым дальше работает UI.
  Именно это поле нужно использовать:
  - как идентификатор вкладки камеры;
  - как ключ виджета preview;
  - как ключ для команд `stop_source`, `resume_source` и т.д.

- `input_source: Optional[Any]`
  Путь к файлу, URL или другой источник.
  Используется в основном для:
  - файловых источников;
  - RTSP-потоков;
  - диагностики и отображения “откуда приходит видео”.

- `camera_id: Optional[int]`
  Индекс камеры.
  Обычно используется только для локальных камер.
  Не рекомендуется использовать `camera_id` как основной ключ UI, потому что у file/RTSP source его может не быть.

- `base_priority: float = 1.0`
  Базовый вес источника в scheduler.
  Чем выше значение, тем агрессивнее scheduler будет выбирать этот source при прочих равных.
  Для UI это полезно, если нужно дать оператору возможность “поднять приоритет” критичной камеры.

- `capture_queue_size: int = 8`
  Размер очереди захвата.
  Это лимит кадров, ожидающих обслуживания после чтения из source и до dispatch в infer pipeline.
  Слишком маленькое значение дает ранний drop, слишком большое увеличивает lag.

- `drop_policy: str = "drop_oldest"`
  Допустимые значения:
  - `"drop_oldest"`
  - `"drop_newest"`
  Значение определяет, какие кадры выбрасывать при переполнении source queue.
  Обычно:
  - `drop_oldest` лучше для “живого” интерфейса;
  - `drop_newest` лучше, если важнее не терять уже накопленный backlog.

- `detector_schedule: Dict[str, Dict[str, int]] = {}`
  Runtime-расписание отдельных детекторов на source.
  Это точная настройка частоты запуска детекторов без выключения самих детекторов.
  UI может использовать это для режима “экономии” или “повышенной точности”.

- `enabled_detectors: List[str] = ["all"]`
  Список логически включенных детекторов на source.
  Если указан `"all"`, runtime пытается включить все поддерживаемые детекторы, но часть из них может быть недоступна из-за отсутствия `hub` или других ограничений.

### Расшифровка `detector_schedule`

`detector_schedule` задается по имени детектора. Для каждого детектора можно передать:

- `every_n_frames`
  Запускать детектор только на каждом N-м кадре.
  Пример:
  - `1` означает запуск на каждом кадре;
  - `5` означает запуск примерно раз в пять кадров.

- `min_interval_ms`
  Минимальный интервал между двумя запусками детектора в миллисекундах.
  Даже если пришел “подходящий” кадр, детектор не будет вызван раньше этого интервала.

- `priority`
  Локальный приоритет детектора внутри собственных scheduler-эвристик.
  Сейчас это скорее runtime tuning параметр для согласованной настройки, чем UI-поле ежедневного использования.

- `result_ttl_frames`
  Сколько кадров разрешено переиспользовать последний результат детектора без нового запуска.
  Полезно для тяжелых детекторов вроде `yolo`, `forbidden`, `dms`.

### SchedulerConfig

Поля:

- `policy: str = "weighted_round_robin"`
  Допустимые значения:
  - `"weighted_round_robin"`
  - `"lag_aware"`
  `weighted_round_robin`:
  - старается честно обслуживать все sources с учетом их `base_priority`.
  `lag_aware`:
  - сильнее реагирует на backlog и лаг, что полезно при перегрузке.

- `aging_factor: float = 2.0`
  Коэффициент “старения” ожидания source.
  Чем выше значение, тем быстрее растет шанс давно не обслуживаемого source быть выбранным scheduler.

- `backlog_factor: float = 0.7`
  Вклад текущего backlog source queue в решение scheduler.
  Чем выше значение, тем сильнее scheduler реагирует на переполнение очередей.

- `starvation_threshold_sec: float = 1.0`
  Порог времени, после которого source считается голодающим.

- `starvation_boost: float = 10.0`
  Дополнительный boost приоритета для source, который давно не обслуживался.

- `dispatch_sleep_sec: float = 0.003`
  Пауза scheduler в idle-состоянии, когда dispatch нечего делать.
  Меньше значение:
  - уменьшает задержки;
  - увеличивает нагрузку на CPU.

- `infer_queue_size: int = 64`
  Максимальная длина очереди задач на inference.
  Это важный параметр latency/backlog tradeoff.

- `infer_overflow_strategy: str = "drop_oldest"`
  Стратегия поведения при переполнении infer queue.

### RuntimeTopology

Поля:

- `sources: List[SourceConfig]`
- `scheduler: SchedulerConfig`
- `infer_workers: int = 1`
- `postprocess_workers: int = 1`
- `use_centralized_runtime: bool = True`

### Создание topology из словаря

```python
topology = RuntimeTopology.from_dict({
    "sources": [
        {
            "source_id": "camera_0",
            "camera_id": 0,
            "base_priority": 1.0,
            "capture_queue_size": 8,
            "drop_policy": "drop_oldest",
            "detectors": ["movement", "forbidden"],
            "detector_schedule": {
                "movement": {
                    "every_n_frames": 2,
                    "min_interval_ms": 0,
                    "priority": 95,
                    "result_ttl_frames": 0,
                }
            },
        }
    ],
    "scheduler": {
        "policy": "weighted_round_robin",
        "dispatch_sleep_sec": 0.003,
        "infer_queue_size": 64,
    },
    "infer_workers": 1,
    "postprocess_workers": 1,
})
```

## 8. Методы управления жизненным циклом

### `start() -> bool`

Запускает runtime.

Возвращает:
- `True`, если запуск инициирован;
- `False`, если runtime уже запущен.

Пример:

```python
started = controller.start()
```

### `stop(timeout: float = 5.0) -> bool`

Останавливает runtime.

Возвращает:
- `True`, если был активный runtime и остановка запрошена;
- `False`, если runtime не был запущен.

### `wait(timeout: Optional[float] = None) -> None`

Ждет завершения runtime-потока.

### `is_running() -> bool`

Проверяет, активен ли runtime.

## 9. Методы per-source управления

Эти методы доступны полноценно в `centralized` режиме.

В `legacy` режиме они либо недоступны, либо возвращают `unsupported_runtime`.

### `set_source_detectors(source_id: str, detectors: List[str]) -> SourceCommandResult`

Изменяет набор активных детекторов только для одного source.

Пример:

```python
result = controller.set_source_detectors("camera_0", ["movement", "forbidden"])
```

Что приходит в `result.data` при успехе:

- `enabled_detectors`
  Нормализованный список реально активированных детекторов после применения алиасов и runtime-нормализации.

### `get_source_detectors(source_id: str) -> Dict[str, Any]`

Возвращает:
- `source_id`
- `enabled_detectors`
- `detector_statuses`
- `detector_schedule`
- опционально `error`

Расшифровка:

- `enabled_detectors`
  Логически включенные детекторы для source.

- `detector_statuses`
  Runtime-состояние экземпляров детекторов, обычно список словарей вида:
  - `name`
  - `enabled`
  - `active`

- `detector_schedule`
  Текущее фактическое расписание запусков детекторов на этом source.

### `stop_source(source_id: str) -> SourceCommandResult`

Останавливает один source.

Семантика:
- команда идемпотентна;
- не останавливает весь runtime;
- не ломает общий `stop()`.

Что полезно для UI:
- если source уже был остановлен, команда все равно считается успешной;
- в `data` может прийти `already_stopped=True`.

### `resume_source(source_id: str) -> SourceCommandResult`

Возобновляет source.

Семантика:
- если source уже работает, вернется `already_running`;
- если runtime не запущен, вернется `runtime_not_running`;
- для файлового источника воспроизведение стартует заново с начала.

Что это значит для UI:
- у file source после `resume` нужно ожидать новый поток кадров с `frame_index` снова от начала;
- не нужно пытаться “продолжить с того же места” на стороне интерфейса.

### `reset_movement_reference(source_id: str) -> SourceCommandResult`

Сбрасывает reference frame для movement detector.

Если кадр еще не доступен, вернется `no_reference_frame`.

Обычно кнопку reset reference лучше активировать только после появления первых preview/state обновлений.

### `set_source_visual_config(source_id: str, config: Dict[str, Any]) -> SourceCommandResult`

Изменяет visual-настройки одного source.

Поддерживаемые поля:
- `preview_width`
- `show_fps`
- `show_indicators`
- `show_stats_panel`
- `show_movement_arrow`
- `show_violation_labels`
- `show_boxes`

Все поля можно передавать частично.
Runtime делает merge с текущим config источника, поэтому UI может менять только один флаг без отправки всей структуры.

### `get_source_visual_config(source_id: str) -> Dict[str, Any]`

Возвращает:
- `source_id`
- `visual_config`
- опционально `error`

## 10. Методы discovery и introspection

Эти методы нужны UI, чтобы не хардкодить доступные команды и формы настройки.

### `get_runtime_configuration() -> Dict[str, Any]`

Возвращает полную активную конфигурацию runtime в сериализуемом виде.

Включает:
- тип движка;
- флаги preview/event callback;
- настройки scheduler;
- настройки writer;
- список sources;
- active visual config;
- detector schedule по каждому source.

Гарантированные поля `scheduler` в текущем ядре:

- `policy`
- `aging_factor`
- `backlog_factor`
- `starvation_threshold_sec`
- `starvation_boost`
- `dispatch_sleep_sec`
- `infer_queue_size`
- `infer_overflow_strategy`

Практически это “источник истины” для формы настроек.
Если UI открыл редактор параметров, именно этот метод должен быть исходным snapshot.

Этот метод можно использовать:
- до запуска runtime;
- после запуска runtime;
- при открытии окна настроек;
- после hot-reload, чтобы обновить форму UI.

### `describe_ui_capabilities() -> Dict[str, Any]`

Возвращает capabilities snapshot:
- что runtime вообще умеет;
- какие поля можно менять без рестарта;
- какие поля требуют полного рестарта;
- список поддерживаемых команд;
- список поддерживаемых visual-полей;
- каталог допустимых детекторов.

Это основной метод для построения адаптивного UI.

Гарантированные `visual_config_fields`:

- `preview_width`
- `show_fps`
- `show_indicators`
- `show_stats_panel`
- `show_movement_arrow`
- `show_violation_labels`
- `show_boxes`

Гарантированные `source_command_names` для `centralized`:

- `set_source_detectors`
- `get_source_detectors`
- `stop_source`
- `resume_source`
- `reset_movement_reference`
- `set_source_visual_config`
- `get_source_visual_config`

Гарантированные `mutable_runtime_fields` для `centralized`:

- `scheduler.policy`
- `scheduler.aging_factor`
- `scheduler.backlog_factor`
- `scheduler.starvation_threshold_sec`
- `scheduler.starvation_boost`
- `scheduler.dispatch_sleep_sec`
- `scheduler.infer_overflow_strategy`

Гарантированные `mutable_source_fields` для `centralized`:

- `detectors`
- `detector_schedule`
- `visual_config`
- `base_priority`

Для `legacy`:

- `source_command_names == []`
- `mutable_runtime_fields == []`
- `mutable_source_fields == []`
- `restart_required_fields == {"runtime": [], "source": []}`

### Практический порядок использования discovery-методов

Если вы строите UI с нуля, используйте их именно в таком порядке:

1. `describe_ui_capabilities()`
   Чтобы понять:
   - какие controls вообще показывать;
   - какие разделы скрыть;
   - что можно менять без рестарта.

2. `get_runtime_configuration()`
   Чтобы заполнить:
   - текущие значения полей;
   - таблицу камер;
   - настройки scheduler;
   - visual config;
   - detector schedule.

3. `get_state()`
   Чтобы после запуска показать:
   - что реально сейчас работает;
   - какие cameras stopped/error/degraded;
   - какие очереди перегружены.

### `describe_restart_required_updates() -> Dict[str, List[str]]`

Возвращает группы полей, которые нельзя безопасно менять на лету.

Использование в UI:
- серые/disabled поля в окне настроек;
- пометка “требует перезапуска”;
- предупреждение перед применением batch-изменений.

## 11. Снимки состояния и статистики

### `get_state() -> Dict[str, Any]`

Возвращает текущее runtime-состояние.

На верхнем уровне:

- `running`
- `frame_count`
- `uptime_sec`
- `writer_queue_size`
- `last_error`

Для `centralized` дополнительно:

- `infer_queue_size`
- `postprocess_queue_size`
- `max_infer_queue_depth`
- `max_postprocess_queue_depth`
- `command_queue_size`
- `sources`

Расшифровка верхнего уровня:

- `running`
  Запущен ли сейчас runtime.

- `frame_count`
  Сумма `captured_frames` по всем sources, рассчитанная на уровне контроллера.
  Это скорее общий счетчик активности, а не точная метрика производительности.

- `uptime_sec`
  Время жизни текущего запуска runtime в секундах.

- `writer_queue_size`
  Суммарный размер writer queues по всем sources.
  Полезен для глобального индикатора “не успеваем сохранять”.

- `last_error`
  Последняя ошибка уровня runtime/controller.
  Если ошибка относится к конкретному source, ее лучше брать из `sources[source_id].last_error`.

- `infer_queue_size`
  Текущая длина центральной infer queue.

- `postprocess_queue_size`
  Текущая длина очереди постобработки.

- `max_infer_queue_depth`
  Максимальное значение `infer_queue_size`, достигнутое с начала запуска.

- `max_postprocess_queue_depth`
  Максимальная глубина очереди постобработки.

- `command_queue_size`
  Сколько UI-команд сейчас ждут обработки внутри runtime.

- `sources`
  Словарь per-source состояния, индексируемый по `source_id`.

#### `sources[source_id]` в `get_state()`

Поля:

- `source_queue_size`
- `served_count`
- `dropped_before_infer`
- `captured_frames`
- `capture_dropped_frames`
- `processed_frames`
- `max_source_queue_depth`
- `stop_requested`
- `preview_closed`
- `enabled_detectors`
- `detector_statuses`
- `detector_schedule`
- `visual_config`
- `writer_queue_size`
- `capture_running`
- `health_status`
- `last_error`

Расшифровка полей `sources[source_id]` в `get_state()`:

- `source_queue_size`
  Сколько кадров сейчас лежит в очереди этого source до dispatch.

- `served_count`
  Сколько кадров уже было отдано scheduler в дальнейший pipeline.

- `dropped_before_infer`
  Сколько кадров было выброшено до инференса.
  Если это число растет, источник не успевает обслуживаться.

- `captured_frames`
  Сколько кадров уже прочитано из source.

- `capture_dropped_frames`
  Сколько кадров было отброшено еще на этапе capture worker.

- `processed_frames`
  Сколько кадров прошло полный pipeline до postprocess/render этапа.

- `max_source_queue_depth`
  Максимальная глубина source queue с начала запуска.

- `stop_requested`
  Был ли source остановлен пользовательской командой.

- `preview_closed`
  Закрыт ли preview этого source.
  Для внешнего UI это чаще диагностический флаг, чем основной сигнал.

- `enabled_detectors`
  Текущий логический набор включенных детекторов.

- `detector_statuses`
  Runtime-статусы экземпляров детекторов.
  Полезно для панели “что реально работает”.

- `detector_schedule`
  Текущее runtime-расписание вызова детекторов.

- `visual_config`
  Активная конфигурация отображения для source.

- `writer_queue_size`
  Сколько задач записи нарушений ждут сохранения для этого source.

- `capture_running`
  Активен ли capture worker прямо сейчас.

- `health_status`
  Агрегированный статус source:
  - `running`
  - `idle`
  - `stopped`
  - `degraded`
  - `error`
  - `unknown`

- `last_error`
  Последняя ошибка уровня source.
  Это основное поле для показа ошибки на карточке камеры.

Назначение `get_state()`:
- отрисовка текущего статуса камеры;
- статус кнопок UI;
- online-индикаторы;
- queue/health snapshot;
- диагностика остановленных/ошибочных sources.

### `get_stats() -> Dict[str, Any]`

Возвращает накопленную статистику runtime.

На верхнем уровне:

- `running`
- `total_sources`
- `total_processed_frames`
- `infer_queue_size`
- `postprocess_queue_size`
- `max_infer_queue_depth`
- `max_postprocess_queue_depth`
- `command_queue_size`
- `hub_metrics`
- `sources`
- `last_error`

Расшифровка верхнего уровня `get_stats()`:

- `total_sources`
  Общее число sources в runtime.

- `total_processed_frames`
  Суммарное число кадров, полностью прошедших обработку.

- `hub_metrics`
  Метрики общего inference hub.
  Конкретный состав зависит от реализации hub, но типично там есть счетчики вызовов и cache hits.

#### `sources[source_id]` в `get_stats()`

Поля:

- `captured_frames`
- `capture_dropped_frames`
- `processed_frames`
- `served_count`
- `dropped_before_infer`
- `postprocess_errors`
- `max_source_queue_depth`
- `avg_scheduling_latency_ms`
- `avg_infer_time_ms`
- `avg_end_to_end_lag_ms`
- `writer_queue_size`
- `runtime_stats`
- `enabled_detectors`
- `detector_statuses`
- `detector_schedule`
- `visual_config`
- `capture_running`
- `health_status`
- `last_error`

Расшифровка полей `sources[source_id]` в `get_stats()`:

- `postprocess_errors`
  Количество ошибок в postprocess/pipeline части для source.

- `avg_scheduling_latency_ms`
  Средняя задержка от захвата кадра до его dispatch в infer pipeline.

- `avg_infer_time_ms`
  Среднее время shared inference для source.

- `avg_end_to_end_lag_ms`
  Средний end-to-end lag кадра от захвата до завершения обработки.

- `runtime_stats`
  Domain-specific счетчики менеджеров.
  Обычно это:
  - число сохраненных нарушений;
  - число кадров;
  - профильные счетчики конкретных детекторов.

Назначение `get_stats()`:
- панели производительности;
- latency-графики;
- диагностика pipeline;
- накопительные счетчики.

## 12. Callback preview кадров

### Как подключить

Передайте `preview_callback` в `RuntimeConfig`.

Важно:
- `show_preview` должен быть `True`;
- callback будет вызываться только если source не остановлен и preview не закрыт;
- callback не должен блокировать надолго UI-поток runtime.

Дополнительно важно:
- кадр в `payload.frame` приходит в формате `BGR`, как обычно в OpenCV;
- не стоит модифицировать этот массив “на месте”, если тот же callback-путь может использовать его дальше;
- для UI лучше воспринимать `frame` как read-only входные данные;
- если нужна дальнейшая обработка, лучше делать копию на стороне интерфейса.

### Новый контракт

Новый рекомендуемый формат:

```python
def on_preview(payload: PreviewFramePayload) -> None:
    ...
```

### Legacy-совместимость

Временно поддерживается и старый формат:

```python
def on_preview(source_id, frame) -> None:
    ...
```

Если callback принимает 1 позиционный аргумент, runtime передаст `PreviewFramePayload`.

Если callback принимает 2 позиционных аргумента, runtime передаст `(source_id, frame)`.

### Поля PreviewFramePayload

- `source_id: str`
- `frame: Any`
- `frame_index: int`
- `video_timestamp: float`
- `captured_at: float`
- `processed_at: float`
- `metadata: Dict[str, Any]`

Расшифровка:

- `source_id`
  Какому source принадлежит кадр.

- `frame`
  Готовый BGR-кадр preview после visual overlays.

- `frame_index`
  Индекс кадра внутри source.
  Для file source после `resume` может снова начинаться сначала.

- `video_timestamp`
  Timestamp в координатах самого видеоисточника, а не wall-clock.
  Полезно для таймлиний и синхронизации с архивом.

- `captured_at`
  Wall-clock время захвата кадра.

- `processed_at`
  Wall-clock время, когда payload был готов для UI.
  Разница `processed_at - captured_at` дает приближенную оценку свежести кадра.

### Поля `metadata`

На текущий момент гарантированно присутствуют:

- `processed_frames`
- `writer_queue_size`
- `show_fps`
- `preview_width`
- `overlays_enabled`

Расшифровка:

- `processed_frames`
  Сколько кадров этот source уже полностью обработал к моменту отправки preview.

- `writer_queue_size`
  Текущий backlog записи нарушений для source.

- `show_fps`
  Включена ли отрисовка FPS на самом preview-кадре.

- `preview_width`
  Активная ширина preview для этого source после учета per-source config.

- `overlays_enabled`
  Упрощенный флаг: включен ли хотя бы один визуальный overlay.
  Удобно для UI-переключателя “preview only / preview with overlays”.

Что в `metadata` сейчас не гарантируется:
- любые пользовательские поля;
- доменные результаты детекции;
- статистика нарушений;
- координаты объектов.

Если UI нужны такие данные, он должен брать их из других контрактов или отдельного расширения API, а не предполагать, что они появятся в `metadata`.

### Практическая рекомендация

Не рендерьте `numpy`-кадр прямо в callback. Лучше:

1. быстро скопировать ссылку/данные в thread-safe очередь UI;
2. уже в UI-потоке конвертировать в `QImage/QPixmap`.

## 13. Callback runtime-событий

### Как подключить

Передайте `event_callback` в `RuntimeConfig`.

Рекомендуемый формат:

```python
def on_event(event: RuntimeEvent) -> None:
    ...
```

### Поля RuntimeEvent

- `event_type: str`
- `source_id: Optional[str]`
- `timestamp: float`
- `severity: str`
- `data: Dict[str, Any]`

Расшифровка:

- `event_type`
  Тип события. Именно по нему UI обычно маршрутизирует логику реакции.

- `source_id`
  `None`, если событие относится ко всему runtime, иначе идентификатор source.

- `timestamp`
  Wall-clock время события.

- `severity`
  Уровень важности:
  - `info`
  - `warning`
  - `error`

- `data`
  Дополнительная нагрузка события.
  Формат зависит от `event_type`.

Важно:
- `data` не является строго одинаковым для всех типов событий;
- UI должен сначала смотреть на `event_type`, и только потом читать ожидаемые поля из `data`;
- не стоит писать код, который предполагает одинаковую схему `data` для всех событий.

### Основные `event_type`

Поддерживаемые типы событий:

- `source_started`
- `source_stopped`
- `source_resumed`
- `source_reconnect_open`
- `source_reconnect_read`
- `infer_queue_overflow`
- `source_queue_drop`
- `detector_error`
- `violation_saved`
- `runtime_warning`
- `runtime_error`

### Подробно по `event_type`

#### `source_started`

Когда приходит:
- при старте runtime, когда source начал работу.

Зачем нужен UI:
- включить индикатор “камера активна”;
- снять статус “offline”.

Типичный `data`:
- обычно пустой словарь.

#### `source_stopped`

Когда приходит:
- после `stop_source(...)`.

Зачем нужен UI:
- перевести карточку камеры в состояние “остановлена пользователем”;
- отключить элементы, зависящие от live stream.

Обычно после этого полезно:
- дождаться обновления `get_state()`;
- выключить preview-индикатор;
- оставить доступной кнопку `resume`.

#### `source_resumed`

Когда приходит:
- после успешного `resume_source(...)`.

Зачем нужен UI:
- вернуть source в live-состояние;
- сбросить локальный статус “stopped”.

После этого разумно:
- очистить локальный баннер ошибки/стопа;
- ожидать новые кадры и новый state snapshot.

#### `source_reconnect_open`

Когда приходит:
- если capture worker пытается переподключить source после потери соединения/ошибки открытия.

Зачем нужен UI:
- показать “переподключение...”;
- не трактовать это сразу как фатальную ошибку.

Типичный `data`:
- `backoff_sec`

На стороне UI это обычно не фатальная ошибка, а переходное состояние.

#### `source_reconnect_read`

Когда приходит:
- если source был открыт, но чтение кадров сорвалось и runtime ушел в reconnect.

Зачем нужен UI:
- различать “не открылся вообще” и “в процессе чтения потерялся”.

Если в интерфейсе есть отдельные статусы соединения, это полезно показывать как `reconnecting`, а не как окончательный `error`.

Типичный `data`:
- `backoff_sec`

#### `infer_queue_overflow`

Когда приходит:
- при переполнении infer queue.

Зачем нужен UI:
- показать предупреждение о перегрузке;
- предложить уменьшить число активных детекторов или изменить scheduler tuning.

Типичный `data`:
- `infer_queue_size`
- `strategy`

`source_id` для этого события в текущем ядре всегда указывает на конкретный source, из которого не удалось принять кадр в infer queue.

Это сильный сигнал для UI:
- уменьшить нагрузку;
- отключить часть детекторов;
- снизить частоту тяжелых детекторов;
- поднять машинные ресурсы.

#### `source_queue_drop`

Когда приходит:
- если source queue переполнена и кадры были выброшены.

Зачем нужен UI:
- показать деградацию качества потока;
- подсветить, что детекторы/машина не успевают.

Типичный `data`:
- drop policy;
- `dropped_frames`

В текущем коде гарантируются:
- `drop_policy`
- `dropped_frames`

Рост таких событий означает, что preview может оставаться “живым”, но детекция и/или аналитика уже идет с потерями кадров.

#### `detector_error`

Когда приходит:
- если в postprocess/detection pipeline произошла ошибка.

Зачем нужен UI:
- показать source-level ошибку;
- дать ссылку на диагностику;
- не завершать весь runtime, если ошибка локальная.

Типичный `data`:
- `error`
- `frame_index`

UI должен считать это source-level ошибкой, а не обязательно фатальным падением всего runtime.

#### `violation_saved`

Когда приходит:
- после фактического успешного сохранения нарушения.

Зачем нужен UI:
- обновить таблицу нарушений;
- показать toast/уведомление;
- обновить счетчики сохраненных событий.

Типичный `data`:
- `violation_kind`
- `violation_id`
- `media_file`

`severity` для этого события в текущем коде равен `info`.

`violation_kind` обычно показывает, какой именно тип нарушения сохранен:
- `obstruction`
- `movement`
- `forbidden`
- `dms`

#### `runtime_warning`

Когда приходит:
- для нефатальных проблем runtime.

Например:
- ошибка внутри `preview_callback`;
- частичная деградация интеграции.

Типичный `data`:
- `stage`
- `error`

Это warning, а не обязательно авария. Чаще всего UI должен показать мягкое уведомление, а не переводить весь runtime в красный статус.

#### `runtime_error`

Когда приходит:
- при ошибке уровня scheduler/inference worker/postprocess worker/runtime loop.

Зачем нужен UI:
- показать глобальную ошибку runtime;
- инициировать аварийный статус;
- предложить перезапуск.

Типичный `data`:
- `stage`
- `error`

Это уже кандидат на глобальный error-banner или предложение пользователю перезапустить runtime.

### Что важно про `violation_saved`

Событие эмитится по факту успешной записи артефактов нарушения, а не просто по факту обнаружения.

Это важно для UI:
- можно безопасно обновлять список сохраненных нарушений;
- не нужно перепроверять файловую систему;
- не нужно парсить логи.

## 14. Hot-reload topology и runtime-настроек

### `apply_topology_updates(updates: Dict[str, Any]) -> TopologyUpdateResult`

Позволяет безопасно применять часть настроек без рестарта процесса.

Поддерживаемые категории:

- обновление scheduler-параметров;
- обновление detector config;
- обновление detector schedule;
- обновление visual config;
- обновление `base_priority`.

### Общий формат `updates`

```python
updates = {
    "scheduler": {
        "dispatch_sleep_sec": 0.01,
        "aging_factor": 1.5,
    },
    "sources": [
        {
            "source_id": "camera_0",
            "detectors": ["movement"],
            "detector_schedule": {
                "movement": {
                    "every_n_frames": 5,
                    "min_interval_ms": 10,
                    "priority": 90,
                    "result_ttl_frames": 1,
                }
            },
            "visual_config": {
                "show_fps": True,
                "show_boxes": False,
            },
            "base_priority": 2.0,
        }
    ]
}
```

### Формат результата TopologyUpdateResult

- `success: bool`
- `applied: Dict[str, Any]`
- `rejected: Dict[str, str]`
- `restart_required: List[str]`
- `message: Optional[str]`

Как использовать в UI:

- `applied`
  Показывать как список реально принятых изменений.

- `rejected`
  Показывать как список полей, которые runtime не принял.

- `restart_required`
  Использовать для баннера “часть изменений требует рестарта”.

### Как интерпретировать `success`

- `True`: хотя бы часть изменений применена или отклонений нет;
- `False`: изменения отклонены полностью.

### Как интерпретировать `rejected`

Ключи выглядят как path-подобные имена, например:

- `scheduler.infer_queue_size`
- `sources.camera_0.camera_id`
- `queue_limits.postprocess_queue_size`

Значения:

- `restart_required`
- `unsupported`
- `invalid:<описание>`
- `source_not_found`

## 15. Структуры ответов и dataclass-контракты

Контракты находятся в `src/runtime/contracts.py`.

### SourceCommandResult

Универсальный результат per-source команды.

Поля:

- `success: bool`
- `source_id: Optional[str]`
- `error: Optional[str]`
- `message: Optional[str]`
- `data: Dict[str, Any]`

Как использовать:
- `success` использовать как первичный флаг выполнения;
- `error` как машинно-обрабатываемый код;
- `message` как готовый текст для лога или debug UI;
- `data` как payload специфики конкретной команды.

Примеры `data`:
- для `set_source_detectors`: `enabled_detectors`
- для `set_source_visual_config`: `visual_config`
- для повторного `stop_source`: `already_stopped`

Практический шаблон обработки:

```python
result = controller.stop_source("camera_0")
if not result.success:
    handle_command_error(result.error, result.message)
else:
    apply_success_ui_state(result.data)
```

### SourceVisualConfig

Поля:

- `preview_width: Optional[int]`
- `show_fps: bool`
- `show_indicators: bool`
- `show_stats_panel: bool`
- `show_movement_arrow: bool`
- `show_violation_labels: bool`
- `show_boxes: bool`

Практический смысл полей:

- `preview_width`
  Итоговая ширина preview для source.
  Если `None`, используется runtime default.

- `show_fps`
  Показывать ли FPS на кадре.

- `show_indicators`
  Показывать ли блок индикаторов состояний.

- `show_stats_panel`
  Показывать ли статистическую панель поверх preview.

- `show_movement_arrow`
  Показывать ли стрелку/направление движения камеры.

- `show_violation_labels`
  Показывать ли текстовые подписи нарушений.

- `show_boxes`
  Рисовать ли bounding boxes/объекты.

### SourceConfigurationSnapshot

Поля:

- `source_id`
- `source_type`
- `input_source`
- `camera_id`
- `base_priority`
- `capture_queue_size`
- `drop_policy`
- `enabled_detectors`
- `detector_schedule`
- `visual_config`
- `output_dir`

Что это дает UI:
- готовый snapshot для формы “настройки камеры”;
- понимание типа source (`camera`, `rtsp`, `file`);
- готовый путь сохранения артефактов именно для этого source.

### RuntimeConfigurationSnapshot

Поля:

- `engine`
- `running`
- `save_dir`
- `show_preview`
- `default_visual_config`
- `async_violation_writes`
- `writer_queue_max_size`
- `writer_overflow_strategy`
- `preview_callback_attached`
- `event_callback_attached`
- `command_timeout_sec`
- `infer_workers`
- `postprocess_workers`
- `scheduler`
- `sources`

Что это дает UI:
- готовый snapshot для окна глобальных настроек;
- truth source для scheduler tuning;
- понимание, подключены ли callback-интеграции.

### RuntimeCapabilitiesSnapshot

Поля:

- `engine`
- `supports_per_source_control`
- `supports_preview_callback`
- `supports_event_callback`
- `supports_hot_topology_updates`
- `supports_runtime_configuration_snapshot`
- `supports_detector_schedule_updates`
- `detector_catalog`
- `visual_config_fields`
- `source_command_names`
- `mutable_runtime_fields`
- `mutable_source_fields`
- `restart_required_fields`

Что это дает UI:
- можно динамически строить доступные controls;
- можно не показывать unsupported functionality;
- можно маркировать поля “требует рестарта”.

Для нового разработчика это один из самых важных объектов.
Если сомневаетесь, можно ли что-то менять или показывать в интерфейсе, сначала смотрите именно сюда.

## 16. Коды ошибок и причины отказов

Основные коды, которые UI должен уметь обрабатывать:

### `unsupported_runtime`

Означает, что операция доступна только в `centralized`, а вызвана в `legacy`.

### `source_not_found`

Указанный `source_id` не существует.

### `runtime_not_running`

Команда требует активного runtime, но он остановлен.

### `command_queue_full`

Команда не была принята, потому что внутренняя runtime command queue переполнена.
UI должен считать это сигналом backpressure:

- не ретраить мгновенно в tight loop;
- дождаться следующего `get_state()`/`get_stats()`;
- повторить действие с небольшой задержкой.

### `command_timeout`

Команда была отправлена в runtime command queue, но подтверждение не пришло за `command_timeout_sec`.

### `unknown_command`

Внутренняя runtime-команда не распознана.

### `detector_manager_unavailable`

У данного source недоступен `DetectionManager`.

### `already_running`

Попытка `resume_source(...)` для уже активного source.

### `already_stopped`

Повторная остановка уже остановленного source.

На практике приходит как:

```json
{
  "success": true,
  "data": {
    "already_stopped": true
  }
}
```

### `no_reference_frame`

Для `reset_movement_reference(...)` пока нет кадра, который можно использовать как reference.

### `reset_unavailable`

У source нет доступного механизма reset reference для movement detector.

### `restart_required`

Поле нельзя безопасно обновить в работающем runtime.

### `unsupported`

Поле не поддерживается hot-reload API.

### `invalid:<...>`

Передано некорректное значение, и runtime отклонил обновление.

## 17. Потоковая модель и гарантии безопасности

### Важное правило

Публичные UI-команды в `centralized` режиме сериализуются через command queue runtime.

Это означает:
- нет необходимости строить свой mutex вокруг `stop_source/resume_source/...`;
- одновременные команды от UI не должны ломать внутреннее состояние;
- результат каждой команды приходит как `SourceCommandResult`.

### Что это не гарантирует

Это не означает, что UI может игнорировать порядок собственных действий.

Например, если пользователь быстро нажмет:

1. `stop_source`
2. `resume_source`
3. `set_source_visual_config`

команды будут обработаны последовательно, но UI все равно должен обновлять экран по фактическому `result` и по `get_state()`, а не только по факту клика.

## 18. Что UI может менять без рестарта

### Runtime-level

Сейчас поддерживаются:

- `scheduler.policy`
- `scheduler.aging_factor`
- `scheduler.backlog_factor`
- `scheduler.starvation_threshold_sec`
- `scheduler.starvation_boost`
- `scheduler.dispatch_sleep_sec`
- `scheduler.infer_overflow_strategy`

### Source-level

Сейчас поддерживаются:

- `detectors`
- `detector_schedule`
- `visual_config`
- `base_priority`

## 19. Что требует полного рестарта

### Runtime-level

- `infer_workers`
- `postprocess_workers`
- `scheduler.infer_queue_size`
- `queue_limits.infer_queue_size`
- `queue_limits.postprocess_queue_size`

### Source-level

- `capture_queue_size`
- `drop_policy`
- `input_source`
- `camera_id`

## 20. Ограничения текущей реализации

Важно честно учитывать следующие ограничения:

### Нет динамического add/remove source во время работы

Сейчас runtime не поддерживает безопасное добавление и удаление источников на лету.

Можно:
- управлять уже существующими sources;
- останавливать и возобновлять их;
- менять часть настроек.

Нельзя:
- добавить новую камеру в работающий runtime;
- удалить source из topology без полного рестарта.

### Legacy не равен centralized

`legacy` режим нужен для совместимости, но новый UI должен ориентироваться на `centralized`.

### Preview callback зависит от `show_preview`

Если `show_preview=False`, preview callback вызываться не будет.

## 21. Практические рекомендации для интерфейса

### Рекомендуемый startup-flow

1. Создать `RuntimeController`.
2. Сразу вызвать `describe_ui_capabilities()`.
3. Сразу вызвать `get_runtime_configuration()`.
4. На основе этого построить панели управления.
5. После `start()` включить периодический polling `get_state()` и `get_stats()`.

### Рекомендуемый runtime-flow экрана камеры

Для каждого виджета камеры удобно держать такую схему:

1. `source_id` как главный ключ.
2. Последний `PreviewFramePayload` для картинки.
3. Последний `get_state()["sources"][source_id]` для статусов.
4. Последний `get_stats()["sources"][source_id]` для метрик.
5. Последний `RuntimeEvent` для реактивных уведомлений.

Тогда UI не будет зависеть только от callback-ов или только от polling-а, а сможет сочетать оба канала.

### Что использовать для чего

Для кнопок и online-статусов:
- `get_state()`

Для графиков и метрик:
- `get_stats()`

Для экрана настроек:
- `get_runtime_configuration()`

Для условного показа контролов:
- `describe_ui_capabilities()`

Для реактивных уведомлений:
- `event_callback`

Для живого видео:
- `preview_callback`

Если нужна форма настроек:
- `get_runtime_configuration()`

Если нужно понять, что именно разрешено менять:
- `describe_ui_capabilities()`

### Как хранить идентичность камер в UI

Используйте только `source_id`.

Не полагайтесь на:
- индекс в массиве;
- `camera_id` как на универсальный ключ;
- имя файла как на ключ источника.

### Что делать, если событие и snapshot противоречат друг другу

Источником истины должен быть snapshot.

Практическое правило:
- событие использовать как триггер обновления;
- `get_state()` и `get_stats()` использовать как окончательное подтверждение текущего состояния.

Пример:
- пришел `source_stopped`;
- UI сразу показывает промежуточный статус;
- затем берет `get_state()` и проверяет `stop_requested=True`.

### Что не нужно делать в UI

Не нужно:
- напрямую трогать внутренние объекты runtime;
- строить логику на парсинге `.log` файлов;
- хардкодить поля, которые можно узнать из `describe_ui_capabilities()`;
- считать, что callbacks приходят из главного UI-потока;
- использовать `camera_id` вместо `source_id` как основной идентификатор.

### Как читать этот документ дальше

Если вы уже поняли общую картину, дальше используйте документ как справочник:

- разделы 6-7: структура конфигурации;
- раздел 9: команды;
- раздел 11: polling snapshots;
- разделы 12-13: callbacks;
- раздел 14: hot-reload;
- раздел 16: обработка ошибок.

## 22. Примеры

### Пример: остановка и возобновление камеры

```python
stop_result = controller.stop_source("camera_0")
if stop_result.success:
    print("Камера остановлена")

resume_result = controller.resume_source("camera_0")
if resume_result.success:
    print("Камера возобновлена")
```

### Пример: изменение visual config

```python
result = controller.set_source_visual_config(
    "camera_0",
    {
        "show_fps": True,
        "show_boxes": False,
        "show_stats_panel": True,
        "show_indicators": True,
    },
)
```

### Пример: изменение detector schedule

```python
result = controller.apply_topology_updates(
    {
        "sources": [
            {
                "source_id": "camera_0",
                "detector_schedule": {
                    "movement": {
                        "every_n_frames": 4,
                        "min_interval_ms": 50,
                        "priority": 90,
                        "result_ttl_frames": 1,
                    }
                },
            }
        ]
    }
)

if result.success:
    print("Обновление применено")
print(result.applied)
print(result.rejected)
```

### Пример: построение UI по capabilities

```python
caps = controller.describe_ui_capabilities()

if caps["supports_per_source_control"]:
    show_source_controls()

if "detector_schedule" in caps["mutable_source_fields"]:
    show_detector_schedule_editor()
```

### Пример: чтение runtime configuration

```python
cfg = controller.get_runtime_configuration()

for source_id, source_cfg in cfg["sources"].items():
    print(source_id, source_cfg["source_type"], source_cfg["enabled_detectors"])
```

### Пример: обработка событий

```python
def on_event(event):
    if event.event_type == "violation_saved":
        refresh_violations_table()
    elif event.event_type == "source_queue_drop":
        show_warning_badge(event.source_id)
    elif event.event_type == "runtime_error":
        show_runtime_error(event.data.get("error"))
```

### Пример: обработка preview payload

```python
def on_preview(payload):
    source_id = payload.source_id
    frame = payload.frame
    frame_index = payload.frame_index
    writer_queue_size = payload.metadata["writer_queue_size"]
    overlays_enabled = payload.metadata["overlays_enabled"]
    update_preview_widget(source_id, frame, frame_index, writer_queue_size, overlays_enabled)
```
