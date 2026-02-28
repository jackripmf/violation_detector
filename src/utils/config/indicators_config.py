"""Файл: src/utils/config/indicators_config.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable


@dataclass
class IndicatorConfig:
    """Класс: IndicatorConfig
Назначение: описывает структуру конфигурации и типизированные данные компонента.
Поля класса:
- `color_active` (`Tuple[int, int, int]`): параметр визуализации, влияющий на внешний вид оверлеев и текста.
- `color_inactive` (`Tuple[int, int, int]`): цвет индикатора для неактивного состояния.
- `condition` (`Optional[Callable[[Dict], bool]]`): функция-предикат, определяющая условие отображения/активации.
- `id` (`str`): идентификатор сущности, используемый для сопоставления и доступа.
- `label` (`str`): подпись объекта/нарушения для отображения на кадре.
- `visible` (`bool`): флаг отображения элемента в UI/визуализации.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    id: str  # Уникальный идентификатор
    label: str  # Буква для отображения
    visible: bool = True  # Показывать ли индикатор
    color_active: Tuple[int, int, int] = (255, 0, 0)  # Цвет когда активен
    color_inactive: Tuple[int, int, int] = (100, 100, 100)  # Цвет когда неактивен
    condition: Optional[Callable[[Dict], bool]] = None  # Функция проверки активности


@dataclass
class IndicatorsLayout:
    """Класс: IndicatorsLayout
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `indicator_size` (`int`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `indicator_spacing` (`int`): параметр визуализации, влияющий на внешний вид оверлеев и текста.
- `indicators` (`Dict[str, IndicatorConfig]`): параметр визуализации, влияющий на внешний вид оверлеев и текста.
- `left_margin` (`int`): координаты области или точки, используемые для геометрических вычислений.
- `show_indicators` (`bool`): параметр визуализации, влияющий на внешний вид оверлеев и текста.
- `text_start_x` (`int`): координаты области или точки, используемые для геометрических вычислений.
- `top_margin` (`int`): координаты области или точки, используемые для геометрических вычислений.
Ключевые методы:
- `get_visible_indicators()`, `get_indicator_field_width()`, `add_indicator()`, `remove_indicator()`, `set_indicator_visible()`, `show_all_indicators()`, `hide_all_indicators()`, `clone()`"""
    # Поле индикаторов
    show_indicators: bool = True  # Показывать ли вообще индикаторы
    indicator_size: int = 15  # Размер индикатора (будет масштабироваться)
    indicator_spacing: int = 5  # Отступ между индикаторами
    left_margin: int = 5  # Отступ слева
    top_margin: int = 5  # Отступ сверху
    
    # Поле для текста (автоматически вычисляется)
    text_start_x: int = 0
    
    # Индикаторы
    indicators: Dict[str, IndicatorConfig] = field(default_factory=lambda: {
        "obstruction": IndicatorConfig(
            id="obstruction",
            label="O",
            visible=True,
            color_active=(0, 0, 255),
            color_inactive=(0, 255, 0),
        ),
        "movement": IndicatorConfig(
            id="movement",
            label="M",
            visible=True,
            color_active=(0, 0, 255),
            color_inactive=(0, 255, 0),
        ),
        "forbidden": IndicatorConfig(
            id="forbidden",
            label="F",
            visible=True,
            color_active=(255, 0, 255),
            color_inactive=(100, 100, 100),
        ),
        "dms": IndicatorConfig(
            id="dms",
            label="D",
            visible=True,
            color_active=(255, 0, 0),
            color_inactive=(100, 100, 100),
        ),
        "cigarette": IndicatorConfig(
            id="cigarette",
            label="C",
            visible=True,
            color_active=(0, 165, 255),
            color_inactive=(100, 100, 100),
        ),
        "phone": IndicatorConfig(
            id="phone",
            label="P",
            visible=True,
            color_active=(255, 0, 255),
            color_inactive=(100, 100, 100),
        ),
    })
    
    def get_visible_indicators(self) -> List[IndicatorConfig]:
        """Функция: get_visible_indicators()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: List[IndicatorConfig]: результат шага обработки, который используется следующим этапом пайплайна."""
        return [ind for ind in self.indicators.values() if ind.visible]
    
    def get_indicator_field_width(self) -> int:
        """Функция: get_indicator_field_width()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: int: результат шага обработки, который используется следующим этапом пайплайна."""
        if not self.show_indicators:
            return 0
        
        visible = self.get_visible_indicators()
        if not visible:
            return 0
        
        return self.left_margin + self.indicator_size + self.indicator_spacing
    
    def add_indicator(self, id: str, label: str, color_active: Tuple[int, int, int], 
                     color_inactive: Tuple[int, int, int] = (100, 100, 100),
                     visible: bool = True) -> None:
        """Функция: add_indicator()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `id` (`str`): идентификатор сущности, используемый для сопоставления и доступа.
- `label` (`str`): подпись объекта/нарушения для отображения на кадре.
- `color_active` (`Tuple[int, int, int]`): параметр визуализации, влияющий на внешний вид оверлеев и текста.
- `color_inactive` (`Tuple[int, int, int]`): цвет индикатора для неактивного состояния.
- `visible` (`bool`): флаг отображения элемента в UI/визуализации.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        self.indicators[id] = IndicatorConfig(
            id=id,
            label=label,
            visible=visible,
            color_active=color_active,
            color_inactive=color_inactive,
        )
    
    def remove_indicator(self, id: str) -> None:
        """Функция: remove_indicator()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `id` (`str`): идентификатор сущности, используемый для сопоставления и доступа.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if id in self.indicators:
            del self.indicators[id]
    
    def set_indicator_visible(self, id: str, visible: bool) -> None:
        """Функция: set_indicator_visible()
Назначение: обновляет состояние объекта или конфигурацию во время выполнения.
Параметры функции:
- `id` (`str`): идентификатор сущности, используемый для сопоставления и доступа.
- `visible` (`bool`): флаг отображения элемента в UI/визуализации.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        if id in self.indicators:
            self.indicators[id].visible = visible
    
    def show_all_indicators(self) -> None:
        """Функция: show_all_indicators()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        for ind in self.indicators.values():
            ind.visible = True
    
    def hide_all_indicators(self) -> None:
        """Функция: hide_all_indicators()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: None: результат шага обработки, который используется следующим этапом пайплайна."""
        for ind in self.indicators.values():
            ind.visible = False
    
    def clone(self) -> 'IndicatorsLayout':
        """Функция: clone()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: 'IndicatorsLayout': результат шага обработки, который используется следующим этапом пайплайна."""
        import copy
        return copy.deepcopy(self)
