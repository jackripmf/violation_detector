"""Файл: src/utils/config/visualizer_config.py
Тип: вспомогательный модуль.
Назначение: предоставляет утилиты логирования, визуализации, конфигурации и файловых операций.
Связи: подключается из processors/detectors/start_scripts для общей инфраструктурной логики.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import cv2


@dataclass
class TextStyle:
    """Класс: TextStyle
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `bg_alpha` (`float`): прозрачность фоновой подложки под текст/индикатор.
- `bg_color` (`Optional[Tuple[int, int, int]]`): цвет фоновой подложки под текст/индикатор.
- `color` (`Tuple[int, int, int]`): цвет визуализации элемента на кадре.
- `font` (`int`): идентификатор шрифта OpenCV для вывода текста.
- `scale` (`float`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `thickness` (`int`): толщина линии при рисовании прямоугольников и индикаторов.
Ключевые методы:
- `clone()`"""
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    scale: float = 0.7          # Базовый масштаб (будет умножен на коэффициент)
    thickness: int = 2
    color: Tuple[int, int, int] = (255, 255, 255)  # BGR
    bg_color: Optional[Tuple[int, int, int]] = None  # Цвет фона (None = без фона)
    bg_alpha: float = 0.6       # Прозрачность фона
    
    def clone(self) -> 'TextStyle':
        """Функция: clone()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: 'TextStyle': результат шага обработки, который используется следующим этапом пайплайна."""
        return TextStyle(
            font=self.font,
            scale=self.scale,
            thickness=self.thickness,
            color=self.color,
            bg_color=self.bg_color,
            bg_alpha=self.bg_alpha
        )


@dataclass
class TextPosition:
    """Класс: TextPosition
Назначение: содержит предметную логику текущего модуля.
Поля класса:
- `anchor` (`str`): параметр визуализации, влияющий на внешний вид оверлеев и текста.
- `is_relative` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `x` (`int`): координата X точки или левого края области.
- `y` (`int`): координата Y точки или верхнего края области.
Ключевые методы:
- `clone()`"""
    x: int = 10                 # X координата (абсолютная или процент)
    y: int = 30                 # Y координата (абсолютная или процент)
    is_relative: bool = False   # Если True, x и y в процентах (0.0-1.0)
    anchor: str = "top-left"    # top-left, top-right, bottom-left, bottom-right, center
    
    def clone(self) -> 'TextPosition':
        """Функция: clone()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: 'TextPosition': результат шага обработки, который используется следующим этапом пайплайна."""
        return TextPosition(
            x=self.x,
            y=self.y,
            is_relative=self.is_relative,
            anchor=self.anchor
        )


@dataclass
class ElementConfig:
    """Класс: ElementConfig
Назначение: описывает структуру конфигурации и типизированные данные компонента.
Поля класса:
- `line_spacing` (`int`): межстрочный интервал при рисовании текстовых блоков.
- `position` (`TextPosition`): идентификатор/индекс для адресации и сопоставления сущностей.
- `show_background` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `style` (`TextStyle`): набор параметров визуального стиля отрисовки.
- `title` (`str`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `visible` (`bool`): флаг отображения элемента в UI/визуализации.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    visible: bool = True
    title: str = ""
    position: TextPosition = field(default_factory=TextPosition)
    style: TextStyle = field(default_factory=TextStyle)
    line_spacing: int = 25      # Отступ между строками
    show_background: bool = True


@dataclass
class BBoxConfig:
    """Класс: BBoxConfig
Назначение: описывает структуру конфигурации и типизированные данные компонента.
Поля класса:
- `font_scale` (`float`): масштаб шрифта OpenCV для вывода текста.
- `show_area` (`bool`): логический флаг, включающий/отключающий соответствующее поведение.
- `show_confidence` (`bool`): пороговое/нормирующее значение, влияющее на фильтрацию и принятие решения.
- `show_label` (`bool`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `thickness` (`int`): толщина линии при рисовании прямоугольников и индикаторов.
- `visible` (`bool`): флаг отображения элемента в UI/визуализации.
Ключевые методы:
- Методы не объявлены явно в теле класса."""
    visible: bool = True
    thickness: int = 2
    font_scale: float = 0.5
    show_label: bool = True
    show_confidence: bool = True
    show_area: bool = False


@dataclass  
class VisualizerConfig:
    """Класс: VisualizerConfig
Назначение: описывает структуру конфигурации и типизированные данные компонента.
Поля класса:
- `base_height` (`int`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `base_width` (`int`): геометрический размер или коэффициент масштабирования для расчетов и отрисовки.
- `colors` (`Dict[str, Tuple[int, int, int]]`): словарь/палитра цветов для разных типов нарушений или объектов.
- `dms_bbox` (`BBoxConfig`): данные детектора нарушений, используемые для итогового решения по кадру.
- `forbidden_bbox` (`BBoxConfig`): данные детектора нарушений, используемые для итогового решения по кадру.
- `fps` (`ElementConfig`): частота кадров (кадров/с), используемая для таймингов и видео-вывода.
- `movement` (`ElementConfig`): результат детектора движения/смещения камеры.
- `obstruction` (`ElementConfig`): результат проверки перекрытия объектива/обструкции камеры.
- `video_time` (`ElementConfig`): временной параметр, определяющий интервалы, задержки или длительность этапа.
- `yolo_bbox` (`BBoxConfig`): координаты области или точки, используемые для геометрических вычислений.
Ключевые методы:
- `create_scaled()`, `from_dict()`"""
    
    # Базовое разрешение для масштабирования
    base_width: int = 1280
    base_height: int = 720
    
    # Элементы
    movement: ElementConfig = field(default_factory=lambda: ElementConfig(
        visible=True,
        title="MOVEMENT",
        position=TextPosition(x=10, y=30),
        style=TextStyle(color=(0, 255, 255), scale=0.8)
    ))
    
    obstruction: ElementConfig = field(default_factory=lambda: ElementConfig(
        visible=True,
        title="OBSTRUCTION",
        position=TextPosition(x=10, y=180),
        style=TextStyle(color=(0, 255, 0), scale=0.8)
    ))
    
    video_time: ElementConfig = field(default_factory=lambda: ElementConfig(
        visible=True,
        position=TextPosition(x=-10, y=-50, is_relative=False, anchor="bottom-right"),
        style=TextStyle(color=(255, 255, 0), scale=0.7, bg_color=(0, 0, 0))
    ))
    
    fps: ElementConfig = field(default_factory=lambda: ElementConfig(
        visible=False,  # По умолчанию выключен
        position=TextPosition(x=-10, y=-20, is_relative=False, anchor="bottom-right"),
        style=TextStyle(color=(0, 255, 0), scale=0.7, bg_color=(0, 0, 0))
    ))
    
    # Bounding boxes
    yolo_bbox: BBoxConfig = field(default_factory=lambda: BBoxConfig(
        visible=True,
        thickness=2,
        font_scale=0.5,
        show_label=True,
        show_confidence=True,
        show_area=True
    ))
    
    forbidden_bbox: BBoxConfig = field(default_factory=lambda: BBoxConfig(
        visible=True,
        thickness=2,
        font_scale=0.6,
        show_label=True,
        show_confidence=False,
        show_area=False
    ))
    
    dms_bbox: BBoxConfig = field(default_factory=lambda: BBoxConfig(
        visible=True,
        thickness=2,
        font_scale=0.5,
        show_label=True,
        show_confidence=True,
        show_area=False
    ))
    
    # Цвета
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "white": (255, 255, 255),
        "blue": (255, 0, 0),
        "purple": (255, 0, 255),
        "cyan": (255, 255, 0),
        "dark_green": (0, 100, 0),
        "dark_blue": (139, 0, 0),
        "dark_red": (0, 0, 139),
    })
    
    def create_scaled(self, frame_width: int, frame_height: int) -> 'VisualizerConfig':
        """Функция: create_scaled()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `frame_width` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
- `frame_height` (`int`): кадр/изображение, которое передается на обработку текущему этапу.
Возвращаемое значение: 'VisualizerConfig': результат шага обработки, который используется следующим этапом пайплайна."""
        scale_x = frame_width / self.base_width
        scale_y = frame_height / self.base_height
        scale = min(scale_x, scale_y)
        
        # Создаём копию с применённым масштабом
        scaled = VisualizerConfig(
            base_width=frame_width,
            base_height=frame_height,
        )
        
        # Масштабируем позиции и размеры
        for attr_name in ['movement', 'obstruction', 'video_time', 'fps']:
            src_config = getattr(self, attr_name)
            dst_config = getattr(scaled, attr_name)
            
            # Копируем видимость
            dst_config.visible = src_config.visible
            
            # Масштабируем позицию
            if src_config.position.is_relative:
                dst_config.position.x = int(src_config.position.x * frame_width)
                dst_config.position.y = int(src_config.position.y * frame_height)
            else:
                # Абсолютная позиция с якорем
                if src_config.position.anchor == "top-left":
                    dst_config.position.x = int(src_config.position.x * scale_x)
                    dst_config.position.y = int(src_config.position.y * scale_y)
                elif src_config.position.anchor == "top-right":
                    dst_config.position.x = int(frame_width - src_config.position.x * scale_x)
                    dst_config.position.y = int(src_config.position.y * scale_y)
                elif src_config.position.anchor == "bottom-left":
                    dst_config.position.x = int(src_config.position.x * scale_x)
                    dst_config.position.y = int(frame_height - src_config.position.y * scale_y)
                elif src_config.position.anchor == "bottom-right":
                    dst_config.position.x = int(frame_width + src_config.position.x * scale_x)
                    dst_config.position.y = int(frame_height + src_config.position.y * scale_y)
            
            # Масштабируем стиль
            dst_config.style.scale = src_config.style.scale * scale
            dst_config.style.thickness = max(1, int(src_config.style.thickness * scale))
            dst_config.line_spacing = int(src_config.line_spacing * scale)
        
        # Масштабируем bbox конфиги
        for attr_name in ['yolo_bbox', 'forbidden_bbox', 'dms_bbox']:
            src_bbox = getattr(self, attr_name)
            dst_bbox = getattr(scaled, attr_name)
            dst_bbox.visible = src_bbox.visible
            dst_bbox.thickness = max(1, int(src_bbox.thickness * scale))
            dst_bbox.font_scale = src_bbox.font_scale * scale
        
        return scaled
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualizerConfig':
        """Функция: from_dict()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `data` (`Dict[str, Any]`): словарь данных, передаваемый между функциями или в конфиг-объект.
Возвращаемое значение: 'VisualizerConfig': результат шага обработки, который используется следующим этапом пайплайна."""
        # TODO: Парсинг из YAML/JSON
        return cls()
