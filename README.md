# Детектор перекрытий камеры и поворотов 
**Функционал:**
1. Детекция перекрытий камеры:
- Компьютерное зрение (анализ яркости, резкости, контраста)
- Детекция темных областей
- YOLO - детекция
2. Детекция движений камеры:
- Анализ поворотов (детекция угловых смещений)
- Детекция смещений (обнаружение линейных перемещений)
- Фильтрация ложных срабатываний (подтверждение движений по времени и кадрам)
3. Система сохранений нарушений:
- Скриншоты нарушений
- Видео отрезки с движениями камеры
- Текстовые отчеты

**Требования:**
- OpenCV
- Ultralytics
- NumPy

**Структура**
- violation_detector: *# основная директория проекта*
  - main.py *# MAIN*
  - README.md *# README*
  - requirements.txt *# REQIREMENTS*
  - examples: *# Директория с примерами запуска*
    - test.py
    - your_video.mp4
    - yolov8n.pt *# Легковесная версия yolo*
  - obstruction_detector: *# директория модулей для детектора*
  - __init__.py
  - utils.py *# Вспомогательные функции (получение информации о видео, времени...)*
    - detectors: *# Директория детекторов*
      - __init__.py
      - base_detector.py *# Базовый детектор*
      - movement_detector.py *# Детектор движений*
      - obstruction_detector.py *# Детектор перекрытий*
    - outputs *# Директория выходных данных (отчеты, визуализация)*
      - __init__.py
      - file_manager.py *# модули для отчетов*
      - visualizer.py *# отрисовка информации на кадрах*
    - processors *# процессоры программы*
      - __init__.py
      - video_processor.py *# процессор для анализа предзаписанного видео*
      - webcam_processor.py *# процессор для анализа видеопотока (например с вебкамеры)*

# ОСНОВНЫЕ ПАРАМЕТРЫ
**С помощью этих параметров можно калибровать работу скрипта в соответствии с задачей**

```python
class CVDetector(BaseDetector):
    ...
        self.brightness_thresh = brightness_thresh # Порог яркости (по умолчанию = 25)
        self.contrast_thresh = contrast_thresh # Порог контраста (по умолчанию = 10)
        self.sharpness_thresh = sharpness_thresh # Порог резкости/блюр (по умолчанию = 20)
    
class DarkAreaDetector(BaseDetector):
    ...
        self.dark_area_threshold = dark_area_threshold # Порог темных областей (по умолчанию = 0.8)
    
class YOLODetector(BaseDetector):
    ...
        self.area_threshold = area_threshold # Порог по площади перекрытия кадра объектом (по умолчанию = 0.7)
        self.conf_threshold = conf_threshold # Уверенность в идентификации класса объекта (по умолчанию = 0.5)
```
```python
class CameraMovementDetector(BaseDetector):
    ...
        self.rotation_treshold = rotation_treshold      # порог поворота (по умолчанию = 3.0)
        self.translation_treshold = translation_treshold     # порог смещения (по умолчанию = 0.05)
        self.min_matches = min_matches # минимальное совпадение точек необходимых для расчета (по умолчанию = 10)
        # Параметры фильтрации
        self.min_movement_duration = min_movement_duration      # минимальная продолжительность движения (по умолчанию = 1.0)
        self.confirmation_frames = confirmation_frames      # последовательные кадры с движением (по умолчанию = 3)
        self.movement_start_time = None     # начало движения
        self.consecutive_movement_frames = 0    # счетчик последовательности кадров с движением
        self.last_movement_detected = False     # конец движения
        # Инициализация детектора особенностей
        self.orb = cv2.ORB_create(nfeatures=1000) # Количество точек интереса
        # Референсный кадр
        self.reference_frame = None # Референсный кадр
        self.reference_kp = None # массив точек интереса на референсном кадре
        self.reference_des = None # вектор дескрипторов точек интереса на референсном кадре
```
```python
        # Инициализация детекторов
        self.movement_detector = CameraMovementDetector() # движение
        self.cv_detector = CVDetector() # детектор по метрикам OpenCV
        self.dark_area_detector = DarkAreaDetector() # детектор темных областей
        self.yolo_detector = YOLODetector() # детектор yolo

        # Управление файлами
        self.file_manager = FileManager(save_dir) # выбор директории для отчетов
        self.visualizer = Visualizer() # инициализация отрисовщика информации

        # Состояние системы
        self.alert_count = 0 # Счетчик нарушений
        self.is_obstruction_active = False # Присутствует ли перекрытие в кадре
        self.last_violation_time = None # Последняя временная метка нарушения
        self.violation_cooldown = 5 # Задержка между нарушениями
        self.obstruction_start_time = None # Временная метка начала перекрытия
        self.min_obstruction_duration = 5 # Минимальная продолжительность перекрытия, которую нужно фиксировать
        self.gap_threshold = 3 # Максимальный разрыв между сегментов видео с нарушениями (если меньше = 1 нарушение) 
        self.capture_frame_at = 3 # Скриншот нарушения (не ранее этой временной метки и с уверенностью HIGH)

        # Статистика
        self.stats = {
            "total_detections": 0, # Найдено нарушений перекрытия
            "saved_violations": 0, # Сохранено нарушений перекрытия
            "camera_movements": 0, # Найдено движений камеры
            "movement_violations_saved": 0 # Сохранено движений камеры
        }
        self.frame_count = 0 # Счетчик кадров

        # Сегменты нарушений
        self.obstruction_segments = [] # Сегменты, где фиксируется движение
        self.current_segment_start = None # Текущий сегмент с движением
        self.frame_to_save = None # Сохранение нарушения
        self.saved_for_current_violation = False # Флаг, показывающий сохраняли это нарушение или еще нет

        # Запись видео
        self.movement_video_writer = None # Запись видео
        self.movement_video_start_time = None # Метка начала записи
        self.is_recording_movement = False # Флаг, показывающий ведется запись или нет
        self.movement_violation_count = 0 # Счетчик движений
        self.current_video_path = None # Путь к файлу

        # Превью
        self.preview_writer = None # Запись превью
```
# Базовый запуск
*Для предзаписанного видео:*

*Терминал:*

python main.py --input test_video.mp4

или

```python
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import process_video

if __name__== "__main__":
    process_video(
        input_video='WIN_20251026_16_26_39_Pro.mp4',
        save_dir='my_violations',
        show_preview=True
    )
```
*Для веб-камеры:*

*Терминал:*

python main.py --input 0

или
```python
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import process_webcam

if __name__ == "__main__":
  process_webcam(
    camera_id=0,
    save_dir='webcam_violations',
    show_preview=True
  )
```

Результат:
По завершению анализа видео, в директории, указанной для сохранения нарушений, сохранятся скриншоты
с перекрытиями камеры и отчеты формата .txt к ним, а так же отрезки видео, где фиксировалось движение
 и так же отчет. Кроме того, по умолчанию записывается превью всего цикла работы программы с визуализацией

# Возможные проблемы
Если веб камера или видеопоток не читаются, нужно передать в класс вторым аргументом
в cv2.VideoCapture *(self.camera_id, cv2.CAP_DSHOW)* и *(self.camera_id, cv2.CAP_FFMPEG)* соответственно