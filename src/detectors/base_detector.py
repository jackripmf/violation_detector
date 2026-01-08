from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Базовый класс для всех детекторов"""

    def __init__(self, name):
        self.name = name
        self.is_active = True

    @abstractmethod
    def detect(self, frame):
        """Основной метод детекции"""
        pass

    def enable(self):
        """Активация детектора"""
        self.is_active = True

    def disable(self):
        """Деактивация детектора"""
        self.is_active = False
