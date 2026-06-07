"""
Модуль для расчета пикетажа (расстояния от начальной точки дороги)
Используется в дорожном строительстве для определения местоположения объектов
"""

from typing import Tuple, Optional
import cv2


class PicketTracker:
    """Класс для отслеживания пикетажа во время обработки видео"""
    
    def __init__(self, video_path: str, speed_kmh: float = 50.0, start_picket_m: float = 0.0):
        """
        Инициализация трекера пикетажа
        
        Args:
            video_path: Путь к видео файлу
            speed_kmh: Средняя скорость движения в км/ч (по умолчанию 50)
            start_picket_m: Начальный пикетаж в метрах (по умолчанию 0)
        """
        self.video_path = video_path
        self.speed_kmh = speed_kmh
        self.start_picket_m = start_picket_m
        
        # Параметры видео
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None
        
        print(f"📏 Пикетаж: начальная точка {self.format_picket(start_picket_m)}, скорость {speed_kmh} км/ч")
    
    def setup_video(self, cap: cv2.VideoCapture):
        """
        Настройка параметров видео
        
        Args:
            cap: Открытый VideoCapture объект
        """
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.total_frames > 0:
            total_distance = self.calculate_distance_for_frame(self.total_frames)
            print(f"📐 Видео: {self.total_frames} кадров, "
                  f"примерная длина участка {total_distance:.0f}м "
                  f"({self.format_picket(self.start_picket_m)} - {self.format_picket(self.start_picket_m + total_distance)})")
    
    def calculate_distance_for_frame(self, frame_number: int) -> float:
        """
        Вычисление пройденного расстояния от начальной точки для кадра
        
        Args:
            frame_number: Номер кадра (начиная с 0)
        
        Returns:
            float: Расстояние в метрах от начальной точки
        """
        if self.fps is None or self.fps == 0:
            return 0.0
        
        # Время в секундах
        time_seconds = frame_number / self.fps
        
        # Скорость в метрах в секунду
        speed_ms = self.speed_kmh / 3.6
        
        # Пройденное расстояние в метрах
        distance_m = speed_ms * time_seconds
        
        return distance_m
    
    def get_frame_picket(self, frame_number: int) -> float:
        """
        Получение пикетажа для текущего кадра
        
        Args:
            frame_number: Номер кадра (начиная с 0)
        
        Returns:
            float: Пикетаж в метрах (абсолютное значение от начала дороги)
        """
        distance = self.calculate_distance_for_frame(frame_number)
        return self.start_picket_m + distance
    
    def get_object_picket(self, bbox: Tuple[int, int, int, int], frame_number: int) -> float:
        """
        Получение пикетажа объекта с учетом его положения на кадре
        
        Args:
            bbox: Координаты bbox (x1, y1, x2, y2)
            frame_number: Номер кадра
        
        Returns:
            float: Пикетаж объекта в метрах
        """
        # Базовый пикетаж для кадра
        frame_picket = self.get_frame_picket(frame_number)
        
        # Корректировка на основе положения объекта на кадре
        # Объекты в верхней части кадра находятся дальше по дороге
        if self.height:
            x1, y1, x2, y2 = bbox
            object_y = (y1 + y2) / 2
            
            # Нормализованная позиция по вертикали (0 = верх кадра, 1 = низ кадра)
            normalized_y = object_y / self.height
            
            # Оценка дополнительного расстояния (объекты вверху на 50-100м дальше)
            # Используем обратную зависимость: чем выше объект, тем он дальше
            additional_distance = (1.0 - normalized_y) * 75.0  # До 75 метров корректировки
            
            return frame_picket + additional_distance
        
        return frame_picket
    
    @staticmethod
    def format_picket(picket_m: float) -> str:
        """
        Форматирование пикетажа в стандартный вид КМ+метры
        
        Args:
            picket_m: Пикетаж в метрах
        
        Returns:
            str: Форматированный пикетаж (например: "КМ 0+325", "КМ 1+528")
        """
        km = int(picket_m // 1000)
        m = int(picket_m % 1000)
        return f"КМ {km}+{m:03d}"
    
    @staticmethod
    def format_picket_simple(picket_m: float) -> str:
        """
        Упрощенное форматирование пикетажа (только метры)
        
        Args:
            picket_m: Пикетаж в метрах
        
        Returns:
            str: Форматированный пикетаж (например: "325м", "1528м")
        """
        return f"{int(picket_m)}м"
    
    def set_speed(self, speed_kmh: float):
        """
        Изменение скорости движения
        
        Args:
            speed_kmh: Новая скорость в км/ч
        """
        self.speed_kmh = speed_kmh
        print(f"📏 Скорость изменена на {speed_kmh} км/ч")
    
    def set_start_picket(self, start_picket_m: float):
        """
        Изменение начального пикетажа
        
        Args:
            start_picket_m: Новый начальный пикетаж в метрах
        """
        self.start_picket_m = start_picket_m
        print(f"📏 Начальный пикетаж изменен на {self.format_picket(start_picket_m)}")
