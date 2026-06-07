"""
Модуль для отслеживания GPS координат во время обработки видео
Привязывает GPS координаты к каждому обнаруженному объекту
"""

from typing import Optional, Tuple, List
from pathlib import Path
import cv2

try:
    from .gps_extractor import GPSExtractor
except ImportError:
    from gps_extractor import GPSExtractor


class GPSTracker:
    """Класс для отслеживания GPS координат во время обработки видео"""
    
    def __init__(self, video_path: str, start_coords: Optional[Tuple[float, float]] = None,
                 end_coords: Optional[Tuple[float, float]] = None, speed_kmh: float = 50.0):
        """
        Инициализация GPS трекера
        
        Args:
            video_path: Путь к видео файлу
            start_coords: Начальные GPS координаты (lat, lon). Если None - попытается извлечь из видео
            end_coords: Конечные GPS координаты (lat, lon). Если None - будет интерполяция
            speed_kmh: Средняя скорость в км/ч для интерполяции (если нет конечных координат)
        """
        self.video_path = video_path
        self.gps_extractor = GPSExtractor()
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.speed_kmh = speed_kmh
        
        # Параметры камеры для вычисления координат объектов
        self.focal_length = 700.0  # Фокусное расстояние (пиксели)
        self.altitude = 5.0  # Высота камеры над землей (метры)
        
        # Извлекаем GPS из метаданных видео, если возможно
        self._extract_gps_from_video()
        
        # Параметры видео (будут установлены при открытии)
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None
    
    def _extract_gps_from_video(self):
        """Попытка извлечь GPS координаты из метаданных видео"""
        if self.start_coords is None:
            coords = self.gps_extractor.extract_from_video_metadata(self.video_path)
            if coords:
                self.start_coords = coords
                print(f"✅ GPS координаты извлечены из видео: {self.start_coords}")
            else:
                # Координаты по умолчанию (можно задать вручную)
                self.start_coords = (55.7558, 37.6173)  # Москва
                print(f"⚠️  GPS координаты не найдены в видео. Используются координаты по умолчанию: {self.start_coords}")
    
    def setup_video(self, cap: cv2.VideoCapture):
        """
        Настройка параметров видео для GPS трекинга
        
        Args:
            cap: Открытый VideoCapture объект
        """
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Если есть конечные координаты, используем их для интерполяции
        if self.end_coords is None and self.total_frames > 0:
            # Вычисляем примерные конечные координаты на основе скорости
            # (упрощенная модель - движение по прямой)
            time_seconds = self.total_frames / self.fps
            speed_ms = self.speed_kmh / 3.6
            distance_m = speed_ms * time_seconds
            
            # Предполагаем движение на север (можно улучшить, используя направление)
            lat_delta = distance_m / 111000.0
            lon_delta = 0.0  # Для упрощения
            
            self.end_coords = (
                self.start_coords[0] + lat_delta,
                self.start_coords[1] + lon_delta
            )
    
    def get_frame_coordinates(self, frame_number: int) -> Tuple[float, float]:
        """
        Получение GPS координат центра текущего кадра
        
        Args:
            frame_number: Номер кадра (начиная с 0)
        
        Returns:
            Tuple[float, float]: GPS координаты (lat, lon) центра кадра
        """
        if self.total_frames is None or self.total_frames == 0:
            return self.start_coords
        
        if self.end_coords:
            # Линейная интерполяция между начальными и конечными координатами
            progress = frame_number / self.total_frames
            lat = self.start_coords[0] + (self.end_coords[0] - self.start_coords[0]) * progress
            lon = self.start_coords[1] + (self.end_coords[1] - self.start_coords[1]) * progress
            return (lat, lon)
        else:
            # Интерполяция на основе скорости
            return self.gps_extractor.interpolate_coordinates(
                frame_number=frame_number,
                fps=self.fps,
                start_coords=self.start_coords,
                speed_kmh=self.speed_kmh,
                initial_direction=0.0  # Направление на север (можно улучшить)
            )
    
    def get_object_coordinates(
        self,
        bbox: Tuple[int, int, int, int],
        frame_number: int
    ) -> Tuple[float, float]:
        """
        Получение GPS координат объекта на основе его bbox и номера кадра
        
        Args:
            bbox: Координаты bbox (x1, y1, x2, y2)
            frame_number: Номер кадра
        
        Returns:
            Tuple[float, float]: GPS координаты объекта (lat, lon)
        """
        # Получаем координаты центра кадра
        frame_coords = self.get_frame_coordinates(frame_number)
        
        # Преобразуем bbox в GPS координаты
        if self.width and self.height:
            object_coords = self.gps_extractor.bbox_to_gps(
                bbox=bbox,
                frame_center_coords=frame_coords,
                frame_width=self.width,
                frame_height=self.height,
                focal_length=self.focal_length,
                altitude=self.altitude
            )
            return object_coords
        
        # Если размеры кадра не установлены, возвращаем координаты центра кадра
        return frame_coords
    
    def set_camera_params(self, focal_length: float = None, altitude: float = None):
        """
        Установка параметров камеры для более точного вычисления координат
        
        Args:
            focal_length: Фокусное расстояние в пикселях
            altitude: Высота камеры над землей в метрах
        """
        if focal_length is not None:
            self.focal_length = focal_length
        if altitude is not None:
            self.altitude = altitude
