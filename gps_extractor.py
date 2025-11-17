"""
Модуль для извлечения GPS координат из видео и работы с координатами
"""
import cv2
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import math


class GPSExtractor:
    """Класс для извлечения и обработки GPS координат"""
    
    def __init__(self, video_path: str = None):
        self.video_path = video_path
        self.gps_data = []
    
    def extract_from_video_metadata(self, video_path: str) -> Optional[Tuple[float, float]]:
        """Извлечение GPS координат из метаданных видео"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Получаем метаданные видео
            metadata = {}
            # OpenCV не всегда может извлечь GPS из видео, используем exifread для дополнительной информации
            try:
                import exifread
                with open(video_path, 'rb') as f:
                    tags = exifread.process_file(f)
                    # Ищем GPS теги
                    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                        lat = self._parse_gps_coordinate(tags['GPS GPSLatitude'].values)
                        lon = self._parse_gps_coordinate(tags['GPS GPSLongitude'].values)
                        
                        # Проверяем направление
                        if 'GPS GPSLatitudeRef' in tags:
                            if tags['GPS GPSLatitudeRef'].values == 'S':
                                lat = -lat
                        if 'GPS GPSLongitudeRef' in tags:
                            if tags['GPS GPSLongitudeRef'].values == 'W':
                                lon = -lon
                        
                        return (lat, lon)
            except ImportError:
                pass
            except Exception as e:
                print(f"Ошибка при извлечении GPS из метаданных: {e}")
            
            cap.release()
        except Exception as e:
            print(f"Ошибка при открытии видео: {e}")
        
        return None
    
    def _parse_gps_coordinate(self, values: List) -> float:
        """Парсинг GPS координат из формата градусы/минуты/секунды"""
        degrees = float(values[0].num) / float(values[0].den)
        minutes = float(values[1].num) / float(values[1].den)
        seconds = float(values[2].num) / float(values[2].den)
        return degrees + minutes / 60.0 + seconds / 3600.0
    
    def calculate_position_by_time(
        self, 
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        current_frame: int,
        total_frames: int,
        speed: float = None
    ) -> Tuple[float, float]:
        """Вычисление позиции на основе времени и начальных/конечных координат"""
        if total_frames == 0:
            return start_coords
        
        progress = current_frame / total_frames
        lat = start_coords[0] + (end_coords[0] - start_coords[0]) * progress
        lon = start_coords[1] + (end_coords[1] - start_coords[1]) * progress
        
        return (lat, lon)
    
    def interpolate_coordinates(
        self,
        frame_number: int,
        fps: float,
        start_coords: Optional[Tuple[float, float]] = None,
        end_coords: Optional[Tuple[float, float]] = None,
        speed_kmh: float = 50.0,
        initial_direction: float = 0.0
    ) -> Tuple[float, float]:
        """
        Интерполяция координат на основе скорости и направления
        
        Args:
            frame_number: Номер кадра
            fps: Частота кадров в секунду
            start_coords: Начальные координаты (если известны)
            end_coords: Конечные координаты (если известны)
            speed_kmh: Скорость в км/ч
            initial_direction: Начальное направление в градусах
        
        Returns:
            Tuple[float, float]: (широта, долгота)
        """
        if start_coords is None:
            # Примерные координаты (можно задать вручную)
            start_coords = (55.7558, 37.6173)  # Москва по умолчанию
        
        # Вычисляем время в секундах
        time_seconds = frame_number / fps
        
        # Скорость в метрах в секунду
        speed_ms = speed_kmh / 3.6
        
        # Пройденное расстояние в метрах
        distance_m = speed_ms * time_seconds
        
        # Конвертируем расстояние в градусы (приблизительно)
        # 1 градус широты ≈ 111 км
        # 1 градус долготы ≈ 111 км * cos(широта)
        lat_delta = distance_m / 111000.0
        lon_delta = distance_m / (111000.0 * math.cos(math.radians(start_coords[0])))
        
        # Вычисляем новые координаты
        direction_rad = math.radians(initial_direction)
        new_lat = start_coords[0] + lat_delta * math.cos(direction_rad)
        new_lon = start_coords[1] + lon_delta * math.sin(direction_rad)
        
        return (new_lat, new_lon)
    
    def bbox_to_gps(
        self,
        bbox: Tuple[int, int, int, int],
        frame_center_coords: Tuple[float, float],
        frame_width: int,
        frame_height: int,
        focal_length: float = 700.0,
        altitude: float = 5.0
    ) -> Tuple[float, float]:
        """
        Преобразование координат bbox на изображении в GPS координаты
        
        Args:
            bbox: Координаты bbox (x1, y1, x2, y2)
            frame_center_coords: GPS координаты центра кадра
            frame_width: Ширина кадра
            frame_height: Высота кадра
            focal_length: Фокусное расстояние камеры
            altitude: Высота камеры над землей в метрах
        
        Returns:
            Tuple[float, float]: GPS координаты объекта
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Смещение от центра кадра в пикселях
        dx = center_x - frame_width / 2
        dy = center_y - frame_height / 2
        
        # Преобразуем в углы (упрощенная модель)
        angle_x = math.atan(dx / focal_length)
        angle_y = math.atan(dy / focal_length)
        
        # Вычисляем смещение в метрах
        distance_x = altitude * math.tan(angle_x)
        distance_y = altitude * math.tan(angle_y)
        
        # Конвертируем в градусы
        lat_delta = distance_y / 111000.0
        lon_delta = distance_x / (111000.0 * math.cos(math.radians(frame_center_coords[0])))
        
        # Новые координаты
        new_lat = frame_center_coords[0] + lat_delta
        new_lon = frame_center_coords[1] + lon_delta
        
        return (new_lat, new_lon)

