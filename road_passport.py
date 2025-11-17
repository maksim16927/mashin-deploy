"""
Модуль для создания паспорта дороги
"""
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import asdict
from road_detector import RoadObject, RoadMarking, LaneInfo
from gps_extractor import GPSExtractor


class RoadPassport:
    """Класс для формирования паспорта дороги"""
    
    def __init__(self):
        self.objects = []
        self.markings = []
        self.lane_infos = []
        self.gps_extractor = GPSExtractor()
    
    def add_object(
        self,
        obj: RoadObject,
        frame_coordinates: Optional[Tuple[float, float]] = None,
        frame_number: int = None
    ):
        """Добавление объекта в паспорт"""
        # Если координаты не заданы, пытаемся вычислить
        if obj.coordinates is None and frame_coordinates:
            # Упрощенное вычисление координат объекта на основе координат кадра
            obj.coordinates = self._estimate_object_coordinates(
                obj.bbox,
                frame_coordinates
            )
        
        # Добавляем временную метку
        if obj.timestamp is None:
            obj.timestamp = datetime.now()
        
        self.objects.append(obj)
    
    def add_marking(
        self,
        marking: RoadMarking,
        frame_coordinates: Optional[Tuple[float, float]] = None
    ):
        """Добавление разметки в паспорт"""
        if frame_coordinates:
            # Упрощенное вычисление координат для начальной и конечной точек
            marking.coordinates_start = self._estimate_object_coordinates(
                (marking.start_point[0], marking.start_point[1], 
                 marking.start_point[0] + 10, marking.start_point[1] + 10),
                frame_coordinates
            )
            marking.coordinates_end = self._estimate_object_coordinates(
                (marking.end_point[0], marking.end_point[1],
                 marking.end_point[0] + 10, marking.end_point[1] + 10),
                frame_coordinates
            )
        
        if marking.timestamp is None:
            marking.timestamp = datetime.now()
        
        self.markings.append(marking)
    
    def add_lane_info(self, lane_info: LaneInfo):
        """Добавление информации о полосах"""
        if lane_info.timestamp is None:
            lane_info.timestamp = datetime.now()
        
        self.lane_infos.append(lane_info)
    
    def _estimate_object_coordinates(
        self,
        bbox: Tuple[int, int, int, int],
        frame_coords: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Оценка GPS координат объекта на основе координат кадра"""
        # Упрощенная модель - возвращаем координаты кадра
        # В реальной системе нужно учитывать положение объекта в кадре,
        # высоту камеры, угол обзора и т.д.
        if frame_coords is None:
            return None
        return frame_coords
    
    def to_dataframe(self) -> pd.DataFrame:
        """Преобразование паспорта в DataFrame"""
        data = []
        
        # Добавляем объекты
        for obj in self.objects:
            if obj.coordinates and len(obj.coordinates) == 2:
                coords_str = f"{obj.coordinates[0]:.6f}, {obj.coordinates[1]:.6f}"
            else:
                coords_str = "неизвестно"
            data.append({
                'Тип объекта': obj.object_type,
                'Название': obj.class_name,
                'Координаты': coords_str,
                'Дата и время': obj.timestamp.strftime('%Y-%m-%d %H:%M:%S') if obj.timestamp else '',
                'Уверенность': f"{obj.confidence:.2f}",
                'Доп. информация': json.dumps(obj.additional_info) if obj.additional_info else ''
            })
        
        # Добавляем разметку
        for marking in self.markings:
            if marking.coordinates_start and len(marking.coordinates_start) == 2:
                coords_start = f"{marking.coordinates_start[0]:.6f}, {marking.coordinates_start[1]:.6f}"
            else:
                coords_start = "неизвестно"
            
            if marking.coordinates_end and len(marking.coordinates_end) == 2:
                coords_end = f"{marking.coordinates_end[0]:.6f}, {marking.coordinates_end[1]:.6f}"
            else:
                coords_end = "неизвестно"
            
            if marking.marking_type == "пешеходный_переход":
                coords_full = f"От: {coords_start}, До: {coords_end}"
            else:
                coords_full = f"От: {coords_start}, До: {coords_end}"
            
            data.append({
                'Тип объекта': 'разметка',
                'Название': self._translate_marking_type(marking.marking_type),
                'Координаты': coords_full,
                'Дата и время': marking.timestamp.strftime('%Y-%m-%d %H:%M:%S') if marking.timestamp else '',
                'Уверенность': '',
                'Доп. информация': ''
            })
        
        # Добавляем информацию о полосах
        for lane_info in self.lane_infos:
            if lane_info.lane_count > 0:
                data.append({
                    'Тип объекта': 'инфраструктура',
                    'Название': f'Количество полос: {lane_info.lane_count}',
                    'Координаты': 'по всей дороге',
                    'Дата и время': lane_info.timestamp.strftime('%Y-%m-%d %H:%M:%S') if lane_info.timestamp else '',
                    'Уверенность': '',
                    'Доп. информация': f'Ширина полосы (пиксели): {lane_info.lane_width_pixels:.1f}'
                })
        
        return pd.DataFrame(data)
    
    def _translate_marking_type(self, marking_type: str) -> str:
        """Перевод типа разметки на русский"""
        translations = {
            'сплошная': 'Сплошная линия',
            'прерывистая': 'Прерывистая линия',
            'двойная_сплошная': 'Двойная сплошная линия',
            'пешеходный_переход': 'Пешеходный переход'
        }
        return translations.get(marking_type, marking_type)
    
    def to_json(self, filepath: str = None) -> str:
        """Экспорт паспорта в JSON"""
        passport_data = {
            'objects': [asdict(obj) for obj in self.objects],
            'markings': [asdict(marking) for marking in self.markings],
            'lane_infos': [asdict(lane_info) for lane_info in self.lane_infos]
        }
        
        # Конвертируем datetime в строки
        for obj in passport_data['objects']:
            if obj.get('timestamp'):
                obj['timestamp'] = obj['timestamp'].isoformat() if isinstance(obj['timestamp'], datetime) else str(obj['timestamp'])
        
        for marking in passport_data['markings']:
            if marking.get('timestamp'):
                marking['timestamp'] = marking['timestamp'].isoformat() if isinstance(marking['timestamp'], datetime) else str(marking['timestamp'])
        
        for lane_info in passport_data['lane_infos']:
            if lane_info.get('timestamp'):
                lane_info['timestamp'] = lane_info['timestamp'].isoformat() if isinstance(lane_info['timestamp'], datetime) else str(lane_info['timestamp'])
        
        json_str = json.dumps(passport_data, ensure_ascii=False, indent=2)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def to_csv(self, filepath: str):
        """Экспорт паспорта в CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    def to_excel(self, filepath: str):
        """Экспорт паспорта в Excel"""
        df = self.to_dataframe()
        df.to_excel(filepath, index=False, engine='openpyxl')

