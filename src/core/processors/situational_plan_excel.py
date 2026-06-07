"""
Модуль создания Excel отчета ситуационного плана
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import os
from pathlib import Path


class SituationalPlanExcelGenerator:
    """Генератор Excel отчета ситуационного плана"""
    
    def __init__(self):
        """Инициализация генератора"""
        self.object_type_mapping = {
            'sign': 'Дорожный знак',
            'barrier': 'Барьер/ограждение',
            'exit': 'Съезд с дороги',
            'noise_strip': 'Шумовая полоса',
            'gantry': 'Опора/портал',
            'vehicle': 'Транспорт',
            'traffic_light': 'Светофор'
        }
    
    def create_excel_report(self, 
                           detections: List[Dict], 
                           video_info: Dict,
                           output_path: Optional[str] = None) -> str:
        """
        Создание Excel отчета ситуационного плана
        
        Args:
            detections: Список всех детекций с пикетажем
            video_info: Информация о видео
            output_path: Путь для сохранения отчета
            
        Returns:
            str: Путь к созданному Excel файлу
        """
        if not detections:
            raise ValueError("Нет детекций для создания отчета")
        
        # Подготавливаем данные
        report_data = self._prepare_report_data(detections)
        
        # Создаем Excel файл с несколькими листами
        if output_path is None:
            # Получаем корневую директорию проекта
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent  # До mashin/
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = project_root / "videos" / "excel_reports" / f"situational_plan_{timestamp}.xlsx"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
            # Лист 1: Общая сводка
            self._create_summary_sheet(report_data, video_info, writer)
            
            # Лист 2: Детальный список всех объектов
            self._create_detailed_sheet(report_data, writer)
            
            # Лист 3: Группировка по типам объектов
            self._create_grouped_sheet(report_data, writer)
            
            # Лист 4: Статистика по участкам дороги
            self._create_statistics_sheet(report_data, writer)
        
        return str(output_path)
    
    def _prepare_report_data(self, detections: List[Dict]) -> List[Dict]:
        """Подготовка данных для отчета (данные уже дедуплицированы)"""
        report_data = []
        
        for i, detection in enumerate(detections, 1):
            # Базовая информация
            obj_type = detection.get('object_type', 'unknown')
            class_name = detection.get('class_name', 'Неизвестно')
            confidence = detection.get('confidence', 0.0)
            picket = detection.get('picket', 0.0)
            frame_number = detection.get('frame_number', 0)
            timestamp = detection.get('timestamp', 0.0)
            bbox = detection.get('bbox', (0, 0, 0, 0))
            
            # Форматируем пикетаж
            km = int(picket // 1000)
            m = int(picket % 1000)
            picket_formatted = f"КМ {km}+{m:03d}"
            
            # Форматируем время
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_formatted = f"{minutes:02d}:{seconds:02d}"
            
            # Определяем сторону дороги
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x_center = (bbox[0] + bbox[2]) / 2
                if x_center < 213:  # Левая треть кадра
                    road_side = "Левая сторона"
                elif x_center > 427:  # Правая треть кадра  
                    road_side = "Правая сторона"
                else:
                    road_side = "Центр дороги"
                
                # Размер объекта
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
            else:
                road_side = "Неизвестно"
                x_center = 320
                width = height = area = 0
            
            report_item = {
                '№': i,
                'Тип объекта': self.object_type_mapping.get(obj_type, obj_type),
                'Класс': class_name,
                'Пикетаж': picket_formatted,
                'Пикетаж (м)': int(picket),
                'Время видео': time_formatted,
                'Кадр №': frame_number,
                'Сторона дороги': road_side,
                'Уверенность (%)': round(confidence * 100, 1),
                'Координаты X': int(x_center),
                'Размер (пикс²)': int(area),
                'Ширина': int(width),
                'Высота': int(height)
            }
            
            report_data.append(report_item)
        
        return report_data
    
    def _create_summary_sheet(self, report_data: List[Dict], video_info: Dict, writer):
        """Создание листа общей сводки"""
        # Статистика по типам объектов
        type_stats = {}
        for item in report_data:
            obj_type = item['Тип объекта']
            if obj_type not in type_stats:
                type_stats[obj_type] = 0
            type_stats[obj_type] += 1
        
        # Создаем DataFrame для сводки
        summary_data = []
        
        # Общая информация
        summary_data.append(['ОБЩАЯ ИНФОРМАЦИЯ', ''])
        summary_data.append(['Дата создания отчета', datetime.now().strftime("%d.%m.%Y %H:%M")])
        summary_data.append(['Название видео', video_info.get('filename', 'Неизвестно')])
        summary_data.append(['Длительность видео', f"{video_info.get('duration', 0):.1f} сек"])
        summary_data.append(['Уникальных объектов (после дедупликации)', len(report_data)])
        summary_data.append(['', ''])
        
        # Диапазон пикетажа
        if report_data:
            min_picket = min(item['Пикетаж (м)'] for item in report_data)
            max_picket = max(item['Пикетаж (м)'] for item in report_data)
            distance = max_picket - min_picket
            
            summary_data.append(['УЧАСТОК ДОРОГИ', ''])
            summary_data.append(['Начальный пикетаж', f"КМ {min_picket//1000}+{min_picket%1000:03.0f}"])
            summary_data.append(['Конечный пикетаж', f"КМ {max_picket//1000}+{max_picket%1000:03.0f}"])
            summary_data.append(['Протяженность участка', f"{distance:.0f} м"])
            summary_data.append(['', ''])
        
        # Статистика по типам
        summary_data.append(['СТАТИСТИКА ПО ТИПАМ ОБЪЕКТОВ', ''])
        summary_data.append(['(после дедупликации близких объектов)', ''])
        for obj_type, count in sorted(type_stats.items()):
            percentage = (count / len(report_data)) * 100
            summary_data.append([obj_type, f"{count} шт. ({percentage:.1f}%)"])
        
        summary_df = pd.DataFrame(summary_data, columns=['Параметр', 'Значение'])
        summary_df.to_excel(writer, sheet_name='Общая сводка', index=False)
        
        # Форматирование
        worksheet = writer.sheets['Общая сводка']
        worksheet.column_dimensions['A'].width = 30
        worksheet.column_dimensions['B'].width = 25
    
    def _create_detailed_sheet(self, report_data: List[Dict], writer):
        """Создание листа с детальным списком"""
        df = pd.DataFrame(report_data)
        df.to_excel(writer, sheet_name='Детальный список', index=False)
        
        # Форматирование
        worksheet = writer.sheets['Детальный список']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    def _create_grouped_sheet(self, report_data: List[Dict], writer):
        """Создание листа с группировкой по типам"""
        df = pd.DataFrame(report_data)
        
        # Группируем по типам объектов
        grouped_data = []
        
        for obj_type in df['Тип объекта'].unique():
            type_data = df[df['Тип объекта'] == obj_type].copy()
            type_data = type_data.sort_values('Пикетаж (м)')
            
            # Добавляем заголовок группы
            grouped_data.append({
                '№': f"=== {obj_type.upper()} ({len(type_data)} шт.) ===",
                'Тип объекта': '',
                'Класс': '',
                'Пикетаж': '',
                'Время видео': '',
                'Сторона дороги': '',
                'Уверенность (%)': ''
            })
            
            # Добавляем объекты этого типа
            for _, row in type_data.iterrows():
                grouped_data.append({
                    '№': row['№'],
                    'Тип объекта': row['Тип объекта'],
                    'Класс': row['Класс'],
                    'Пикетаж': row['Пикетаж'],
                    'Время видео': row['Время видео'],
                    'Сторона дороги': row['Сторона дороги'],
                    'Уверенность (%)': row['Уверенность (%)']
                })
            
            # Добавляем пустую строку
            grouped_data.append({col: '' for col in ['№', 'Тип объекта', 'Класс', 'Пикетаж', 'Время видео', 'Сторона дороги', 'Уверенность (%)']})
        
        grouped_df = pd.DataFrame(grouped_data)
        grouped_df.to_excel(writer, sheet_name='По типам объектов', index=False)
        
        # Форматирование
        worksheet = writer.sheets['По типам объектов']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 25)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    def _create_statistics_sheet(self, report_data: List[Dict], writer):
        """Создание листа со статистикой по участкам"""
        if not report_data:
            return
        
        df = pd.DataFrame(report_data)
        
        # Разбиваем на участки по 500 метров
        min_picket = df['Пикетаж (м)'].min()
        max_picket = df['Пикетаж (м)'].max()
        
        segment_size = 500  # метров
        segments = []
        
        current_start = int(min_picket // segment_size) * segment_size
        
        while current_start <= max_picket:
            current_end = current_start + segment_size
            
            # Фильтруем объекты в этом сегменте
            segment_data = df[(df['Пикетаж (м)'] >= current_start) & (df['Пикетаж (м)'] < current_end)]
            
            if len(segment_data) > 0:
                # Статистика по типам в сегменте
                type_counts = segment_data['Тип объекта'].value_counts()
                
                segment_info = {
                    'Участок': f"КМ {current_start//1000}+{current_start%1000:03.0f} - КМ {current_end//1000}+{current_end%1000:03.0f}",
                    'Протяженность (м)': segment_size,
                    'Всего объектов': len(segment_data),
                    'Плотность (объектов/км)': round((len(segment_data) / segment_size) * 1000, 1)
                }
                
                # Добавляем количество по каждому типу
                for obj_type in self.object_type_mapping.values():
                    segment_info[obj_type] = type_counts.get(obj_type, 0)
                
                segments.append(segment_info)
            
            current_start += segment_size
        
        if segments:
            segments_df = pd.DataFrame(segments)
            segments_df.to_excel(writer, sheet_name='Статистика по участкам', index=False)
            
            # Форматирование
            worksheet = writer.sheets['Статистика по участкам']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 25)
                worksheet.column_dimensions[column_letter].width = adjusted_width


def create_situational_plan_excel(detections: List[Dict], 
                                 video_info: Dict,
                                 output_path: Optional[str] = None) -> str:
    """
    Создание Excel отчета ситуационного плана
    
    Args:
        detections: Список всех детекций с пикетажем
        video_info: Информация о видео
        output_path: Путь для сохранения отчета
        
    Returns:
        str: Путь к созданному Excel файлу
    """
    generator = SituationalPlanExcelGenerator()
    return generator.create_excel_report(detections, video_info, output_path)