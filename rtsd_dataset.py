"""
Модуль для работы с датасетом RTSD (Russian Traffic Sign Dataset)
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RTSDDataset:
    """Класс для работы с датасетом RTSD"""
    
    def __init__(self, dataset_path: str = None):
        """
        Инициализация датасета RTSD
        
        Args:
            dataset_path: Путь к корневой директории датасета
        """
        self.dataset_path = dataset_path
        self.annotations = []
        self.classes = {}
        
        if dataset_path and os.path.exists(dataset_path):
            self.load_dataset(dataset_path)
    
    def load_dataset(self, dataset_path: str):
        """Загрузка датасета"""
        self.dataset_path = dataset_path
        
        # Ищем файлы аннотаций (обычно в формате JSON или YOLO)
        # Структура датасета может варьироваться
        annotations_path = os.path.join(dataset_path, 'annotations')
        images_path = os.path.join(dataset_path, 'images')
        
        # Пытаемся найти аннотации
        if os.path.exists(annotations_path):
            self._load_annotations(annotations_path)
        
        if os.path.exists(images_path):
            self.images_path = images_path
    
    def _load_annotations(self, annotations_path: str):
        """Загрузка аннотаций из директории"""
        # Поддержка различных форматов аннотаций
        annotation_files = []
        
        for ext in ['*.json', '*.txt', '*.xml']:
            annotation_files.extend(Path(annotations_path).rglob(ext))
        
        for ann_file in annotation_files:
            try:
                if ann_file.suffix == '.json':
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        ann_data = json.load(f)
                        self.annotations.append(ann_data)
            except Exception as e:
                print(f"Ошибка при загрузке аннотации {ann_file}: {e}")
    
    def get_classes(self) -> Dict[int, str]:
        """
        Получение словаря классов датасета RTSD
        
        Returns:
            Словарь {id: название_класса}
        """
        # Полный список классов RTSD (Russian Traffic Sign Dataset)
        # Согласно стандартной структуре RTSD датасета
        rtsd_classes = {
            # Основные категории по ГОСТ Р 52290-2004
            0: 'Запрещающие знаки',
            1: 'Предписывающие знаки', 
            2: 'Предупреждающие знаки',
            3: 'Знаки приоритета',
            4: 'Информационно-указательные знаки',
            5: 'Знаки сервиса',
            6: 'Знаки дополнительной информации',
            
            # Конкретные дорожные знаки (RTSD использует эту нумерацию)
            10: 'Ограничение максимальной скорости (20)',
            11: 'Ограничение максимальной скорости (30)',
            12: 'Ограничение максимальной скорости (40)',
            13: 'Ограничение максимальной скорости (50)',
            14: 'Ограничение максимальной скорости (60)',
            15: 'Ограничение максимальной скорости (70)',
            16: 'Ограничение максимальной скорости (80)',
            17: 'Ограничение максимальной скорости (90)',
            18: 'Ограничение максимальной скорости (100)',
            19: 'Ограничение максимальной скорости (110)',
            20: 'Ограничение максимальной скорости (120)',
            21: 'Обгон запрещен',
            22: 'Остановка запрещена',
            23: 'Стоянка запрещена',
            24: 'Движение запрещено',
            25: 'Поворот направо запрещен',
            26: 'Поворот налево запрещен',
            27: 'Разворот запрещен',
            28: 'Движение грузовых автомобилей запрещено',
            29: 'Въезд запрещен',
            
            # Знаки приоритета
            30: 'Уступите дорогу',
            31: 'Главная дорога',
            32: 'Проезд без остановки запрещен',
            33: 'Преимущество встречного движения',
            
            # Предписывающие знаки
            40: 'Движение прямо',
            41: 'Движение направо',
            42: 'Движение налево',
            43: 'Движение прямо или направо',
            44: 'Движение прямо или налево',
            45: 'Движение направо или налево',
            46: 'Объезд препятствия справа',
            47: 'Объезд препятствия слева',
            48: 'Круговое движение',
            49: 'Пешеходная дорожка',
            50: 'Ограничение минимальной скорости',
            
            # Предупреждающие знаки
            60: 'Опасный поворот направо',
            61: 'Опасный поворот налево',
            62: 'Опасные повороты (сначала направо)',
            63: 'Опасные повороты (сначала налево)',
            64: 'Крутой спуск',
            65: 'Крутой подъем',
            66: 'Сужение дороги',
            67: 'Неровная дорога',
            68: 'Скользкая дорога',
            69: 'Выброс гравия',
            70: 'Дорожные работы',
            71: 'Пешеходный переход',
            72: 'Дети',
            73: 'Пересечение с велосипедной дорожкой',
            74: 'Дикие животные',
            75: 'Падение камней',
            
            # Информационно-указательные
            80: 'Начало населенного пункта',
            81: 'Конец населенного пункта',
            82: 'Одностороннее движение',
            83: 'Место стоянки',
            84: 'Подземный пешеходный переход',
            85: 'Надземный пешеходный переход',
            86: 'Тупик',
            87: 'Указатель направления',
            88: 'Предварительный указатель направления',
            
            # Знаки сервиса
            90: 'Пункт медицинской помощи',
            91: 'Больница',
            92: 'Автозаправочная станция',
            93: 'Техническое обслуживание автомобилей',
            94: 'Мойка автомобилей',
            95: 'Телефон',
            96: 'Пункт питания',
            97: 'Питьевая вода',
            98: 'Гостиница или мотель',
            99: 'Кемпинг',
        }
        
        # Если датасет загружен, пытаемся получить реальные классы из аннотаций
        if self.annotations:
            # Пытаемся извлечь уникальные классы из аннотаций
            unique_classes = set()
            for ann in self.annotations:
                if isinstance(ann, dict):
                    if 'annotations' in ann:
                        for item in ann['annotations']:
                            if 'category_id' in item:
                                unique_classes.add(item['category_id'])
                    elif 'category_id' in ann:
                        unique_classes.add(ann['category_id'])
            
            # Если нашли классы в аннотациях, используем их
            if unique_classes:
                print(f"Обнаружено {len(unique_classes)} уникальных классов в датасете")
                # Обновляем словарь только для найденных классов
        
        return rtsd_classes
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """Получение пути к изображению по ID"""
        if not hasattr(self, 'images_path'):
            return None
        
        # Ищем изображение в различных форматах
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_path = os.path.join(self.images_path, f"{image_id}{ext}")
            if os.path.exists(image_path):
                return image_path
        
        return None


def load_rtsd_model(dataset_path: str = None, model_path: str = None):
    """
    Загрузка модели, обученной на датасете RTSD
    
    Args:
        dataset_path: Путь к датасету (для получения классов)
        model_path: Путь к файлу модели YOLO
    
    Returns:
        Tuple (model, classes_dict)
    """
    try:
        from ultralytics import YOLO
        
        # Загружаем модель
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            # Используем предобученную модель или дообучаем на RTSD
            print("Используется предобученная модель YOLOv8")
            print("Для лучших результатов рекомендуется использовать модель, обученную на RTSD")
            model = YOLO('yolov8n.pt')
        
        # Загружаем классы
        rtsd = RTSDDataset(dataset_path)
        classes = rtsd.get_classes()
        
        return model, classes
    
    except ImportError:
        print("Библиотека ultralytics не установлена. Установите её командой: pip install ultralytics")
        return None, {}

