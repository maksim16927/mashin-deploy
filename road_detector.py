"""
Модуль для распознавания элементов дорожной инфраструктуры
"""
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Предупреждение: OpenCV не установлен. Некоторые функции могут быть недоступны.")

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import math


@dataclass
class RoadObject:
    """Класс для представления объекта дорожной инфраструктуры"""
    object_type: str
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    coordinates: Optional[Tuple[float, float]] = None  # GPS координаты
    timestamp: Optional[datetime] = None
    additional_info: Dict = None


@dataclass
class RoadMarking:
    """Класс для представления дорожной разметки"""
    marking_type: str  # "сплошная", "прерывистая", "двойная_сплошная", "пешеходный_переход"
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    coordinates_start: Optional[Tuple[float, float]] = None
    coordinates_end: Optional[Tuple[float, float]] = None
    timestamp: Optional[datetime] = None


@dataclass
class LaneInfo:
    """Информация о полосах движения"""
    lane_count: int
    lane_markings: List[RoadMarking]
    lane_width_pixels: float
    timestamp: Optional[datetime] = None


class RoadSignDetector:
    """Детектор дорожных знаков с использованием YOLO и RTSD датасета"""
    
    def __init__(self, model_path: str = None, rtsd_dataset_path: str = None):
        """
        Инициализация детектора дорожных знаков
        Если model_path не указан, используется предобученная модель YOLOv8
        
        Args:
            model_path: Путь к модели YOLO (обученной на RTSD)
            rtsd_dataset_path: Путь к датасету RTSD для получения классов
        """
        # Базовые классы дорожных знаков (используются по умолчанию)
        self.sign_classes = {
            0: 'Знак ограничения скорости',
            1: 'Запрещающий знак',
            2: 'Предписывающий знак',
            3: 'Предупреждающий знак',
            4: 'Знак приоритета',
            5: 'Знак пешеходный переход',
            6: 'Знак уступи дорогу',
            7: 'Знак главная дорога',
            8: 'Знак "Остановка запрещена"',
            9: 'Знак "Стоянка запрещена"',
            10: 'Знак "Поворот запрещен"',
            11: 'Знак "Движение прямо"',
            12: 'Знак "Движение направо"',
            13: 'Знак "Движение налево"',
            14: 'Знак "Круговое движение"',
            15: 'Знак "Опасный поворот"',
            16: 'Знак "Пешеходная дорожка"',
            17: 'Знак "Дорожные работы"',
            # Можно расширить список классов
        }
        
        try:
            from ultralytics import YOLO
            
            if model_path:
                self.model = YOLO(model_path)
            else:
                # Используем YOLOv8, обученную на COCO и дообученную на дорожных знаках
                # Для лучших результатов рекомендуется использовать модель, обученную на RTSD
                self.model = YOLO('yolov8n.pt')
            
            # Загружаем классы из RTSD датасета, если доступен
            if rtsd_dataset_path:
                try:
                    from rtsd_dataset import RTSDDataset
                    rtsd = RTSDDataset(rtsd_dataset_path)
                    rtsd_classes = rtsd.get_classes()
                    if rtsd_classes:
                        self.sign_classes = rtsd_classes
                        print(f"Загружено {len(rtsd_classes)} классов из RTSD датасета")
                except Exception as e:
                    print(f"Ошибка при загрузке RTSD датасета: {e}. Используются базовые классы.")
        except ImportError:
            print("Предупреждение: ultralytics не установлен. Распознавание знаков будет упрощено.")
            self.model = None
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[RoadObject]:
        """Обнаружение дорожных знаков на кадре"""
        if self.model is None:
            return []
        
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detected_signs = []
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                class_name = self.sign_classes.get(cls, f'Знак класс {cls}')
                
                road_obj = RoadObject(
                    object_type='дорожный_знак',
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf
                )
                detected_signs.append(road_obj)
        
        return detected_signs


class RoadMarkingDetector:
    """Детектор дорожной разметки"""
    
    def __init__(self):
        self.line_threshold = 100
        self.min_line_length = 50
        self.max_line_gap = 10
    
    def detect_lanes(self, frame: np.ndarray) -> LaneInfo:
        """Обнаружение полос движения и их количество"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Область интереса (нижняя часть кадра - дорога)
        height, width = gray.shape
        roi_top = height // 2
        roi = gray[roi_top:, :]
        
        # Применяем размытие
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        
        # Детекция краев
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Преобразование Хафа для поиска линий
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.line_threshold,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        
        if lines is None:
            return LaneInfo(lane_count=0, lane_markings=[], lane_width_pixels=0.0)
        
        # Фильтрация и классификация линий
        left_lines = []
        right_lines = []
        center_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Нормализуем координаты относительно всего кадра
            y1 += roi_top
            y2 += roi_top
            
            # Определяем направление линии
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Фильтруем почти горизонтальные линии
            if abs(angle) < 10 or abs(angle) > 170:
                continue
            
            line_center_x = (x1 + x2) / 2
            line_type = self._classify_line_type(edges, line[0], roi_top)
            
            marking = RoadMarking(
                marking_type=line_type,
                start_point=(x1, y1),
                end_point=(x2, y2)
            )
            
            # Распределяем линии по категориям
            if line_center_x < width * 0.4:
                left_lines.append(marking)
            elif line_center_x > width * 0.6:
                right_lines.append(marking)
            else:
                center_lines.append(marking)
        
        # Подсчет полос (между левой и правой границами + центральные линии)
        lane_count = max(1, len(center_lines) + 1)
        
        all_markings = left_lines + right_lines + center_lines
        
        # Оцениваем среднюю ширину полосы
        avg_lane_width = self._estimate_lane_width(all_markings, width, height)
        
        return LaneInfo(
            lane_count=lane_count,
            lane_markings=all_markings,
            lane_width_pixels=avg_lane_width
        )
    
    def _classify_line_type(self, edges: np.ndarray, line: Tuple, roi_top: int) -> str:
        """Определение типа линии разметки"""
        x1, y1, x2, y2 = line
        # Упрощенная классификация - проверяем толщину и сплошность
        # Для полной реализации нужна более сложная логика
        return "прерывистая"  # По умолчанию
    
    def _estimate_lane_width(self, markings: List[RoadMarking], width: int, height: int) -> float:
        """Оценка средней ширины полосы в пикселях"""
        if len(markings) < 2:
            return width / 3.0  # Примерная оценка
        
        # Вычисляем средние расстояния между линиями
        distances = []
        for i in range(len(markings) - 1):
            m1 = markings[i]
            m2 = markings[i + 1]
            dist = math.sqrt((m1.start_point[0] - m2.start_point[0])**2 + 
                           (m1.start_point[1] - m2.start_point[1])**2)
            distances.append(dist)
        
        if distances:
            return np.mean(distances)
        return width / 3.0
    
    def detect_crosswalk(self, frame: np.ndarray) -> List[RoadMarking]:
        """Обнаружение пешеходных переходов (зебра)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Применяем пороговую обработку для поиска белых полос
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции для выделения полос
        kernel = np.ones((3, 15), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        crosswalks = []
        min_area = 500  # Минимальная площадь для пешеходного перехода
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Проверяем, что это горизонтальные полосы (соотношение сторон)
                if w > h * 2:
                    marking = RoadMarking(
                        marking_type="пешеходный_переход",
                        start_point=(x, y),
                        end_point=(x + w, y + h)
                    )
                    crosswalks.append(marking)
        
        return crosswalks


class TrafficLightDetector:
    """Детектор светофоров"""
    
    def __init__(self):
        try:
            from ultralytics import YOLO
            # Используем предобученную модель или специализированную модель для светофоров
            self.model = YOLO('yolov8n.pt')
        except ImportError:
            self.model = None
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[RoadObject]:
        """Обнаружение светофоров на кадре"""
        if self.model is None:
            # Упрощенная детекция по цвету
            return self._detect_by_color(frame)
        
        # YOLO детекция
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detected_lights = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                # Класс 9 в COCO - traffic light
                if cls == 9:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Определяем состояние светофора
                    state = self._detect_light_state(frame[y1:y2, x1:x2])
                    
                    road_obj = RoadObject(
                        object_type='светофор',
                        class_name=f'Светофор ({state})',
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        additional_info={'state': state}
                    )
                    detected_lights.append(road_obj)
        
        return detected_lights
    
    def _detect_by_color(self, frame: np.ndarray) -> List[RoadObject]:
        """Упрощенная детекция светофоров по цвету"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Определяем маски для красного, желтого и зеленого
        red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([50, 100, 100]), np.array([70, 255, 255]))
        
        # Находим контуры
        lights = []
        for mask, color in [(red_mask, 'красный'), (yellow_mask, 'желтый'), (green_mask, 'зеленый')]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    road_obj = RoadObject(
                        object_type='светофор',
                        class_name=f'Светофор ({color})',
                        bbox=(x, y, x + w, y + h),
                        confidence=0.7
                    )
                    lights.append(road_obj)
        
        return lights
    
    def _detect_light_state(self, roi: np.ndarray) -> str:
        """Определение состояния светофора (красный/желтый/зеленый)"""
        if roi.size == 0:
            return "неизвестно"
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Подсчет пикселей каждого цвета
        red_pixels = np.sum(cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])))
        yellow_pixels = np.sum(cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255])))
        green_pixels = np.sum(cv2.inRange(hsv, np.array([50, 100, 100]), np.array([70, 255, 255])))
        
        total_pixels = red_pixels + yellow_pixels + green_pixels
        if total_pixels == 0:
            return "неизвестно"
        
        # Определяем доминирующий цвет
        if red_pixels > yellow_pixels and red_pixels > green_pixels:
            return "красный"
        elif yellow_pixels > green_pixels:
            return "желтый"
        else:
            return "зеленый"


class StreetLightDetector:
    """Детектор уличных фонарей"""
    
    def detect(self, frame: np.ndarray) -> List[RoadObject]:
        """Обнаружение уличных фонарей"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Применяем размытие
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Детекция ярких объектов (фонари обычно яркие)
        _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lights = []
        min_area = 50
        max_area = 5000
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Фонари обычно вертикальные
                if h > w * 1.5:
                    road_obj = RoadObject(
                        object_type='фонарь',
                        class_name='Уличный фонарь',
                        bbox=(x, y, x + w, y + h),
                        confidence=0.6
                    )
                    lights.append(road_obj)
        
        return lights


class FenceDetector:
    """Детектор ограждений"""
    
    def detect(self, frame: np.ndarray) -> List[RoadObject]:
        """Обнаружение ограждений"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Применяем детекцию краев
        edges = cv2.Canny(gray, 50, 150)
        
        # Преобразование Хафа для поиска вертикальных линий
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return []
        
        fences = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Вертикальные линии (угол близок к 90 градусам)
            if 75 < angle < 105:
                vertical_lines.append((x1, y1, x2, y2))
        
        # Группируем близкие вертикальные линии как ограждения
        if len(vertical_lines) > 3:
            # Упрощенная логика - считаем скопление вертикальных линий ограждением
            x, y = vertical_lines[0][0], vertical_lines[0][1]
            w = max([abs(line[0] - line[2]) for line in vertical_lines])
            h = max([abs(line[1] - line[3]) for line in vertical_lines])
            
            road_obj = RoadObject(
                object_type='ограждение',
                class_name='Дорожное ограждение',
                bbox=(x, y, x + w, y + h),
                confidence=0.5
            )
            fences.append(road_obj)
        
        return fences

