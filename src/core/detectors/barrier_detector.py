"""
Детектор дорожных барьеров и ограждений
Использует комбинацию COCO модели и компьютерного зрения
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

def detect_barriers_opencv(frame: np.ndarray) -> List[Dict]:
    """
    Детекция дорожных барьеров через OpenCV (улучшенная версия для боковых ограждений)
    Барьеры - это вертикальные и горизонтальные структуры по краям дороги
    
    Args:
        frame: Входной кадр (BGR)
    
    Returns:
        List[Dict]: Список детектированных барьеров с bbox
    """
    height, width = frame.shape[:2]
    barriers = []
    
    # Анализируем всю нижнюю половину кадра для лучшего захвата боковых ограждений
    roi_y_start = int(height * 0.2)  # Увеличиваем область поиска
    roi_frame = frame[roi_y_start:, :]
    roi_height = roi_frame.shape[0]
    
    # Преобразуем в градации серого
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    
    # Улучшаем контраст для лучшего выделения ограждений
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Применяем адаптивную бинаризацию
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    # Морфологические операции для соединения линий
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))  # Для вертикальных структур
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))  # Для горизонтальных ограждений
    
    # Обрабатываем вертикальные и горизонтальные структуры отдельно
    vertical_struct = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_vertical, iterations=2)
    horizontal_struct = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_horizontal, iterations=2)
    
    # Объединяем результаты
    combined = cv2.bitwise_or(vertical_struct, horizontal_struct)
    
    # Ищем как вертикальные, так и горизонтальные линии (ограждения)
    # 1. Вертикальные линии (столбы, опоры)
    vertical_lines = cv2.HoughLinesP(vertical_struct, rho=1, theta=np.pi/180, 
                                   threshold=int(roi_height * 0.2),  # Снижаем порог
                                   minLineLength=int(roi_height * 0.15),  # Минимум 15% высоты
                                   maxLineGap=20)
    
    # 2. Горизонтальные линии (перила, ограждения)
    horizontal_lines = cv2.HoughLinesP(horizontal_struct, rho=1, theta=np.pi/180, 
                                     threshold=int(width * 0.15),  # Адаптивный порог по ширине
                                     minLineLength=int(width * 0.1),  # Минимум 10% ширины
                                     maxLineGap=30)
    
    all_barrier_lines = []
    
    # Обрабатываем вертикальные линии
    if vertical_lines is not None:
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            # Вычисляем угол линии
            if x2 != x1:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            else:
                angle = 90.0
            
            # Вертикальные линии (70-110 градусов - расширяем диапазон)
            if 70 <= angle <= 110:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > roi_height * 0.1:  # Снижаем требования к длине
                    all_barrier_lines.append(('vertical', x1, y1, x2, y2, length))
    
    # Обрабатываем горизонтальные линии
    if horizontal_lines is not None:
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            # Вычисляем угол линии
            if x2 != x1:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            else:
                angle = 90.0
            
            # Горизонтальные линии (0-20 и 160-180 градусов)
            if angle <= 20 or angle >= 160:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > width * 0.08:  # Минимум 8% ширины кадра
                    # Проверяем что линия находится по краям (боковые ограждения)
                    center_x = (x1 + x2) / 2
                    if center_x < width * 0.25 or center_x > width * 0.75:  # По краям кадра
                        all_barrier_lines.append(('horizontal', x1, y1, x2, y2, length))
    
    if len(all_barrier_lines) == 0:
        return barriers
    
    # Группируем линии по типу и близости
    vertical_groups = []
    horizontal_groups = []
    
    # Сортируем вертикальные линии по X координате
    vertical_lines_only = [line for line in all_barrier_lines if line[0] == 'vertical']
    if vertical_lines_only:
        vertical_lines_only.sort(key=lambda l: l[1])  # Сортируем по x1
        
        current_group = [vertical_lines_only[0]]
        for line in vertical_lines_only[1:]:
            # Группируем линии, которые находятся близко по X (в пределах 60 пикселей)
            if abs(line[1] - current_group[-1][1]) < 60:
                current_group.append(line)
            else:
                if len(current_group) >= 1:  # Достаточно одной линии для барьера
                    vertical_groups.append(current_group)
                current_group = [line]
        
        if len(current_group) >= 1:
            vertical_groups.append(current_group)
    
    # Сортируем горизонтальные линии по Y координате
    horizontal_lines_only = [line for line in all_barrier_lines if line[0] == 'horizontal']
    if horizontal_lines_only:
        horizontal_lines_only.sort(key=lambda l: l[2])  # Сортируем по y1
        
        current_group = [horizontal_lines_only[0]]
        for line in horizontal_lines_only[1:]:
            # Группируем линии, которые находятся близко по Y (в пределах 40 пикселей)
            if abs(line[2] - current_group[-1][2]) < 40:
                current_group.append(line)
            else:
                if len(current_group) >= 1:
                    horizontal_groups.append(current_group)
                current_group = [line]
        
        if len(current_group) >= 1:
            horizontal_groups.append(current_group)
    
    # Создаем bbox для вертикальных групп (столбы, опоры)
    for group in vertical_groups:
        x_coords = [l[1] for l in group] + [l[3] for l in group]  # x1, x2
        y_coords = [l[2] for l in group] + [l[4] for l in group]  # y1, y2
        
        x1 = max(0, min(x_coords) - 20)
        y1 = max(0, min(y_coords)) + roi_y_start
        x2 = min(width, max(x_coords) + 20)
        y2 = min(height, max(y_coords) + roi_y_start)
        
        barrier_width = x2 - x1
        barrier_height = y2 - y1
        
        # Проверяем размеры вертикального барьера
        if (barrier_height > height * 0.08 and  # Снижаем требования
            barrier_width < width * 0.2 and     # Увеличиваем допустимую ширину
            y2 > height * 0.3):                 # Может быть выше
            
            # Проверяем что барьер по краям или в характерных местах
            center_x = (x1 + x2) / 2
            is_side_barrier = (center_x < width * 0.2) or (center_x > width * 0.8)
            
            if is_side_barrier:
                barriers.append({
                    'bbox': (x1, y1, x2, y2),
                    'type': 'barrier_vertical',
                    'confidence': 0.7
                })
    
    # Создаем bbox для горизонтальных групп (ограждения, перила)
    for group in horizontal_groups:
        x_coords = [l[1] for l in group] + [l[3] for l in group]  # x1, x2
        y_coords = [l[2] for l in group] + [l[4] for l in group]  # y1, y2
        
        x1 = max(0, min(x_coords) - 10)
        y1 = max(0, min(y_coords)) + roi_y_start
        x2 = min(width, max(x_coords) + 10)
        y2 = min(height, max(y_coords) + roi_y_start + 30)  # Добавляем высоту для горизонтальных
        
        barrier_width = x2 - x1
        barrier_height = y2 - y1
        
        # Проверяем размеры горизонтального ограждения
        if (barrier_width > width * 0.06 and    # Минимальная ширина
            barrier_height < height * 0.3 and   # Не слишком высокое
            barrier_height > 20):               # Минимальная высота в пикселях
            
            barriers.append({
                'bbox': (x1, y1, x2, y2),
                'type': 'barrier_horizontal',
                'confidence': 0.65
            })
    
    return barriers


def detect_barriers_from_coco(frame: np.ndarray, coco_model, conf_threshold: float = 0.3) -> List[Dict]:
    """
    Детекция барьеров через COCO модель (улучшенная версия)
    Ищем классы: fence (ограждение), pole (столб/опора)
    
    Args:
        frame: Входной кадр
        coco_model: YOLO COCO модель
        conf_threshold: Порог уверенности
    
    Returns:
        List[Dict]: Список детектированных барьеров
    """
    barriers = []
    height, width = frame.shape[:2]
    
    # Запускаем детекцию с более низким порогом для лучшего обнаружения
    results = coco_model(frame, conf=max(conf_threshold, 0.25), verbose=False)
    
    # Классы COCO, которые могут быть барьерами
    # COCO class IDs: 10=fence (ограждение), 5=parking meter (можно считать опорой)
    # Проверяем по именам классов для надежности
    barrier_keywords = ['fence', 'pole']
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = coco_model.names[cls_id].lower()
                conf = float(box.conf[0])
                
                # Проверяем, является ли это барьером/ограждением
                is_barrier = False
                for keyword in barrier_keywords:
                    if keyword in class_name:
                        # Для барьеров требуем уверенность минимум 0.3
                        if conf >= 0.3:
                            is_barrier = True
                            break
                
                if is_barrier:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Дополнительная фильтрация по геометрии
                    # Барьеры обычно вертикальные и находятся в нижней части кадра
                    box_width = x2 - x1
                    box_height = y2 - y1
                    aspect_ratio = box_height / box_width if box_width > 0 else 0
                    center_y = (y1 + y2) / 2
                    
                    # Барьеры должны быть:
                    # - Вертикальными (высота > ширины, aspect_ratio > 1.2)
                    # - Или горизонтальными заграждениями (ширина значительно больше высоты)
                    # - Расположены в нижней половине кадра (для вертикальных опор)
                    
                    is_valid = False
                    if 'fence' in class_name:
                        # Ограждения могут быть горизонтальными или вертикальными
                        # Главное - достаточная уверенность и разумный размер
                        if conf >= 0.35 and box_height > height * 0.05:
                            is_valid = True
                    elif 'pole' in class_name:
                        # Столбы/опоры должны быть вертикальными
                        if aspect_ratio > 1.2 and center_y > height * 0.3:
                            is_valid = True
                    
                    if is_valid:
                        barriers.append({
                            'bbox': (x1, y1, x2, y2),
                            'type': 'barrier',
                            'confidence': conf,
                            'class': class_name
                        })
    
    return barriers


def detect_barriers_combined(frame: np.ndarray, coco_model=None, conf_threshold: float = 0.3) -> List[Dict]:
    """
    Комбинированная детекция барьеров (OpenCV + COCO)
    
    Args:
        frame: Входной кадр
        coco_model: YOLO COCO модель (опционально)
        conf_threshold: Порог уверенности
    
    Returns:
        List[Dict]: Список детектированных барьеров
    """
    all_barriers = []
    
    # 1. Детекция через OpenCV (вертикальные структуры)
    cv_barriers = detect_barriers_opencv(frame)
    all_barriers.extend(cv_barriers)
    
    # 2. Детекция через COCO модель (если доступна)
    if coco_model is not None:
        coco_barriers = detect_barriers_from_coco(frame, coco_model, conf_threshold)
        all_barriers.extend(coco_barriers)
    
    # Объединяем дублирующиеся детекции (NMS)
    if len(all_barriers) > 1:
        # Простой NMS - убираем пересекающиеся bbox
        filtered_barriers = []
        for barrier in all_barriers:
            x1, y1, x2, y2 = barrier['bbox']
            is_duplicate = False
            
            for existing in filtered_barriers:
                ex1, ey1, ex2, ey2 = existing['bbox']
                # Вычисляем пересечение
                inter_x1 = max(x1, ex1)
                inter_y1 = max(y1, ey1)
                inter_x2 = min(x2, ex2)
                inter_y2 = min(y2, ey2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # Есть пересечение - оставляем тот, у которого выше confidence
                    if barrier['confidence'] > existing['confidence']:
                        filtered_barriers.remove(existing)
                        filtered_barriers.append(barrier)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_barriers.append(barrier)
        
        return filtered_barriers
    
    return all_barriers
