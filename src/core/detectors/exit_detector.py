"""
Детектор съездов с дороги
Определяет места где дорога разветвляется или есть съезды
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class ExitDetector:
    """Класс для детекции съездов и развязок на дороге"""
    
    def __init__(self):
        self.min_exit_width = 80   # Минимальная ширина съезда
        self.min_exit_length = 100  # Минимальная длина съезда
        self.min_confidence = 0.5   # Минимальная уверенность для детекции
        
    def detect_exits(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция съездов на кадре
        
        Args:
            frame: Входной кадр
            
        Returns:
            List[Dict]: Список найденных съездов с координатами и типом
        """
        exits = []
        
        # Конвертируем в серый
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Анализируем среднюю часть кадра (где видны съезды)
        roi = gray[h//3:, :]
        
        # Детекция по изменению ширины дороги
        width_exits = self._detect_by_road_width_change(roi, frame)
        exits.extend(width_exits)
        
        # Детекция по направлению разметки  
        marking_exits = self._detect_by_marking_direction(roi, frame)
        exits.extend(marking_exits)
        
        # Фильтруем дубликаты и объединяем близкие съезды
        exits = self._filter_and_merge_exits(exits)
        
        return exits
    
    def _detect_by_road_width_change(self, roi: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Детекция съездов по изменению ширины дороги"""
        exits = []
        h, w = roi.shape
        
        # Применяем адаптивную бинаризацию
        binary = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 15, 2)
        
        # Анализируем ширину дороги по горизонтальным срезам
        road_widths = []
        for y in range(0, h, 10):  # Каждые 10 пикселей
            row = binary[y, :]
            
            # Находим границы дороги (темные участки)
            dark_pixels = np.where(row < 128)[0]
            if len(dark_pixels) > w * 0.3:  # Если есть достаточно дорожного покрытия
                road_width = len(dark_pixels)
                road_widths.append((y, road_width))
        
        # Ищем резкие изменения ширины (признак съезда)
        if len(road_widths) > 5:
            for i in range(1, len(road_widths) - 1):
                prev_width = road_widths[i-1][1]
                curr_width = road_widths[i][1]
                next_width = road_widths[i+1][1]
                
                # Резкое увеличение ширины = возможный съезд
                if curr_width > prev_width * 1.5 and curr_width > next_width * 1.3:
                    y_pos = road_widths[i][0] + frame.shape[0]//3
                    exits.append({
                        'type': 'width_change',
                        'bbox': (w//4, y_pos, 3*w//4, y_pos + 50),
                        'confidence': 0.6,
                        'description': 'Расширение дороги (возможный съезд)'
                    })
        
        return exits
    
    def _detect_by_marking_direction(self, roi: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Детекция съездов по направлению дорожной разметки"""
        exits = []
        h, w = roi.shape
        
        # Детекция линий Хафа
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            # Анализируем направления линий
            diagonal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Вычисляем угол линии
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Ищем диагональные линии (съездные полосы)
                if 20 < abs(angle) < 70 and length > 50:
                    diagonal_lines.append({
                        'line': (x1, y1, x2, y2),
                        'angle': angle,
                        'length': length
                    })
            
            # Группируем близкие диагональные линии (минимум 3 для обнаружения)
            if len(diagonal_lines) >= 3:
                # Находим кластеры линий
                for i, line1 in enumerate(diagonal_lines):
                    similar_lines = [line1]
                    x1, y1, x2, y2 = line1['line']
                    
                    for j, line2 in enumerate(diagonal_lines[i+1:], i+1):
                        x3, y3, x4, y4 = line2['line']
                        
                        # Проверяем близость линий
                        dist = min(
                            np.sqrt((x1-x3)**2 + (y1-y3)**2),
                            np.sqrt((x2-x4)**2 + (y2-y4)**2)
                        )
                        
                        if dist < 120 and abs(line1['angle'] - line2['angle']) < 30:
                            similar_lines.append(line2)
                    
                    # Если найден кластер диагональных линий - это съезд (минимум 3 линии)
                    if len(similar_lines) >= 3:
                        # Вычисляем bbox для съезда
                        all_x = [p for line in similar_lines for p in [line['line'][0], line['line'][2]]]
                        all_y = [p for line in similar_lines for p in [line['line'][1], line['line'][3]]]
                        
                        min_x, max_x = min(all_x), max(all_x)
                        min_y, max_y = min(all_y), max(all_y)
                        
                        # Корректируем координаты относительно полного кадра
                        min_y += frame.shape[0]//3
                        max_y += frame.shape[0]//3
                        
                        # Проверяем, что съезд не в центре дороги
                        center_x = (min_x + max_x) / 2
                        frame_width = frame.shape[1]
                        
                        # Съезды должны быть по краям (левые 40% или правые 40%) - смягчили критерий
                        if center_x < frame_width * 0.4 or center_x > frame_width * 0.6:
                            exits.append({
                                'type': 'marking_direction',
                                'bbox': (min_x, min_y, max_x, max_y),
                                'confidence': 0.7,  # Разумная уверенность для разметочного метода
                                'description': 'Съезд по направлению разметки'
                            })
                        break  # Один съезд на кадр
        
        return exits
    
    def _filter_and_merge_exits(self, exits: List[Dict]) -> List[Dict]:
        """Фильтрация дубликатов и объединение близких съездов"""
        if not exits:
            return exits
        
        filtered = []
        
        for exit1 in exits:
            is_duplicate = False
            x1, y1, x2, y2 = exit1['bbox']
            center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            for exit2 in filtered:
                x3, y3, x4, y4 = exit2['bbox']
                center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
                
                # Если центры близко - это дубликат
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                if distance < 150:  # Порог для объединения
                    # Объединяем съезды - берем с большей уверенностью
                    if exit1.get('confidence', 0) > exit2.get('confidence', 0):
                        # Заменяем на новый
                        filtered.remove(exit2)
                        filtered.append(exit1)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(exit1)
        
        return filtered
    
    def _detect_by_road_contours(self, roi: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Детекция съездов по контурам дороги"""
        exits = []
        h, w = roi.shape
        
        # Размытие и бинаризация
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Морфологические операции для выделения дорожного полотна
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Анализируем большие контуры (дорожное полотно) - более строгие критерии
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > w * h * 0.2:  # Очень большой контур (увеличено с 0.1 до 0.2)
                
                # Аппроксимируем контур
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Ищем "выступы" в контуре (съезды) - очень сложная форма
                if len(approx) > 8:  # Очень сложная форма (увеличено с 6 до 8)
                    # Анализируем выпуклые дефекты
                    hull = cv2.convexHull(contour, returnPoints=False)
                    if len(hull) > 3:
                        defects = cv2.convexityDefects(contour, hull)
                        
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(contour[s][0])
                                end = tuple(contour[e][0])
                                far = tuple(contour[f][0])
                                
                                # Если дефект очень глубокий - это может быть съезд
                                if d > 2000:  # Глубина дефекта (увеличено с 1000 до 2000)
                                    # Корректируем координаты
                                    y_offset = frame.shape[0]//3
                                    
                                    exits.append({
                                        'type': 'contour_defect',
                                        'bbox': (min(start[0], end[0], far[0]) - 20,
                                               min(start[1], end[1], far[1]) + y_offset - 20,
                                               max(start[0], end[0], far[0]) + 20,
                                               max(start[1], end[1], far[1]) + y_offset + 20),
                                        'confidence': 0.5,  # Низкая уверенность для контурного метода
                                        'description': 'Съезд по контуру дороги'
                                    })
        
        return exits
    
    def _filter_duplicate_exits(self, exits: List[Dict]) -> List[Dict]:
        """Фильтрация дублирующихся съездов"""
        if not exits:
            return exits
        
        filtered = []
        
        for exit1 in exits:
            is_duplicate = False
            x1, y1, x2, y2 = exit1['bbox']
            center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            for exit2 in filtered:
                x3, y3, x4, y4 = exit2['bbox']
                center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
                
                # Если центры близко - это дубликат
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                if distance < 100:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(exit1)
        
        return filtered


def detect_road_exits(frame: np.ndarray) -> List[Dict]:
    """
    Основная функция для детекции съездов
    
    Args:
        frame: Входной кадр
        
    Returns:
        List[Dict]: Список найденных съездов
    """
    detector = ExitDetector()
    return detector.detect_exits(frame)