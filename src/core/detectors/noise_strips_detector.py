"""
Детектор шумовых полос на дороге
Определяет поперечные линии (шумовые полосы) на дорожном покрытии
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class NoiseStripsDetector:
    """Класс для детекции шумовых полос (поперечных линий) на дороге"""
    
    def __init__(self):
        self.min_strip_width = 20   # Минимальная ширина полосы
        self.max_strip_width = 100  # Максимальная ширина полосы
        self.min_strip_length = 150 # Минимальная длина полосы
        self.min_confidence = 0.50  # Минимальная уверенность для детекции
        
    def detect_noise_strips(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция шумовых полос на кадре
        
        Args:
            frame: Входной кадр
            
        Returns:
            List[Dict]: Список найденных шумовых полос
        """
        strips = []
        
        # Конвертируем в серый
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Анализируем нижнюю часть кадра (где видны шумовые полосы)
        roi = gray[h//2:, :]
        
        # Проверяем общий контраст в ROI - если низкий, то шумовых полос скорее всего нет
        roi_std = np.std(roi)
        if roi_std < 10:  # Умеренный порог контраста
            return []
        
        # 1. Детекция по периодическим паттернам
        pattern_strips = self._detect_by_periodic_patterns(roi, frame)
        strips.extend(pattern_strips)
        
        # 2. Детекция по горизонтальным линиям Хафа
        hough_strips = self._detect_by_hough_lines(roi, frame)
        strips.extend(hough_strips)
        
        # 3. Детекция по текстурному анализу
        texture_strips = self._detect_by_texture_analysis(roi, frame)
        strips.extend(texture_strips)
        
        # Фильтруем дубликаты и объединяем близкие полосы
        strips = self._filter_and_merge_strips(strips)
        
        # Дополнительная фильтрация по уверенности
        strips = [s for s in strips if s.get('confidence', 0) >= self.min_confidence]
        
        return strips
    
    def _detect_by_hough_lines(self, roi: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Детекция шумовых полос по горизонтальным линиям Хафа"""
        strips = []
        h, w = roi.shape
        
        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)
        
        # Детекция краев
        edges = cv2.Canny(enhanced, 30, 100, apertureSize=3)
        
        # Морфологические операции для выделения горизонтальных структур
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Детекция линий Хафа (более строгие параметры)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=self.min_strip_length, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Вычисляем угол линии
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Ищем строго горизонтальные линии (шумовые полосы)
                if abs(angle) < 5 and length > self.min_strip_length:  # Строго горизонтальные
                    horizontal_lines.append({
                        'line': (x1, y1, x2, y2),
                        'angle': angle,
                        'length': length,
                        'y_center': (y1 + y2) // 2
                    })
            
            # Группируем близкие горизонтальные линии в полосы
            if horizontal_lines:
                # Сортируем по Y координате
                horizontal_lines.sort(key=lambda x: x['y_center'])
                
                current_group = [horizontal_lines[0]]
                
                for i in range(1, len(horizontal_lines)):
                    current_line = horizontal_lines[i]
                    prev_line = horizontal_lines[i-1]
                    
                    # Если линии близко по Y - добавляем в группу
                    if abs(current_line['y_center'] - prev_line['y_center']) < 30:
                        current_group.append(current_line)
                    else:
                        # Обрабатываем текущую группу
                        if len(current_group) >= 3:  # Минимум 3 линии для полосы
                            strip = self._create_strip_from_lines(current_group, frame, h)
                            if strip:
                                strips.append(strip)
                        
                        current_group = [current_line]
                
                # Обрабатываем последнюю группу
                if len(current_group) >= 3:
                    strip = self._create_strip_from_lines(current_group, frame, h)
                    if strip:
                        strips.append(strip)
        
        return strips
    
    def _detect_by_texture_analysis(self, roi: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Детекция шумовых полос по текстурному анализу"""
        strips = []
        h, w = roi.shape
        
        # Применяем фильтр Габора для выделения горизонтальных текстур
        kernel_size = 31
        sigma = 4
        theta = 0  # Горизонтальное направление
        lambd = 10
        gamma = 0.5
        
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        gabor_response = cv2.filter2D(roi, cv2.CV_8UC3, kernel)
        
        # Бинаризация ответа фильтра
        _, binary = cv2.threshold(gabor_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Морфологические операции
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Получаем ограничивающий прямоугольник
            x, y, width, height = cv2.boundingRect(contour)
            
            # Проверяем размеры (должно быть горизонтально вытянутым)
            if (width > self.min_strip_length and 
                self.min_strip_width < height < self.max_strip_width and
                width > height * 4):  # Соотношение сторон 4:1
                
                y_offset = frame.shape[0]//2
                
                strips.append({
                    'type': 'texture_analysis',
                    'bbox': (x, y + y_offset, x + width, y + height + y_offset),
                    'confidence': 0.65,
                    'description': 'Шумовая полоса (текстурный анализ)'
                })
        
        return strips
    
    def _detect_by_periodic_patterns(self, roi: np.ndarray, frame: np.ndarray) -> List[Dict]:
        """Детекция шумовых полос по периодическим паттернам"""
        strips = []
        h, w = roi.shape
        
        # Анализируем вертикальные профили интенсивности
        # Шумовые полосы создают периодические изменения яркости
        
        # Усредняем по горизонтали для получения вертикального профиля
        vertical_profile = np.mean(roi, axis=1)
        
        # Сглаживаем профиль
        kernel = np.ones(5) / 5
        smoothed_profile = np.convolve(vertical_profile, kernel, mode='same')
        
        # Находим локальные минимумы (темные полосы)
        try:
            from scipy.signal import find_peaks
        except ImportError:
            # Простая заглушка если scipy недоступна
            def find_peaks(data, **kwargs):
                return [], {}
        
        # Инвертируем для поиска минимумов как пиков
        inverted_profile = 255 - smoothed_profile
        
        peaks, properties = find_peaks(inverted_profile, 
                                     height=15,      # Минимальная высота пика
                                     distance=15,    # Минимальное расстояние между пиками
                                     width=5)        # Минимальная ширина пика
        
        if len(peaks) >= 5:  # Нужно минимум 5 полос для периодичности
            # Проверяем периодичность
            distances = np.diff(peaks)
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # Проверка периодичности — вариативность < 15%
            if (std_distance < mean_distance * 0.15 and
                10 <= mean_distance <= 50):  # Разумное расстояние между полосами
                
                # Создаем полосы для каждого пика
                for peak in peaks:
                    # Определяем ширину полосы
                    peak_width = properties['widths'][np.where(peaks == peak)[0][0]] if 'widths' in properties else 10
                    
                    y_start = max(0, int(peak - peak_width))
                    y_end = min(h, int(peak + peak_width))
                    
                    # Корректируем координаты относительно полного кадра
                    y_offset = frame.shape[0]//2
                    
                    strips.append({
                        'type': 'periodic_pattern',
                        'bbox': (w//4, y_start + y_offset, 3*w//4, y_end + y_offset),
                        'confidence': 0.75,
                        'description': f'Шумовая полоса (периодический паттерн, пик {peak})'
                    })
        
        return strips
    
    def _create_strip_from_lines(self, lines_group: List[Dict], frame: np.ndarray, roi_height: int) -> Optional[Dict]:
        """Создание полосы из группы линий"""
        if not lines_group:
            return None
        
        # Вычисляем общий bbox для группы линий
        all_x = []
        all_y = []
        
        for line_info in lines_group:
            x1, y1, x2, y2 = line_info['line']
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Проверяем размеры
        width = max_x - min_x
        height = max_y - min_y
        
        if width > self.min_strip_length and height < self.max_strip_width:
            # Корректируем координаты относительно полного кадра
            y_offset = frame.shape[0]//2
            
            # Дополнительная проверка контраста в области полосы
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            strip_region = gray[min_y + y_offset:max_y + y_offset, min_x:max_x]
            
            if strip_region.size > 0:
                strip_mean = np.mean(strip_region)
                # Проверяем окружающую область
                surrounding_y1 = max(0, min_y + y_offset - 20)
                surrounding_y2 = min(frame.shape[0], max_y + y_offset + 20)
                surrounding_region = gray[surrounding_y1:surrounding_y2, min_x:max_x]
                surrounding_mean = np.mean(surrounding_region)
                
                # Шумовые полосы должны быть заметно темнее окружения
                contrast_ratio = (surrounding_mean - strip_mean) / surrounding_mean if surrounding_mean > 0 else 0
                
                if contrast_ratio > 0.08:  # Минимум 8% контраст
                    confidence = min(0.9, 0.45 + contrast_ratio * 2)  # Уверенность зависит от контраста
                    
                    return {
                        'type': 'hough_lines',
                        'bbox': (min_x, min_y + y_offset, max_x, max_y + y_offset),
                        'confidence': confidence,
                        'description': f'Шумовая полоса ({len(lines_group)} линий, контраст: {contrast_ratio:.2f})'
                    }
        
        return None
    
    def _filter_and_merge_strips(self, strips: List[Dict]) -> List[Dict]:
        """Фильтрация дубликатов и объединение близких полос"""
        if not strips:
            return strips
        
        filtered = []
        
        for strip1 in strips:
            merged = False
            x1, y1, x2, y2 = strip1['bbox']
            
            for i, strip2 in enumerate(filtered):
                x3, y3, x4, y4 = strip2['bbox']
                
                # Проверяем пересечение по Y (горизонтальные полосы)
                y_overlap = max(0, min(y2, y4) - max(y1, y3))
                y_union = max(y2, y4) - min(y1, y3)
                
                if y_overlap > 0 and y_overlap / y_union > 0.5:  # Значительное пересечение
                    # Объединяем полосы
                    merged_bbox = (
                        min(x1, x3),
                        min(y1, y3),
                        max(x2, x4),
                        max(y2, y4)
                    )
                    
                    filtered[i] = {
                        'type': 'merged',
                        'bbox': merged_bbox,
                        'confidence': max(strip1['confidence'], strip2['confidence']),
                        'description': 'Объединенная шумовая полоса'
                    }
                    merged = True
                    break
            
            if not merged:
                filtered.append(strip1)
        
        return filtered


def detect_noise_strips(frame: np.ndarray) -> List[Dict]:
    """
    Основная функция для детекции шумовых полос
    
    Args:
        frame: Входной кадр
        
    Returns:
        List[Dict]: Список найденных шумовых полос
    """
    detector = NoiseStripsDetector()
    return detector.detect_noise_strips(frame)