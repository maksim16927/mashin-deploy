#!/usr/bin/env python3
"""
Улучшенный детектор дорожной разметки
Основано на: https://github.com/Serzho/Road-marking-recognition

Комбинирует классический CV подход с YOLO:
- ROI обработка (нижняя часть кадра)
- Медианное размытие -> Пороговое -> Гауссовское -> Canny
- Детекция контуров для сплошных/пунктирных линий
- Template matching для стрелок направления
- YOLO для барьеров и других объектов
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class RoadMarkingDetector:
    """Детектор дорожной разметки с продвинутой обработкой"""
    
    def __init__(self, model_path=None):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к обученной YOLO модели (опционально)
        """
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"✅ YOLO модель загружена: {model_path}")
        
        # Параметры обработки (из Serzho/Road-marking-recognition)
        self.roi_start_ratio = 0.5  # Начало ROI (50% высоты)
        self.roi_end_ratio = 0.7    # Конец ROI (70% высоты)
        self.left_zone = (0.3, 0.5)   # Левая зона разметки (30-50% ширины)
        self.right_zone = (0.75, 0.95) # Правая зона разметки (75-95% ширины)
        self.median_blur_size = 5
        self.threshold_value = 210
        self.gaussian_kernel = (7, 7)
        self.gaussian_sigma = 1.5
        self.canny_low = 1
        self.canny_high = 50
        
        # Буферы для стабилизации
        self.control_frames = []  # Последние N кадров для анализа сплошности
        self.buffer_size = 10
        
        # Пороги классификации
        self.solid_threshold = 1000  # Порог для сплошной линии
    
    def preprocess_frame(self, frame):
        """
        Предобработка кадра (метод из Serzho repo)
        
        Args:
            frame: Исходный кадр BGR
            
        Returns:
            processed_frame: Обработанный кадр
            roi_frame: ROI для детекции
        """
        height, width = frame.shape[:2]
        
        # Определяем ROI (region of interest) - нижняя часть кадра
        roi_y_start = int(height * self.roi_start_ratio)
        roi_y_end = int(height * self.roi_end_ratio)
        roi_frame = frame[roi_y_start:roi_y_end, 0:width]
        
        # Шаг 1: Медианное размытие (убирает шум)
        blurred = cv2.medianBlur(roi_frame, self.median_blur_size)
        
        # Шаг 2: Пороговое преобразование (выделяет белую разметку)
        _, threshold = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Шаг 3: Конвертация в grayscale
        gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
        
        # Шаг 4: Гауссовское размытие
        gaussian = cv2.GaussianBlur(gray, self.gaussian_kernel, self.gaussian_sigma)
        
        # Шаг 5: Детекция краев Canny
        edges = cv2.Canny(gaussian, self.canny_low, self.canny_high)
        
        return edges, roi_frame
    
    def detect_contours(self, edges_frame):
        """
        Детекция контуров разметки
        
        Args:
            edges_frame: Кадр после Canny
            
        Returns:
            contours: Список контуров
            hierarchy: Иерархия контуров
        """
        contours, hierarchy = cv2.findContours(
            edges_frame, 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_NONE
        )
        return contours, hierarchy
    
    def analyze_line_solidity(self, contours, width):
        """
        Анализ сплошности линий (метод из Serzho)
        
        Args:
            contours: Список контуров
            width: Ширина кадра
            
        Returns:
            (sum_left_zone, sum_right_zone): Количество пикселей в зонах
        """
        sum_left_zone = 0
        sum_right_zone = 0
        
        for contour in contours:
            for dot_packed in contour:
                x = int(dot_packed[0][0])
                y = int(dot_packed[0][1])
                
                # Проверка попадания в левую зону
                if int(width * self.left_zone[0]) <= x <= int(width * self.left_zone[1]):
                    sum_left_zone += 1
                # Проверка попадания в правую зону
                elif int(width * self.right_zone[0]) <= x <= int(width * self.right_zone[1]):
                    sum_right_zone += 1
        
        return sum_left_zone, sum_right_zone
    
    def classify_lines(self, left_sum, right_sum):
        """
        Классификация линий как сплошные/пунктирные
        
        Args:
            left_sum: Сумма пикселей левой зоны
            right_sum: Сумма пикселей правой зоны
            
        Returns:
            (left_type, right_type): Типы линий
        """
        left_type = "Solid" if left_sum > self.solid_threshold else "Intermittent"
        right_type = "Solid" if right_sum > self.solid_threshold else "Intermittent"
        return left_type, right_type
    
    def process_frame(self, frame, use_yolo=True):
        """
        Полная обработка кадра
        
        Args:
            frame: Исходный кадр BGR
            use_yolo: Использовать YOLO для дополнительной детекции
            
        Returns:
            annotated_frame: Кадр с аннотациями
            detection_info: Информация о детекции
        """
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Рисуем ROI область
        roi_y_start = int(height * self.roi_start_ratio)
        roi_y_end = int(height * self.roi_end_ratio)
        cv2.rectangle(annotated_frame, (0, roi_y_start), (width, roi_y_end), (0, 0, 255), 2)
        
        # Рисуем зоны детекции
        cv2.rectangle(
            annotated_frame,
            (int(width * self.left_zone[0]), roi_y_start),
            (int(width * self.left_zone[1]), roi_y_end),
            (0, 255, 255), 2
        )
        cv2.rectangle(
            annotated_frame,
            (int(width * self.right_zone[0]), roi_y_start),
            (int(width * self.right_zone[1]), roi_y_end),
            (0, 255, 255), 2
        )
        
        # Обработка для детекции разметки
        edges, roi_frame = self.preprocess_frame(frame)
        contours, hierarchy = self.detect_contours(edges)
        
        # Рисуем контуры
        cv2.drawContours(
            annotated_frame[roi_y_start:roi_y_end, 0:width],
            contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1
        )
        
        # Анализ сплошности линий
        left_sum, right_sum = self.analyze_line_solidity(contours, width)
        
        # Добавляем в буфер для стабилизации
        self.control_frames.append((left_sum, right_sum))
        if len(self.control_frames) > self.buffer_size:
            self.control_frames.pop(0)
        
        # Вычисляем среднее
        if len(self.control_frames) >= self.buffer_size:
            avg_left = sum(f[0] for f in self.control_frames) / self.buffer_size
            avg_right = sum(f[1] for f in self.control_frames) / self.buffer_size
            left_type, right_type = self.classify_lines(avg_left, avg_right)
        else:
            avg_left, avg_right = left_sum, right_sum
            left_type, right_type = self.classify_lines(left_sum, right_sum)
        
        # Аннотации
        cv2.putText(
            annotated_frame,
            f"LeftZone: {int(avg_left)}, RightZone: {int(avg_right)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.putText(
            annotated_frame,
            f"Left: {left_type}, Right: {right_type}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        
        # YOLO детекция (если модель загружена)
        yolo_detections = []
        if use_yolo and self.model:
            results = self.model.predict(frame, verbose=False, conf=0.3)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    yolo_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls
                    })
                    
                    # Рисуем bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"Class {cls}: {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
        
        detection_info = {
            'left_sum': avg_left,
            'right_sum': avg_right,
            'left_type': left_type,
            'right_type': right_type,
            'contours_count': len(contours),
            'yolo_detections': yolo_detections
        }
        
        return annotated_frame, detection_info

def main():
    """Тестирование детектора на видео"""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python improved_road_marking_detector.py <video_path> [model_path]")
        return
    
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 80)
    print("🚗 ДЕТЕКТОР ДОРОЖНОЙ РАЗМЕТКИ (Улучшенный)")
    print("=" * 80)
    
    # Инициализация детектора
    detector = RoadMarkingDetector(model_path)
    
    # Открытие видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n📹 Видео: {width}x{height} @ {fps:.1f} FPS")
    print("🎯 Нажмите 'q' для выхода, SPACE для паузы\n")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Обработка кадра
            annotated_frame, info = detector.process_frame(frame, use_yolo=True)
            
            # Показываем кадр
            cv2.imshow("Road Marking Detection", annotated_frame)
            
            # Логируем каждые 30 кадров
            if frame_count % 30 == 0:
                print(f"Кадр {frame_count}: "
                      f"Контуров={info['contours_count']}, "
                      f"Left={info['left_type']} ({info['left_sum']:.0f}), "
                      f"Right={info['right_type']} ({info['right_sum']:.0f}), "
                      f"YOLO={len(info['yolo_detections'])}")
        
        # Управление
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Обработано {frame_count} кадров")

if __name__ == "__main__":
    main()
