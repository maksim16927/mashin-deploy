#!/usr/bin/env python3
"""
Автоматическая генерация лейблов для дорожной разметки
Использует Hough line detection для создания YOLO меток

Запуск:
    python scripts/utils/auto_label_road_marking.py
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Пути
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "road_marking_dataset"
IMAGES_TRAIN = DATASET_PATH / "images" / "train"
LABELS_TRAIN = DATASET_PATH / "labels" / "train"

# Классы (по data.yaml)
# 0: solid_line, 1: dashed_line, 8: lane_marking
CLASS_SOLID_LINE = 0
CLASS_DASHED_LINE = 1
CLASS_LANE_MARKING = 8

def detect_lane_lines(image_path, min_length=50):
    """Детекция линий разметки с помощью Hough transform"""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    height, width = img.shape[:2]
    
    # ROI - нижняя 75% изображения
    roi_y = int(height * 0.25)
    roi = img[roi_y:height, 0:width]
    
    # Преобразование в серый
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Выделение белых линий
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Размытие и Canny
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges1 = cv2.Canny(blur, 30, 100)
    edges2 = cv2.Canny(white_thresh, 50, 150)
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Морфология
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Hough lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                           threshold=25, minLineLength=20, maxLineGap=10)
    
    if lines is None:
        return []
    
    detected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Переводим координаты в полные координаты изображения
        y1 += roi_y
        y2 += roi_y
        
        # Вычисляем угол и длину
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Фильтр: вертикальные линии (40-140 градусов), длина > min_length
        if (40 < abs(angle) < 140) and length > min_length:
            detected_lines.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'angle': angle, 'length': length
            })
    
    return detected_lines

def line_to_yolo_bbox(line, img_width, img_height, padding=10):
    """Конвертация линии в YOLO формат bbox"""
    x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
    
    # Находим bounding box вокруг линии с отступами
    x_min = max(0, min(x1, x2) - padding)
    x_max = min(img_width, max(x1, x2) + padding)
    y_min = max(0, min(y1, y2) - padding)
    y_max = min(img_height, max(y1, y2) + padding)
    
    # YOLO формат: x_center, y_center, width, height (нормализованные 0-1)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height
    
    return x_center, y_center, bbox_width, bbox_height

def determine_line_class(line):
    """Определение класса линии (сплошная/пунктирная)"""
    # Простая эвристика: короткие линии = пунктир, длинные = сплошная
    if line['length'] < 80:
        return CLASS_DASHED_LINE
    else:
        return CLASS_SOLID_LINE

def generate_labels_for_image(image_path, output_label_path):
    """Генерация YOLO лейблов для одного изображения"""
    # Детектируем линии
    lines = detect_lane_lines(image_path)
    
    if not lines:
        # Создаем пустой файл (background)
        output_label_path.write_text("")
        return 0
    
    # Читаем размеры изображения
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    
    # Генерируем лейблы
    labels = []
    for line in lines:
        class_id = determine_line_class(line)
        x_center, y_center, bbox_width, bbox_height = line_to_yolo_bbox(line, width, height)
        
        # YOLO формат: class x_center y_center width height
        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
    
    # Сохраняем лейблы
    output_label_path.write_text("\n".join(labels) + "\n")
    return len(labels)

def main():
    """Основная функция"""
    print("=" * 80)
    print("🏷️  АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ ЛЕЙБЛОВ ДЛЯ ДОРОЖНОЙ РАЗМЕТКИ")
    print("=" * 80)
    
    if not IMAGES_TRAIN.exists():
        print(f"❌ Папка с изображениями не найдена: {IMAGES_TRAIN}")
        return
    
    # Создаем папку для лейблов
    LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
    
    # Получаем список изображений
    image_files = list(IMAGES_TRAIN.glob("*.jpg")) + list(IMAGES_TRAIN.glob("*.png"))
    image_files = [f for f in image_files if f.name != "__init__.py"]
    
    print(f"\n📊 Найдено изображений: {len(image_files)}")
    print(f"📁 Результаты будут сохранены в: {LABELS_TRAIN}")
    print("\n🔄 Обработка изображений...")
    
    total_labels = 0
    images_with_labels = 0
    
    for i, image_path in enumerate(image_files, 1):
        # Путь к файлу лейблов
        label_path = LABELS_TRAIN / (image_path.stem + ".txt")
        
        # Генерируем лейблы
        num_labels = generate_labels_for_image(image_path, label_path)
        total_labels += num_labels
        
        if num_labels > 0:
            images_with_labels += 1
        
        # Прогресс
        if i % 10 == 0 or i == len(image_files):
            print(f"   Обработано: {i}/{len(image_files)} изображений "
                  f"({images_with_labels} с метками, {total_labels} объектов)")
    
    print("\n" + "=" * 80)
    print("✅ ГЕНЕРАЦИЯ ЛЕЙБЛОВ ЗАВЕРШЕНА")
    print("=" * 80)
    print(f"\n📈 Статистика:")
    print(f"   - Всего изображений: {len(image_files)}")
    print(f"   - С метками: {images_with_labels}")
    print(f"   - Без меток (background): {len(image_files) - images_with_labels}")
    print(f"   - Всего объектов: {total_labels}")
    print(f"   - Среднее на изображение: {total_labels / len(image_files):.1f}")
    
    print(f"\n💡 Теперь можно запустить обучение:")
    print(f"   python scripts/training/train_road_marking_20epochs.py")

if __name__ == "__main__":
    main()
