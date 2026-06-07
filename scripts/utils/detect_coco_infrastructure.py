"""
Скрипт для быстрого тестирования базовой модели COCO на дорожной инфраструктуре
COCO модель уже обучена на 80 классах включая:
- bus (автобус)
- traffic light (светофор)
- stop sign (знак стоп)
- truck (грузовик)
- car (машина)
- person (человек)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

# Классы COCO, связанные с дорожной инфраструктурой
ROAD_RELATED_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13
}

# Цвета для разных категорий
COLORS = {
    'bus': (0, 255, 255),  # Желтый
    'traffic light': (0, 255, 0),  # Зеленый
    'stop sign': (0, 0, 255),  # Красный
    'car': (255, 0, 0),  # Синий
    'truck': (255, 165, 0),  # Оранжевый
    'person': (128, 128, 128),  # Серый
    'bicycle': (255, 0, 255),  # Фиолетовый
    'motorcycle': (0, 255, 255),  # Голубой
    'default': (147, 20, 255)  # Розовый
}

def detect_road_infrastructure(video_path, output_path=None, model_path='yolov8n.pt'):
    """
    Распознавание дорожной инфраструктуры используя базовую модель COCO
    """
    
    print("=" * 70)
    print("🚗 ДЕТЕКЦИЯ ДОРОЖНОЙ ИНФРАСТРУКТУРЫ (YOLO COCO)")
    print("=" * 70)
    
    # Загружаем модель
    print(f"\n⏳ Загрузка модели {model_path}...")
    model = YOLO(model_path)
    
    print(f"✅ Модель загружена")
    print(f"\n🎯 Классы для детекции:")
    for class_name in ROAD_RELATED_CLASSES.keys():
        print(f"   • {class_name}")
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return
    
    # Параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Видео: {Path(video_path).name}")
    print(f"   Разрешение: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Кадров: {total_frames}")
    
    # Выходное видео
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"videos/detected/{video_name}_coco_infrastructure.mp4"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"💾 Сохранение в: {output_path}")
    print("\n🚀 Обработка видео...")
    
    # Статистика
    detections_by_class = {}
    frame_num = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Детекция только интересующих классов
        results = model(frame, classes=list(ROAD_RELATED_CLASSES.values()), conf=0.3, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # Цвет по классу
                color = COLORS.get(class_name, COLORS['default'])
                
                # Рисуем bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Метка
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Статистика
                detections_by_class[class_name] = detections_by_class.get(class_name, 0) + 1
                total_detections += 1
        
        # Информация на кадре
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {total_detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        
        if frame_num % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"   Прогресс: {progress:.1f}% ({frame_num}/{total_frames})")
    
    cap.release()
    out.release()
    
    print("\n" + "=" * 70)
    print("✅ ОБРАБОТКА ЗАВЕРШЕНА!")
    print("=" * 70)
    
    print(f"\n📊 Статистика детекций:")
    print(f"   Всего кадров: {frame_num}")
    print(f"   Всего детекций: {total_detections}")
    print(f"\n📈 Детекций по классам:")
    for class_name, count in sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True):
        print(f"   {class_name}: {count}")
    
    print(f"\n💾 Результат: {output_path}")
    
    # Размер файла
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"📦 Размер: {file_size:.1f} MB")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Детекция дорожной инфраструктуры с COCO моделью")
    parser.add_argument("video", type=str, help="Путь к видео")
    parser.add_argument("-o", "--output", type=str, default=None, help="Путь для сохранения")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Путь к модели")
    
    args = parser.parse_args()
    
    output = detect_road_infrastructure(args.video, args.output, args.model)
    
    if output:
        print(f"\n✅ Готово! Откройте: {output}")
