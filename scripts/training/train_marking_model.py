#!/usr/bin/env python3
"""
Скрипт для обучения модели распознавания дорожной разметки
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def create_marking_dataset():
    """
    Создает минимальный датасет для тестового обучения
    """
    print("\n📦 Создание тестового датасета для дорожной разметки...")
    
    dataset_dir = Path("road_marking_dataset")
    dataset_dir.mkdir(exist_ok=True)
    
    # Создаем структуру
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Создаем конфигурационный файл
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 6,  # количество классов
        'names': [
            'solid_line',      # сплошная линия
            'dashed_line',     # пунктирная линия
            'double_line',     # двойная линия
            'stop_line',       # стоп-линия
            'crosswalk',       # пешеходный переход
            'road_edge'        # край дороги
        ]
    }
    
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ Датасет создан: {dataset_dir}")
    print(f"✓ Конфигурация: {yaml_path}")
    print(f"✓ Классы: {', '.join(data_yaml['names'])}")
    
    return yaml_path


def extract_frames_from_video(video_path, output_dir, max_frames=50):
    """
    Извлекает кадры из видео для создания датасета
    """
    import cv2
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Видео не найдено: {video_path}")
        return 0
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📹 Извлечение кадров из {video_path.name}...")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Извлекаем каждый N-й кадр
    step = max(1, total_frames // max_frames)
    
    count = 0
    frame_idx = 0
    
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            output_path = output_dir / f"frame_{count:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"✓ Извлечено {count} кадров")
    
    return count


def train_marking_model(epochs=30, imgsz=640):
    """
    Обучает модель распознавания дорожной разметки
    """
    print("=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ ДОРОЖНОЙ РАЗМЕТКИ")
    print("=" * 70)
    
    # Путь к базовой модели
    base_model = "models/yolov8n-seg.pt"
    if not Path(base_model).exists():
        print(f"❌ Базовая модель не найдена: {base_model}")
        return None
    
    # Создаем датасет
    data_yaml = create_marking_dataset()
    
    # Извлекаем кадры из видео для обучения
    dataset_dir = Path("road_marking_dataset")
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    
    # Проверяем наличие видео
    videos = list(Path(".").glob("*.mp4")) + list(Path(".").glob("*.MOV"))
    if videos:
        video = videos[0]  # Берем первое видео
        print(f"\n📹 Используем видео для создания датасета: {video.name}")
        
        # Извлекаем кадры для обучения и валидации
        train_count = extract_frames_from_video(video, train_dir, max_frames=40)
        val_count = extract_frames_from_video(video, val_dir, max_frames=10)
        
        if train_count == 0:
            print("\n⚠️  Нет кадров для обучения!")
            print("💡 Добавьте изображения в:")
            print(f"   - Обучение: {train_dir}")
            print(f"   - Валидация: {val_dir}")
            print("\n💡 Для каждого изображения нужен файл разметки YOLO format:")
            print("   example.jpg -> example.txt")
            return None
    else:
        print("\n⚠️  Видео не найдено! Создана структура датасета.")
        print("💡 Добавьте изображения и разметку:")
        print(f"   - {train_dir}")
        print(f"   - {val_dir}")
        return None
    
    # Создаем пустые файлы разметки (для примера)
    print("\n📝 Создание примеров разметки...")
    for img_path in train_dir.glob("*.jpg"):
        label_path = dataset_dir / "labels" / "train" / f"{img_path.stem}.txt"
        label_path.touch()
    
    for img_path in val_dir.glob("*.jpg"):
        label_path = dataset_dir / "labels" / "val" / f"{img_path.stem}.txt"
        label_path.touch()
    
    print(f"✓ Создано {train_count} примеров для обучения")
    print(f"✓ Создано {val_count} примеров для валидации")
    
    print("\n" + "=" * 70)
    print("⚠️  ВАЖНО: Необходима разметка!")
    print("=" * 70)
    print("\n💡 Для полноценного обучения нужно:")
    print("1. Разметить извлеченные кадры (используйте labelImg, CVAT, или Roboflow)")
    print("2. Добавить файлы разметки в формате YOLO (*.txt)")
    print("3. Запустить обучение снова")
    
    print("\n💡 Альтернатива - скачать готовый датасет:")
    print("   - Roboflow Universe: https://universe.roboflow.com/search?q=road+marking")
    print("   - Kaggle: https://www.kaggle.com/search?q=road+marking")
    
    # Пробуем обучить на том что есть (даже с пустой разметкой для демонстрации)
    print("\n" + "=" * 70)
    print("ЗАПУСК ОБУЧЕНИЯ (демонстрационный режим)")
    print("=" * 70)
    
    try:
        # Загружаем базовую модель
        model = YOLO(base_model)
        
        print(f"\n⏳ Обучение модели...")
        print(f"📊 Параметры:")
        print(f"   - Эпох: {epochs}")
        print(f"   - Размер изображения: {imgsz}")
        print(f"   - Датасет: {data_yaml}")
        
        # Обучаем модель
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=4,
            device='mps',  # Apple Silicon
            project='runs/marking_train',
            name='yolov8n_marking',
            exist_ok=True,
            patience=10,
            save=True,
            plots=True,
            verbose=True
        )
        
        # Сохраняем обученную модель
        output_model = Path("models/yolov8n_road_marking.pt")
        best_model = Path("runs/marking_train/yolov8n_marking/weights/best.pt")
        
        if best_model.exists():
            import shutil
            shutil.copy(best_model, output_model)
            print(f"\n✅ Модель сохранена: {output_model}")
            print(f"📊 Размер: {output_model.stat().st_size / (1024*1024):.1f} MB")
            return str(output_model)
        else:
            print("\n⚠️  Обученная модель не найдена")
            return None
            
    except Exception as e:
        print(f"\n❌ Ошибка обучения: {e}")
        print("\n💡 Это нормально - нет размеченных данных!")
        print("   Скачайте готовый датасет или разметьте кадры вручную.")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение модели разметки")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Количество эпох")
    parser.add_argument("-s", "--imgsz", type=int, default=640, help="Размер изображения")
    args = parser.parse_args()
    
    model_path = train_marking_model(epochs=args.epochs, imgsz=args.imgsz)
    
    if model_path:
        print("\n" + "=" * 70)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 70)
        print(f"\n💾 Модель: {model_path}")
        print("\n💡 Тестирование:")
        print(f"   python3 detect_marking.py 'video.mp4' -m {model_path}")
    else:
        print("\n" + "=" * 70)
        print("⚠️  ТРЕБУЕТСЯ ДАТАСЕТ")
        print("=" * 70)
        print("\n💡 Используйте готовый датасет с Roboflow Universe")


if __name__ == "__main__":
    main()
