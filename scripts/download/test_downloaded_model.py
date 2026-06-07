#!/usr/bin/env python3
"""
Скрипт для тестирования скачанной модели дорожных знаков
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ ДОРОЖНЫХ ЗНАКОВ")
    print("=" * 70)
    
    # Путь к модели
    model_path = Path(__file__).parent.parent.parent / "models" / "traffic_signs_yolov8.pt"
    
    if not model_path.exists():
        print(f"\n❌ Модель не найдена: {model_path}")
        sys.exit(1)
    
    print(f"\n📦 Загрузка модели: {model_path.name}")
    print(f"   Размер: {model_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Загружаем модель
    model = YOLO(str(model_path))
    
    print("\n✅ Модель успешно загружена!")
    
    # Информация о модели
    print(f"\n📊 Информация о модели:")
    print(f"   - Тип задачи: {model.task}")
    print(f"   - Количество классов: {len(model.names)}")
    
    # Выводим названия классов
    print(f"\n🏷️  Классы дорожных знаков:")
    for idx, name in model.names.items():
        print(f"   {idx}: {name}")
    
    # Создаем тестовое изображение
    print("\n🖼️  Создание тестового изображения...")
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.circle(test_image, (320, 320), 100, (0, 0, 255), -1)
    cv2.circle(test_image, (320, 320), 80, (255, 255, 255), -1)
    
    # Тест детекции
    print("\n🔍 Тест детекции на пустом изображении...")
    results = model(test_image, verbose=False)
    
    print(f"   ✓ Детекция выполнена успешно")
    print(f"   Найдено объектов: {len(results[0].boxes)}")
    
    print("\n" + "=" * 70)
    print("✅ МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
    print("=" * 70)
    
    print("\n💡 Для использования модели:")
    print("   from ultralytics import YOLO")
    print(f"   model = YOLO('{model_path.name}')")
    print("   results = model('path/to/image.jpg')")
    
except ImportError as e:
    print(f"\n❌ Ошибка импорта: {e}")
    print("\n💡 Установите необходимые зависимости:")
    print("   pip install ultralytics opencv-python")
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()
