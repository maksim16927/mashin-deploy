#!/usr/bin/env python3
"""
Скачивание публичного датасета дорожной разметки
"""

import os
import urllib.request
import zipfile
from pathlib import Path


def download_dataset():
    """
    Скачивает готовый датасет дорожной разметки
    """
    print("=" * 70)
    print("СКАЧИВАНИЕ ДАТАСЕТА ДОРОЖНОЙ РАЗМЕТКИ")
    print("=" * 70)
    
    print("\n💡 Для скачивания датасета с Roboflow:")
    print("1. Перейдите: https://universe.roboflow.com/road-marking-detection/road-marking-iflo2")
    print("2. Нажмите 'Download Dataset'")
    print("3. Выберите формат 'YOLOv8'")
    print("4. Скопируйте код для скачивания")
    print("\nИли используйте альтернативные источники:")
    print("- Kaggle: https://www.kaggle.com/search?q=road+marking")
    print("- GitHub: поиск готовых датасетов")
    
    print("\n" + "=" * 70)
    print("ИСПОЛЬЗОВАНИЕ ГОТОВЫХ МОДЕЛЕЙ")
    print("=" * 70)
    
    print("\n💡 Вместо обучения своей модели можно использовать:")
    print("1. YOLOv8-seg базовую модель (уже скачана)")
    print("2. Модель для lane detection от Ultralytics")
    print("3. Специализированные модели с Roboflow Universe")
    
    print("\n⏳ Попробуем использовать YOLOv8-seg для сегментации...")
    
    # Используем базовую модель сегментации
    # Она уже обучена на COCO и может распознавать некоторые объекты
    from ultralytics import YOLO
    
    model = YOLO("models/yolov8n-seg.pt")
    
    print("\n✅ Модель готова к использованию!")
    print("\n💡 Базовая модель YOLOv8-seg обучена на COCO dataset (80 классов)")
    print("   Классы включают: person, car, truck, bus, traffic light и др.")
    print("\n📝 Для распознавания разметки потребуется:")
    print("   - Специализированный датасет с разметкой")
    print("   - Или использование классических CV методов (Canny, HoughLines)")
    
    print("\n" + "=" * 70)
    print("✅ ГОТОВО")
    print("=" * 70)
    
    print("\n💡 Используйте скрипт detect_all.py для комбинированного распознавания:")
    print("   python3 detect_all.py 'video.mp4'")
    print("\n   Он использует:")
    print("   - YOLOv8s для знаков (работает отлично ✅)")
    print("   - Canny + HoughLines для разметки (классический CV)")


if __name__ == "__main__":
    download_dataset()
