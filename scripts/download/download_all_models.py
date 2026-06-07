#!/usr/bin/env python3
"""
Скрипт для загрузки всех необходимых моделей из интернета
Включает модели для:
- Дорожных знаков (уже есть)
- Автобусных остановок
- П-образных опор (gantry)
- Дорожной инфраструктуры
- COCO объектов
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def download_model(url_or_name, save_path, model_type="yolo"):
    """
    Загрузка модели YOLO
    
    Args:
        url_or_name: Имя модели или URL для загрузки
        save_path: Путь для сохранения модели
        model_type: Тип модели (yolo/huggingface)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists():
        print(f"✅ Модель уже существует: {save_path.name}")
        return str(save_path)
    
    try:
        print(f"⏳ Загрузка модели: {url_or_name}...")
        
        if model_type == "yolo":
            # Загружаем через YOLO
            model = YOLO(url_or_name)
            # Сохраняем модель
            model.save(str(save_path))
            print(f"✅ Модель загружена: {save_path.name}")
            return str(save_path)
        else:
            print(f"⚠️  Неподдерживаемый тип модели: {model_type}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке {url_or_name}: {e}")
        return None

def download_all_models():
    """
    Загрузка всех необходимых моделей
    """
    
    print("=" * 80)
    print("📥 ЗАГРУЗКА ВСЕХ МОДЕЛЕЙ ДЛЯ ОБРАБОТКИ ВИДЕО".center(80))
    print("=" * 80)
    print()
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Список моделей для загрузки
    models_to_download = [
        {
            "name": "YOLOv8n (COCO)",
            "url": "yolov8n.pt",
            "path": models_dir / "yolov8n.pt",
            "description": "Базовая COCO модель для детекции общих объектов (автобусы, машины, люди, светофоры)"
        },
        {
            "name": "YOLOv8s (COCO)",
            "url": "yolov8s.pt",
            "path": models_dir / "yolov8s.pt",
            "description": "Более точная COCO модель для детекции инфраструктуры"
        },
        {
            "name": "YOLOv8n-seg (Segmentation)",
            "url": "yolov8n-seg.pt",
            "path": models_dir / "yolov8n-seg.pt",
            "description": "Сегментационная модель для детальной детекции объектов"
        },
        {
            "name": "YOLOv8n-road-marking",
            "url": "yolov8n_road_marking.pt",
            "path": models_dir / "yolov8n_road_marking.pt",
            "description": "Модель для детекции дорожной разметки",
            "optional": True  # Может не существовать
        }
    ]
    
    downloaded_models = []
    
    for model_info in models_to_download:
        print(f"\n📦 {model_info['name']}")
        print(f"   {model_info['description']}")
        
        if model_info.get('optional', False) and not model_info['path'].exists():
            print(f"   ⚠️  Пропущено (опциональная модель)")
            continue
            
        result = download_model(model_info['url'], model_info['path'])
        if result:
            downloaded_models.append(result)
        
        print()
    
    # Проверяем существующие модели
    print("=" * 80)
    print("📋 ПРОВЕРКА СУЩЕСТВУЮЩИХ МОДЕЛЕЙ".center(80))
    print("=" * 80)
    print()
    
    existing_models = {
        "Дорожные знаки (RTSD-155)": models_dir / "yolov8s_35epochs_rtsd155.pt",
        "YOLOv8n (COCO)": models_dir / "yolov8n.pt",
        "YOLOv8s (COCO)": models_dir / "yolov8s.pt",
        "YOLOv8n-seg": models_dir / "yolov8n-seg.pt",
        "Дорожная разметка": models_dir / "yolov8n_road_marking.pt"
    }
    
    available_models = []
    for name, path in existing_models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {name}: {path.name} ({size_mb:.1f} MB)")
            available_models.append((name, str(path)))
        else:
            print(f"❌ {name}: не найдена")
    
    print()
    print("=" * 80)
    print("✅ ЗАГРУЗКА ЗАВЕРШЕНА!".center(80))
    print("=" * 80)
    print()
    
    if available_models:
        print(f"📊 Доступно моделей: {len(available_models)}")
        print("\n💡 Использование моделей:")
        print("   python scripts/detect_full_system.py video.mp4")
        print("   python scripts/process_all_videos_full.py")
    
    return available_models

if __name__ == "__main__":
    models = download_all_models()
    
    if models:
        print(f"\n🎉 Готово! Доступно {len(models)} моделей для обработки видео.")
        sys.exit(0)
    else:
        print("\n⚠️  Не удалось загрузить модели. Проверьте подключение к интернету.")
        sys.exit(1)
