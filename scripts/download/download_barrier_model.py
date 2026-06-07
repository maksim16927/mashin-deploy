#!/usr/bin/env python3
"""
Скрипт для загрузки модели детекции дорожных барьеров и ограждений
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def download_barrier_model():
    """
    Загрузка модели для детекции барьеров
    Использует COCO модель, которая уже содержит классы fence, pole
    """
    
    print("=" * 80)
    print("📥 ЗАГРУЗКА МОДЕЛИ ДЛЯ ДЕТЕКЦИИ БАРЬЕРОВ".center(80))
    print("=" * 80)
    print()
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Для барьеров используем существующую COCO модель
    # Она уже содержит классы: fence (ограждение), pole (столб)
    
    model_name = "yolov8n.pt"  # Базовая COCO модель
    model_path = models_dir / model_name
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Модель уже существует: {model_name}")
        print(f"   Размер: {size_mb:.1f} MB")
        print(f"   Путь: {model_path}")
        print()
        print(f"💡 Модель содержит классы COCO, включая:")
        print(f"   - fence (ограждение)")
        print(f"   - pole (столб)")
        print(f"   - другие объекты инфраструктуры")
        print()
        return str(model_path)
    
    # Загружаем модель если её нет
    print(f"⏳ Загрузка модели: {model_name}...")
    try:
        model = YOLO(model_name)
        # Модель автоматически скачается в кеш, нужно скопировать в models/
        # Попробуем найти её в кеше и скопировать
        from pathlib import Path
        import shutil
        
        # Ultralytics кеширует модели в HOME/.ultralytics/weights/
        home_dir = Path.home()
        cache_dir = home_dir / ".ultralytics" / "weights"
        
        if cache_dir.exists():
            cached_model = cache_dir / model_name
            if cached_model.exists():
                shutil.copy(cached_model, model_path)
                print(f"✅ Модель скопирована в: {model_path}")
            else:
                # Сохраняем модель напрямую
                model.save(str(model_path))
                print(f"✅ Модель сохранена в: {model_path}")
        else:
            # Сохраняем модель напрямую
            model.save(str(model_path))
            print(f"✅ Модель сохранена в: {model_path}")
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   Размер: {size_mb:.1f} MB")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return None
    
    print("=" * 80)
    print("✅ МОДЕЛЬ ДЛЯ БАРЬЕРОВ ГОТОВА!".center(80))
    print("=" * 80)
    print()
    
    return str(model_path)


def check_models():
    """Проверка всех доступных моделей"""
    
    print("=" * 80)
    print("📋 ПРОВЕРКА ДОСТУПНЫХ МОДЕЛЕЙ".center(80))
    print("=" * 80)
    print()
    
    models_dir = Path("models")
    required_models = {
        "yolov8s_35epochs_rtsd155.pt": "Дорожные знаки (RTSD-155)",
        "yolov8n.pt": "COCO инфраструктура (включая барьеры)",
    }
    
    available = []
    missing = []
    
    for model_file, description in required_models.items():
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_file}")
            print(f"   {description}")
            print(f"   Размер: {size_mb:.1f} MB")
            print()
            available.append(model_file)
        else:
            print(f"❌ {model_file}")
            print(f"   {description}")
            print(f"   ⚠️  Не найдена!")
            print()
            missing.append(model_file)
    
    print("=" * 80)
    if missing:
        print(f"⚠️  Отсутствует моделей: {len(missing)}/{len(required_models)}")
        print()
        print("💡 Для загрузки запустите:")
        print("   python scripts/download_all_models.py")
    else:
        print(f"✅ Все модели доступны: {len(available)}/{len(required_models)}")
    print("=" * 80)
    print()
    
    return available, missing


if __name__ == "__main__":
    # Проверяем существующие модели
    available, missing = check_models()
    
    # Загружаем модель для барьеров если нужно
    if "yolov8n.pt" in missing:
        print()
        model_path = download_barrier_model()
        if model_path:
            print(f"✅ Модель готова: {model_path}")
        else:
            print(f"❌ Не удалось загрузить модель")
            sys.exit(1)
    else:
        print()
        print("✅ Модель для барьеров уже доступна!")
    
    print()
    print("💡 Теперь бот может детектировать:")
    print("   🚦 Дорожные знаки (RTSD-155)")
    print("   🏗️  Инфраструктуру (автобусы, машины, светофоры)")
    print("   🚧 Дорожные барьеры/ограждения (fence, pole)")
    print("   🛣️  Дорожную разметку")
    print()
    sys.exit(0)
