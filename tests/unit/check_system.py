#!/usr/bin/env python3
"""
Демонстрационный скрипт - проверка всех компонентов системы
Запустите его для проверки, что всё работает правильно
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """Красивый заголовок"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def check_dependencies():
    """Проверка установленных зависимостей"""
    print_header("🔍 ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    
    dependencies = {
        'ultralytics': 'YOLOv8',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'torch': 'PyTorch'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'ultralytics':
                from ultralytics import YOLO
            elif module == 'numpy':
                import numpy
            elif module == 'torch':
                import torch
            print(f"✅ {name} - установлен")
        except ImportError:
            print(f"❌ {name} - НЕ установлен")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Установите отсутствующие пакеты:")
        print("   pip install -r requirements.txt\n")
        return False
    
    return True

def check_models():
    """Проверка наличия моделей"""
    print_header("🧠 ПРОВЕРКА МОДЕЛЕЙ")
    
    models = {
        'models/yolov8s_35epochs_rtsd155.pt': 'Модель дорожных знаков (RTSD)',
        'models/yolov8n-seg.pt': 'Базовая модель сегментации'
    }
    
    found_models = []
    missing_models = []
    
    for model_path, description in models.items():
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {description}")
            print(f"   Путь: {model_path}")
            print(f"   Размер: {size:.1f} MB")
            found_models.append(model_path)
        else:
            print(f"❌ {description} - не найдена")
            print(f"   Ожидается: {model_path}")
            missing_models.append(model_path)
    
    # Поиск обученных моделей инфраструктуры
    infra_models = list(Path('models').glob('infrastructure_model_*.pt'))
    if infra_models:
        print(f"\n✅ Найдено обученных моделей инфраструктуры: {len(infra_models)}")
        for model in infra_models[:3]:  # Показываем первые 3
            size = os.path.getsize(model) / (1024 * 1024)
            print(f"   • {model.name} ({size:.1f} MB)")
    else:
        print(f"\n⚠️  Модели инфраструктуры не найдены")
        print("   Обучите модель: python3 scripts/train_infrastructure.py")
    
    return len(found_models) > 0

def check_videos():
    """Проверка наличия видео"""
    print_header("📹 ПРОВЕРКА ВИДЕО")
    
    original_videos = list(Path('videos/original').glob('*'))
    detected_videos = list(Path('videos/detected').glob('*'))
    
    print(f"Исходные видео: {len(original_videos)}")
    for video in original_videos[:5]:
        if video.is_file():
            size = os.path.getsize(video) / (1024 * 1024)
            print(f"   • {video.name} ({size:.1f} MB)")
    
    print(f"\nОбработанные видео: {len(detected_videos)}")
    for video in detected_videos[:5]:
        if video.is_file():
            size = os.path.getsize(video) / (1024 * 1024)
            print(f"   • {video.name} ({size:.1f} MB)")
    
    return len(original_videos) > 0

def check_scripts():
    """Проверка наличия скриптов"""
    print_header("📝 ПРОВЕРКА СКРИПТОВ")
    
    scripts = [
        'scripts/detect_video.py',
        'scripts/detect_all.py',
        'scripts/detect_infrastructure.py',
        'scripts/train_infrastructure.py',
        'scripts/download_infrastructure_dataset.py'
    ]
    
    all_found = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {os.path.basename(script)}")
        else:
            print(f"❌ {os.path.basename(script)} - не найден")
            all_found = False
    
    return all_found

def check_datasets():
    """Проверка загруженных датасетов"""
    print_header("📦 ПРОВЕРКА ДАТАСЕТОВ")
    
    datasets_dir = Path('datasets')
    if not datasets_dir.exists():
        print("⚠️  Папка datasets не найдена")
        print("   Датасеты еще не загружены")
        return False
    
    datasets = list(datasets_dir.glob('*'))
    if not datasets:
        print("⚠️  Датасеты не найдены")
        print("   Загрузите датасеты:")
        print("   python3 scripts/download_infrastructure_dataset.py --api-key YOUR_KEY")
        return False
    
    print(f"Найдено датасетов: {len(datasets)}")
    for dataset in datasets:
        if dataset.is_dir():
            # Подсчет изображений
            train_images = list(dataset.glob('train/images/*'))
            val_images = list(dataset.glob('val/images/*'))
            print(f"\n✅ {dataset.name}")
            print(f"   Train: {len(train_images)} изображений")
            print(f"   Val: {len(val_images)} изображений")
    
    return len(datasets) > 0

def show_next_steps():
    """Следующие шаги"""
    print_header("🚀 СЛЕДУЮЩИЕ ШАГИ")
    
    print("1️⃣  Если датасеты не загружены:")
    print("   • Получите API ключ: https://app.roboflow.com/settings/api")
    print("   • Загрузите: python3 scripts/download_infrastructure_dataset.py --api-key YOUR_KEY")
    
    print("\n2️⃣  Обучите модель инфраструктуры:")
    print("   • Базовое: python3 scripts/train_infrastructure.py")
    print("   • С параметрами: python3 scripts/train_infrastructure.py --epochs 100 --model yolov8s.pt")
    
    print("\n3️⃣  Протестируйте на видео:")
    print("   • Знаки+разметка: python3 scripts/detect_all.py 'video.mp4'")
    print("   • Полная детекция: python3 scripts/detect_infrastructure.py 'video.mp4' --infra-model models/infrastructure_model_*.pt")
    
    print("\n4️⃣  Документация:")
    print("   • Быстрый старт: cat QUICK_START_INFRASTRUCTURE.txt")
    print("   • Полное руководство: cat README_INFRASTRUCTURE.md")

def main():
    """Главная функция"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  🚗 ПРОВЕРКА СИСТЕМЫ РАСПОЗНАВАНИЯ ДОРОЖНЫХ ОБЪЕКТОВ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Проверки
    deps_ok = check_dependencies()
    models_ok = check_models()
    videos_ok = check_videos()
    scripts_ok = check_scripts()
    datasets_ok = check_datasets()
    
    # Итоги
    print_header("📊 ИТОГИ ПРОВЕРКИ")
    
    checks = [
        ("Зависимости", deps_ok),
        ("Модели", models_ok),
        ("Видео", videos_ok),
        ("Скрипты", scripts_ok),
        ("Датасеты", datasets_ok)
    ]
    
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
    
    # Статус готовности
    print("\n" + "=" * 70)
    ready_count = sum(1 for _, status in checks if status)
    total_count = len(checks)
    
    if ready_count == total_count:
        print("🎉 ВСЁ ГОТОВО! Система полностью готова к работе")
    elif ready_count >= 3:
        print("⚠️  ЧАСТИЧНО ГОТОВО - можно работать, но не все компоненты установлены")
    else:
        print("❌ НЕ ГОТОВО - требуется установка зависимостей и моделей")
    
    print(f"Готовность: {ready_count}/{total_count} ({ready_count*100//total_count}%)")
    print("=" * 70)
    
    # Следующие шаги
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
