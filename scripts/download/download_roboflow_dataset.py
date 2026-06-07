#!/usr/bin/env python3
"""
Скрипт для скачивания готового датасета дорожной разметки с Roboflow
"""

import os
import sys
from pathlib import Path


def download_roboflow_dataset():
    """
    Инструкции по скачиванию датасета с Roboflow
    """
    print("=" * 70)
    print("СКАЧИВАНИЕ ДАТАСЕТА ДОРОЖНОЙ РАЗМЕТКИ С ROBOFLOW")
    print("=" * 70)
    
    print("\n📦 Доступные датасеты на Roboflow Universe:")
    print("\n1. Road Marking Detection (788 images)")
    print("   URL: https://universe.roboflow.com/road-marking-detection/road-marking-iflo2")
    print("   Классы: Дорожная разметка")
    
    print("\n2. HP-RMDv2 (1920+ images)")
    print("   URL: https://universe.roboflow.com/road-marking-detection/hp-rmdv2")
    print("   Классы: Различные типы разметки")
    
    print("\n" + "=" * 70)
    print("АВТОМАТИЧЕСКОЕ СКАЧИВАНИЕ")
    print("=" * 70)
    
    try:
        # Проверяем установлен ли roboflow
        import roboflow
        print("\n✓ Библиотека roboflow установлена")
    except ImportError:
        print("\n⚠️  Требуется установка библиотеки roboflow")
        print("\n💡 Установите:")
        print("   pip install roboflow")
        
        response = input("\n❓ Установить сейчас? (y/n): ")
        if response.lower() == 'y':
            import subprocess
            print("\n⏳ Установка roboflow...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
            print("✓ Установлено!")
            import roboflow
        else:
            print("\n💡 Для ручного скачивания:")
            print("1. Зарегистрируйтесь на https://roboflow.com")
            print("2. Перейдите на страницу датасета")
            print("3. Нажмите 'Download' -> выберите 'YOLOv8'")
            print("4. Скачайте и распакуйте в проект")
            return None
    
    print("\n" + "=" * 70)
    print("ВХОД В АККАУНТ ROBOFLOW")
    print("=" * 70)
    
    print("\n💡 Для скачивания нужен API ключ:")
    print("1. Войдите на https://app.roboflow.com/")
    print("2. Откройте Settings -> API")
    print("3. Скопируйте Private API Key")
    
    api_key = input("\n🔑 Введите API ключ (или Enter для пропуска): ").strip()
    
    if not api_key:
        print("\n⚠️  API ключ не указан")
        print("\n💡 Для скачивания без API:")
        print("1. Перейдите: https://universe.roboflow.com/road-marking-detection/road-marking-iflo2")
        print("2. Нажмите 'Download Dataset'")
        print("3. Выберите формат 'YOLOv8'")
        print("4. Распакуйте в папку road_marking_dataset/")
        return None
    
    print("\n⏳ Подключение к Roboflow...")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        print("✓ Успешное подключение!")
        
        print("\n⏳ Скачивание датасета 'road-marking-iflo2'...")
        
        # Скачиваем датасет
        project = rf.workspace("road-marking-detection").project("road-marking-iflo2")
        dataset = project.version(1).download("yolov8", location="./road_marking_dataset")
        
        print("\n✅ Датасет скачан!")
        print(f"📂 Расположение: {Path('road_marking_dataset').absolute()}")
        
        # Проверяем структуру
        dataset_path = Path("road_marking_dataset")
        if dataset_path.exists():
            train_images = list((dataset_path / "train" / "images").glob("*")) if (dataset_path / "train" / "images").exists() else []
            val_images = list((dataset_path / "valid" / "images").glob("*")) if (dataset_path / "valid" / "images").exists() else []
            
            print(f"\n📊 Статистика датасета:")
            print(f"   - Обучающих изображений: {len(train_images)}")
            print(f"   - Валидационных изображений: {len(val_images)}")
            
            # Ищем data.yaml
            yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("data.yaml"))
            if yaml_files:
                yaml_path = yaml_files[0]
                print(f"   - Конфигурация: {yaml_path}")
                
                print("\n" + "=" * 70)
                print("✅ ГОТОВО К ОБУЧЕНИЮ!")
                print("=" * 70)
                print(f"\n💡 Запустите обучение:")
                print(f"   python3 train_marking_simple.py")
                
                return str(yaml_path)
        
        return "road_marking_dataset/data.yaml"
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Попробуйте скачать вручную:")
        print("   https://universe.roboflow.com/road-marking-detection/road-marking-iflo2")
        return None


if __name__ == "__main__":
    download_roboflow_dataset()
