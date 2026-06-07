#!/usr/bin/env python3
"""
Скрипт для скачивания модели русских дорожных знаков с Roboflow
"""

import os
import sys
from pathlib import Path


def download_russian_signs_model():
    """
    Скачивает обученную модель русских дорожных знаков с Roboflow
    """
    print("=" * 70)
    print("СКАЧИВАНИЕ МОДЕЛИ РУССКИХ ДОРОЖНЫХ ЗНАКОВ")
    print("=" * 70)
    
    try:
        from roboflow import Roboflow
        
        print("\n📦 Доступные датасеты с русскими дорожными знаками:")
        print("\n1. Russian Traffic Signs Recognition (mguogareva)")
        print("   - 2400 изображений")
        print("   - 165 классов знаков")
        print("   - Обученная модель доступна")
        
        print("\n2. Russian Signs (cchegeu)")
        print("   - С предобученной моделью")
        
        print("\n3. YOLO v8 Russian Road Signs (Skoltech)")
        print("   - 2000 изображений")
        
        print("\n" + "=" * 70)
        print("АВТОМАТИЧЕСКОЕ СКАЧИВАНИЕ")
        print("=" * 70)
        
        # Используем публичный API без ключа для первой попытки
        print("\n⏳ Подключение к Roboflow...")
        
        # Пробуем получить доступ к публичному датасету
        print("\n💡 Для скачивания моделей с Roboflow потребуется API ключ")
        print("   Получите его на: https://app.roboflow.com/")
        print("   Settings -> API -> Private API Key")
        
        api_key = input("\n🔑 Введите API ключ Roboflow (или Enter для пропуска): ").strip()
        
        if not api_key:
            print("\n⚠️  Без API ключа можно только скачать датасет вручную")
            print("\n💡 Варианты скачивания:")
            print("1. Roboflow Universe - mguogareva/russian-traffic-signs-recognition")
            print("   https://universe.roboflow.com/mguogareva/russian-traffic-signs-recognition")
            print("\n2. Roboflow Universe - cchegeu/russian-signs")
            print("   https://universe.roboflow.com/cchegeu/russian-signs")
            print("\n3. GitHub репозитории с обученными моделями")
            print("   - https://github.com/kth-vyu/traffic_sign_detection_yolov8s")
            
            return None
        
        rf = Roboflow(api_key=api_key)
        print("✓ Успешное подключение!")
        
        # Пробуем скачать датасет с моделью
        print("\n⏳ Скачивание датасета с моделью...")
        
        try:
            # Пробуем russian-traffic-signs-recognition
            project = rf.workspace("mguogareva").project("russian-traffic-signs-recognition")
            version = project.version(1)
            
            # Скачиваем датасет в формате YOLOv8
            dataset_path = Path("./data/datasets/russian_traffic_signs")
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            dataset = version.download("yolov8", location=str(dataset_path))
            
            print(f"\n✅ Датасет скачан!")
            print(f"📂 Расположение: {dataset_path.absolute()}")
            
            # Проверяем наличие модели в датасете
            model_files = list(dataset_path.rglob("*.pt"))
            if model_files:
                print(f"\n🎯 Найдены модели:")
                for model_file in model_files:
                    print(f"   - {model_file}")
                    # Копируем в models/
                    import shutil
                    dest = Path("./models") / f"russian_signs_{model_file.name}"
                    shutil.copy(model_file, dest)
                    print(f"   ✓ Скопирована в {dest}")
            
            # Информация о датасете
            yaml_file = dataset_path / "data.yaml"
            if yaml_file.exists():
                print(f"\n📄 Конфигурация: {yaml_file}")
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    print(f.read())
            
            return str(dataset_path)
            
        except Exception as e:
            print(f"\n⚠️  Ошибка при скачивании: {e}")
            print("\n💡 Попробуйте другой проект или скачайте вручную")
            return None
            
    except ImportError:
        print("\n❌ Библиотека roboflow не установлена")
        print("\n💡 Установите:")
        print("   pip install roboflow")
        return None
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return None


if __name__ == "__main__":
    download_russian_signs_model()
