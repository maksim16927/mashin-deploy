#!/usr/bin/env python3
"""
Скрипт для скачивания обученной модели русских дорожных знаков RTSD с Roboflow
"""

import os
import sys
from pathlib import Path


def main():
    print("=" * 80)
    print("СКАЧИВАНИЕ МОДЕЛИ RTSD (Русские дорожные знаки) С ROBOFLOW")
    print("=" * 80)
    
    print("\n📦 Лучшие датасеты с русскими дорожными знаками на Roboflow:")
    print("\n1. Russian Traffic Signs Recognition (mguogareva)")
    print("   • 2,400 изображений")
    print("   • 165 классов русских дорожных знаков")
    print("   • Обученная модель YOLOv8 доступна")
    print("   • URL: https://universe.roboflow.com/mguogareva/russian-traffic-signs-recognition")
    
    print("\n2. Russian Signs (cchegeu)")
    print("   • Предобученная модель YOLOv8")
    print("   • URL: https://universe.roboflow.com/cchegeu/russian-signs")
    
    print("\n3. YOLO v8 Russian Road Signs (Skoltech)")
    print("   • 2,000 изображений")
    print("   • Лицензия: MIT")
    print("   • URL: https://universe.roboflow.com/skoltech-zlr4k/yolo-v8-russian-road-signs")
    
    print("\n" + "=" * 80)
    print("АВТОМАТИЧЕСКОЕ СКАЧИВАНИЕ")
    print("=" * 80)
    
    try:
        from roboflow import Roboflow
        print("\n✅ Библиотека roboflow установлена")
    except ImportError:
        print("\n❌ Библиотека roboflow не установлена")
        print("\n💡 Установите командой:")
        print("   pip install roboflow")
        return
    
    print("\n📝 Для скачивания моделей требуется API ключ Roboflow")
    print("\n🔑 Как получить API ключ:")
    print("   1. Зарегистрируйтесь на https://roboflow.com")
    print("   2. Войдите в аккаунт: https://app.roboflow.com/")
    print("   3. Перейдите в Settings -> API")
    print("   4. Скопируйте Private API Key")
    
    api_key = input("\n🔑 Введите API ключ Roboflow (или Enter для пропуска): ").strip()
    
    if not api_key:
        print("\n⚠️  Пропущено - API ключ не введен")
        print("\n💡 АЛЬТЕРНАТИВНЫЙ СПОСОБ - Ручное скачивание:")
        print("\n   Шаг 1: Перейдите на страницу датасета:")
        print("   https://universe.roboflow.com/mguogareva/russian-traffic-signs-recognition")
        print("\n   Шаг 2: Нажмите кнопку 'Download Dataset'")
        print("\n   Шаг 3: Выберите формат 'YOLOv8 PyTorch'")
        print("\n   Шаг 4: Скачайте и распакуйте в:")
        print("   data/datasets/russian_traffic_signs/")
        print("\n   Шаг 5: После обучения модель будет в:")
        print("   runs/detect/train/weights/best.pt")
        print("\n   Шаг 6: Скопируйте best.pt в:")
        print("   models/rtsd_yolov8_165classes.pt")
        return
    
    print("\n⏳ Подключение к Roboflow...")
    
    try:
        rf = Roboflow(api_key=api_key)
        print("✅ Подключение успешно!")
        
        print("\n📦 Выберите датасет для скачивания:")
        print("1. mguogareva (165 классов) - Рекомендуется")
        print("2. cchegeu (с предобученной моделью)")
        print("3. Skoltech (базовый, 2 класса)")
        
        choice = input("\nВведите номер (1-3) [1]: ").strip() or "1"
        
        if choice == "1":
            workspace = "mguogareva"
            project_name = "russian-traffic-signs-recognition"
            output_dir = "data/datasets/russian_traffic_signs_mguogareva"
        elif choice == "2":
            workspace = "cchegeu"
            project_name = "russian-signs"
            output_dir = "data/datasets/russian_signs_cchegeu"
        elif choice == "3":
            workspace = "skoltech-zlr4k"
            project_name = "yolo-v8-russian-road-signs"
            output_dir = "data/datasets/russian_signs_skoltech"
        else:
            print("❌ Неверный выбор")
            return
        
        print(f"\n⏳ Скачивание датасета {workspace}/{project_name}...")
        
        project = rf.workspace(workspace).project(project_name)
        version = project.version(1)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset = version.download("yolov8", location=str(output_path))
        
        print(f"\n✅ Датасет успешно скачан!")
        print(f"📂 Расположение: {output_path.absolute()}")
        
        # Проверяем наличие data.yaml
        yaml_file = output_path / "data.yaml"
        if yaml_file.exists():
            print(f"\n📄 Конфигурация: {yaml_file}")
            
            # Читаем информацию о датасете
            import yaml
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
            
            print(f"\n📊 Информация о датасете:")
            print(f"   • Классов: {data_config.get('nc', 'N/A')}")
            if 'names' in data_config:
                print(f"   • Названия классов: {len(data_config['names'])} шт.")
            
            train_dir = output_path / "train" / "images"
            valid_dir = output_path / "valid" / "images"
            
            if train_dir.exists():
                train_count = len(list(train_dir.glob("*.*")))
                print(f"   • Обучающих изображений: {train_count}")
            
            if valid_dir.exists():
                valid_count = len(list(valid_dir.glob("*.*")))
                print(f"   • Валидационных изображений: {valid_count}")
        
        print("\n" + "=" * 80)
        print("✅ ГОТОВО К ОБУЧЕНИЮ!")
        print("=" * 80)
        
        print(f"\n💡 Для обучения модели на скачанном датасете:")
        print(f"\n   yolo detect train \\")
        print(f"     data={yaml_file} \\")
        print(f"     model=yolov8s.pt \\")
        print(f"     epochs=100 \\")
        print(f"     imgsz=640 \\")
        print(f"     batch=16")
        
        print(f"\n📝 После обучения модель будет в:")
        print(f"   runs/detect/train/weights/best.pt")
        
        print(f"\n💡 Скопируйте обученную модель:")
        print(f"   cp runs/detect/train/weights/best.pt models/rtsd_russian_signs.pt")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Проверьте:")
        print("   • Правильность API ключа")
        print("   • Доступ к интернету")
        print("   • Права доступа к датасету на Roboflow")


if __name__ == "__main__":
    main()
