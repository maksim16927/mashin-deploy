#!/usr/bin/env python3
"""
Скрипт для тестирования всех моделей
"""

import os
from pathlib import Path
from ultralytics import YOLO

def test_models():
    """Тестирование всех моделей"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ ДЕТЕКЦИИ")
    print("=" * 70)
    
    models_dir = Path("models")
    
    # Список моделей для тестирования
    models = [
        ("yolov8s_35epochs_rtsd155.pt", "Модель дорожных знаков (155 классов RTSD)"),
        ("yolov8n_road_marking.pt", "Модель дорожной разметки"),
        ("yolov8n-seg.pt", "Базовая модель сегментации YOLOv8"),
    ]
    
    print("\n🔍 Поиск моделей...\n")
    
    available_models = []
    missing_models = []
    
    for model_name, description in models:
        model_path = models_dir / model_name
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_name}")
            print(f"   📝 {description}")
            print(f"   📦 Размер: {size_mb:.1f} MB")
            print(f"   💾 Путь: {model_path}")
            available_models.append((model_path, description))
        else:
            print(f"❌ {model_name}")
            print(f"   📝 {description}")
            print(f"   ⚠️  Файл не найден")
            missing_models.append((model_name, description))
        print()
    
    # Тестирование доступных моделей
    if available_models:
        print("\n" + "=" * 70)
        print("ЗАГРУЗКА И ПРОВЕРКА МОДЕЛЕЙ")
        print("=" * 70 + "\n")
        
        for model_path, description in available_models:
            try:
                print(f"⏳ Загрузка: {model_path.name}...")
                model = YOLO(str(model_path))
                
                print(f"✅ Модель загружена успешно!")
                print(f"   📊 Тип задачи: {model.task}")
                print(f"   🎯 Количество классов: {len(model.names) if hasattr(model, 'names') else 'N/A'}")
                
                # Показываем первые 10 классов если доступны
                if hasattr(model, 'names') and model.names:
                    print(f"   📋 Примеры классов:")
                    for i, (key, name) in enumerate(list(model.names.items())[:10]):
                        print(f"      {key}: {name}")
                    if len(model.names) > 10:
                        print(f"      ... и еще {len(model.names) - 10} классов")
                
                print()
            except Exception as e:
                print(f"❌ Ошибка при загрузке модели: {e}\n")
    
    # Информация о недостающих моделях
    if missing_models:
        print("\n" + "=" * 70)
        print("НЕДОСТАЮЩИЕ МОДЕЛИ")
        print("=" * 70 + "\n")
        
        for model_name, description in missing_models:
            print(f"⚠️  {model_name}")
            print(f"   {description}")
        
        print("\n💡 Для скачивания моделей используйте:")
        print("   python scripts/download_marking_model.py")
    
    print("\n" + "=" * 70)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)
    print(f"\n📊 Статистика:")
    print(f"   ✅ Доступно моделей: {len(available_models)}")
    print(f"   ❌ Отсутствует моделей: {len(missing_models)}")
    
    if len(available_models) == len(models):
        print("\n🎉 Все модели готовы к использованию!")
    elif available_models:
        print("\n✅ Основные модели доступны, можно начинать работу!")
    else:
        print("\n⚠️  Необходимо скачать модели перед началом работы.")

if __name__ == "__main__":
    test_models()
