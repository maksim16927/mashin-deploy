#!/usr/bin/env python3
"""
Простое обучение модели разметки (если датасет уже скачан)
"""

from pathlib import Path
from ultralytics import YOLO
import shutil


def train_simple():
    print("=" * 70)
    print("БЫСТРОЕ ОБУЧЕНИЕ МОДЕЛИ РАЗМЕТКИ")
    print("=" * 70)
    
    # Проверяем датасет
    dataset_path = Path("road_marking_dataset")
    yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("data.yaml"))
    
    if not yaml_files:
        print("\n❌ Датасет не найден!")
        print("💡 Сначала скачайте датасет:")
        print("   python3 download_roboflow_dataset.py")
        return
    
    data_yaml = yaml_files[0]
    print(f"\n✓ Датасет найден: {data_yaml}")
    
    # Загружаем базовую модель
    base_model = "models/yolov8n-seg.pt"
    if not Path(base_model).exists():
        print(f"\n❌ Базовая модель не найдена: {base_model}")
        print("💡 Скачайте модель:")
        print("   python3 download_marking_model.py")
        return
    
    print(f"✓ Базовая модель: {base_model}")
    
    # Параметры обучения
    epochs = 50
    imgsz = 640
    batch = 8
    
    print(f"\n📊 Параметры обучения:")
    print(f"   - Эпохи: {epochs}")
    print(f"   - Размер изображения: {imgsz}")
    print(f"   - Batch size: {batch}")
    
    print("\n🚀 Запуск обучения...")
    print("⏳ Это займет несколько минут...\n")
    
    try:
        model = YOLO(base_model)
        
        # Обучение
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device='mps',
            project='runs/marking',
            name='exp',
            exist_ok=True,
            patience=15,
            save=True,
            plots=True,
            val=True,
            verbose=True
        )
        
        # Копируем лучшую модель
        best_model = Path("runs/marking/exp/weights/best.pt")
        output_model = Path("models/yolov8n_road_marking.pt")
        
        if best_model.exists():
            shutil.copy(best_model, output_model)
            
            print("\n" + "=" * 70)
            print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print("=" * 70)
            print(f"\n💾 Модель сохранена: {output_model}")
            print(f"📊 Размер: {output_model.stat().st_size / (1024*1024):.1f} MB")
            
            print("\n💡 Тестирование на видео:")
            print(f"   python3 detect_marking.py '2026-01-15 10.00.52.mp4' -m {output_model}")
            
            return str(output_model)
        else:
            print("\n❌ Обученная модель не найдена")
            return None
            
    except Exception as e:
        print(f"\n❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    train_simple()
