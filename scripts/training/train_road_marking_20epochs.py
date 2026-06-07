#!/usr/bin/env python3
"""
Скрипт обучения YOLOv8 модели для детекции дорожной разметки
Включает: разметку, барьеры, шумовые полосы, полосы движения и т.д.

Основано на подходе: https://github.com/Serzho/Road-marking-recognition
- Использует ROI (нижняя часть кадра)
- Медианное размытие + пороговое преобразование
- Гауссовское размытие + Canny для контуров
- YOLOv8 для детекции вместо классического CV

Использование:
    python scripts/training/train_road_marking_20epochs.py
    
Результат:
    - Обученная модель в runs/detect/train/weights/best.pt
    - Метрики и графики в runs/detect/train/
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

# Настройка путей
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "road_marking_dataset"
DATA_YAML = DATASET_PATH / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "marking_train"

# Параметры обучения
EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 640
MODEL_SIZE = 'n'  # nano - быстрая модель, можно изменить на 's', 'm', 'l', 'x'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 10  # Early stopping patience
SAVE_PERIOD = 5  # Сохранять веса каждые 5 эпох

# Гиперпараметры оптимизации
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

def check_dataset():
    """Проверка наличия датасета"""
    if not DATA_YAML.exists():
        print(f"❌ Файл данных не найден: {DATA_YAML}")
        print("📋 Создайте датасет или укажите правильный путь")
        return False
    
    # Проверяем директории с изображениями
    train_dir = DATASET_PATH / "images" / "train"
    val_dir = DATASET_PATH / "images" / "val"
    
    if not train_dir.exists():
        print(f"❌ Папка train не найдена: {train_dir}")
        return False
    
    if not val_dir.exists():
        print(f"⚠️  Папка val не найдена: {val_dir}")
        print("💡 Будет использована только train выборка")
    
    # Подсчет изображений
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")) if val_dir.exists() else []
    
    print(f"📊 Датасет статистика:")
    print(f"   - Train изображений: {len(train_images)}")
    print(f"   - Val изображений: {len(val_images)}")
    
    if len(train_images) == 0:
        print("❌ Нет изображений для обучения!")
        return False
    
    return True

def download_pretrained_weights():
    """Скачивание предобученных весов YOLOv8"""
    model_name = f'yolov8{MODEL_SIZE}.pt'
    model_path = PROJECT_ROOT / "data" / "models" / model_name
    
    if model_path.exists():
        print(f"✅ Используем существующие веса: {model_name}")
        return str(model_path)
    
    print(f"⬇️  Скачиваем предобученные веса {model_name}...")
    return model_name  # YOLO автоматически скачает если не найдет

def train_model():
    """Основная функция обучения"""
    print("=" * 80)
    print("🚀 ОБУЧЕНИЕ YOLOV8 ДЛЯ ДЕТЕКЦИИ ДОРОЖНОЙ РАЗМЕТКИ")
    print("=" * 80)
    
    # Проверка датасета
    print("\n📋 Проверка датасета...")
    if not check_dataset():
        return
    
    # Загрузка модели
    print(f"\n🤖 Загрузка модели YOLOv8{MODEL_SIZE}...")
    pretrained_weights = download_pretrained_weights()
    model = YOLO(pretrained_weights)
    
    print(f"💻 Устройство: {DEVICE}")
    print(f"📦 Размер батча: {BATCH_SIZE}")
    print(f"🖼️  Размер изображений: {IMG_SIZE}x{IMG_SIZE}")
    print(f"🔄 Эпохи: {EPOCHS}")
    print(f"⏱️  Patience (early stopping): {PATIENCE}")
    
    # Создание директории для результатов
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Результаты будут сохранены в: {OUTPUT_DIR}")
    print("\n" + "=" * 80)
    print("🏋️  НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 80 + "\n")
    
    # Обучение модели
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            
            # Оптимизация
            lr0=LEARNING_RATE,
            lrf=0.01,  # Финальная learning rate (lr0 * lrf)
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Сохранение
            save=True,
            save_period=SAVE_PERIOD,
            patience=PATIENCE,
            project=str(OUTPUT_DIR.parent),
            name=OUTPUT_DIR.name,
            exist_ok=True,
            
            # Аугментация
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            
            # Прочее
            verbose=True,
            plots=True,
            val=True,
            cache=False,  # Не кэшировать в RAM (экономия памяти)
            workers=8,
        )
        
        print("\n" + "=" * 80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 80)
        
        # Путь к лучшей модели
        best_model_path = OUTPUT_DIR / "weights" / "best.pt"
        last_model_path = OUTPUT_DIR / "weights" / "last.pt"
        
        if best_model_path.exists():
            print(f"\n🏆 Лучшая модель: {best_model_path}")
            print(f"📊 Метрики и графики: {OUTPUT_DIR}")
            
            # Копируем в основную директорию моделей
            final_model_path = PROJECT_ROOT / "data" / "models" / "yolov8_road_marking_20epochs.pt"
            import shutil
            shutil.copy(best_model_path, final_model_path)
            print(f"\n✅ Модель скопирована в: {final_model_path}")
            print(f"💾 Размер модели: {final_model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        if last_model_path.exists():
            print(f"📌 Последняя модель: {last_model_path}")
        
        # Показываем метрики
        print("\n📈 Основные метрики:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
        
        print("\n💡 Для использования модели обновите MARKING_MODEL_PATH в bot.py:")
        print(f"   MARKING_MODEL_PATH = str(MODEL_DIR / 'yolov8_road_marking_20epochs.pt')")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
        print("📁 Промежуточные результаты сохранены в:", OUTPUT_DIR)
    
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)

def validate_model():
    """Валидация обученной модели"""
    best_model_path = OUTPUT_DIR / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print("❌ Обученная модель не найдена. Сначала запустите обучение.")
        return
    
    print("\n🔍 ВАЛИДАЦИЯ МОДЕЛИ")
    print("=" * 80)
    
    model = YOLO(str(best_model_path))
    results = model.val(data=str(DATA_YAML), imgsz=IMG_SIZE, batch=BATCH_SIZE)
    
    print("✅ Валидация завершена")
    print(f"📊 Результаты: {OUTPUT_DIR}/val")

def test_model_on_image(image_path: str):
    """Тестирование модели на одном изображении"""
    best_model_path = OUTPUT_DIR / "weights" / "best.pt"
    
    if not best_model_path.exists():
        print("❌ Обученная модель не найдена. Сначала запустите обучение.")
        return
    
    print(f"\n🧪 ТЕСТ НА ИЗОБРАЖЕНИИ: {image_path}")
    print("=" * 80)
    
    model = YOLO(str(best_model_path))
    results = model.predict(image_path, save=True, imgsz=IMG_SIZE, conf=0.25)
    
    print(f"✅ Результат сохранен в: runs/detect/predict")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            validate_model()
        elif command == "test" and len(sys.argv) > 2:
            test_model_on_image(sys.argv[2])
        else:
            print("Использование:")
            print("  python train_road_marking_20epochs.py          # Обучение")
            print("  python train_road_marking_20epochs.py validate # Валидация")
            print("  python train_road_marking_20epochs.py test <image_path>  # Тест")
    else:
        # Запуск обучения
        train_model()
