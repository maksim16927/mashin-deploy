"""
Скрипт для обучения модели распознавания дорожной инфраструктуры
Поддерживает:
- Автобусные остановки
- П-образные опоры (gantries)
- Электрические столбы
- Повреждения инфраструктуры (ямы, знаки и т.д.)
"""

from ultralytics import YOLO
import torch
import os
from datetime import datetime

def train_infrastructure_model(
    dataset_path=None,
    base_model='yolov8n.pt',
    epochs=50,
    img_size=640,
    batch_size=16,
    device=None
):
    """
    Обучение модели для распознавания дорожной инфраструктуры
    
    Args:
        dataset_path: путь к data.yaml файлу датасета
        base_model: базовая модель YOLOv8 (n, s, m, l, x)
        epochs: количество эпох
        img_size: размер изображения
        batch_size: размер батча
        device: устройство (mps/cpu/cuda)
    """
    
    print("=" * 70)
    print("ОБУЧЕНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ ДОРОЖНОЙ ИНФРАСТРУКТУРЫ")
    print("=" * 70)
    
    # Автоопределение устройства
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print(f"\n📱 Устройство: {device}")
    print(f"🧠 Базовая модель: {base_model}")
    print(f"📊 Эпох: {epochs}")
    print(f"🖼️  Размер изображения: {img_size}")
    print(f"📦 Размер батча: {batch_size}")
    
    # Если датасет не указан, используем загруженный
    if dataset_path is None:
        # Проверяем возможные местоположения датасетов
        possible_paths = [
            "datasets/infrastructure_dataset.yaml",
            "datasets/Bus-Stop-Detection-1/data.yaml",
            "datasets/road-infrastructure-predict-1/data.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"\n✅ Найден датасет: {dataset_path}")
                break
        
        if dataset_path is None:
            print("\n❌ Датасет не найден!")
            print("Сначала запустите: python3 scripts/download_infrastructure_dataset.py")
            return None
    
    print(f"📂 Датасет: {dataset_path}")
    
    # Загружаем базовую модель
    print(f"\n⏳ Загрузка базовой модели {base_model}...")
    model = YOLO(base_model)
    
    # Создаем директорию для сохранения результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"infrastructure_training_{timestamp}"
    
    print(f"\n🚀 Начинаем обучение...")
    print(f"📁 Результаты будут сохранены в: runs/detect/{project_name}")
    print("=" * 70)
    
    try:
        # Обучаем модель
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project="runs/detect",
            name=project_name,
            patience=10,  # Early stopping
            save=True,
            save_period=10,  # Сохранение каждые 10 эпох
            plots=True,  # Графики обучения
            verbose=True,
            amp=False if device == 'mps' else True,  # AMP для CUDA
            lr0=0.01,  # Начальная скорость обучения
            lrf=0.01,  # Конечная скорость обучения
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,  # Вес loss для bbox
            cls=0.5,  # Вес loss для классификации
            dfl=1.5,  # Вес loss для DFL
            pose=12.0,  # Вес loss для keypoints (если используется)
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.015,  # Аугментация HSV hue
            hsv_s=0.7,  # Аугментация HSV saturation
            hsv_v=0.4,  # Аугментация HSV value
            degrees=0.0,  # Поворот изображений
            translate=0.1,  # Смещение
            scale=0.5,  # Масштабирование
            shear=0.0,  # Сдвиг
            perspective=0.0,  # Перспектива
            flipud=0.0,  # Вертикальное отражение
            fliplr=0.5,  # Горизонтальное отражение
            mosaic=1.0,  # Мозаичная аугментация
            mixup=0.0,  # Mixup аугментация
            copy_paste=0.0  # Copy-paste аугментация
        )
        
        print("\n" + "=" * 70)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 70)
        
        # Путь к обученной модели
        best_model_path = os.path.join("runs/detect", project_name, "weights/best.pt")
        last_model_path = os.path.join("runs/detect", project_name, "weights/last.pt")
        
        print(f"\n📦 Лучшая модель: {best_model_path}")
        print(f"📦 Последняя модель: {last_model_path}")
        
        # Копируем лучшую модель в models/
        os.makedirs("models", exist_ok=True)
        final_model_path = f"models/infrastructure_model_{timestamp}.pt"
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, final_model_path)
            print(f"\n✅ Модель скопирована в: {final_model_path}")
        
        # Валидация модели
        print("\n" + "=" * 70)
        print("🔍 ВАЛИДАЦИЯ МОДЕЛИ")
        print("=" * 70)
        
        best_model = YOLO(best_model_path)
        metrics = best_model.val(data=dataset_path, device=device)
        
        print(f"\n📊 Метрики валидации:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        print("\n" + "=" * 70)
        print("🎉 ГОТОВО! Модель готова к использованию")
        print("=" * 70)
        
        print(f"\nДля тестирования на видео используйте:")
        print(f"python3 scripts/detect_infrastructure.py 'video.mp4' --model {final_model_path}")
        
        return final_model_path
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_custom_dataset_yaml(classes_dict, dataset_name="custom_infrastructure"):
    """
    Создание кастомного data.yaml для датасета
    
    Args:
        classes_dict: словарь {id: class_name}
        dataset_name: имя датасета
    """
    yaml_content = f"""# Датасет дорожной инфраструктуры
path: ./datasets/{dataset_name}
train: train/images
val: val/images
test: test/images

# Классы
names:
"""
    
    for class_id, class_name in classes_dict.items():
        yaml_content += f"  {class_id}: {class_name}\n"
    
    yaml_path = f"datasets/{dataset_name}/data.yaml"
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Создан файл конфигурации: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Обучение модели распознавания дорожной инфраструктуры"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Путь к data.yaml файлу датасета"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Базовая модель YOLOv8"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Количество эпох обучения"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Размер входного изображения"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Размер батча"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["mps", "cpu", "cuda", "0", "1"],
        help="Устройство для обучения"
    )
    
    args = parser.parse_args()
    
    # Запускаем обучение
    model_path = train_infrastructure_model(
        dataset_path=args.dataset,
        base_model=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        device=args.device
    )
    
    if model_path:
        print(f"\n✅ Обучение завершено успешно!")
        print(f"📦 Модель сохранена: {model_path}")
    else:
        print("\n❌ Обучение не удалось")
