#!/usr/bin/env python3
"""
Полное обучение модели YOLO на RTSD датасете
"""
import os
import json
import shutil
from pathlib import Path
import yaml

def download_rtsd_dataset():
    """Загрузка RTSD датасета с Kaggle"""
    download_path = "rtsd_dataset"
    
    # Проверяем, есть ли уже датасет
    if os.path.exists(download_path) and any(Path(download_path).rglob('*.jpg')) or any(Path(download_path).rglob('*.png')):
        print(f"✓ Датасет уже существует: {download_path}")
        return download_path
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("=" * 70)
        print("ЗАГРУЗКА RTSD ДАТАСЕТА С KAGGLE")
        print("=" * 70)
        
        api = KaggleApi()
        try:
            api.authenticate()
        except:
            print("⚠ Kaggle API не настроен. Скачайте датасет вручную:")
            print("1. https://www.kaggle.com/datasets/watchman/rtsd-dataset")
            print(f"2. Распакуйте в директорию: {download_path}")
            return None
        
        dataset_name = "watchman/rtsd-dataset"
        Path(download_path).mkdir(exist_ok=True)
        
        print(f"Скачивание датасета {dataset_name}...")
        print("Это может занять некоторое время...")
        
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        print(f"✓ Датасет загружен: {download_path}")
        return download_path
        
    except ImportError:
        print("⚠ Kaggle API не установлен")
        print("Скачайте датасет вручную:")
        print("1. https://www.kaggle.com/datasets/watchman/rtsd-dataset")
        print(f"2. Распакуйте в директорию: {download_path}")
        return None
    except Exception as e:
        print(f"⚠ Ошибка загрузки через API: {e}")
        print("\nСкачайте датасет вручную:")
        print("1. https://www.kaggle.com/datasets/watchman/rtsd-dataset")
        print(f"2. Распакуйте в директорию: {download_path}")
        return None


def convert_rtsd_to_yolo(rtsd_path: str):
    """Конвертация RTSD датасета в формат YOLO"""
    print("\n" + "=" * 70)
    print("КОНВЕРТАЦИЯ RTSD В ФОРМАТ YOLO")
    print("=" * 70)
    
    yolo_path = "rtsd_yolo"
    Path(yolo_path).mkdir(exist_ok=True)
    
    # Структура YOLO датасета
    for split in ['train', 'val', 'test']:
        Path(f"{yolo_path}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{yolo_path}/{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Ищем аннотации и изображения в RTSD
    annotations_file = None
    images_dir = None
    
    # Ищем файлы аннотаций в корне датасета
    possible_annos = [
        os.path.join(rtsd_path, 'train_anno.json'),
        os.path.join(rtsd_path, 'val_anno.json'),
        os.path.join(rtsd_path, 'train_anno_reduced.json'),
    ]
    
    for ann_path in possible_annos:
        if os.path.exists(ann_path):
            annotations_file = ann_path
            print(f"✓ Найдены аннотации: {annotations_file}")
            break
    
    # Если не нашли, ищем в директориях
    if not annotations_file:
        for root, dirs, files in os.walk(rtsd_path):
            for file in files:
                if 'annotation' in file.lower() or 'anno' in file.lower():
                    if file.endswith('.json'):
                        annotations_file = os.path.join(root, file)
                        break
            if annotations_file:
                break
    
    # Ищем директорию с изображениями
    possible_img_dirs = [
        os.path.join(rtsd_path, 'rtsd-frames', 'rtsd-frames'),
        os.path.join(rtsd_path, 'images'),
        os.path.join(rtsd_path, 'train'),
    ]
    
    for img_dir in possible_img_dirs:
        if os.path.exists(img_dir) and any(Path(img_dir).glob('*.jpg')) or any(Path(img_dir).glob('*.png')):
            images_dir = img_dir
            print(f"✓ Найдены изображения: {images_dir}")
            break
    
    if not images_dir:
        # Пробуем найти любую директорию с изображениями
        for root, dirs, files in os.walk(rtsd_path):
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files[:5]):
                images_dir = root
                print(f"✓ Найдены изображения: {images_dir}")
                break
    
    print(f"Аннотации: {annotations_file if annotations_file else 'не найдены'}")
    print(f"Изображения: {images_dir if images_dir else 'не найдены'}")
    
    # Загружаем аннотации
    if annotations_file and os.path.exists(annotations_file):
        if annotations_file.endswith('.json'):
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        else:
            import pandas as pd
            annotations = pd.read_csv(annotations_file).to_dict('records')
        
        # Конвертируем аннотации в YOLO формат
        print("Конвертация аннотаций...")
        
        # Если структура COCO
        if isinstance(annotations, dict) and 'images' in annotations:
            # COCO формат
            images_data = {img['id']: img for img in annotations.get('images', [])}
            annotations_data = annotations.get('annotations', [])
            categories = {cat['id']: cat for cat in annotations.get('categories', [])}
            
            for ann in annotations_data:
                img_id = ann['image_id']
                if img_id not in images_data:
                    continue
                
                img_info = images_data[img_id]
                img_filename = img_info['file_name']
                img_width = img_info['width']
                img_height = img_info['height']
                
                # Bbox в формате COCO [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Конвертируем в YOLO формат [class, x_center, y_center, width, height] (нормализовано)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                category_id = ann['category_id']
                
                # Ищем и копируем изображение
                # Путь может быть относительным или абсолютным
                img_basename = os.path.basename(img_filename)
                
                # Пробуем разные варианты путей
                possible_paths = [
                    os.path.join(images_dir, img_basename),
                    os.path.join(rtsd_path, img_filename),
                    os.path.join(images_dir, img_filename),
                ]
                
                source_img = None
                for path in possible_paths:
                    if os.path.exists(path):
                        source_img = path
                        break
                
                if source_img and os.path.exists(source_img):
                    # Определяем split (80% train, 15% val, 5% test)
                    img_id_int = hash(img_filename) % 100
                    if img_id_int < 80:
                        split = 'train'
                    elif img_id_int < 95:
                        split = 'val'
                    else:
                        split = 'test'
                    
                    dest_img = f"{yolo_path}/{split}/images/{img_basename}"
                    
                    # Создаем директорию если нужно
                    os.makedirs(os.path.dirname(dest_img), exist_ok=True)
                    
                    # Копируем только если еще не скопировано
                    if not os.path.exists(dest_img):
                        shutil.copy2(source_img, dest_img)
                    
                    # Создаем файл аннотации YOLO
                    label_file = f"{yolo_path}/{split}/labels/{Path(img_basename).stem}.txt"
                    
                    # Открываем в режиме append и добавляем аннотацию
                    with open(label_file, 'a') as f:
                        f.write(f"{category_id} {x_center} {y_center} {w_norm} {h_norm}\n")
        
        # Создаем файл classes.txt
        classes_file = f"{yolo_path}/classes.txt"
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cat_id in sorted(categories.keys()):
                cat_name = categories[cat_id].get('name', f'class_{cat_id}')
                f.write(f"{cat_name}\n")
        
        print(f"✓ Конвертация завершена: {yolo_path}")
        return yolo_path
    
    else:
        print("⚠ Аннотации не найдены, используем структуру как есть")
        # Если датасет уже в формате YOLO
        if os.path.exists(os.path.join(rtsd_path, 'train')):
            return rtsd_path
        
        return None


def create_yolo_config(yolo_path: str, num_classes: int):
    """Создание конфигурационного файла для YOLO"""
    config_file = f"{yolo_path}/dataset.yaml"
    
    # Получаем полный путь
    abs_path = os.path.abspath(yolo_path)
    
    config = {
        'path': abs_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_classes,
        'names': {}
    }
    
    # Загружаем имена классов
    classes_file = f"{yolo_path}/classes.txt"
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines()]
            config['names'] = {i: name for i, name in enumerate(classes)}
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"✓ Конфигурация создана: {config_file}")
    return config_file


def train_yolo_model(yolo_config: str, epochs: int = 100):
    """Обучение модели YOLO на RTSD датасете"""
    try:
        from ultralytics import YOLO
        
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ МОДЕЛИ YOLO НА RTSD ДАТАСЕТЕ")
        print("=" * 70)
        
        # Загружаем предобученную модель YOLOv8
        print("Загрузка базовой модели YOLOv8...")
        model = YOLO('yolov8n.pt')
        
        print(f"\nНачало обучения...")
        print(f"Эпохи: {epochs}")
        print(f"Конфигурация: {yolo_config}")
        print("\nЭто займет много времени...")
        
        # Определяем устройство: CUDA > MPS (Apple Metal) > CPU
        device = 'cpu'
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"✓ Используется CUDA (NVIDIA GPU)")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'  # Apple Metal для Mac M1/M2/M3
                print(f"✓ Используется MPS (Apple Metal GPU)")
            else:
                print(f"⚠ Используется CPU (медленно, рекомендуется GPU)")
        except:
            print(f"⚠ Используется CPU (не удалось определить GPU)")
        
        # Обучаем модель
        results = model.train(
            data=yolo_config,
            epochs=epochs,
            imgsz=640,
            batch=16,
            patience=50,
            save=True,
            project='rtsd_training',
            name='rtsd_yolov8n',
            device=device
        )
        
        # Путь к лучшей модели
        best_model = "rtsd_training/rtsd_yolov8n/weights/best.pt"
        
        if os.path.exists(best_model):
            # Копируем в models
            Path("models").mkdir(exist_ok=True)
            final_model = "models/rtsd_yolov8n_best.pt"
            shutil.copy2(best_model, final_model)
            print(f"\n✓ Обучение завершено!")
            print(f"✓ Модель сохранена: {final_model}")
            return final_model
        else:
            print("❌ Модель не найдена после обучения")
            return None
            
    except ImportError:
        print("❌ Ultralytics не установлен")
        print("Установите: pip install ultralytics")
        return None
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Полный pipeline: загрузка -> конвертация -> обучение"""
    print("\n" + "=" * 70)
    print("ПОЛНОЕ ОБУЧЕНИЕ МОДЕЛИ НА RTSD ДАТАСЕТЕ")
    print("=" * 70)
    
    # Шаг 1: Загрузка датасета
    rtsd_path = download_rtsd_dataset()
    if not rtsd_path:
        # Пробуем найти датасет в стандартных местах
        possible_paths = ["rtsd_dataset", "rtsd-dataset", "../rtsd_dataset"]
        for path in possible_paths:
            if os.path.exists(path) and (any(Path(path).rglob('*.jpg')) or any(Path(path).rglob('*.png'))):
                rtsd_path = path
                print(f"✓ Найден датасет: {rtsd_path}")
                break
        
        if not rtsd_path:
            print("\n❌ Датасет не найден!")
            print("Скачайте датасет с Kaggle:")
            print("https://www.kaggle.com/datasets/watchman/rtsd-dataset")
            print("И распакуйте в директорию: rtsd_dataset")
            return
    
    # Шаг 2: Конвертация в YOLO формат
    yolo_path = convert_rtsd_to_yolo(rtsd_path)
    if not yolo_path:
        print("❌ Не удалось конвертировать датасет")
        return
    
    # Шаг 3: Создание конфигурации
    # Определяем количество классов
    classes_file = f"{yolo_path}/classes.txt"
    num_classes = 77  # По умолчанию для RTSD
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            num_classes = len([l for l in f.readlines() if l.strip()])
    
    config_file = create_yolo_config(yolo_path, num_classes)
    
    # Шаг 4: Обучение модели
    epochs = 100
    print(f"\nКоличество эпох обучения: {epochs}")
    print("Начало обучения...")
    
    model_path = train_yolo_model(config_file, epochs)
    
    if model_path:
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 70)
        print(f"\n✓ Обученная модель: {model_path}")
        print("\nИспользуйте в коде:")
        print(f'sign_detector = RoadSignDetector(model_path="{model_path}")')
        print("\nИли укажите путь в road_analysis.ipynb")
    else:
        print("\n❌ Обучение не завершено")


if __name__ == "__main__":
    main()

