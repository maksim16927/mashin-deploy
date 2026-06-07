#!/usr/bin/env python3
"""
Тестирование готовой обученной модели YOLOv8s (35 эпох, 155 классов RTSD)
"""
import os
from pathlib import Path

def test_model():
    """Тестирование модели на валидационном датасете"""
    try:
        from ultralytics import YOLO
        import torch
        
        print("=" * 70)
        print("ТЕСТИРОВАНИЕ ГОТОВОЙ МОДЕЛИ YOLOV8S (35 ЭПОХ, RTSD-155)")
        print("=" * 70)
        
        # Путь к модели
        model_path = "models/yolov8s_35epochs_rtsd155.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Модель не найдена: {model_path}")
            return None
        
        print(f"\n📦 Модель: {model_path}")
        print(f"📏 Размер: {os.path.getsize(model_path) / (1024**2):.1f} MB")
        
        # Загрузка модели
        print("\nЗагрузка модели...")
        model = YOLO(model_path)
        
        # Проверка устройства
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ Используется: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("✓ Используется: Apple Metal (MPS)")
        else:
            print("⚠ Используется: CPU")
        
        # Информация о модели
        print(f"\n📊 Информация о модели:")
        print(f"  Имена классов: {len(model.names)} классов")
        print(f"  Архитектура: YOLOv8s")
        print(f"  Обучена на: 35 эпох")
        print(f"  Датасет: RTSD (Russian Traffic Sign Dataset)")
        
        # Показываем первые 20 классов
        print(f"\n🏷️  Примеры классов (первые 20 из {len(model.names)}):")
        for i in range(min(20, len(model.names))):
            print(f"  {i}: {model.names[i]}")
        if len(model.names) > 20:
            print(f"  ... и ещё {len(model.names) - 20} классов")
        
        # Проверяем наличие датасета для валидации
        dataset_yaml = "rtsd_yolo/dataset.yaml"
        if os.path.exists(dataset_yaml):
            print(f"\n✓ Датасет найден: {dataset_yaml}")
            
            # Запускаем валидацию
            print("\n🧪 Запуск валидации на тестовой выборке...")
            print("Это может занять несколько минут...\n")
            
            results = model.val(
                data=dataset_yaml,
                split='test',  # Используем тестовую выборку
                device=device,
                batch=8,
                imgsz=640,
                verbose=True
            )
            
            print("\n" + "=" * 70)
            print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
            print("=" * 70)
            
            # Метрики
            if hasattr(results, 'box'):
                box_metrics = results.box
                print(f"\n📈 Метрики качества:")
                print(f"  mAP50: {box_metrics.map50:.4f}")
                print(f"  mAP50-95: {box_metrics.map:.4f}")
                print(f"  Precision: {box_metrics.mp:.4f}")
                print(f"  Recall: {box_metrics.mr:.4f}")
            
            print(f"\n✓ Валидация завершена!")
            print(f"  Протестировано на: test split датасета RTSD")
            
        else:
            print(f"\n⚠ Датасет не найден: {dataset_yaml}")
            print("  Валидация пропущена")
            print("\nДля валидации убедитесь что:")
            print("  1. Датасет конвертирован в формат YOLO")
            print("  2. Файл dataset.yaml существует")
        
        # Пример использования
        print("\n" + "=" * 70)
        print("ПРИМЕР ИСПОЛЬЗОВАНИЯ")
        print("=" * 70)
        
        print(f"\n1. В Python коде:")
        print(f"```python")
        print(f"from ultralytics import YOLO")
        print(f"model = YOLO('{model_path}')")
        print(f"results = model.predict('image.jpg')")
        print(f"```")
        
        print(f"\n2. Через командную строку:")
        print(f"```bash")
        print(f"yolo predict model={model_path} source=image.jpg")
        print(f"```")
        
        print(f"\n3. В road_detector.py:")
        print(f"```python")
        print(f"sign_detector = RoadSignDetector(model_path='{model_path}')")
        print(f"```")
        
        return model
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("\nУстановите зависимости:")
        print("pip install ultralytics torch")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_on_sample_images():
    """Тест на примерах изображений"""
    try:
        from ultralytics import YOLO
        import cv2
        import torch
        
        model_path = "models/yolov8s_35epochs_rtsd155.pt"
        
        if not os.path.exists(model_path):
            print(f"❌ Модель не найдена: {model_path}")
            return
        
        print("\n" + "=" * 70)
        print("ТЕСТ НА ПРИМЕРАХ ИЗОБРАЖЕНИЙ")
        print("=" * 70)
        
        # Загружаем модель
        model = YOLO(model_path)
        
        # Ищем тестовые изображения
        test_images_dir = "rtsd_yolo/test/images"
        if os.path.exists(test_images_dir):
            test_images = list(Path(test_images_dir).glob("*.jpg"))[:5]  # Берём 5 изображений
            
            if test_images:
                print(f"\n✓ Найдено {len(test_images)} тестовых изображений")
                print("Запуск детекции...\n")
                
                # Определяем устройство
                device = 'cpu'
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                
                for i, img_path in enumerate(test_images, 1):
                    print(f"{i}. Обработка: {img_path.name}")
                    
                    # Детекция
                    results = model.predict(
                        source=str(img_path),
                        device=device,
                        conf=0.25,  # Порог уверенности
                        save=True,  # Сохраняем результат
                        project='test_results',
                        name='pretrained_model',
                        exist_ok=True
                    )
                    
                    # Выводим результаты
                    if results and len(results) > 0:
                        result = results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            print(f"   ✓ Найдено объектов: {len(result.boxes)}")
                            
                            # Показываем топ-3 обнаружения
                            for j, box in enumerate(result.boxes[:3]):
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = model.names[cls]
                                print(f"      • {class_name} ({conf:.2%})")
                        else:
                            print(f"   - Объектов не найдено")
                    print()
                
                print("=" * 70)
                print("✓ ТЕСТ ЗАВЕРШЁН")
                print("=" * 70)
                print(f"\n📁 Результаты сохранены в: test_results/pretrained_model/")
                print("\nОткройте папку чтобы посмотреть визуализацию детекций")
                
            else:
                print("⚠ Тестовые изображения не найдены")
        else:
            print(f"⚠ Директория не найдена: {test_images_dir}")
            print("\nДля тестирования на изображениях:")
            print("  1. Подготовьте датасет RTSD в формате YOLO")
            print("  2. Или укажите свои изображения")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование готовой модели YOLOv8s')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Запустить валидацию на датасете')
    parser.add_argument('--test-images', '-t', action='store_true',
                       help='Тест на примерах изображений')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Запустить все тесты')
    
    args = parser.parse_args()
    
    if args.all or (not args.validate and not args.test_images):
        # Запускаем оба теста по умолчанию
        model = test_model()
        if model:
            test_on_sample_images()
    else:
        if args.validate:
            test_model()
        if args.test_images:
            test_on_sample_images()
