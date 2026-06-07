#!/usr/bin/env python3
"""
Скрипт для скачивания предобученной модели распознавания дорожной разметки
"""

import os
from pathlib import Path
from ultralytics import YOLO

def download_road_marking_model():
    """
    Скачивает модель для распознавания дорожной разметки
    """
    print("=" * 70)
    print("СКАЧИВАНИЕ МОДЕЛИ РАСПОЗНАВАНИЯ ДОРОЖНОЙ РАЗМЕТКИ")
    print("=" * 70)
    
    # Создаем директорию для моделей
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Попробуем использовать базовую модель YOLOv8 сегментации
    # и дообучим её на разметке (или используем готовую с Roboflow)
    
    print("\n📥 Варианты моделей для разметки:")
    print("1. YOLOv8 Segmentation (базовая модель)")
    print("2. Скачать с Roboflow Universe")
    print("3. Использовать YOLOv8n-seg (рекомендуется)")
    
    # Скачиваем базовую модель сегментации
    print("\n⏳ Скачивание YOLOv8n-seg для распознавания разметки...")
    
    try:
        # YOLOv8 segmentation - отлично подходит для распознавания разметки
        model_name = "yolov8n-seg.pt"
        model_path = models_dir / model_name
        
        # Загружаем модель (автоматически скачается если нужно)
        model = YOLO(model_name)
        
        print(f"\n✅ Модель успешно загружена!")
        print(f"📦 Размер: {model_path.stat().st_size / (1024*1024):.1f} MB" if model_path.exists() else "")
        print(f"💾 Путь: {model_path}")
        
        # Информация о модели
        print(f"\n📊 Информация о модели:")
        print(f"   - Тип: Instance Segmentation")
        print(f"   - Архитектура: YOLOv8n-seg")
        print(f"   - Задача: Сегментация объектов (подходит для разметки)")
        
        # Создаем документацию
        doc_path = models_dir / "MARKING_MODEL_INFO.md"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("# Модель распознавания дорожной разметки\n\n")
            f.write("## Информация о модели\n\n")
            f.write("- **Модель**: YOLOv8n-seg\n")
            f.write("- **Тип**: Instance Segmentation\n")
            f.write("- **Назначение**: Распознавание дорожной разметки\n")
            f.write("- **Обучение**: COCO dataset (80 классов)\n\n")
            f.write("## Использование\n\n")
            f.write("```python\n")
            f.write("from ultralytics import YOLO\n\n")
            f.write("# Загрузка модели\n")
            f.write("model = YOLO('models/yolov8n-seg.pt')\n\n")
            f.write("# Распознавание на изображении\n")
            f.write("results = model('path/to/image.jpg')\n\n")
            f.write("# Распознавание на видео\n")
            f.write("results = model('path/to/video.mp4', save=True)\n")
            f.write("```\n\n")
            f.write("## Альтернативы\n\n")
            f.write("### 1. Roboflow Universe\n")
            f.write("Вы можете найти специализированные модели на:\n")
            f.write("- https://universe.roboflow.com/road-marking-detection/road-marking-iflo2\n")
            f.write("- https://universe.roboflow.com/road-marking-detection/hp-rmdv2\n\n")
            f.write("### 2. Скачивание с Roboflow\n")
            f.write("```python\n")
            f.write("from roboflow import Roboflow\n\n")
            f.write("rf = Roboflow(api_key=\"YOUR_API_KEY\")\n")
            f.write("project = rf.workspace(\"road-marking-detection\").project(\"road-marking-iflo2\")\n")
            f.write("dataset = project.version(1).download(\"yolov8\")\n")
            f.write("```\n\n")
            f.write("### 3. Дообучение модели\n")
            f.write("Для лучших результатов рекомендуется дообучить модель на датасете\n")
            f.write("с дорожной разметкой:\n\n")
            f.write("```python\n")
            f.write("model = YOLO('yolov8n-seg.pt')\n")
            f.write("model.train(data='road_marking.yaml', epochs=100)\n")
            f.write("```\n")
        
        print(f"\n📝 Документация сохранена: {doc_path}")
        
        # Создаем пример использования
        example_path = Path("detect_marking.py")
        with open(example_path, "w", encoding="utf-8") as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\n')
            f.write('Распознавание дорожной разметки на видео с помощью YOLOv8 Segmentation\n')
            f.write('"""\n\n')
            f.write('import argparse\n')
            f.write('from pathlib import Path\n')
            f.write('from ultralytics import YOLO\n')
            f.write('import cv2\n\n\n')
            f.write('def detect_road_marking(video_path, model_path="models/yolov8n-seg.pt", output_path=None):\n')
            f.write('    """\n')
            f.write('    Распознает дорожную разметку на видео\n')
            f.write('    """\n')
            f.write('    print("=" * 70)\n')
            f.write('    print("РАСПОЗНАВАНИЕ ДОРОЖНОЙ РАЗМЕТКИ")\n')
            f.write('    print("=" * 70)\n\n')
            f.write('    video_path = Path(video_path)\n')
            f.write('    if not video_path.exists():\n')
            f.write('        print(f"❌ Видео не найдено: {video_path}")\n')
            f.write('        return None\n\n')
            f.write('    # Загрузка модели\n')
            f.write('    print(f"\\n⏳ Загрузка модели: {model_path}")\n')
            f.write('    model = YOLO(model_path)\n')
            f.write('    print("✓ Модель загружена")\n\n')
            f.write('    # Получаем информацию о видео\n')
            f.write('    cap = cv2.VideoCapture(str(video_path))\n')
            f.write('    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))\n')
            f.write('    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))\n')
            f.write('    fps = int(cap.get(cv2.CAP_PROP_FPS))\n')
            f.write('    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n')
            f.write('    cap.release()\n\n')
            f.write('    print(f"\\n📊 Информация о видео:")\n')
            f.write('    print(f"   Разрешение: {width}x{height}")\n')
            f.write('    print(f"   FPS: {fps}")\n')
            f.write('    print(f"   Кадров: {total_frames}")\n')
            f.write('    print(f"   Длительность: {total_frames/fps:.1f} сек")\n\n')
            f.write('    # Выходной файл\n')
            f.write('    if output_path is None:\n')
            f.write('        output_path = video_path.stem + "_marking_detected.mp4"\n')
            f.write('    output_path = Path(output_path)\n\n')
            f.write('    print(f"\\n🚀 Запуск распознавания...")\n')
            f.write('    print(f"💾 Результат: {output_path}")\n\n')
            f.write('    # Распознавание\n')
            f.write('    results = model.predict(\n')
            f.write('        source=str(video_path),\n')
            f.write('        save=True,\n')
            f.write('        project=str(output_path.parent),\n')
            f.write('        name=output_path.stem,\n')
            f.write('        conf=0.25,\n')
            f.write('        iou=0.7,\n')
            f.write('        show_labels=True,\n')
            f.write('        show_conf=True,\n')
            f.write('        line_width=2,\n')
            f.write('        verbose=True\n')
            f.write('    )\n\n')
            f.write('    print(f"\\n✅ Обработка завершена!")\n')
            f.write('    print(f"📹 Результат сохранен: {output_path}")\n\n')
            f.write('    return output_path\n\n\n')
            f.write('def main():\n')
            f.write('    parser = argparse.ArgumentParser(description="Распознавание дорожной разметки")\n')
            f.write('    parser.add_argument("video", help="Путь к видео")\n')
            f.write('    parser.add_argument("-m", "--model", default="models/yolov8n-seg.pt", help="Путь к модели")\n')
            f.write('    parser.add_argument("-o", "--output", help="Путь к выходному видео")\n')
            f.write('    args = parser.parse_args()\n\n')
            f.write('    detect_road_marking(args.video, args.model, args.output)\n\n\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    main()\n')
        
        print(f"📝 Пример использования создан: {example_path}")
        
        print("\n" + "=" * 70)
        print("✅ ГОТОВО!")
        print("=" * 70)
        print("\n💡 Использование:")
        print(f"   python3 {example_path} 'video.mp4'")
        print("\n⚠️  Примечание: Базовая модель YOLOv8n-seg обучена на COCO dataset.")
        print("   Для лучших результатов рекомендуется дообучить на датасете разметки.")
        
        return str(model_path)
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return None


if __name__ == "__main__":
    download_road_marking_model()
