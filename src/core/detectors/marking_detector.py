#!/usr/bin/env python3
"""
Распознавание дорожной разметки на видео с помощью YOLOv8 Segmentation
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def detect_road_marking(video_path, model_path="models/yolov8n-seg.pt", output_path=None):
    """
    Распознает дорожную разметку на видео
    """
    print("=" * 70)
    print("РАСПОЗНАВАНИЕ ДОРОЖНОЙ РАЗМЕТКИ")
    print("=" * 70)

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Видео не найдено: {video_path}")
        return None

    # Загрузка модели
    print(f"\n⏳ Загрузка модели: {model_path}")
    model = YOLO(model_path)
    print("✓ Модель загружена")

    # Получаем информацию о видео
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\n📊 Информация о видео:")
    print(f"   Разрешение: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Кадров: {total_frames}")
    print(f"   Длительность: {total_frames/fps:.1f} сек")

    # Выходной файл
    if output_path is None:
        output_path = video_path.stem + "_marking_detected.mp4"
    output_path = Path(output_path)

    print(f"\n🚀 Запуск распознавания...")
    print(f"💾 Результат: {output_path}")

    # Распознавание
    results = model.predict(
        source=str(video_path),
        save=True,
        project=str(output_path.parent),
        name=output_path.stem,
        conf=0.25,
        iou=0.7,
        show_labels=True,
        show_conf=True,
        line_width=2,
        verbose=True
    )

    print(f"\n✅ Обработка завершена!")
    print(f"📹 Результат сохранен: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Распознавание дорожной разметки")
    parser.add_argument("video", help="Путь к видео")
    parser.add_argument("-m", "--model", default="models/yolov8n-seg.pt", help="Путь к модели")
    parser.add_argument("-o", "--output", help="Путь к выходному видео")
    args = parser.parse_args()

    detect_road_marking(args.video, args.model, args.output)


if __name__ == "__main__":
    main()
