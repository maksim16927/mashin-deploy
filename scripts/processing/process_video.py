#!/usr/bin/env python3
"""
Простой скрипт для обработки видео с распознаванием дорожных знаков и разметки
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

def process_video(video_path):
    """
    Обрабатывает видео с помощью моделей детекции
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        print(f"❌ Видео не найдено: {video_path}")
        return
    
    print("=" * 70)
    print("ОБРАБОТКА ВИДЕО С ДЕТЕКЦИЕЙ ДОРОЖНОЙ ИНФРАСТРУКТУРЫ")
    print("=" * 70)
    print(f"\n📹 Входное видео: {video_path}")
    print(f"📦 Размер: {video_path.stat().st_size / (1024*1024):.1f} MB\n")
    
    # Загрузка моделей
    print("⏳ Загрузка моделей...")
    
    try:
        # Модель для дорожных знаков (выбираем первую доступную)
        signs_candidates = [
            "models/yolov8s_35epochs_rtsd155.pt",
            "models/russian_signs_yolo12m.pt",
            "models/russian_rtsd_yolov8.pt",
        ]
        signs_path = next((p for p in signs_candidates if Path(p).exists()), None)
        if signs_path is None:
            print("❌ Не найдена ни одна модель дорожных знаков в models/")
            return
        signs_model = YOLO(signs_path)
        print(f"✅ Модель дорожных знаков загружена: {signs_path} ({len(signs_model.names)} классов)")
        
        # Модель для разметки
        marking_model = YOLO("models/yolov8n_road_marking.pt")
        print("✅ Модель дорожной разметки загружена (6 классов)\n")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке моделей: {e}")
        return
    
    # Создание выходного файла
    output_dir = Path("videos/detected")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / f"{video_path.stem}_processed.mp4"
    
    print(f"🎬 Начало обработки...")
    print(f"💾 Результат будет сохранен в: {output_path}\n")
    
    # Обработка с помощью модели знаков (основная)
    try:
        results = signs_model.predict(
            source=str(video_path),
            save=True,
            project=str(output_dir.parent),
            name="detected",
            conf=0.25,  # Порог уверенности
            save_txt=False,
            save_conf=True,
            line_width=2,
            show_labels=True,
            show_conf=True,
        )
        
        print("\n✅ Обработка завершена!")
        print(f"📊 Обработано кадров: {len(results) if isinstance(results, list) else 'N/A'}")
        print(f"💾 Результат: {output_path}")
        
        # Информация о детекциях
        if isinstance(results, list) and results:
            total_detections = sum(len(r.boxes) if hasattr(r, 'boxes') else 0 for r in results)
            print(f"🎯 Всего детекций: {total_detections}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при обработке: {e}")
        return
    
    print("\n" + "=" * 70)
    print("✅ ГОТОВО!")
    print("=" * 70)
    print(f"\n💡 Откройте видео: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Использование: python process_video.py <путь_к_видео>")
        print("\nПримеры:")
        print("  python process_video.py videos/original/test.mp4")
        print("  python process_video.py videos/original/2026-01-15\\ 10.00.52.mp4")
        
        # Показываем доступные видео
        original_dir = Path("videos/original")
        if original_dir.exists():
            videos = list(original_dir.glob("*.mp4")) + list(original_dir.glob("*.MOV"))
            if videos:
                print("\n📹 Доступные видео:")
                for i, video in enumerate(videos, 1):
                    size_mb = video.stat().st_size / (1024*1024)
                    print(f"  {i}. {video.name} ({size_mb:.1f} MB)")
        return
    
    video_path = sys.argv[1]
    process_video(video_path)

if __name__ == "__main__":
    main()
