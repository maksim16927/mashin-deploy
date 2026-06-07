"""
Полная комплексная система детекции
Все модели работают вместе:
- YOLOv8s RTSD-155 (дорожные знаки)
- YOLOv8n COCO (инфраструктура: автобусы, машины, люди)
- OpenCV (дорожная разметка)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

# Цвета для разных категорий
COLORS = {
    'sign': (0, 255, 0),          # Зеленый - дорожные знаки
    'bus': (0, 255, 255),         # Желтый - автобусы
    'car': (255, 0, 0),           # Синий - машины
    'truck': (255, 165, 0),       # Оранжевый - грузовики
    'person': (128, 128, 128),    # Серый - люди
    'traffic_light': (0, 128, 255), # Голубой - светофоры
    'stop_sign': (0, 0, 255),     # Красный - знаки Stop
    'bicycle': (255, 0, 255),     # Фиолетовый - велосипеды
    'motorcycle': (255, 255, 0),  # Голубой - мотоциклы
    'marking': (255, 0, 0),       # Синий - разметка
    'noise_strip': (0, 215, 255), # Золотистый - шумовые полосы
    'barrier': (0, 165, 255),     # Оранжевый - барьеры/ограждения
    'gantry': (255, 0, 255),      # Фиолетовый - П-образные опоры
    'exit': (255, 255, 0),        # Желтый - съезды
    'roi': (0, 0, 255),           # Красный - область интереса (ROI)
    'zone': (0, 255, 255),        # Желтый - зоны анализа
    'contour': (255, 0, 0),       # Синий - контуры разметки
    'default': (147, 20, 255)     # Розовый - остальное
}

# Классы COCO инфраструктуры (расширенный список)
INFRA_CLASSES = {
    # Транспорт
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    # Инфраструктура
    9: 'traffic_light', 10: 'fire_hydrant', 11: 'stop_sign', 
    12: 'parking_meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    # Дорожная инфраструктура
    56: 'chair', 57: 'couch', 59: 'potted_plant', 
    60: 'bed', 61: 'dining_table', 62: 'toilet',
    # Дополнительные объекты
    64: 'tv', 67: 'mouse', 68: 'remote', 69: 'keyboard',
    70: 'cell_phone', 72: 'microwave', 73: 'oven', 74: 'toaster',
    75: 'sink', 76: 'refrigerator', 77: 'book', 78: 'clock',
    79: 'vase', 80: 'scissors', 81: 'teddy_bear', 82: 'hair_drier',
    83: 'toothbrush'
}

# Специальные классы для дорожной инфраструктуры
ROAD_INFRA_CLASSES = {
    'bus': 5,  # Автобусы (могут указывать на остановки)
    'car': 2,  # Машины
    'truck': 7,  # Грузовики
    'traffic_light': 9,  # Светофоры
    'stop_sign': 11,  # Знаки остановки
    'person': 0,  # Люди (могут быть на остановках)
    'motorcycle': 3,  # Мотоциклы
    'bicycle': 1  # Велосипеды
}

def detect_lane_marking(frame, roi_ratio=0.75):
    """Улучшенная детекция дорожной разметки для соседних полос"""
    height, width = frame.shape[:2]
    roi_y = int(height * (1 - roi_ratio))
    roi = frame[roi_y:height, 0:width]
    
    # Преобразуем в серый и улучшаем контраст
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Выделяем белые линии разметки
    _, white_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Размытие и детекция краев
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges1 = cv2.Canny(blur, 30, 100)
    edges2 = cv2.Canny(white_thresh, 50, 150)
    
    # Объединяем результаты
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Морфологические операции для соединения разорванных линий
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Детекция линий с более чувствительными параметрами
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                           threshold=25,      # Снижаем порог
                           minLineLength=20,  # Уменьшаем минимальную длину
                           maxLineGap=10)     # Уменьшаем разрыв
    
    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Вычисляем угол линии
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Фильтруем по углу (разметка вертикальная/диагональная)
            # Исключаем только горизонтальные линии
            if (abs(angle) > 20 and abs(angle) < 160) and length > 15:
                # Проверяем яркость области вокруг линии
                try:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 6)
                    mean_brightness = cv2.mean(gray, mask=mask)[0]
                    
                    # Разметка должна быть достаточно яркой (белая)
                    if mean_brightness > 100:
                        lane_lines.append((x1, y1 + roi_y, x2, y2 + roi_y))
                except:
                    lane_lines.append((x1, y1 + roi_y, x2, y2 + roi_y))
    
    return lane_lines

def detect_gantries_and_structures(frame):
    """
    Детекция П-образных опор и больших структур над дорогой
    Использует простую эвристику для поиска горизонтальных структур в верхней части кадра
    """
    height, width = frame.shape[:2]
    
    # Область интереса - верхняя треть кадра (где обычно находятся опоры)
    roi_top = int(height * 0.1)
    roi_bottom = int(height * 0.5)
    roi = frame[roi_top:roi_bottom, :]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Ищем горизонтальные линии (характерны для П-образных опор)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                           threshold=80, minLineLength=width*0.3, maxLineGap=50)
    
    gantries = []
    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Проверяем, что линия почти горизонтальная
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170:  # Горизонтальная линия
                y1 += roi_top
                y2 += roi_top
                horizontal_lines.append((x1, y1, x2, y2))
        
        # Если найдено несколько горизонтальных линий в верхней части - возможно это опора
        if len(horizontal_lines) >= 2:
            # Группируем близкие линии
            for i, (x1, y1, x2, y2) in enumerate(horizontal_lines):
                if i == 0 or abs(y1 - horizontal_lines[i-1][1]) > 20:  # Новая группа
                    gantries.append({
                        'bbox': (min(x1, x2), y1, max(x1, x2), y1 + 10),
                        'type': 'gantry',
                        'confidence': 0.6
                    })
    
    return gantries

def full_detection(video_path, output_path=None,
                   sign_model_path="models/yolov8s_35epochs_rtsd155.pt",
                   infra_model_path="models/yolov8n.pt",
                   conf_threshold=0.25):
    """
    Полная комплексная детекция всех объектов
    """
    
    print("\n" + "=" * 80)
    print("🚀 ПОЛНАЯ КОМПЛЕКСНАЯ СИСТЕМА ДЕТЕКЦИИ".center(80))
    print("=" * 80)
    
    # Загружаем модели
    print("\n⏳ Загрузка моделей...")
    
    sign_model = YOLO(sign_model_path)
    infra_model = YOLO(infra_model_path)
    
    print(f"✅ Модель знаков: {Path(sign_model_path).name} (155 классов RTSD)")
    print(f"✅ Модель инфраструктуры: {Path(infra_model_path).name} (COCO - автобусы, машины, светофоры)")
    print(f"✅ Детекция П-образных опор: Эвристический метод (горизонтальные структуры)")
    print(f"✅ Детекция разметки: OpenCV (Canny + HoughLines)")
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {video_path}")
        return
    
    # Параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Видео: {Path(video_path).name}")
    print(f"   Разрешение: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Кадров: {total_frames}")
    
    # Выходное видео
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"videos/detected/{video_name}_FULL_DETECTION.mp4"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"💾 Сохранение в: {output_path}")
    print("\n" + "=" * 80)
    print("🚀 НАЧИНАЕМ ОБРАБОТКУ...")
    print("=" * 80 + "\n")
    
    # Статистика
    stats = {
        'signs': {},
        'infrastructure': {},
        'marking_frames': 0,
        'total_signs': 0,
        'total_infra': 0,
        'frames_with_signs': 0,
        'frames_with_infra': 0
    }
    
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        frame_has_signs = False
        frame_has_infra = False
        frame_has_marking = False
        
        # 1. ДЕТЕКЦИЯ ДОРОЖНЫХ ЗНАКОВ (RTSD-155)
        sign_results = sign_model(frame, conf=conf_threshold, verbose=False)
        
        for result in sign_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = sign_model.names[cls_id]
                
                # Зеленый для знаков
                color = COLORS['sign']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Метка
                label = f"SIGN: {class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                stats['signs'][class_name] = stats['signs'].get(class_name, 0) + 1
                stats['total_signs'] += 1
                frame_has_signs = True
        
        # 2. ДЕТЕКЦИЯ ИНФРАСТРУКТУРЫ (COCO) - все релевантные классы
        # Используем классы дорожной инфраструктуры для лучшей детекции
        infra_classes_to_detect = list(ROAD_INFRA_CLASSES.values())
        infra_results = infra_model(frame, classes=infra_classes_to_detect, 
                                    conf=conf_threshold, verbose=False)
        
        for result in infra_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = infra_model.names[cls_id]
                
                # Цвет по типу объекта
                class_key = class_name.replace(' ', '_').lower()
                color = COLORS.get(class_key, COLORS.get(class_name.lower(), COLORS['default']))
                
                # Специальная обработка для автобусов (могут указывать на остановки)
                if class_name.lower() == 'bus':
                    # Желтый для автобусов и остановок
                    color = COLORS['bus']
                    label_text = f"BUS/BUS_STOP: {class_name} {conf:.2f}"
                # Специальная обработка для светофоров
                elif class_name.lower() in ['traffic_light', 'traffic light']:
                    color = COLORS['traffic_light']
                    label_text = f"TRAFFIC_LIGHT: {class_name} {conf:.2f}"
                # Специальная обработка для знаков остановки
                elif class_name.lower() in ['stop_sign', 'stop sign']:
                    color = COLORS['stop_sign']
                    label_text = f"STOP_SIGN: {class_name} {conf:.2f}"
                else:
                    label_text = f"INFRA: {class_name} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Метка
                label = label_text
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                stats['infrastructure'][class_name] = stats['infrastructure'].get(class_name, 0) + 1
                stats['total_infra'] += 1
                frame_has_infra = True
        
        # 3. ДЕТЕКЦИЯ П-ОБРАЗНЫХ ОПОР И СТРУКТУР
        gantries = detect_gantries_and_structures(frame)
        frame_has_gantries = False
        if gantries:
            for gantry in gantries:
                x1, y1, x2, y2 = gantry['bbox']
                # Фиолетовый для П-образных опор
                color = (255, 0, 255)  # Фиолетовый
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                label = f"GANTRY/P-SHAPED_SUPPORT: {gantry['confidence']:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                stats['infrastructure']['gantry'] = stats['infrastructure'].get('gantry', 0) + 1
                stats['total_infra'] += 1
                frame_has_gantries = True
        
        # 4. ДЕТЕКЦИЯ РАЗМЕТКИ (OpenCV)
        lane_lines = detect_lane_marking(frame)
        if lane_lines:
            for x1, y1, x2, y2 in lane_lines:
                cv2.line(frame, (x1, y1), (x2, y2), COLORS['marking'], 2)
            frame_has_marking = True
            stats['marking_frames'] += 1
        
        # Обновляем счетчики
        if frame_has_signs:
            stats['frames_with_signs'] += 1
        if frame_has_infra:
            stats['frames_with_infra'] += 1
        
        # Информационная панель
        info_y = 30
        
        # Знаки
        cv2.putText(frame, f"SIGNS: {stats['total_signs']}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['sign'], 2)
        info_y += 30
        
        # Инфраструктура
        cv2.putText(frame, f"INFRA: {stats['total_infra']}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['bus'], 2)
        info_y += 30
        
        # Разметка
        marking_status = "YES" if frame_has_marking else "NO"
        cv2.putText(frame, f"MARKING: {marking_status}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['marking'], 2)
        info_y += 30
        
        # Прогресс
        cv2.putText(frame, f"FRAME: {frame_num}/{total_frames}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        # Прогресс в консоли
        if frame_num % 50 == 0 or frame_num == total_frames:
            progress = (frame_num / total_frames) * 100
            print(f"   📊 {progress:.1f}% | Кадр {frame_num}/{total_frames} | "
                  f"Знаки: {stats['total_signs']} | Инфра: {stats['total_infra']}")
    
    cap.release()
    out.release()
    
    # ИТОГИ
    print("\n" + "=" * 80)
    print("✅ ОБРАБОТКА ЗАВЕРШЕНА!".center(80))
    print("=" * 80)
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"   Всего кадров: {frame_num}")
    print(f"\n🚦 ДОРОЖНЫЕ ЗНАКИ (RTSD-155):")
    print(f"   Всего детекций: {stats['total_signs']}")
    print(f"   Кадров со знаками: {stats['frames_with_signs']} ({stats['frames_with_signs']/frame_num*100:.1f}%)")
    if stats['signs']:
        print(f"   Топ-5 знаков:")
        for sign, count in sorted(stats['signs'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      • {sign}: {count}")
    
    print(f"\n🏗️  ИНФРАСТРУКТУРА (COCO):")
    print(f"   Всего детекций: {stats['total_infra']}")
    print(f"   Кадров с объектами: {stats['frames_with_infra']} ({stats['frames_with_infra']/frame_num*100:.1f}%)")
    if stats['infrastructure']:
        print(f"   Топ-5 объектов:")
        for obj, count in sorted(stats['infrastructure'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      • {obj}: {count}")
    
    print(f"\n🛣️  ДОРОЖНАЯ РАЗМЕТКА (OpenCV):")
    print(f"   Кадров с разметкой: {stats['marking_frames']} ({stats['marking_frames']/frame_num*100:.1f}%)")
    
    print(f"\n💾 Результат: {output_path}")
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"📦 Размер файла: {file_size:.1f} MB")
    
    print("\n" + "=" * 80)
    print(f"🎉 ГОТОВО! Все 3 модели работали вместе!".center(80))
    print("=" * 80 + "\n")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Полная комплексная детекция всех объектов")
    parser.add_argument("video", type=str, help="Путь к видео")
    parser.add_argument("-o", "--output", type=str, default=None, help="Путь для сохранения")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности")
    
    args = parser.parse_args()
    
    output = full_detection(args.video, args.output, conf_threshold=args.conf)
    
    if output:
        print(f"✅ Откройте: {output}")
