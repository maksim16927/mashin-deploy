#!/usr/bin/env python3
"""
Простая детекция - всё работает
"""
import cv2
from ultralytics import YOLO
from pathlib import Path

# Модели
sign_model = YOLO("models/yolov8s_35epochs_rtsd155.pt")
infra_model = YOLO("models/yolov8n.pt")

# Видео
video = "videos/original/2026-01-15 10.00.58.mp4"
output = "videos/detected/result.mp4"

Path("videos/detected").mkdir(exist_ok=True)

# Открываем
cap = cv2.VideoCapture(video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Выход
out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print(f"📹 {Path(video).name}: {w}x{h}, {fps}fps, {total} frames")
print(f"💾 {output}\n")

n = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    n += 1
    
    # Знаки (зеленые)
    for r in sign_model(frame, conf=0.35, verbose=False):
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    
    # Транспорт (синие)
    for r in infra_model(frame, classes=[2,3,5,6,7], conf=0.35, verbose=False):
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    
    out.write(frame)
    
    if n % 50 == 0:
        print(f"{n}/{total} ({n/total*100:.0f}%)")

cap.release()
out.release()

print(f"\n✅ Готово: {output}")
print(f"Открыть: open '{output}'")
