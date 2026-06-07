import cv2, os
from ultralytics import YOLO

input_dir = 'data/results/videos/telegram_input'
videos = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
videos.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
video_path = os.path.join(input_dir, videos[0])
print(f'Video: {video_path}')

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Total frames: {total}')
cap.release()

new_models = [
    'russian_rtsd_yolov8.pt',
    'russian_signs_yolo12m.pt',
    'russian_traffic_signs.pt',
    'traffic_signs_yolov8.pt',
]

for model_name in new_models:
    model_path = f'models/{model_name}'
    if not os.path.exists(model_path):
        continue
    print(f'\n=== {model_name} ({os.path.getsize(model_path)/1024/1024:.0f}MB) ===')
    try:
        model = YOLO(model_path)
        print(f'  Classes ({len(model.names)}): {dict(list(model.names.items())[:10])}...')
    except Exception as e:
        print(f'  LOAD ERROR: {e}')
        continue

    cap = cv2.VideoCapture(video_path)
    det_count = 0
    for frame_idx in range(0, total, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=0.15, verbose=False)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls_id]
                    det_count += 1
                    if det_count <= 15:
                        print(f'  Fr{frame_idx}: {name} conf={conf:.3f}')
    cap.release()
    print(f'  >>> TOTAL: {det_count} detections')

print('\nDone')
