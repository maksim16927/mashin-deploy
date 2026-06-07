#!/usr/bin/env python3
"""
Бот для мессенджера Max (max.ru) — обработка видео с детекцией дорожной инфраструктуры.
Полный перенос функционала Telegram-бота.
"""

import os
import sys
import asyncio
import logging
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ── maxapi ────────────────────────────────────────────────────────────────────
from maxapi import Bot, Dispatcher
from maxapi.types import (
    MessageCreated, MessageCallback, BotStarted,
    Command, CommandStart,
    CallbackButton, ButtonsPayload,
    InputMedia,
)
from maxapi.types.input_media import UploadType
from maxapi.types.attachments.attachment import AttachmentType
from maxapi.context import MemoryContext

# ── CV / ML ───────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── пути проекта ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── опциональные модули ───────────────────────────────────────────────────────
try:
    from core.detectors.barrier_detector import detect_barriers_combined
    BARRIER_AVAILABLE = True
except ImportError:
    BARRIER_AVAILABLE = False
    def detect_barriers_combined(frame, coco_model=None, conf_threshold=0.3):
        return []

try:
    from core.processors.excel_exporter import create_excel_report, format_detections_for_excel
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    from core.trackers.gps_tracker import GPSTracker
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

try:
    from core.trackers.picket_tracker import PicketTracker
    PICKET_AVAILABLE = True
except ImportError:
    PICKET_AVAILABLE = False

try:
    from core.detectors.exit_detector import detect_road_exits
    EXIT_DETECTOR_AVAILABLE = True
except ImportError:
    EXIT_DETECTOR_AVAILABLE = False
    def detect_road_exits(frame):
        return []

try:
    from core.detectors.noise_strips_detector import detect_noise_strips
    NOISE_DETECTOR_AVAILABLE = True
except ImportError:
    NOISE_DETECTOR_AVAILABLE = False
    def detect_noise_strips(frame):
        return []

try:
    from core.processors.situational_plan import create_situational_plan
    SITUATIONAL_PLAN_AVAILABLE = True
except ImportError:
    SITUATIONAL_PLAN_AVAILABLE = False
    def create_situational_plan(detections, video_info, output_path=None):
        return None

# ── логгер ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── конфиг ────────────────────────────────────────────────────────────────────
BOT_TOKEN    = os.getenv("MAX_BOT_TOKEN", "")           # токен из dev.max.ru
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")

MODEL_DIR    = PROJECT_ROOT / "models"
if not MODEL_DIR.exists():
    MODEL_DIR = PROJECT_ROOT / "data" / "models"

SIGN_MODEL_PATH    = str(MODEL_DIR / "russian_signs_yolo12m.pt")
INFRA_MODEL_PATH   = str(MODEL_DIR / "yolov8n.pt")
MARKING_MODEL_PATH = None
for _m in ["yolov8_road_marking_20epochs.pt", "yolov8n_road_marking.pt"]:
    if (MODEL_DIR / _m).exists():
        MARKING_MODEL_PATH = str(MODEL_DIR / _m)
        break

DOWNLOAD_DIR = PROJECT_ROOT / "data" / "results" / "videos" / "max_input"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "results" / "videos" / "max_output"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── цвета детекций (BGR) ──────────────────────────────────────────────────────
COLORS = {
    "sign":        (0,   0,   255),
    "infra":       (255, 255,   0),
    "barrier":     (0,   140, 255),
    "marking":     (0,   255,   0),
    "exit":        (255,   0, 255),
    "noise_strip": (0,   200, 255),
}

ROAD_INFRA_CLASSES = {"traffic_light": 9, "stop_sign": 11, "bus": 5, "truck": 7,
                      "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3}


# ═══════════════════════════════════════════════════════════════════════════════
#  VideoProcessor — ядро обработки (без зависимостей от мессенджера)
# ═══════════════════════════════════════════════════════════════════════════════
class VideoProcessor:
    def __init__(self, sign_model_path, infra_model_path, marking_model_path=None):
        # ── Устройство: по умолчанию CPU (целевой сервер Ubuntu без GPU) ─────────
        # При необходимости можно переопределить через BOT_DEVICE=cuda.
        self.device = os.getenv("BOT_DEVICE", "cpu").strip().lower()
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA недоступна — откатываюсь на CPU")
            self.device = "cpu"
        logger.info(f"Устройство: {self.device}")

        # Необязательное ограничение CPU-потоков (BOT_CPU_THREADS=N).
        # По умолчанию не ограничиваем — на сервере нужны все ядра для скорости.
        _threads = os.getenv("BOT_CPU_THREADS")
        if _threads:
            try:
                _n = max(1, int(_threads))
                torch.set_num_threads(_n)
                cv2.setNumThreads(_n)
                logger.info(f"CPU-потоки ограничены: {_n}")
            except Exception:
                pass

        self.sign_model  = YOLO(sign_model_path)
        self.infra_model = YOLO(infra_model_path)

        self._speed_classes    = {i for i, n in self.sign_model.names.items() if "speed_over" in n or "speed_limit" in n}
        self._overtake_classes = {i for i, n in self.sign_model.names.items() if "overtake" in n}
        self._priority_classes = {i for i, n in self.sign_model.names.items() if "prio_" in n}

        if marking_model_path and Path(marking_model_path).exists():
            self.marking_model = YOLO(marking_model_path)
        else:
            self.marking_model = None

        if self.device != "cpu":
            if self.device == "cuda":
                torch.cuda.set_per_process_memory_fraction(0.7)
            self.sign_model.to(self.device)
            self.infra_model.to(self.device)
            if self.marking_model:
                self.marking_model.to(self.device)
        logger.info("✅ Модели загружены")

    def _merge_sign_detections(self, all_results):
        all_boxes = []
        for result in all_results:
            if hasattr(result, "boxes") and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    all_boxes.append({"bbox": (x1, y1, x2, y2), "conf": conf, "cls": cls_id})
        unique = []
        for box in sorted(all_boxes, key=lambda b: -b["conf"]):
            x1, y1, x2, y2 = box["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not any(((cx - (u["bbox"][0]+u["bbox"][2])//2)**2 + (cy - (u["bbox"][1]+u["bbox"][3])//2)**2)**0.5 < 50 for u in unique):
                unique.append(box)
        return unique

    def _same_sign_group(self, cls_a, cls_b):
        if cls_a == cls_b:
            return True
        for grp in [self._speed_classes, self._overtake_classes, self._priority_classes]:
            if cls_a in grp and cls_b in grp:
                return True
        return False

    async def process_video(
        self,
        video_path: str,
        output_path: str,
        progress_cb,          # async callback(pct, text)
        detector_settings: dict = None,
        gps_coords=None,
    ):
        """
        Обрабатывает видео.
        progress_cb — async-функция (pct: int, text: str)
        Возвращает (stats, all_detections, output_mp4_path)
        """
        if detector_settings is None:
            detector_settings = {k: True for k in
                                  ["signs", "infrastructure", "barriers", "marking", "exits", "noise_strips"]}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        if fps < 5 or fps > 120:
            fps = 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width % 2 != 0: width -= 1
        if height % 2 != 0: height -= 1

        # GPS трекер
        gps_tracker = None
        if GPS_AVAILABLE:
            try:
                gps_tracker = GPSTracker(video_path, start_coords=gps_coords, speed_kmh=50.0)
                gps_tracker.setup_video(cap)
                logger.info("✅ GPS трекер инициализирован")
            except Exception as e:
                logger.warning(f"GPS трекер недоступен: {e}")

        # Пикетаж трекер
        picket_tracker = None
        if PICKET_AVAILABLE:
            try:
                picket_tracker = PicketTracker(video_path=str(video_path), speed_kmh=60.0, start_picket_m=0.0)
                picket_tracker.setup_video(cap)
                logger.info("✅ Пикетаж трекер инициализирован")
            except Exception as e:
                logger.warning(f"Пикетаж трекер недоступен: {e}")

        frames_dir = Path(output_path).parent / f"frames_{Path(output_path).stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(Path(output_path).with_suffix(".mp4"))

        FRAME_SKIP = max(1, fps // 20)

        # Константы дедупликации
        NOISE_STRIP_CONF              = 0.35
        NOISE_STRIP_MATCH_DISTANCE    = 250
        NOISE_STRIP_DUPLICATE_DISTANCE= 120
        NOISE_STRIP_PICKET_MATCH_METERS = 25
        INFRA_MATCH_DISTANCE          = 150
        INFRA_DUPLICATE_DISTANCE      = 60
        INFRA_PICKET_MATCH_METERS     = 8
        BARRIER_MATCH_DISTANCE        = 180
        BARRIER_DUPLICATE_DISTANCE    = 80
        BARRIER_PICKET_MATCH_METERS   = 10
        EXIT_MATCH_DISTANCE           = 200
        EXIT_DUPLICATE_DISTANCE       = 100
        EXIT_PICKET_MATCH_METERS      = 15

        stats = {k: 0 for k in ["signs","sign_frames","infra","infra_frames",
                                 "marking_frames","barriers","barrier_frames",
                                 "exits","exit_frames","noise_strips","noise_strip_frames"]}
        tracked_objects = {k: [] for k in ["signs","infra","barriers","exits","noise_strips"]}
        all_detections  = []
        saved_frames    = []
        frame_num       = 0
        processed_frames= 0
        last_progress   = 0

        await progress_cb(0, "Загрузка моделей завершена, начинаю обработку...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            # Кадры без детекции — сохраняем как есть
            if frame_num % FRAME_SKIP != 0:
                frame = np.ascontiguousarray(frame)
                fp = frames_dir / f"frame_{frame_num:06d}.png"
                cv2.imwrite(str(fp), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                saved_frames.append(fp)
                continue

            processed_frames += 1

            # Пикетаж overlay
            if picket_tracker:
                try:
                    p_val = picket_tracker.get_frame_picket(frame_num)
                    p_text = picket_tracker.format_picket(p_val)
                    overlay = f"PICKET: {p_text}"
                    (tw, th), _ = cv2.getTextSize(overlay, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (10, 10), (20+tw, 40+th), (0,0,0), -1)
                    cv2.putText(frame, overlay, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                except Exception:
                    pass

            # ── Детекция знаков ──────────────────────────────────────────────
            frame_signs = 0
            if detector_settings.get("signs", True):
                sign_results = self.sign_model(frame, conf=0.08, imgsz=1536, augment=True, verbose=False, device=self.device)
                unique_boxes = self._merge_sign_detections(list(sign_results))
            else:
                unique_boxes = []

            for bd in unique_boxes:
                x1, y1, x2, y2 = bd["bbox"]
                conf   = bd["conf"]
                cls_id = bd["cls"]
                if conf < 0.12:
                    continue
                class_name = self.sign_model.names[cls_id]
                center = ((x1+x2)//2, (y1+y2)//2)

                picket = None
                picket_lbl = ""
                if picket_tracker:
                    try:
                        picket = picket_tracker.get_object_picket((x1,y1,x2,y2), frame_num)
                        picket_lbl = f" ({int(picket)}m)"
                    except Exception:
                        pass

                is_new = True
                for tr in tracked_objects["signs"]:
                    if frame_num - tr["frame"] > fps * 1.0:
                        continue
                    if not self._same_sign_group(tr.get("cls"), cls_id):
                        continue
                    tp = tr.get("picket")
                    if picket and tp and abs(picket-tp) <= 15:
                        is_new = False; tr.update({"frame":frame_num,"center":center,"picket":picket}); break
                    tx,ty = tr["center"]
                    if ((center[0]-tx)**2+(center[1]-ty)**2)**0.5 < 150:
                        is_new = False; tr.update({"frame":frame_num,"center":center}); break

                if is_new:
                    tracked_objects["signs"].append({"center":center,"frame":frame_num,"cls":cls_id,"picket":picket})
                    stats["signs"] += 1
                    if EXCEL_AVAILABLE:
                        ts = frame_num/fps if fps else None
                        coords = None
                        if gps_tracker:
                            try: coords = gps_tracker.get_object_coordinates((x1,y1,x2,y2), frame_num)
                            except Exception: pass
                        all_detections.append(format_detections_for_excel(
                            frame_number=frame_num, object_type="sign",
                            class_name=class_name, confidence=conf,
                            bbox=(x1,y1,x2,y2), timestamp=ts,
                            coordinates=coords, picket=picket))

                color = COLORS["sign"]
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 5)
                lbl = f"SIGN: {class_name}{picket_lbl}"
                ls, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                cv2.rectangle(frame, (x1, y1-ls[1]-10), (x1+ls[0], y1), color, -1)
                cv2.putText(frame, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3)
                frame_signs += 1
            if frame_signs: stats["sign_frames"] += 1

            # ── Инфраструктура ────────────────────────────────────────────────
            frame_infra = 0
            frame_infra_centers = []
            if detector_settings.get("infrastructure", True):
                infra_results = self.infra_model(frame, classes=list(ROAD_INFRA_CLASSES.values()),
                                                 conf=0.60, augment=True, verbose=False, device=self.device)
                for result in infra_results:
                    for box in result.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        conf   = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        if conf < 0.60: continue
                        class_name = self.infra_model.names[cls_id]
                        center = ((x1+x2)//2,(y1+y2)//2)
                        if any(((center[0]-c[0])**2+(center[1]-c[1])**2)**0.5 < INFRA_DUPLICATE_DISTANCE for c in frame_infra_centers):
                            continue
                        frame_infra_centers.append(center)

                        picket = None
                        if picket_tracker:
                            try: picket = picket_tracker.get_object_picket((x1,y1,x2,y2), frame_num)
                            except Exception: pass

                        is_new = True
                        for tr in tracked_objects["infra"]:
                            if frame_num - tr["frame"] > fps*2: continue
                            if tr.get("cls") != cls_id: continue
                            tp = tr.get("picket")
                            if picket and tp and abs(picket-tp) <= INFRA_PICKET_MATCH_METERS:
                                is_new=False; tr.update({"center":center,"frame":frame_num,"picket":picket}); break
                            tx,ty=tr["center"]
                            if ((center[0]-tx)**2+(center[1]-ty)**2)**0.5 < INFRA_MATCH_DISTANCE:
                                is_new=False; tr.update({"center":center,"frame":frame_num}); break

                        if is_new:
                            tracked_objects["infra"].append({"center":center,"frame":frame_num,"cls":cls_id,"picket":picket})
                            stats["infra"] += 1
                            if EXCEL_AVAILABLE:
                                ts = frame_num/fps if fps else None
                                coords = None
                                if gps_tracker:
                                    try: coords = gps_tracker.get_object_coordinates((x1,y1,x2,y2), frame_num)
                                    except Exception: pass
                                all_detections.append(format_detections_for_excel(
                                    frame_number=frame_num, object_type="infra",
                                    class_name=class_name, confidence=conf,
                                    bbox=(x1,y1,x2,y2), timestamp=ts,
                                    coordinates=coords, picket=picket))

                        color = COLORS["infra"]
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
                        lbl = f"{class_name} {conf:.2f}"
                        ls,_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1-ls[1]-8), (x1+ls[0], y1), color, -1)
                        cv2.putText(frame, lbl, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                        frame_infra += 1
            if frame_infra: stats["infra_frames"] += 1

            # ── Барьеры ───────────────────────────────────────────────────────
            frame_barriers = 0
            if detector_settings.get("barriers", True) and BARRIER_AVAILABLE:
                barriers = detect_barriers_combined(frame, coco_model=self.infra_model, conf_threshold=0.3)
                frame_barrier_centers = []
                for barrier in barriers:
                    x1,y1,x2,y2 = barrier["bbox"]
                    conf = barrier.get("confidence", 0.5)
                    if conf < 0.3: continue
                    center = ((x1+x2)//2,(y1+y2)//2)
                    if any(((center[0]-c[0])**2+(center[1]-c[1])**2)**0.5 < BARRIER_DUPLICATE_DISTANCE for c in frame_barrier_centers):
                        continue
                    frame_barrier_centers.append(center)

                    picket = None; picket_lbl = ""
                    if picket_tracker:
                        try:
                            picket = picket_tracker.get_object_picket((x1,y1,x2,y2), frame_num)
                            picket_lbl = f" ({int(picket)}m)"
                        except Exception: pass

                    is_new = True
                    for tr in tracked_objects["barriers"]:
                        if frame_num - tr["frame"] > fps*2: continue
                        tp = tr.get("picket")
                        if picket and tp and abs(picket-tp) <= BARRIER_PICKET_MATCH_METERS:
                            is_new=False; tr.update({"center":center,"frame":frame_num,"picket":picket}); break
                        tx,ty=tr["center"]
                        if ((center[0]-tx)**2+(center[1]-ty)**2)**0.5 < BARRIER_MATCH_DISTANCE:
                            is_new=False; tr.update({"center":center,"frame":frame_num}); break

                    if is_new:
                        tracked_objects["barriers"].append({"center":center,"frame":frame_num,"picket":picket})
                        stats["barriers"] += 1
                        if EXCEL_AVAILABLE:
                            ts = frame_num/fps if fps else None
                            coords = None
                            if gps_tracker:
                                try: coords = gps_tracker.get_object_coordinates((x1,y1,x2,y2), frame_num)
                                except Exception: pass
                            all_detections.append(format_detections_for_excel(
                                frame_number=frame_num, object_type="barrier",
                                class_name=barrier.get("class","barrier"), confidence=conf,
                                bbox=(x1,y1,x2,y2), timestamp=ts, coordinates=coords, picket=picket))

                    cv2.rectangle(frame, (x1,y1), (x2,y2), COLORS["barrier"], 5)
                    lbl = f"BARRIER {conf:.2f}{picket_lbl}"
                    ls,_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                    cv2.rectangle(frame, (x1, y1-ls[1]-10), (x1+ls[0], y1), COLORS["barrier"], -1)
                    cv2.putText(frame, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3)
                    frame_barriers += 1
            if frame_barriers: stats["barrier_frames"] += 1

            # ── Разметка (метод Serzho) ───────────────────────────────────────
            if detector_settings.get("marking", True):
                h, w = frame.shape[:2]
                roi_y0 = h // 2
                roi_y1 = round((h // 4) * 2.8)
                roi = frame[roi_y0:roi_y1, 0:w]
                proc = cv2.medianBlur(roi, 5)
                _, proc = cv2.threshold(proc, 210, 255, 0)
                proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                proc = cv2.GaussianBlur(proc, (7,7), 1.5)
                proc = cv2.Canny(proc, 1, 50)
                cv2.rectangle(frame, (0, roi_y0), (w, roi_y1), (0,0,255), 2)
                lz0, lz1 = int(w*0.3), int(w*0.5)
                rz0, rz1 = int(w*0.75), int(w*0.95)
                cv2.rectangle(frame, (lz0,roi_y0), (lz1,roi_y1), (0,255,255), 2)
                cv2.rectangle(frame, (rz0,roi_y0), (rz1,roi_y1), (0,255,255), 2)
                contours, hierarchy = cv2.findContours(proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(frame[roi_y0:roi_y1, 0:w], contours, -1, (255,0,0), 5, cv2.LINE_AA, hierarchy, 1)
                sum_l = sum_r = 0
                for cnt in contours:
                    for dp in cnt:
                        x = int(dp[0][0])
                        if lz0 <= x <= lz1: sum_l += 1
                        elif rz0 <= x <= rz1: sum_r += 1
                lt = "Solid" if sum_l > 1000 else "Intermittent"
                rt = "Solid" if sum_r > 1000 else "Intermittent"
                cv2.putText(frame, f"Left: {lt} ({sum_l})", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.putText(frame, f"Right: {rt} ({sum_r})", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                if contours: stats["marking_frames"] += 1

            # ── Съезды ────────────────────────────────────────────────────────
            frame_exits = 0
            if detector_settings.get("exits", True) and EXIT_DETECTOR_AVAILABLE:
                exits = detect_road_exits(frame)
                frame_exit_centers = []
                for ex in exits:
                    x1,y1,x2,y2 = ex["bbox"]
                    conf = ex.get("confidence", 0.7)
                    if conf < 0.70: continue
                    center = ((x1+x2)//2,(y1+y2)//2)
                    if any(((center[0]-c[0])**2+(center[1]-c[1])**2)**0.5 < EXIT_DUPLICATE_DISTANCE for c in frame_exit_centers):
                        continue
                    frame_exit_centers.append(center)

                    picket=None; picket_lbl=""
                    if picket_tracker:
                        try:
                            picket = picket_tracker.get_object_picket((x1,y1,x2,y2), frame_num)
                            picket_lbl = f" ({int(picket)}m)"
                        except Exception: pass

                    is_new=True
                    for tr in tracked_objects["exits"]:
                        if frame_num - tr["frame"] > fps*2: continue
                        tp=tr.get("picket")
                        if picket and tp and abs(picket-tp) <= EXIT_PICKET_MATCH_METERS:
                            is_new=False; tr.update({"center":center,"frame":frame_num,"picket":picket}); break
                        tx,ty=tr["center"]
                        if ((center[0]-tx)**2+(center[1]-ty)**2)**0.5 < EXIT_MATCH_DISTANCE:
                            is_new=False; tr.update({"center":center,"frame":frame_num}); break
                    if is_new:
                        tracked_objects["exits"].append({"center":center,"frame":frame_num,"picket":picket})
                        stats["exits"] += 1
                        if EXCEL_AVAILABLE:
                            ts=frame_num/fps if fps else None
                            coords=None
                            if gps_tracker:
                                try: coords=gps_tracker.get_object_coordinates((x1,y1,x2,y2),frame_num)
                                except Exception: pass
                            all_detections.append(format_detections_for_excel(
                                frame_number=frame_num, object_type="exit",
                                class_name=ex.get("description","road_exit"), confidence=conf,
                                bbox=(x1,y1,x2,y2), timestamp=ts, coordinates=coords, picket=picket))
                    cv2.rectangle(frame, (x1,y1), (x2,y2), COLORS["exit"], 3)
                    lbl=f"EXIT {conf:.2f}{picket_lbl}"
                    ls,_=cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame,(x1,y1-ls[1]-10),(x1+ls[0],y1),COLORS["exit"],-1)
                    cv2.putText(frame,lbl,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                    frame_exits+=1
            if frame_exits: stats["exit_frames"]+=1

            # ── Шумовые полосы ────────────────────────────────────────────────
            frame_noise=0
            if detector_settings.get("noise_strips", True) and NOISE_DETECTOR_AVAILABLE:
                strips = detect_noise_strips(frame)
                frame_noise_centers=[]
                for strip in strips:
                    x1,y1,x2,y2=strip["bbox"]
                    conf=strip.get("confidence",0.7)
                    if conf < NOISE_STRIP_CONF: continue
                    center=((x1+x2)//2,(y1+y2)//2)
                    if any(((center[0]-c[0])**2+(center[1]-c[1])**2)**0.5 < NOISE_STRIP_DUPLICATE_DISTANCE for c in frame_noise_centers):
                        continue
                    frame_noise_centers.append(center)

                    picket=None; picket_lbl=""
                    if picket_tracker:
                        try:
                            picket=picket_tracker.get_object_picket((x1,y1,x2,y2),frame_num)
                            picket_lbl=f" ({int(picket)}m)"
                        except Exception: pass

                    is_new=True
                    for tr in tracked_objects["noise_strips"]:
                        if frame_num-tr["frame"] > fps*2: continue
                        tp=tr.get("picket")
                        if picket and tp and abs(picket-tp)<=NOISE_STRIP_PICKET_MATCH_METERS:
                            is_new=False; tr.update({"center":center,"frame":frame_num,"picket":picket}); break
                        tx,ty=tr["center"]
                        if ((center[0]-tx)**2+(center[1]-ty)**2)**0.5 < NOISE_STRIP_MATCH_DISTANCE:
                            is_new=False; tr.update({"center":center,"frame":frame_num}); break
                    if is_new:
                        tracked_objects["noise_strips"].append({"center":center,"frame":frame_num,"picket":picket})
                        stats["noise_strips"]+=1
                        if EXCEL_AVAILABLE:
                            ts=frame_num/fps if fps else None
                            coords=None
                            if gps_tracker:
                                try: coords=gps_tracker.get_object_coordinates((x1,y1,x2,y2),frame_num)
                                except Exception: pass
                            all_detections.append(format_detections_for_excel(
                                frame_number=frame_num, object_type="noise_strip",
                                class_name=strip.get("description","noise_strip"), confidence=conf,
                                bbox=(x1,y1,x2,y2), timestamp=ts, coordinates=coords, picket=picket))
                    cv2.rectangle(frame,(x1,y1),(x2,y2),COLORS["noise_strip"],3)
                    lbl=f"NOISE {conf:.2f}{picket_lbl}"
                    ls,_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
                    cv2.rectangle(frame,(x1,y1-ls[1]-10),(x1+ls[0],y1),COLORS["noise_strip"],-1)
                    cv2.putText(frame,lbl,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
                    frame_noise+=1
            if frame_noise: stats["noise_strip_frames"]+=1

            # Нормализация и сохранение кадра
            if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
            if len(frame.shape)==2: frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
            elif frame.shape[2]==4: frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
            frame=np.ascontiguousarray(frame)
            fp = frames_dir / f"frame_{frame_num:06d}.png"
            cv2.imwrite(str(fp), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            saved_frames.append(fp)

            # Прогресс
            pct = int(frame_num / max(total,1) * 90)
            if pct > last_progress + 5 or frame_num % 50 == 0:
                last_progress = pct
                await progress_cb(pct,
                    f"📊 {pct}%  кадр {frame_num}/{total}\n"
                    f"🚦 Знаков: {stats['signs']}  🏗️ Инфра: {stats['infra']}\n"
                    f"🚧 Барьеры: {stats['barriers']}  〰️ Шумовые: {stats['noise_strips']}")

        cap.release()

        # ── Сборка видео ──────────────────────────────────────────────────────
        await progress_cb(92, "🎬 Сборка видео через ffmpeg...")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            for c in ["/opt/homebrew/bin/ffmpeg","/usr/local/bin/ffmpeg","/usr/bin/ffmpeg",
                      r"C:\ffmpeg\bin\ffmpeg.exe"]:
                if Path(c).exists():
                    ffmpeg = c; break
        if not ffmpeg:
            raise ValueError("ffmpeg не найден. Установите: brew install ffmpeg")

        result = subprocess.run([
            ffmpeg, "-y", "-framerate", str(fps),
            "-i", str(frames_dir/"frame_%06d.png"),
            "-c:v","libx264","-preset","medium","-crf","23",
            "-pix_fmt","yuv420p","-movflags","+faststart",
            output_path
        ], capture_output=True, text=True, timeout=600,
           stdin=subprocess.DEVNULL, cwd=str(frames_dir.parent))

        if result.returncode != 0:
            raise ValueError(f"ffmpeg ошибка: {result.stderr[:300]}")

        try:
            shutil.rmtree(frames_dir)
        except Exception:
            pass

        await progress_cb(96, "📊 Создание Excel-отчёта...")

        # ── Excel ─────────────────────────────────────────────────────────────
        excel_path = None
        if EXCEL_AVAILABLE and all_detections:
            try:
                excel_path = create_excel_report(
                    all_detections,
                    output_path=str(Path(output_path).parent /
                                   f"{Path(output_path).stem}_report.xlsx"),
                    video_name=Path(video_path).stem,
                )
            except Exception as e:
                logger.warning(f"Excel не создан: {e}")

        # ── Ситуационный план ─────────────────────────────────────────────────
        plan_path = None
        if SITUATIONAL_PLAN_AVAILABLE and all_detections:
            try:
                ASSUMED_SPEED = 10.0
                plan_dets = []
                for d in all_detections:
                    d2 = d.copy()
                    if d2.get("picket") is None and d2.get("timestamp"):
                        d2["picket"] = d2["timestamp"] * ASSUMED_SPEED
                    plan_dets.append(d2)
                plan_dets = [d for d in plan_dets if d.get("picket") is not None]
                if plan_dets:
                    plan_path = create_situational_plan(
                        plan_dets,
                        {"filename": Path(video_path).name,
                         "duration": total/fps if fps else 60,
                         "fps": fps},
                        output_path=str(Path(output_path).parent /
                                       f"{Path(output_path).stem}_plan.png"),
                    )
            except Exception as e:
                logger.warning(f"План не создан: {e}")

        await progress_cb(100, "✅ Готово!")
        return stats, all_detections, output_path, excel_path, plan_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Глобальный процессор (ленивая инициализация)
# ═══════════════════════════════════════════════════════════════════════════════
_processor: VideoProcessor | None = None

def get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        _processor = VideoProcessor(SIGN_MODEL_PATH, INFRA_MODEL_PATH, MARKING_MODEL_PATH)
    return _processor


# ═══════════════════════════════════════════════════════════════════════════════
#  Вспомогательные функции
# ═══════════════════════════════════════════════════════════════════════════════
def _detector_keyboard(settings: dict):
    """Создаёт инлайн-клавиатуру выбора детекторов."""
    def btn(key, label):
        icon = "✅" if settings.get(key, True) else "☐"
        return CallbackButton(text=f"{icon} {label}", payload=f"toggle_{key}")

    buttons = [
        [btn("signs",          "Дорожные знаки")],
        [btn("infrastructure", "Инфраструктура (авто, светофоры)")],
        [btn("barriers",       "Барьеры и ограждения")],
        [btn("marking",        "Дорожная разметка")],
        [btn("exits",          "Съезды с дороги")],
        [btn("noise_strips",   "Шумовые полосы")],
        [CallbackButton(text="🚀 Начать обработку", payload="start_processing")],
    ]
    return ButtonsPayload(buttons=buttons).pack()


async def _download_video_from_message(message, bot: Bot, dest: Path) -> bool:
    """Скачивает видео из сообщения Max. Возвращает True при успехе."""
    import aiohttp

    attachments = message.body.attachments if message.body else []
    if not attachments:
        return False

    url = None
    for att in attachments:
        if att.type == AttachmentType.VIDEO:
            logger.info(f"Video attachment: token={getattr(att,'token',None)}, urls={getattr(att,'urls',None)}, payload={getattr(att,'payload',None)}")
            # 1. Пробуем urls прямо из вложения (когда пользователь отправляет видео)
            urls_obj = getattr(att, "urls", None)
            if urls_obj:
                for q in ["mp4_1080","mp4_720","mp4_480","mp4_360","mp4_240","mp4_144"]:
                    u = getattr(urls_obj, q, None)
                    if u:
                        url = u; break
            # 2. Если urls нет, пробуем через token
            if not url and getattr(att, "token", None):
                try:
                    vid = await bot.get_video(att.token)
                    for q in ["mp4_1080","mp4_720","mp4_480","mp4_360","mp4_240"]:
                        u = getattr(vid.urls, q, None) if vid.urls else None
                        if u:
                            url = u; break
                except Exception as e:
                    logger.warning(f"get_video error: {e}")
            # 3. Fallback: payload.url
            if not url:
                payload = getattr(att, "payload", None)
                if payload and getattr(payload, "url", None):
                    url = payload.url
            break
        elif att.type == AttachmentType.FILE:
            if att.payload and att.payload.url:
                name = getattr(att, "filename", "") or ""
                ext = Path(name).suffix.lower()
                if ext in {".mp4",".mov",".avi",".mkv",".m4v",".3gp",".flv",".wmv",".webm"}:
                    url = att.payload.url; break

    if not url:
        return False

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            if resp.status != 200:
                logger.error(f"Ошибка скачивания: HTTP {resp.status}")
                return False
            with open(dest, "wb") as f:
                async for chunk in resp.content.iter_chunked(65536):
                    f.write(chunk)

    return dest.exists() and dest.stat().st_size > 1024


def _has_video_attachment(message) -> bool:
    attachments = message.body.attachments if message.body else []
    if not attachments:
        return False
    for att in attachments:
        if att.type == AttachmentType.VIDEO:
            return True
        if att.type == AttachmentType.FILE:
            name = getattr(att, "filename", "") or ""
            ext  = Path(name).suffix.lower()
            if ext in {".mp4",".mov",".avi",".mkv",".m4v",".3gp",".flv",".wmv",".webm"}:
                return True
    return False


def _extract_gps_from_video(video_path: str):
    """Извлечение GPS из метаданных (синхронно)."""
    import json, re
    try:
        res = subprocess.run(
            ["ffprobe","-v","quiet","-print_format","json","-show_format","-show_streams",str(video_path)],
            capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            meta = json.loads(res.stdout)
            tags = meta.get("format",{}).get("tags",{})
            lat=lon=None
            for k in ["location-lat","latitude","lat","GPS Latitude"]:
                if k in tags: lat=tags[k]; break
            for k in ["location-lon","longitude","lon","GPS Longitude"]:
                if k in tags: lon=tags[k]; break
            if not lat or not lon:
                for iso_k in ["com.apple.quicktime.location.ISO6709","location"]:
                    v=tags.get(iso_k)
                    if v:
                        m=re.match(r"([+-]?\d+\.?\d*)([+-]\d+\.?\d*)",v)
                        if m: lat,lon=m.group(1),m.group(2); break
            if lat and lon:
                return (float(lat), float(lon))
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Dispatcher + обработчики
# ═══════════════════════════════════════════════════════════════════════════════
dp = Dispatcher()


# ── /start (и при первом запуске бота) ────────────────────────────────────────
START_TEXT = (
    "👋 Добро пожаловать в бот обработки дорожной инфраструктуры!\n\n"
    "📹 Отправьте мне видео — и я применю все модели детекции:\n\n"
    "🚦 Дорожные знаки (155 классов RTSD)\n"
    "🏗️  Инфраструктура (светофоры, машины, автобусы)\n"
    "🚧 Барьеры и ограждения\n"
    "📏 Дорожная разметка\n"
    "🔀 Съезды с дороги\n"
    "〰️ Шумовые полосы\n"
    "📊 Excel-отчёт + ситуационный план\n\n"
    "📋 Команды:\n"
    "/start — это сообщение\n"
    "/gps — показать GPS-координаты последнего видео\n"
    "/plan — создать ситуационный план\n\n"
    "📤 Просто отправьте видеофайл ⬆️"
)


@dp.bot_started()
async def on_bot_started(event: BotStarted, context: MemoryContext):
    bot = event._ensure_bot()
    await bot.send_message(chat_id=event.chat_id, text=START_TEXT)


@dp.message_created(CommandStart())
async def cmd_start(event: MessageCreated, context: MemoryContext):
    await event.message.answer(START_TEXT)


# ── /gps ──────────────────────────────────────────────────────────────────────
@dp.message_created(Command("gps"))
async def cmd_gps(event: MessageCreated, context: MemoryContext):
    data = await context.get_data()
    coords = data.get("video_gps_coords")
    if not coords:
        await event.message.answer(
            "❌ GPS-координаты не найдены.\n\n"
            "📹 Отправьте видео со включённым GPS — координаты определятся автоматически."
        )
        return

    lat, lon = coords
    maps_url = f"https://yandex.ru/maps/?ll={lon},{lat}&z=17&pt={lon},{lat},pm2rdm"
    await event.message.answer(
        f"📍 GPS-координаты видео:\n\n"
        f"🌐 Широта:  {lat:.6f}\n"
        f"🌐 Долгота: {lon:.6f}\n\n"
        f"🗺️ {maps_url}"
    )


# ── /plan ─────────────────────────────────────────────────────────────────────
@dp.message_created(Command("plan"))
async def cmd_plan(event: MessageCreated, context: MemoryContext):
    data = await context.get_data()
    detections = data.get("last_detections")
    if not detections:
        await event.message.answer(
            "❌ Нет данных для плана.\n\n"
            "💡 Сначала обработайте видео, затем используйте /plan."
        )
        return

    sent = await event.message.answer("🗺️ Создаю ситуационный план...")
    status = sent.message if sent else None

    try:
        video_info = data.get("last_video_info", {})
        ASSUMED_SPEED = 10.0
        plan_dets = []
        for d in detections:
            d2 = d.copy()
            if d2.get("picket") is None and d2.get("timestamp"):
                d2["picket"] = d2["timestamp"] * ASSUMED_SPEED
            plan_dets.append(d2)
        plan_dets = [d for d in plan_dets if d.get("picket") is not None]

        if not plan_dets:
            if status: await status.edit(text="❌ Недостаточно данных для плана.")
            return

        plan_path = await asyncio.to_thread(
            create_situational_plan,
            plan_dets,
            video_info,
            str(OUTPUT_DIR / "situational_plan.png"),
        )

        if plan_path and Path(plan_path).exists():
            if status: await status.edit(text="✅ Ситуационный план создан! Отправляю...")
            await event.message.answer(
                text="🗺️ Ситуационный план:",
                attachments=[InputMedia(plan_path, type=UploadType.IMAGE)]
            )
        else:
            if status: await status.edit(text="⚠️ Не удалось создать план.")
    except Exception as e:
        logger.error(f"Ошибка /plan: {e}", exc_info=True)
        if status: await status.edit(text=f"❌ Ошибка: {e}")


# ── Приём видео ───────────────────────────────────────────────────────────────
@dp.message_created()
async def handle_message(event: MessageCreated, context: MemoryContext):
    """Принимает видеофайлы и показывает меню детекторов."""
    msg = event.message

    if not _has_video_attachment(msg):
        # Не видео — игнорируем (команды уже обработаны выше)
        return

    sent = await msg.answer("⏳ Получаю видео...")
    status = sent.message if sent else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = DOWNLOAD_DIR / f"video_{ts}.mp4"

    try:
        bot = event.message.bot
        ok = await _download_video_from_message(msg, bot, input_path)
        if not ok:
            if status: await status.edit(text="❌ Не удалось скачать видео. Попробуйте ещё раз.")
            return

        size_mb = input_path.stat().st_size / 1024 / 1024
        logger.info(f"Видео скачано: {input_path.name} ({size_mb:.1f} MB)")

        if status: await status.edit(text="✅ Видео получено!")

        # Сохраняем в контексте
        output_path = OUTPUT_DIR / f"processed_{ts}.mp4"
        await context.update_data(
            pending_video={"input": str(input_path), "output": str(output_path)},
            detector_settings={k: True for k in
                               ["signs","infrastructure","barriers","marking","exits","noise_strips"]},
        )

        # Показываем меню детекторов
        data = await context.get_data()
        settings = data.get("detector_settings", {k: True for k in
                            ["signs","infrastructure","barriers","marking","exits","noise_strips"]})
        await msg.answer(
            text=(
                "⚙️ Настройка детекторов\n\n"
                "Выберите, что нужно искать:\n"
                "(нажмите кнопку для вкл/выкл)\n\n"
                "После выбора нажмите 🚀 Начать обработку"
            ),
            attachments=[_detector_keyboard(settings)]
        )

    except Exception as e:
        logger.error(f"handle_message error: {e}", exc_info=True)
        if status: await status.edit(text=f"❌ Ошибка: {e}")


# ── Callback — кнопки меню ────────────────────────────────────────────────────
@dp.message_callback()
async def handle_callback(event: MessageCallback, context: MemoryContext):
    payload = event.callback.payload or ""
    data = await context.get_data()

    default_settings = {k: True for k in
                        ["signs","infrastructure","barriers","marking","exits","noise_strips"]}
    settings = data.get("detector_settings", default_settings)

    if payload == "start_processing":
        await event.answer(notification="🚀 Запускаю обработку...")
        await _process_video(event, context, settings)
        return

    # Переключение детекторов
    toggle_map = {
        "toggle_signs":          "signs",
        "toggle_infrastructure": "infrastructure",
        "toggle_barriers":       "barriers",
        "toggle_marking":        "marking",
        "toggle_exits":          "exits",
        "toggle_noise_strips":   "noise_strips",
    }
    if payload in toggle_map:
        key = toggle_map[payload]
        settings[key] = not settings.get(key, True)
        await context.update_data(detector_settings=settings)

    # Обновляем клавиатуру
    await event.answer(
        new_text=(
            "⚙️ Настройка детекторов\n\n"
            "Выберите, что нужно искать:\n"
            "(нажмите кнопку для вкл/выкл)\n\n"
            "После выбора нажмите 🚀 Начать обработку"
        ),
    )
    # Пересылаем обновлённое сообщение с новой клавиатурой
    if event.message:
        await event.message.edit(
            text=(
                "⚙️ Настройка детекторов\n\n"
                "Выберите, что нужно искать:\n"
                "(нажмите кнопку для вкл/выкл)\n\n"
                "После выбора нажмите 🚀 Начать обработку"
            ),
            attachments=[_detector_keyboard(settings)]
        )


# ── Основная обработка видео ──────────────────────────────────────────────────
async def _process_video(event: MessageCallback, context: MemoryContext, settings: dict):
    data = await context.get_data()
    pending = data.get("pending_video")
    if not pending:
        if event.message:
            await event.message.answer("❌ Видео не найдено. Отправьте видео заново.")
        return

    input_path  = Path(pending["input"])
    output_path = Path(pending["output"])
    msg         = event.message  # сообщение с клавиатурой

    if not input_path.exists():
        if msg: await msg.answer("❌ Файл видео не найден. Отправьте видео заново.")
        return

    # GPS из метаданных видео
    gps_coords = await asyncio.to_thread(_extract_gps_from_video, str(input_path))
    if gps_coords:
        await context.update_data(video_gps_coords=gps_coords)
        logger.info(f"📍 GPS: {gps_coords}")

    enabled = [k for k, v in settings.items() if v]
    sent = await (msg.answer if msg else (lambda **kw: None))(
        text=(
            f"{'✅ GPS найден' if gps_coords else '🚫 GPS не найден'}\n"
            f"🚀 Обработка запущена...\n\n"
            f"🔧 Активных детекторов: {len(enabled)}/6\n"
            f"📊 Прогресс: 0%\n"
            f"🎬 Загрузка моделей..."
        )
    )
    status = sent.message if sent else None

    async def progress_cb(pct: int, text: str):
        if status:
            try:
                await status.edit(text=f"⏳ Обработка...\n\n{text}")
            except Exception as e:
                logger.debug(f"edit error: {e}")

    try:
        proc = await asyncio.to_thread(get_processor)

        stats, all_detections, out_mp4, excel_path, plan_path = \
            await proc.process_video(
                str(input_path),
                str(output_path),
                progress_cb=progress_cb,
                detector_settings=settings,
                gps_coords=gps_coords,
            )

        # Сохраняем детекции
        await context.update_data(
            last_detections=all_detections,
            last_video_info={
                "filename": input_path.name,
                "duration": 0,
                "fps": 30,
            }
        )

        if status:
            await status.edit(
                text=(
                    "✅ Обработка завершена!\n\n"
                    f"🚦 Знаков:        {stats['signs']} ({stats['sign_frames']} кадров)\n"
                    f"🏗️  Инфраструктура: {stats['infra']} ({stats['infra_frames']} кадров)\n"
                    f"🚧 Барьеров:       {stats['barriers']} ({stats['barrier_frames']} кадров)\n"
                    f"📏 Кадров с разм.: {stats['marking_frames']}\n"
                    f"🔀 Съездов:        {stats['exits']} ({stats['exit_frames']} кадров)\n"
                    f"〰️ Шумовых полос: {stats['noise_strips']} ({stats['noise_strip_frames']} кадров)\n\n"
                    f"📤 Отправляю результаты..."
                )
            )

        ref_msg = msg  # сообщение куда отвечать

        # Отправляем обработанное видео
        if Path(out_mp4).exists():
            size_mb = Path(out_mp4).stat().st_size / 1024 / 1024
            logger.info(f"Отправка видео: {size_mb:.1f} MB")
            await ref_msg.answer(
                text="🎬 Обработанное видео:",
                attachments=[InputMedia(out_mp4, type=UploadType.VIDEO)]
            )

        # Отправляем Excel
        if excel_path and Path(excel_path).exists():
            await ref_msg.answer(
                text="📊 Excel-отчёт:",
                attachments=[InputMedia(excel_path, type=UploadType.FILE)]
            )

        # Отправляем ситуационный план
        if plan_path and Path(plan_path).exists():
            await ref_msg.answer(
                text="🗺️ Ситуационный план:",
                attachments=[InputMedia(plan_path, type=UploadType.IMAGE)]
            )

        # Очистка входного файла
        try:
            input_path.unlink()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"_process_video error: {e}", exc_info=True)
        if status:
            await status.edit(text=f"❌ Ошибка при обработке видео:\n\n{e}")
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Точка входа
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not BOT_TOKEN:
        print("❌ Не задан токен MAX_BOT_TOKEN!")
        print("💡 Получите токен на dev.max.ru и задайте:")
        print("   export MAX_BOT_TOKEN=ваш_токен")
        return

    print("=" * 70)
    print("🤖  MAX-БОТ — Детекция дорожной инфраструктуры".center(70))
    print("=" * 70)
    print(f"📋 Токен:          {BOT_TOKEN[:20]}...")
    print(f"📁 Входящие видео: {DOWNLOAD_DIR}")
    print(f"📁 Результаты:     {OUTPUT_DIR}")
    print(f"🤖 Модель знаков:  {SIGN_MODEL_PATH}")
    print("=" * 70)

    # Увеличиваем количество попыток и задержку для загрузки медиа (Max обрабатывает видео дольше)
    from maxapi.methods.send_message import SendMessage
    SendMessage.ATTEMPTS_COUNT = 30
    SendMessage.RETRY_DELAY = 5

    bot = Bot(token=BOT_TOKEN, after_input_media_delay=10.0)

    async def _run():
        await dp.start_polling(bot, skip_updates=True)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
