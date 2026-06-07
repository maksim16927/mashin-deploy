#!/usr/bin/env python3
"""
Telegram бот для обработки видео с детекцией дорожной инфраструктуры
Принимает видео, обрабатывает со всеми моделями и отправляет результат
"""

import os
import sys
import logging
import asyncio
import subprocess
import site
from pathlib import Path
from datetime import datetime

# Гарантируем импорт python-telegram-bot даже при локальном пакете telegram
def _prioritize_site_packages():
    src_path = str(Path(__file__).resolve().parents[1])

    if src_path in sys.path:
        sys.path.remove(src_path)

    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass

    user_site = site.getusersitepackages()
    if user_site:
        site_paths.append(user_site)

    for p in reversed([p for p in site_paths if p]):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)

    if src_path not in sys.path:
        sys.path.append(src_path)


_prioritize_site_packages()

def _load_external_telegram_module():
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        site_paths.append(user_site)

    for p in [p for p in site_paths if p]:
        spec = importlib.machinery.PathFinder.find_spec("telegram", [p])
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            _local_telegram = sys.modules.get("telegram")
            sys.modules["telegram"] = module
            try:
                spec.loader.exec_module(module)
            finally:
                if _local_telegram is not None:
                    sys.modules["telegram"] = _local_telegram
                else:
                    sys.modules.pop("telegram", None)
            return module

    raise ImportError("python-telegram-bot не найден в site-packages")


try:
    import importlib.machinery
    import importlib.util

    _external_telegram = _load_external_telegram_module()
    sys.modules["telegram"] = _external_telegram
    _EXTERNAL_TELEGRAM = _external_telegram
    from telegram import Update, Message, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
    from telegram.constants import ParseMode
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️  python-telegram-bot не установлен. Установите: pip install python-telegram-bot")


def _ensure_external_telegram():
    if TELEGRAM_AVAILABLE:
        try:
            sys.modules["telegram"] = _EXTERNAL_TELEGRAM
        except Exception:
            pass

import cv2
from ultralytics import YOLO
import numpy as np
import torch

# Импортируем детектор барьеров и экспортер Excel
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from core.detectors.barrier_detector import detect_barriers_combined
except ImportError:
    # Если не удалось импортировать, используем заглушку
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

# Импортируем новые детекторы с обработкой ошибок
try:
    from core.detectors.exit_detector import detect_road_exits
    EXIT_DETECTOR_AVAILABLE = True
except ImportError as e:
    EXIT_DETECTOR_AVAILABLE = False
    logger.warning(f"ExitDetector недоступен: {e}")
    def detect_road_exits(frame):
        return []

try:
    from core.detectors.noise_strips_detector import detect_noise_strips
    NOISE_DETECTOR_AVAILABLE = True
except ImportError as e:
    NOISE_DETECTOR_AVAILABLE = False
    logger.warning(f"NoiseStripsDetector недоступен: {e}")
    def detect_noise_strips(frame):
        return []

try:
    from core.processors.situational_plan import create_situational_plan
    SITUATIONAL_PLAN_AVAILABLE = True
except ImportError as e:
    SITUATIONAL_PLAN_AVAILABLE = False
    logger.warning(f"SituationalPlan недоступен: {e}")
    def create_situational_plan(detections, video_info):
        return None

try:
    from core.processors.situational_plan_excel import create_situational_plan_excel
    SITUATIONAL_PLAN_EXCEL_AVAILABLE = True
except ImportError as e:
    SITUATIONAL_PLAN_EXCEL_AVAILABLE = False
    logger.warning(f"SituationalPlanExcel недоступен: {e}")
    def create_situational_plan_excel(detections, video_info):
        return None

# Значения по умолчанию
COLORS = {
    'sign': (0, 255, 0),              # Зеленый - дорожные знаки
    'marking': (255, 0, 0),           # Красный - разметка
    'default': (147, 20, 255),        # Розовый - остальное (неизвестные объекты)
    'bus': (0, 255, 255),             # Желтый - автобусы/остановки
    'truck': (0, 165, 255),           # Оранжевый - грузовики
    'traffic_light': (255, 255, 0),   # Голубой - светофоры
    'stop_sign': (0, 0, 255),         # Красный - знаки STOP
    'person': (128, 128, 128),        # Серый - пешеходы
    'bicycle': (255, 0, 255),         # Фиолетовый - велосипеды
    'barrier': (0, 165, 255),         # Оранжевый - барьеры/ограждения
    'gantry': (255, 0, 255),          # Фиолетовый - П-образные опоры
    'exit': (255, 255, 0),            # Желтый - съезды
    'noise_strip': (0, 215, 255),     # Золотистый - шумовые полосы
    'roi': (0, 0, 255),               # Красный - область интереса (ROI)
    'zone': (0, 255, 255),            # Желтый - зоны анализа
    'contour': (255, 0, 0)            # Синий - контуры разметки
}
ROAD_INFRA_CLASSES = {'bus': 5, 'car': 2, 'truck': 7, 'traffic_light': 9, 
                     'stop_sign': 11, 'person': 0, 'motorcycle': 3, 'bicycle': 1}

# Заглушки на случай ошибки импорта
def detect_lane_marking(frame):
    return []

def detect_gantries_and_structures(frame):
    return []

# Импортируем функции детекции
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "scripts" / "utils"))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "detect_full_system",
        project_root / "scripts" / "utils" / "detect_full_system.py"
    )
    detect_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detect_module)
    
    # Переопределяем функции если импорт успешен
    if hasattr(detect_module, 'detect_lane_marking'):
        detect_lane_marking = detect_module.detect_lane_marking
    if hasattr(detect_module, 'detect_gantries_and_structures'):
        detect_gantries_and_structures = detect_module.detect_gantries_and_structures
    if hasattr(detect_module, 'COLORS'):
        COLORS = detect_module.COLORS
    if hasattr(detect_module, 'ROAD_INFRA_CLASSES'):
        ROAD_INFRA_CLASSES = detect_module.ROAD_INFRA_CLASSES
except Exception as e:
    print(f"⚠️  Не удалось импортировать detect_full_system: {e}")

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Логируем доступность модулей после инициализации logger
if not EXCEL_AVAILABLE:
    logger.warning("openpyxl не установлен. Excel экспорт недоступен.")
if not GPS_AVAILABLE:
    logger.warning("GPS трекер недоступен.")
if not PICKET_AVAILABLE:
    logger.warning("Пикетаж трекер недоступен.")

# Конфигурация
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
# Яндекс API ключ: задаётся переменной окружения YANDEX_API_KEY
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
DOWNLOAD_DIR = project_root / "data" / "results" / "videos" / "telegram_input"
OUTPUT_DIR = project_root / "data" / "results" / "videos" / "telegram_output"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Модели — используем папку models/ в корне проекта
MODEL_DIR = project_root / "models"
if not MODEL_DIR.exists():
    MODEL_DIR = project_root / "data" / "models"

SIGN_MODEL_PATH = str(MODEL_DIR / "russian_signs_yolo12m.pt")
INFRA_MODEL_PATH = str(MODEL_DIR / "yolov8n.pt")
# Разметка: приоритет у обученной модели, иначе базовая
if (MODEL_DIR / "yolov8_road_marking_20epochs.pt").exists():
    MARKING_MODEL_PATH = str(MODEL_DIR / "yolov8_road_marking_20epochs.pt")
elif (MODEL_DIR / "yolov8n_road_marking.pt").exists():
    MARKING_MODEL_PATH = str(MODEL_DIR / "yolov8n_road_marking.pt")
else:
    MARKING_MODEL_PATH = None

def extract_gps_from_video(video_path: str):
    """Извлечение GPS координат из метаданных видео"""
    import json
    import re
    
    try:
        # Метод 1: ffprobe с полными метаданными (format + streams)
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', str(video_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            logger.debug(f"📋 Метаданные видео: {json.dumps(metadata, indent=2)}")
            
            # Поиск GPS в format tags
            if 'format' in metadata and 'tags' in metadata['format']:
                tags = metadata['format']['tags']
                logger.debug(f"🏷️  Format tags: {tags}")
                
                # Все возможные варианты ключей GPS
                lat_keys = ['location-lat', 'latitude', 'lat', 'GPS Latitude', 'GPSLatitude']
                lon_keys = ['location-lon', 'longitude', 'lon', 'GPS Longitude', 'GPSLongitude']
                
                lat = None
                lon = None
                
                # Ищем широту
                for key in lat_keys:
                    if key in tags:
                        lat = tags[key]
                        logger.info(f"🔍 Найдена широта в '{key}': {lat}")
                        break
                
                # Ищем долготу
                for key in lon_keys:
                    if key in tags:
                        lon = tags[key]
                        logger.info(f"🔍 Найдена долгота в '{key}': {lon}")
                        break
                
                # Формат ISO6709: +DD.DDDD+DDD.DDDD/ или +DD.DDDD+DDD.DDDD+AAA/
                if not lat or not lon:
                    iso_keys = ['com.apple.quicktime.location.ISO6709', 'location', 'GPS Position']
                    for iso_key in iso_keys:
                        iso_location = tags.get(iso_key)
                        if iso_location:
                            logger.info(f"🔍 Найден ISO6709 в '{iso_key}': {iso_location}")
                            # Парсим: +DD.DDDD+DDD.DDDD/ или +DD.DDDD+DDD.DDDD+ALT/
                            match = re.match(r'([+-]?\d+\.?\d*)([+-]\d+\.?\d*)', iso_location)
                            if match:
                                lat = match.group(1)
                                lon = match.group(2)
                                logger.info(f"✅ Распарсил ISO6709: lat={lat}, lon={lon}")
                                break
            
            # Поиск GPS в streams (особенно для видео с дронов/action камер)
            if not lat or not lon:
                if 'streams' in metadata:
                    for stream in metadata['streams']:
                        if 'tags' in stream:
                            stream_tags = stream['tags']
                            logger.debug(f"🎞️  Stream tags: {stream_tags}")
                            
                            # Ищем в каждом потоке
                            for key in ['location', 'GPS', 'gps', 'GPS Position']:
                                if key in stream_tags:
                                    gps_str = stream_tags[key]
                                    logger.info(f"🔍 Найден GPS в stream '{key}': {gps_str}")
                                    # Парсим различные форматы
                                    match = re.match(r'([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)', gps_str)
                                    if match:
                                        lat = match.group(1)
                                        lon = match.group(2)
                                        break
                
                if lat and lon:
                    lat_float = float(lat)
                    lon_float = float(lon)
                    logger.info(f"📍 GPS найден в видео: {lat_float}, {lon_float}")
                    return (lat_float, lon_float)
        
        # Метод 2: exiftool (если установлен) - более мощный инструмент
        try:
            exif_result = subprocess.run(
                ['exiftool', '-j', '-GPS*', '-Location*', str(video_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if exif_result.returncode == 0:
                exif_data = json.loads(exif_result.stdout)
                if exif_data and len(exif_data) > 0:
                    exif_tags = exif_data[0]
                    logger.debug(f"📸 EXIF tags: {exif_tags}")
                    
                    # Exiftool возвращает более читаемые имена
                    lat = (exif_tags.get('GPSLatitude') or 
                           exif_tags.get('Latitude') or
                           exif_tags.get('LocationLatitude'))
                    lon = (exif_tags.get('GPSLongitude') or 
                           exif_tags.get('Longitude') or
                           exif_tags.get('LocationLongitude'))
                    
                    if lat and lon:
                        # Преобразуем из формата "55 deg 45' 20.88\" N" если нужно
                        if isinstance(lat, str) and ('deg' in lat or '°' in lat):
                            lat = _parse_dms_to_decimal(lat)
                        if isinstance(lon, str) and ('deg' in lon or '°' in lon):
                            lon = _parse_dms_to_decimal(lon)
                        
                        lat_float = float(lat)
                        lon_float = float(lon)
                        logger.info(f"📍 GPS найден через exiftool: {lat_float}, {lon_float}")
                        return (lat_float, lon_float)
        except FileNotFoundError:
            logger.debug("exiftool не установлен (это нормально)")
        except Exception as e:
            logger.debug(f"exiftool error: {e}")
        
        logger.info("📍 GPS координаты не найдены в метаданных видео")
        logger.info("💡 Совет: убедитесь что GPS был включен при съемке видео")
        return None
        
    except Exception as e:
        logger.warning(f"Ошибка извлечения GPS из видео: {e}")
        return None


def _parse_dms_to_decimal(dms_str: str) -> float:
    """Преобразование GPS из формата DMS (градусы минуты секунды) в десятичный"""
    import re
    # Формат: "55 deg 45' 20.88\" N" или "55°45'20.88\"N"
    match = re.match(r"(\d+)[^\d]+(\d+)[^\d]+(\d+\.?\d*)[^\d]*([NSEW])?", dms_str)
    if match:
        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        direction = match.group(4)
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # Если южное или западное направление - инвертируем
        if direction in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    return float(dms_str)  # Fallback


class VideoProcessor:
    """Класс для обработки видео с прогрессом"""
    
    def __init__(self, sign_model_path, infra_model_path, marking_model_path=None):
        # Оптимизация: используем GPU если доступен
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Используем устройство: {self.device}")
        
        # Загружаем модели с оптимизациями
        self.sign_model = YOLO(sign_model_path)
        logger.info(f"✅ Модель знаков загружена: {sign_model_path} (классы: {len(self.sign_model.names)})")
        self.infra_model = YOLO(infra_model_path)
        
        # Предвычисляем группы похожих классов знаков (один раз при загрузке)
        self._speed_classes = {i for i, nm in self.sign_model.names.items() if 'speed_over' in nm or 'speed_limit' in nm}
        self._overtake_classes = {i for i, nm in self.sign_model.names.items() if 'overtake' in nm}
        self._priority_classes = {i for i, nm in self.sign_model.names.items() if 'prio_' in nm}
        
        # Загружаем модель разметки если указана
        if marking_model_path and Path(marking_model_path).exists():
            self.marking_model = YOLO(marking_model_path)
            logger.info(f"✅ Модель разметки загружена: {marking_model_path}")
        else:
            self.marking_model = None
            logger.warning("⚠️  Модель разметки не найдена, используется детекция линий")
        
        # Оптимизации для снижения нагрузки
        if self.device == 'cuda':
            # Ограничиваем использование GPU памяти
            torch.cuda.set_per_process_memory_fraction(0.7)  # Используем максимум 70% GPU памяти
            self.sign_model.to(self.device)
            self.infra_model.to(self.device)
            if self.marking_model:
                self.marking_model.to(self.device)
        
        logger.info("✅ Модели загружены")
    
    def _enhance_frame_for_signs(self, frame):
        """Улучшение кадра для лучшей детекции знаков"""
        # Конвертируем в LAB для лучшей работы с освещением
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Применяем CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Объединяем обратно
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Дополнительное повышение резкости
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced_frame = cv2.filter2D(enhanced_frame, -1, kernel)
        
        return enhanced_frame
    
    def _merge_sign_detections(self, all_results):
        """Объединение результатов из нескольких инференсов, удаление пространственных дубликатов.
        Возвращает список dict: {'bbox', 'conf', 'cls'}
        """
        all_boxes = []
        for result in all_results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    all_boxes.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls_id})
        
        # NMS по расстоянию центров — сортируем по убыванию confidence
        unique_boxes = []
        for box in sorted(all_boxes, key=lambda b: -b['conf']):
            x1, y1, x2, y2 = box['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            is_dup = False
            for u in unique_boxes:
                ux1, uy1, ux2, uy2 = u['bbox']
                ucx, ucy = (ux1 + ux2) // 2, (uy1 + uy2) // 2
                if ((cx - ucx)**2 + (cy - ucy)**2)**0.5 < 50:
                    is_dup = True
                    break
            if not is_dup:
                unique_boxes.append(box)
        
        return unique_boxes
    
    def _same_sign_group(self, cls_a, cls_b):
        """Могут ли два класса быть одним и тем же физическим знаком (модель путает похожие классы)"""
        if cls_a == cls_b:
            return True
        if cls_a in self._speed_classes and cls_b in self._speed_classes:
            return True
        if cls_a in self._overtake_classes and cls_b in self._overtake_classes:
            return True
        if cls_a in self._priority_classes and cls_b in self._priority_classes:
            return True
        return False
    
    def _detect_signs_by_color_shape(self, frame):
        """Дополнительная детекция знаков по цвету и форме"""
        # Создаем фиктивный результат для совместимости с YOLO
        class FakeResult:
            def __init__(self, boxes_data):
                self.boxes = FakeBoxes(boxes_data) if boxes_data else None
        
        class FakeBoxes:
            def __init__(self, boxes_data):
                self.data = boxes_data
            
            def __iter__(self):
                return iter(self.data)
        
        class FakeBox:
            def __init__(self, x1, y1, x2, y2, conf, cls_id):
                self.xyxy = [torch.tensor([x1, y1, x2, y2])]
                self.conf = [torch.tensor(conf)]
                self.cls = [torch.tensor(cls_id)]
        
        detected_boxes = []
        
        # Конвертируем в HSV для лучшей работы с цветом
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Поиск красных знаков (запрещающие, предупреждающие)
        red_signs = self._find_red_signs(frame, hsv)
        detected_boxes.extend(red_signs)
        
        # 2. Поиск синих знаков (предписывающие)
        blue_signs = self._find_blue_signs(frame, hsv)
        detected_boxes.extend(blue_signs)
        
        # 3. Поиск желтых знаков (временные, предупреждающие)
        yellow_signs = self._find_yellow_signs(frame, hsv)
        detected_boxes.extend(yellow_signs)
        
        if detected_boxes:
            fake_boxes = [FakeBox(*box) for box in detected_boxes]
            return [FakeResult(fake_boxes)]
        else:
            return [FakeResult(None)]
    
    def _find_red_signs(self, frame, hsv):
        """Поиск красных дорожных знаков"""
        # Маска для красного цвета (два диапазона из-за цикличности HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        return self._find_signs_by_mask(red_mask, frame, 'red_sign', min_area=200)
    
    def _find_blue_signs(self, frame, hsv):
        """Поиск синих дорожных знаков"""
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        return self._find_signs_by_mask(blue_mask, frame, 'blue_sign', min_area=200)
    
    def _find_yellow_signs(self, frame, hsv):
        """Поиск желтых дорожных знаков"""
        yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255]))
        return self._find_signs_by_mask(yellow_mask, frame, 'yellow_sign', min_area=150)
    
    def _find_signs_by_mask(self, mask, frame, sign_type, min_area=200):
        """Поиск знаков по цветовой маске"""
        # Морфологические операции для очистки маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_signs = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)
            
            # Проверяем соотношение сторон (знаки обычно квадратные или круглые)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.7 <= aspect_ratio <= 1.4):  # Примерно квадратные
                continue
            
            # Проверяем заполненность контура
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            if bbox_area > 0 and contour_area / bbox_area < 0.3:  # Слишком пустой
                continue
            
            # Проверяем, что знак в верхней части кадра (где обычно располагаются знаки)
            frame_height = frame.shape[0]
            if y > frame_height * 0.7:  # Слишком низко
                continue
            
            # Добавляем детекцию
            confidence = min(0.8, 0.4 + (contour_area / bbox_area))  # Уверенность на основе заполненности
            cls_id = 0  # Общий класс для дорожного знака
            
            detected_signs.append((x, y, x + w, y + h, confidence, cls_id))
        
        return detected_signs
    
    async def process_video_with_progress(
        self, 
        video_path: str, 
        output_path: str,
        update_message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        conf_threshold: float = 0.1,  # Максимально низкий порог для поиска всех знаков
        detector_settings: dict = None  # Настройки детекторов
    ):
        """
        Обработка видео с отправкой прогресса в Telegram
        """
        # Настройки детекторов по умолчанию (все включены)
        if detector_settings is None:
            detector_settings = {
                'signs': True,
                'infrastructure': True,
                'barriers': True,
                'marking': True,
                'exits': True,
                'noise_strips': True
            }
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        # Нормализуем FPS в адекватный диапазон, чтобы не получать 3000 fps
        if fps < 5 or fps > 120:
            logger.warning(f"FPS некорректный ({fps}), устанавливаем 30")
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if width == 0 or height == 0:
            raise ValueError(f"Некорректные размеры видео: {width}x{height}")
        
        # Оптимизация: проверяем длительность видео для предотвращения перегрева
        duration_seconds = total_frames / fps if fps > 0 else 0
        duration_minutes = duration_seconds / 60
        
        if duration_minutes > 15:
            cap.release()
            raise ValueError(
                f"Видео слишком длинное: {duration_minutes:.1f} минут\n\n"
                f"💡 Для предотвращения перегрева ПК ограничение: 15 минут\n\n"
                f"📹 Рекомендации:\n"
                f"• Разделите видео на части по 5-10 минут\n"
                f"• Выберите наиболее интересные фрагменты\n"
                f"• Используйте видео меньшей длительности"
            )
        
        # БЕЗ МАСШТАБИРОВАНИЯ - используем оригинальное разрешение для максимального качества!
        scale = None
        
        # КРИТИЧЕСКИ ВАЖНО: делаем размеры четными для совместимости с кодеками!
        # Многие кодеки (особенно на macOS) требуют четные размеры
        if width % 2 != 0:
            width = (width // 2) * 2  # Округляем вниз до четного
        if height % 2 != 0:
            height = (height // 2) * 2  # Округляем вниз до четного
        
        logger.info(f"Используем оригинальное разрешение: {width}x{height}")
        
        # Инициализируем GPS трекер
        gps_tracker = None
        if GPS_AVAILABLE:
            try:
                # Получаем сохраненные GPS координаты пользователя (если есть)
                start_coords = context.user_data.get('gps_coords', None)
                
                if start_coords:
                    logger.info(f"📍 Используем GPS координаты пользователя: {start_coords}")
                    gps_tracker = GPSTracker(video_path, start_coords=start_coords, speed_kmh=50.0)
                else:
                    logger.info("📍 Используем автоматическое определение GPS")
                    gps_tracker = GPSTracker(video_path, speed_kmh=50.0)
                
                gps_tracker.setup_video(cap)
                logger.info("✅ GPS трекер инициализирован")
            except Exception as e:
                logger.warning(f"⚠️  Не удалось инициализировать GPS трекер: {e}")
                gps_tracker = None
        
        # Инициализируем пикетаж трекер
        picket_tracker = None
        if PICKET_AVAILABLE:
            try:
                picket_tracker = PicketTracker(video_path=str(video_path), speed_kmh=60.0, start_picket_m=0.0)
                picket_tracker.setup_video(cap)
                logger.info("✅ Пикетаж трекер инициализирован")
            except Exception as e:
                logger.warning(f"⚠️  Не удалось инициализировать пикетаж трекер: {e}")
                picket_tracker = None
        
        # НОВЫЙ ПОДХОД: Сохраняем кадры как PNG, потом собираем через ffmpeg
        # Это 100% надежный метод, который работает везде
        frames_dir = Path(output_path).parent / f"frames_{Path(output_path).stem}"
        frames_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Сохраняем кадры в: {frames_dir}")
        logger.info(f"Параметры видео: fps={fps}, size={width}x{height}")
        
        # Список для сохранения путей к кадрам
        saved_frames = []
        output_path = str(Path(output_path).with_suffix('.mp4'))  # Финальный MP4
        
        frame_num = 0
        last_progress = 0
        
        # Оптимизации для снижения нагрузки на CPU/GPU
        FRAME_SKIP = max(1, fps // 20)  # ОБРАБАТЫВАЕМ ПРАКТИЧЕСКИ КАЖДЫЙ КАДР (Максимальная точность)
        processed_frames = 0  # Счетчик фактически обработанных кадров
        
        stats = {
            'signs': 0,           # Уникальные объекты (знаки)
            'sign_frames': 0,     # Кадры с знаками
            'infra': 0,           # Уникальные объекты (инфраструктура)
            'infra_frames': 0,    # Кадры с инфраструктурой
            'marking_frames': 0,  # Кадры с разметкой
            'barriers': 0,        # Уникальные объекты (барьеры)
            'barrier_frames': 0,  # Кадры с барьерами
            'exits': 0,           # Уникальные объекты (съезды)
            'exit_frames': 0,     # Кадры со съездами
            'noise_strips': 0,    # Уникальные объекты (шумовые полосы)
            'noise_strip_frames': 0  # Кадры с шумовыми полосами
        }

        # Настройки для более точного подсчета объектов
        # Шумовые полосы (реальных полос обычно 2-5 на видео)
        NOISE_STRIP_CONF = 0.35
        NOISE_STRIP_MATCH_DISTANCE = 250
        NOISE_STRIP_DUPLICATE_DISTANCE = 120
        NOISE_STRIP_PICKET_MATCH_METERS = 25
        # Инфраструктура
        INFRA_MATCH_DISTANCE = 150
        INFRA_DUPLICATE_DISTANCE = 60
        INFRA_PICKET_MATCH_METERS = 8
        # Барьеры
        BARRIER_MATCH_DISTANCE = 180
        BARRIER_DUPLICATE_DISTANCE = 80
        BARRIER_PICKET_MATCH_METERS = 10
        # Съезды
        EXIT_MATCH_DISTANCE = 200
        EXIT_DUPLICATE_DISTANCE = 100
        EXIT_PICKET_MATCH_METERS = 15
        
        # Трекинг уникальных объектов для точного подсчета
        tracked_objects = {
            'signs': [],
            'infra': [],
            'barriers': [],
            'exits': [],
            'noise_strips': []
        }
        
        # Список всех детекций для Excel экспорта
        all_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # БЕЗ RESIZE - используем оригинальное разрешение!
            
            frame_num += 1
            
            # Пропускаем кадры для снижения нагрузки (обрабатываем каждый FRAME_SKIP кадр)
            if frame_num % FRAME_SKIP != 0:
                # Сохраняем кадр БЕЗ изменений - оригинальное качество!
                frame = np.ascontiguousarray(frame)
                
                # Сохраняем кадр как PNG с максимальным качеством
                frame_path = frames_dir / f"frame_{frame_num:06d}.png"
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                saved_frames.append(frame_path)
                continue
            
            processed_frames += 1
            
            # Получаем текущий пикетаж для видео
            current_picket_text = ""
            if picket_tracker:
                p_val = picket_tracker.get_frame_picket(frame_num)
                current_picket_text = picket_tracker.format_picket(p_val)
                
                # Рисуем текущий пикетаж в левом верхнем углу
                overlay_text = f"PICKET: {current_picket_text}"
                (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (10, 10), (20 + tw, 40 + th), (0, 0, 0), -1)
                cv2.putText(frame, overlay_text, (15, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Детекция знаков - модель russian_signs_yolo12m (63 класса РФ)
            frame_signs_count = 0
            if detector_settings.get('signs', True):
                # Максимальное разрешение и аугментация (TTA) для выявления всех знаков 100%
                sign_results = self.sign_model(frame, conf=0.08, imgsz=1536, augment=True, verbose=False)
                unique_boxes = self._merge_sign_detections(list(sign_results))
                if frame_num % 30 == 0 and unique_boxes:
                    logger.info(f"🔍 Кадр {frame_num}: {len(unique_boxes)} детекций знаков")
            else:
                unique_boxes = []
            
            for box_det in unique_boxes:
                x1, y1, x2, y2 = box_det['bbox']
                conf = box_det['conf']
                cls_id = box_det['cls']
                class_name = self.sign_model.names[cls_id]
                
                if conf < 0.12:
                    continue
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Пикетаж
                picket = None
                picket_label_part = ""
                if picket_tracker:
                    try:
                        picket = picket_tracker.get_object_picket((x1, y1, x2, y2), frame_num)
                        picket_label_part = f" ({int(picket)}m)"
                    except Exception as e:
                        logger.debug(f"Ошибка пикетажа знака: {e}")
                
                # Дедупликация между кадрами
                is_new_object = True
                for tracked in tracked_objects['signs']:
                    # Проверка на таймаут трекинга (если не обновлялся больше 1 секунды - это другой знак)
                    if frame_num - tracked['frame'] > fps * 1.0:
                        continue
                    
                    if not self._same_sign_group(tracked.get('cls'), cls_id):
                        continue
                    tracked_picket = tracked.get('picket')
                    if picket is not None and tracked_picket is not None:
                        if abs(picket - tracked_picket) <= 15:
                            is_new_object = False
                            tracked['frame'] = frame_num
                            tracked['center'] = center
                            tracked['picket'] = picket
                            if conf > tracked.get('best_conf', 0):
                                tracked['cls'] = cls_id
                                tracked['best_conf'] = conf
                                tracked['best_name'] = class_name
                                if EXCEL_AVAILABLE and 'detection_idx' in tracked and tracked['detection_idx'] != -1:
                                    idx = tracked['detection_idx']
                                    if idx < len(all_detections):
                                        all_detections[idx]['class_name'] = class_name
                                        all_detections[idx]['confidence'] = conf
                            break
                    else:
                        tx, ty = tracked['center']
                        if ((center[0]-tx)**2 + (center[1]-ty)**2)**0.5 < 150:
                            is_new_object = False
                            tracked['frame'] = frame_num
                            tracked['center'] = center
                            if picket is not None:
                                tracked['picket'] = picket
                            if conf > tracked.get('best_conf', 0):
                                tracked['cls'] = cls_id
                                tracked['best_conf'] = conf
                                tracked['best_name'] = class_name
                                if EXCEL_AVAILABLE and 'detection_idx' in tracked and tracked['detection_idx'] != -1:
                                    idx = tracked['detection_idx']
                                    if idx < len(all_detections):
                                        all_detections[idx]['class_name'] = class_name
                                        all_detections[idx]['confidence'] = conf
                            break
                
                if is_new_object:
                    detection_idx = len(all_detections) if EXCEL_AVAILABLE else -1
                    tracked_objects['signs'].append({
                        'center': center, 'frame': frame_num, 'cls': cls_id,
                        'picket': picket, 'best_conf': conf, 'best_name': class_name,
                        'detection_idx': detection_idx
                    })
                    stats['signs'] += 1
                    logger.info(f"✅ Новый знак #{stats['signs']}: {class_name} conf={conf:.2f} center={center} picket={picket}")
                    # Excel — сохраняем ТОЛЬКО новые уникальные знаки
                    if EXCEL_AVAILABLE:
                        timestamp = frame_num / fps if fps > 0 else None
                        coordinates = None
                        if gps_tracker:
                            try:
                                coordinates = gps_tracker.get_object_coordinates((x1, y1, x2, y2), frame_num)
                            except Exception:
                                pass
                        all_detections.append(format_detections_for_excel(
                            frame_number=frame_num,
                            object_type='sign',
                            class_name=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            coordinates=coordinates,
                            picket=picket
                        ))
                
                # Рисуем рамку на каждом кадре где знак виден
                color = COLORS['sign']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                label = f"SIGN: {class_name}{picket_label_part}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
                frame_signs_count += 1
            
            if frame_signs_count > 0:
                stats['sign_frames'] += 1
            
            # Детекция инфраструктуры с МАКСИМАЛЬНЫМ порогом для 100% точности
            if detector_settings.get('infrastructure', True):
                infra_results = self.infra_model(frame, classes=list(ROAD_INFRA_CLASSES.values()), 
                                                conf=0.60, augment=True, verbose=False)  # Снижен порог для захвата всего
            else:
                infra_results = []
            frame_infra_count = 0
            
            # Логирование инфраструктуры
            if frame_num % 30 == 1:
                infra_total = sum(len(r.boxes) if hasattr(r, 'boxes') and r.boxes is not None else 0 for r in infra_results)
                logger.info(f"🏗️  Кадр {frame_num}: найдено {infra_total} объектов инфраструктуры")
            
            frame_infra_centers = []
            for result in infra_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # МАКСИМАЛЬНАЯ чувствительность
                    if conf < 0.60:
                        continue
                    class_name = self.infra_model.names[cls_id]
                    
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    # Дедупликация в рамках одного кадра
                    is_dup = False
                    for c in frame_infra_centers:
                        if ((center[0]-c[0])**2 + (center[1]-c[1])**2)**0.5 < INFRA_DUPLICATE_DISTANCE:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    frame_infra_centers.append(center)
                    
                    # Вычисляем пикетаж заранее для матчинга
                    picket = None
                    if picket_tracker:
                        try:
                            picket = picket_tracker.get_object_picket((x1, y1, x2, y2), frame_num)
                        except Exception as e:
                            logger.debug(f"Ошибка вычисления пикетажа для инфраструктуры: {e}")
                    
                    # Трекинг уникальных объектов инфраструктуры (по пикетажу + классу)
                    is_new_object = True
                    for tracked in tracked_objects['infra']:
                        if frame_num - tracked['frame'] > fps * 2:
                            continue
                        # Сначала проверяем совпадение класса
                        if tracked.get('cls') != cls_id:
                            continue
                        tracked_picket = tracked.get('picket')
                        if picket is not None and tracked_picket is not None:
                            if abs(picket - tracked_picket) <= INFRA_PICKET_MATCH_METERS:
                                is_new_object = False
                                tracked['center'] = center
                                tracked['frame'] = frame_num
                                tracked['picket'] = picket
                                break
                        else:
                            tx, ty = tracked['center']
                            distance = ((center[0] - tx)**2 + (center[1] - ty)**2)**0.5
                            if distance < INFRA_MATCH_DISTANCE:
                                is_new_object = False
                                tracked['center'] = center
                                tracked['frame'] = frame_num
                                if picket is not None:
                                    tracked['picket'] = picket
                                break
                    
                    # Правильный выбор цвета по типу объекта
                    class_key = class_name.lower().replace(' ', '_')
                    if 'bus' in class_key:
                        color = COLORS['bus']  # Желтый
                        label_text = f"BUS/STOP"
                    elif 'car' in class_key:
                        color = (255, 100, 0)  # Оранжево-синий для машин
                        label_text = f"CAR"
                    elif 'truck' in class_key:
                        color = COLORS['truck']  # Оранжевый
                        label_text = f"TRUCK"
                    elif 'traffic' in class_key and 'light' in class_key:
                        color = COLORS['traffic_light']  # Голубой
                        label_text = f"TRAFFIC LIGHT"
                    elif 'stop' in class_key and 'sign' in class_key:
                        color = COLORS['stop_sign']  # Красный
                        label_text = f"STOP SIGN"
                    elif 'person' in class_key:
                        color = COLORS['person']  # Серый
                        label_text = f"PERSON"
                    elif 'motorcycle' in class_key:
                        color = (0, 200, 255)  # Желто-голубой
                        label_text = f"MOTORCYCLE"
                    elif 'bicycle' in class_key:
                        color = COLORS['bicycle']  # Фиолетовый
                        label_text = f"BICYCLE"
                    else:
                        color = COLORS['default']  # Розовый для остального
                        label_text = f"{class_name.upper()}"

                    if is_new_object:
                        tracked_objects['infra'].append({'center': center, 'frame': frame_num, 'cls': cls_id, 'picket': picket})
                        stats['infra'] += 1  # Только для новых уникальных объектов
                        
                        # Вычисляем GPS координаты для объекта инфраструктуры
                        coordinates = None
                        if gps_tracker:
                            try:
                                coordinates = gps_tracker.get_object_coordinates((x1, y1, x2, y2), frame_num)
                            except Exception as e:
                                logger.debug(f"Ошибка вычисления GPS для инфраструктуры: {e}")
                                
                        # Сохраняем детекцию для Excel
                        if EXCEL_AVAILABLE:
                            timestamp = frame_num / fps if fps > 0 else None
                            all_detections.append(format_detections_for_excel(
                                frame_number=frame_num,
                                object_type=class_key.replace(' ', '_'),
                                class_name=class_name,
                                confidence=conf,
                                bbox=(x1, y1, x2, y2),
                                timestamp=timestamp,
                                coordinates=coordinates,
                                picket=picket
                            ))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                    
                    # Подпись для объекта инфраструктуры - КРУПНЫЙ текст
                    label = f"{label_text} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                    
                    frame_infra_count += 1
            
            if frame_infra_count > 0:
                stats['infra_frames'] += 1
            
            # Детекция дорожных барьеров/ограждений с МАКСИМАЛЬНЫМ порогом для 100% точности
            if detector_settings.get('barriers', True):
                barriers = detect_barriers_combined(frame, self.infra_model, conf_threshold=0.60)
            else:
                barriers = []
            frame_barriers_count = 0
            frame_barrier_centers = []
            for barrier in barriers:
                x1, y1, x2, y2 = barrier['bbox']
                conf = barrier.get('confidence', 0.85)
                
                # МАКСИМАЛЬНО МЯГКАЯ фильтрация
                if conf < 0.65:
                    continue
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Дедупликация в рамках одного кадра
                is_dup = False
                for c in frame_barrier_centers:
                    if ((center[0]-c[0])**2 + (center[1]-c[1])**2)**0.5 < BARRIER_DUPLICATE_DISTANCE:
                        is_dup = True
                        break
                if is_dup:
                    continue
                frame_barrier_centers.append(center)
                
                # Вычисляем пикетаж заранее для матчинга
                picket = None
                picket_label_part = ""
                if picket_tracker:
                    try:
                        picket = picket_tracker.get_object_picket((x1, y1, x2, y2), frame_num)
                        picket_label_part = f" ({int(picket)}m)"
                    except Exception as e:
                        logger.debug(f"Ошибка вычисления пикетажа для барьера: {e}")
                
                # Трекинг уникальных барьеров (по пикетажу при наличии)
                is_new_object = True
                for tracked in tracked_objects['barriers']:
                    if frame_num - tracked['frame'] > fps * 2:
                        continue
                    tracked_picket = tracked.get('picket')
                    if picket is not None and tracked_picket is not None:
                        if abs(picket - tracked_picket) <= BARRIER_PICKET_MATCH_METERS:
                            is_new_object = False
                            tracked['center'] = center
                            tracked['frame'] = frame_num
                            tracked['picket'] = picket
                            break
                    else:
                        tx, ty = tracked['center']
                        distance = ((center[0] - tx)**2 + (center[1] - ty)**2)**0.5
                        if distance < BARRIER_MATCH_DISTANCE:
                            is_new_object = False
                            tracked['center'] = center
                            tracked['frame'] = frame_num
                            if picket is not None:
                                tracked['picket'] = picket
                            break
                
                if is_new_object:
                    tracked_objects['barriers'].append({'center': center, 'frame': frame_num, 'picket': picket})
                    stats['barriers'] += 1  # Только для новых уникальных объектов
                    
                    # Вычисляем GPS координаты для барьера
                    coordinates = None
                    if gps_tracker:
                        try:
                            coordinates = gps_tracker.get_object_coordinates((x1, y1, x2, y2), frame_num)
                        except Exception as e:
                            logger.debug(f"Ошибка вычисления GPS для барьера: {e}")
                    
                    # Сохраняем детекцию барьера для Excel
                    if EXCEL_AVAILABLE:
                        timestamp = frame_num / fps if fps > 0 else None
                        all_detections.append(format_detections_for_excel(
                            frame_number=frame_num,
                            object_type='barrier',
                            class_name=barrier.get('class', 'barrier'),
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            coordinates=coordinates,
                            picket=picket
                        ))
                
                color = (0, 165, 255)  # Оранжевый для барьеров
                
                # ТОЛСТАЯ рамка
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                
                # Подпись для барьера - КРУПНЫЙ текст
                label = f"BARRIER {conf:.2f}{picket_label_part}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

                
                frame_barriers_count += 1
            
            if frame_barriers_count > 0:
                stats['barrier_frames'] += 1
            
            # ОТКЛЮЧЕНО: Детекция П-образных опор (gantry)
            # gantries = detect_gantries_and_structures(frame)
            
            # Детекция разметки - метод StreamHandler (Serzho)
            if detector_settings.get('marking', True):
                # ROI обработка + анализ сплошности линий + детекция направления
                height, width = frame.shape[:2]
            roi_y_start = height // 2
            roi_y_end = round((height // 4) * 2.8)
            roi_frame = frame[roi_y_start:roi_y_end, 0:width]
            
            # Обработка ROI (метод Serzho)
            # Шаг 1: Медианное размытие
            proc_frame = cv2.medianBlur(roi_frame, 5)
            
            # Шаг 2: Пороговое преобразование (белая разметка 210+)
            _, proc_frame = cv2.threshold(proc_frame, 210, 255, 0)
            
            # Шаг 3: Grayscale
            proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            
            # Шаг 4: Гауссовское размытие
            proc_frame = cv2.GaussianBlur(proc_frame, (7, 7), 1.5)
            
            # Шаг 5: Canny
            proc_frame = cv2.Canny(proc_frame, 1, 50)
            
            # Рисуем ROI прямоугольник
            cv2.rectangle(frame, (0, roi_y_start), (width, roi_y_end), (0, 0, 255), 2)
            
            # Рисуем зоны анализа (левая и правая полосы)
            left_zone_start = int(width * 0.3)
            left_zone_end = int(width * 0.5)
            right_zone_start = int(width * 0.75)
            right_zone_end = int(width * 0.95)
            
            cv2.rectangle(frame, (left_zone_start, roi_y_start), (left_zone_end, roi_y_end), (0, 255, 255), 2)
            cv2.rectangle(frame, (right_zone_start, roi_y_start), (right_zone_end, roi_y_end), (0, 255, 255), 2)
            
            # Детекция контуров
            contours, hierarchy = cv2.findContours(proc_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            # Рисуем контуры на исходном кадре (синим цветом)
            cv2.drawContours(
                frame[roi_y_start:roi_y_end, 0:width],
                contours, -1, (255, 0, 0), 5, cv2.LINE_AA, hierarchy, 1
            )
            
            # Анализ сплошности линий (метод Serzho)
            sum_left_zone = 0
            sum_right_zone = 0
            for contour in contours:
                for dot_packed in contour:
                    x = int(dot_packed[0][0])
                    y = int(dot_packed[0][1])
                    
                    # Проверка попадания в зоны
                    if left_zone_start <= x <= left_zone_end:
                        sum_left_zone += 1
                    elif right_zone_start <= x <= right_zone_end:
                        sum_right_zone += 1
            
            # Определение типа линий (сплошная > 1000, иначе пунктирная)
            left_type = "Solid" if sum_left_zone > 1000 else "Intermittent"
            right_type = "Solid" if sum_right_zone > 1000 else "Intermittent"
            
            # Отображение информации
            cv2.putText(frame, f"Left: {left_type} ({sum_left_zone})", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Right: {right_type} ({sum_right_zone})", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if contours:
                stats['marking_frames'] += 1
                
                if frame_num % 50 == 1:
                    logger.info(f"Кадр {frame_num}: {len(contours)} контуров, "
                              f"Left={left_type}({sum_left_zone}), Right={right_type}({sum_right_zone})")
            elif frame_num % 50 == 1:
                logger.info(f"Кадр {frame_num}: контуры не найдены")
                # СТРОГАЯ фильтрация - только вертикальные длинные линии
                filtered_lines = []
            
            # Детекция съездов с МАКСИМАЛЬНО СТРОГИМ порогом (только 100% уверенные)
            if detector_settings.get('exits', True):
                exits = detect_road_exits(frame)
            else:
                exits = []
            frame_exits_count = 0
            frame_exit_centers = []
            for exit_obj in exits:
                x1, y1, x2, y2 = exit_obj['bbox']
                conf = exit_obj.get('confidence', 0.7)
                
                # МАКСИМАЛЬНО МЯГКАЯ фильтрация
                if conf < 0.70:
                    continue
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Дедупликация в рамках одного кадра
                is_dup = False
                for c in frame_exit_centers:
                    if ((center[0]-c[0])**2 + (center[1]-c[1])**2)**0.5 < EXIT_DUPLICATE_DISTANCE:
                        is_dup = True
                        break
                if is_dup:
                    continue
                frame_exit_centers.append(center)
                
                # Вычисляем пикетаж для съезда
                picket = None
                picket_label_part = ""
                if picket_tracker:
                    try:
                        picket = picket_tracker.get_object_picket((x1, y1, x2, y2), frame_num)
                        picket_label_part = f" ({int(picket)}m)"
                    except Exception as e:
                        logger.debug(f"Ошибка вычисления пикетажа для съезда: {e}")
                
                # Трекинг уникальных съездов (по пикетажу при наличии)
                is_new_object = True
                for tracked in tracked_objects['exits']:
                    if frame_num - tracked['frame'] > fps * 2:
                        continue
                    tracked_picket = tracked.get('picket')
                    if picket is not None and tracked_picket is not None:
                        if abs(picket - tracked_picket) <= EXIT_PICKET_MATCH_METERS:
                            is_new_object = False
                            tracked['center'] = center
                            tracked['frame'] = frame_num
                            tracked['picket'] = picket
                            break
                    else:
                        tx, ty = tracked['center']
                        distance = ((center[0] - tx)**2 + (center[1] - ty)**2)**0.5
                        if distance < EXIT_MATCH_DISTANCE:
                            is_new_object = False
                            tracked['center'] = center
                            tracked['frame'] = frame_num
                            if picket is not None:
                                tracked['picket'] = picket
                            break
                
                color = (255, 0, 255)  # Фиолетовый для съездов
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Подпись для съезда
                label = f"EXIT {conf:.2f}{picket_label_part}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if is_new_object:
                    tracked_objects['exits'].append({'center': center, 'frame': frame_num, 'picket': picket})
                    stats['exits'] += 1  # Только для новых уникальных объектов
                    
                    # Вычисляем GPS координаты для съезда
                    coordinates = None
                    if gps_tracker:
                        try:
                            coordinates = gps_tracker.get_object_coordinates((x1, y1, x2, y2), frame_num)
                        except Exception as e:
                            logger.debug(f"Ошибка вычисления GPS для съезда: {e}")
                    
                    # Сохраняем детекцию съезда для Excel
                    if EXCEL_AVAILABLE:
                        timestamp = frame_num / fps if fps > 0 else None
                        all_detections.append(format_detections_for_excel(
                            frame_number=frame_num,
                            object_type='exit',
                            class_name=exit_obj.get('description', 'road_exit'),
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            coordinates=coordinates,
                            picket=picket
                        ))
                
                frame_exits_count += 1
            
            if frame_exits_count > 0:
                stats['exit_frames'] += 1
            
            # Детекция шумовых полос
            if detector_settings.get('noise_strips', True):
                noise_strips = detect_noise_strips(frame)
            else:
                noise_strips = []
            
            # Логирование для отладки (каждые 30 кадров)
            if frame_num % 30 == 1:
                logger.info(f"🔊 Кадр {frame_num}: найдено {len(noise_strips)} шумовых полос")
            
            frame_noise_count = 0
            frame_noise_centers = []
            for strip in noise_strips:
                x1, y1, x2, y2 = strip['bbox']
                conf = strip.get('confidence', 0.7)
                
                # Фильтрация - порог 40% для детекции шумовых полос
                if conf < NOISE_STRIP_CONF:
                    continue
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                # Убираем дубликаты в рамках одного кадра
                is_duplicate_in_frame = False
                for c in frame_noise_centers:
                    d = ((center[0] - c[0])**2 + (center[1] - c[1])**2) ** 0.5
                    if d < NOISE_STRIP_DUPLICATE_DISTANCE:
                        is_duplicate_in_frame = True
                        break
                if is_duplicate_in_frame:
                    continue
                frame_noise_centers.append(center)
                
                # Вычисляем пикетаж для шумовой полосы
                picket = None
                picket_label_part = ""
                if picket_tracker:
                    try:
                        picket = picket_tracker.get_object_picket((x1, y1, x2, y2), frame_num)
                        picket_label_part = f" ({int(picket)}m)"
                    except Exception as e:
                        logger.debug(f"Ошибка вычисления пикетажа для шумовой полосы: {e}")
                
                # Трекинг уникальных шумовых полос (по пикетажу при наличии)
                is_new_object = True
                for tracked in tracked_objects['noise_strips']:
                    if frame_num - tracked['frame'] > fps * 2:
                        continue
                    tracked_picket = tracked.get('picket')
                    if picket is not None and tracked_picket is not None:
                        if abs(picket - tracked_picket) <= NOISE_STRIP_PICKET_MATCH_METERS:
                            is_new_object = False
                            tracked['center'] = center
                            tracked['frame'] = frame_num
                            tracked['picket'] = picket
                            break
                    else:
                        tx, ty = tracked['center']
                        distance = ((center[0] - tx)**2 + (center[1] - ty)**2)**0.5
                        if distance < NOISE_STRIP_MATCH_DISTANCE:
                            is_new_object = False
                            tracked['center'] = center
                            tracked['frame'] = frame_num
                            if picket is not None:
                                tracked['picket'] = picket
                            break
                
                color = COLORS['noise_strip']  # Золотистый для шумовых полос
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                
                # Подпись для шумовой полосы - КРУПНЫЙ текст
                label = f"NOISE STRIP {conf:.2f}{picket_label_part}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
                
                if is_new_object:
                    tracked_objects['noise_strips'].append({'center': center, 'frame': frame_num, 'picket': picket})
                    stats['noise_strips'] += 1  # Только для новых уникальных объектов
                    
                    # Вычисляем GPS координаты для шумовой полосы
                    coordinates = None
                    if gps_tracker:
                        try:
                            coordinates = gps_tracker.get_object_coordinates((x1, y1, x2, y2), frame_num)
                        except Exception as e:
                            logger.debug(f"Ошибка вычисления GPS для шумовой полосы: {e}")
                    
                    # Сохраняем детекцию шумовой полосы для Excel
                    if EXCEL_AVAILABLE:
                        timestamp = frame_num / fps if fps > 0 else None
                        all_detections.append(format_detections_for_excel(
                            frame_number=frame_num,
                            object_type='noise_strip',
                            class_name=strip.get('description', 'noise_strip'),
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            coordinates=coordinates,
                            picket=picket
                        ))
                
                frame_noise_count += 1
            
            if frame_noise_count > 0:
                stats['noise_strip_frames'] += 1
            
            # Сохраняем обработанный кадр БЕЗ resize - оригинальное разрешение!
            
            # Убедимся что кадр в правильном формате BGR uint8
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            frame = np.ascontiguousarray(frame)
            
            # Сохраняем кадр как PNG с МАКСИМАЛЬНЫМ качеством (без сжатия)
            frame_path = frames_dir / f"frame_{frame_num:06d}.png"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 = без сжатия
            saved_frames.append(frame_path)
            
            if frame_num % 100 == 1:
                logger.debug(f"Кадр {frame_num} сохранен: {frame_path.name}")
            
            # Обновление прогресса каждые 5% или каждые 50 кадров
            progress = int((frame_num / total_frames) * 100)
            if progress > last_progress + 5 or frame_num % 50 == 0:
                last_progress = progress
                try:
                    await update_message.edit_text(
                        f"⏳ Обработка видео...\n\n"
                        f"📊 Прогресс: {progress}%\n"
                        f"🎬 Кадров: {frame_num}/{total_frames} (обработано: {processed_frames})\n\n"
                        f"🚦 Знаков: {stats['signs']} (кадров: {stats['sign_frames']})\n"
                        f"🏗️  Инфраструктуры: {stats['infra']} (кадров: {stats['infra_frames']})\n"
                        f"🚧 Барьеров: {stats['barriers']} (кадров: {stats['barrier_frames']})\n"
                        f"🛣️  Разметка: {stats['marking_frames']} кадров\n"
                        f"🚪 Съездов: {stats['exits']} (кадров: {stats['exit_frames']})\n"
                        f"🔊 Шумовых полос: {stats['noise_strips']} (кадров: {stats['noise_strip_frames']})",
                        parse_mode=ParseMode.HTML
                    )
                except Exception as e:
                    logger.warning(f"Не удалось обновить сообщение: {e}")
            
            # Оптимизация: паузы для охлаждения CPU каждые 5 обработанных кадров
            if processed_frames % 5 == 0:
                await asyncio.sleep(0.15)  # 150ms пауза для охлаждения и снижения нагрузки
            
            # Даем возможность другим задачам выполняться
            if frame_num % 100 == 0:
                await asyncio.sleep(0.01)
        
        # Закрываем камеру
        logger.info(f"Завершение обработки: обработано {frame_num} кадров, сохранено {len(saved_frames)} кадров")
        cap.release()
        
        # Собираем видео из кадров через ffmpeg
        logger.info(f"Сборка видео из {len(saved_frames)} кадров...")
        
        try:
            import subprocess
            import shlex
            import shutil as _shutil
            
            # Ищем ffmpeg в системе (Windows + Mac/Linux)
            _ffmpeg_bin = _shutil.which('ffmpeg')
            if not _ffmpeg_bin:
                _candidates = [
                    r'C:\ffmpeg\bin\ffmpeg.exe',
                    r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                    r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
                    r'C:\tools\ffmpeg\bin\ffmpeg.exe',
                    '/opt/homebrew/bin/ffmpeg',
                    '/usr/local/bin/ffmpeg',
                    '/usr/bin/ffmpeg',
                ]
                for _c in _candidates:
                    if Path(_c).exists():
                        _ffmpeg_bin = _c
                        break
            if not _ffmpeg_bin:
                raise FileNotFoundError("ffmpeg not found")
            
            # Безопасные пути (экранируем пробелы)
            frames_pattern = str(frames_dir / 'frame_%06d.png')
            output_file = str(output_path)
            
            # Используем ffmpeg для создания видео из кадров с ВЫСОКИМ качеством
            ffmpeg_cmd = [
                _ffmpeg_bin, '-y',  # Перезаписывать выходной файл
                '-framerate', str(fps),  # FPS видео
                '-i', frames_pattern,  # Шаблон имен кадров
                '-c:v', 'libx264',  # Видео кодек H.264
                '-preset', 'medium',  # Средняя скорость/качество (вместо slow)
                '-crf', '23',  # Нормальное качество (вместо 17)
                '-pix_fmt', 'yuv420p',  # Формат пикселей для совместимости
                '-movflags', '+faststart',  # Оптимизация для стриминга
                output_file  # Используем безопасный путь
            ]
            
            logger.info(f"Запуск ffmpeg: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 минут (вместо 10)
                cwd=str(frames_dir.parent),  # Рабочая директория
                stdin=subprocess.DEVNULL   # Чтобы ffmpeg не читал stdin (избегаем SIGTTOU)
            )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                raise ValueError(f"Не удалось создать видео через ffmpeg: {result.stderr[:500]}")
            
            # Проверяем что видео создано и не пустое
            if not Path(output_path).exists():
                raise ValueError(f"Выходной файл не создан: {output_path}")
            
            file_size = Path(output_path).stat().st_size
            logger.info(f"✅ Видео создано успешно: {file_size / 1024 / 1024:.2f} MB")
            
            if file_size < 100 * 1024:  # Меньше 100KB - подозрительно мало
                logger.warning(f"Выходной файл слишком мал: {file_size} bytes")
                raise ValueError(f"Выходное видео слишком маленькое: {file_size} bytes")
            
            # Удаляем временные кадры для экономии места
            logger.info("Очистка временных кадров...")
            try:
                import shutil
                shutil.rmtree(frames_dir)
                logger.info(f"Временные кадры удалены: {frames_dir}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временные кадры: {e}")
            
            return stats, all_detections, output_path
                
        except FileNotFoundError:
            logger.error("ffmpeg не найден в системе!")
            raise ValueError(
                "ffmpeg не установлен или не найден.\n"
                "Windows: скачайте ZIP с https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip\n"
                "Распакуйте в C:\\ffmpeg (чтобы был C:\\ffmpeg\\bin\\ffmpeg.exe).\n"
                "Mac: brew install ffmpeg"
            )
        except subprocess.TimeoutExpired:
            logger.error("Таймаут при создании видео через ffmpeg")
            raise ValueError("Таймаут при сборке видео")
        except Exception as e:
            logger.error(f"Ошибка при создании видео: {e}")
            raise


# Глобальный процессор
processor = None

def get_processor():
    """Получить или создать процессор"""
    global processor
    if processor is None:
        processor = VideoProcessor(SIGN_MODEL_PATH, INFRA_MODEL_PATH, MARKING_MODEL_PATH)
    return processor


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = (
        "👋 <b>Добро пожаловать в бот обработки дорожной инфраструктуры!</b>\n\n"
        "📹 Отправьте мне видео для обработки, и я применю все модели детекции:\n\n"
        "🚦 Дорожные знаки (155 классов RTSD)\n"
        "🏗️  Инфраструктура (автобусы, машины, светофоры)\n"
        "🛣️  П-образные опоры\n"
        "📏 Дорожная разметка\n"
        "🗺️  Ситуационный план дороги\n\n"
        "📊 <b>Команды:</b>\n"
        "/start - Показать это сообщение\n"
        "/gps - Показать GPS координаты из видео на Яндекс.Картах\n"
        "/plan - Создать ситуационный план (после обработки видео)\n\n"
        "⏳ Во время обработки я буду показывать прогресс в процентах!\n\n"
        "📤 Просто отправьте видео файл ⬆️"
    )
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.HTML)


async def set_gps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /gps - показ GPS координат из видео на Яндекс.Картах"""
    message = update.message
    
    # Проверяем, есть ли сохраненные GPS координаты из последнего видео
    if 'video_gps_coords' not in context.user_data:
        await message.reply_text(
            "❌ <b>GPS координаты не найдены</b>\n\n"
            "📹 Отправьте видео с GPS метаданными для автоматического определения координат\n\n"
            "💡 Большинство видео со смартфонов содержат GPS координаты",
            parse_mode=ParseMode.HTML
        )
        return
    
    latitude, longitude = context.user_data['video_gps_coords']
    
    # Создаем ссылку на Яндекс.Карты
    yandex_maps_url = f"https://yandex.ru/maps/?ll={longitude},{latitude}&z=17&pt={longitude},{latitude},pm2rdm"
    
    # Создаем статическое изображение карты через Яндекс Static API
    static_map_url = (
        f"https://static-maps.yandex.ru/1.x/?"
        f"ll={longitude},{latitude}&"
        f"z=15&"
        f"l=map&"
        f"size=450,450&"
        f"pt={longitude},{latitude},pm2rdm&"
        f"apikey={YANDEX_API_KEY}"
    )
    
    await message.reply_text(
        f"📍 <b>GPS координаты видео:</b>\n\n"
        f"🌐 Широта: <code>{latitude:.6f}</code>\n"
        f"🌐 Долгота: <code>{longitude:.6f}</code>\n\n"
        f"🗺️ <a href='{yandex_maps_url}'>Открыть на Яндекс.Картах</a>\n\n"
        f"📸 Статическая карта:",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=False
    )
    
    # Отправляем статическую карту как изображение
    try:
        import requests
        response = requests.get(static_map_url, timeout=10)
        if response.status_code == 200:
            from io import BytesIO
            await message.reply_photo(
                photo=BytesIO(response.content),
                caption=f"📍 Местоположение: {latitude:.6f}, {longitude:.6f}"
            )
        else:
            logger.warning(f"Не удалось загрузить карту: {response.status_code}")
    except Exception as e:
        logger.warning(f"Ошибка загрузки статической карты: {e}")





async def create_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /plan - создание ситуационного плана"""
    message = update.message
    
    # Проверяем, есть ли сохраненные детекции для этого пользователя
    user_id = update.effective_user.id
    
    # Для демонстрации используем последний обработанный файл
    # В реальной реализации можно хранить детекции в базе данных или файлах
    
    status_msg = await message.reply_text(
        "🗺️ <b>Создание ситуационного плана...</b>\n\n"
        "⏳ Поиск данных обработки...",
        parse_mode=ParseMode.HTML
    )
    
    try:
        # Используем детекции из памяти если есть (сохраняются после обработки видео)
        raw_detections = context.user_data.get('last_detections')
        video_info_saved = context.user_data.get('last_video_info', {})
        
        if not raw_detections:
            await status_msg.edit_text(
                "❌ <b>Нет данных для создания плана</b>\n\n"
                "💡 Сначала обработайте видео, затем сразу используйте /plan",
                parse_mode=ParseMode.HTML
            )
            return
        
        await status_msg.edit_text(
            "🗺️ <b>Создание ситуационного плана...</b>\n\n"
            f"📊 Загружено {len(raw_detections)} детекций из памяти...",
            parse_mode=ParseMode.HTML
        )
        
        # Дедупликация объектов — группируем близкие детекции
        all_detections = raw_detections
        detections = []
        processed = set()
        
        for i, detection in enumerate(all_detections):
            if i in processed:
                continue
            similar_detections = [detection]
            processed.add(i)
            for j, other_detection in enumerate(all_detections[i+1:], i+1):
                if j in processed:
                    continue
                if (detection.get('object_type') == other_detection.get('object_type') and
                    detection.get('class_name') == other_detection.get('class_name') and
                    abs((detection.get('picket') or 0) - (other_detection.get('picket') or 0)) < 50):
                    similar_detections.append(other_detection)
                    processed.add(j)
            best_detection = max(similar_detections, key=lambda x: x.get('confidence', 0))
            detections.append(best_detection)
        
        if not detections:
            await status_msg.edit_text(
                "❌ <b>Нет детекций для создания плана</b>\n\n"
                "💡 Обработайте видео с объектами на дороге",
                parse_mode=ParseMode.HTML
            )
            return
        
        await status_msg.edit_text(
            "🗺️ <b>Создание ситуационного плана...</b>\n\n"
            f"📊 Исходных детекций: {len(all_detections)}\n"
            f"🎯 Уникальных объектов: {len(detections)}\n"
            "🎨 Создаю изображение плана...",
            parse_mode=ParseMode.HTML
        )
        
        # Создаем видео информацию
        video_info = {
            'filename': video_info_saved.get('filename', 'processed_video.mp4'),
            'duration': video_info_saved.get('duration', 60.0),
            'fps': video_info_saved.get('fps', 30.0)
        }
        
        # Создаем ситуационный план (изображение)
        plan_image_path = None
        if SITUATIONAL_PLAN_AVAILABLE:
            try:
                plan_image_path = create_situational_plan(detections, video_info)
            except Exception as e:
                logger.error(f"Ошибка создания изображения плана: {e}")
        
        await status_msg.edit_text(
            "🗺️ <b>Создание ситуационного плана...</b>\n\n"
            f"📊 Исходных детекций: {len(all_detections)}\n"
            f"🎯 Уникальных объектов: {len(detections)}\n"
            "📋 Создаю Excel отчет...",
            parse_mode=ParseMode.HTML
        )
        
        # Создаем Excel отчет ситуационного плана (используем дедуплицированные данные)
        excel_plan_path = None
        if SITUATIONAL_PLAN_EXCEL_AVAILABLE:
            try:
                excel_plan_path = create_situational_plan_excel(detections, video_info)
            except Exception as e:
                logger.error(f"Ошибка создания Excel плана: {e}")
        
        await status_msg.edit_text(
            "🗺️ <b>Ситуационный план готов!</b>\n\n"
            f"📊 Исходных детекций: {len(all_detections)}\n"
            f"🎯 Уникальных объектов: {len(detections)}\n"
            "📤 Отправляю файлы...",
            parse_mode=ParseMode.HTML
        )
        
        # Отправляем изображение плана
        if plan_image_path and Path(plan_image_path).exists():
            with open(plan_image_path, 'rb') as img_file:
                await message.reply_photo(
                    photo=img_file,
                    caption=(
                        "🗺️ <b>Ситуационный план дороги</b>\n\n"
                        f"📊 Уникальных объектов: {len(detections)}\n"
                        f"📈 Исходных детекций: {len(all_detections)}\n"
                        "🎨 Схематичное изображение с расположением объектов"
                    ),
                    parse_mode=ParseMode.HTML
                )
        
        # Отправляем Excel отчет
        if excel_plan_path and Path(excel_plan_path).exists():
            with open(excel_plan_path, 'rb') as excel_file:
                await message.reply_document(
                    document=excel_file,
                    filename="situational_plan.xlsx",
                    caption=(
                        "📋 <b>Детальный отчет ситуационного плана</b>\n\n"
                        "📊 Содержит:\n"
                        "• Общую сводку\n"
                        "• Детальный список объектов\n"
                        "• Группировку по типам\n"
                        "• Статистику по участкам дороги"
                    ),
                    parse_mode=ParseMode.HTML
                )
        
        await status_msg.edit_text(
            "✅ <b>Ситуационный план создан!</b>\n\n"
            f"📊 Уникальных объектов: {len(detections)}\n"
            f"📈 Исходных детекций: {len(all_detections)}\n"
            "🗺️ Изображение плана отправлено\n"
            "📋 Excel отчет отправлен\n\n"
            "💡 Используйте /plan для создания нового плана",
            parse_mode=ParseMode.HTML
        )
        
        # Удаляем временные файлы
        try:
            if plan_image_path and Path(plan_image_path).exists():
                Path(plan_image_path).unlink()
            if excel_plan_path and Path(excel_plan_path).exists():
                Path(excel_plan_path).unlink()
        except:
            pass
        
    except Exception as e:
        logger.error(f"Ошибка создания ситуационного плана: {e}", exc_info=True)
        await status_msg.edit_text(
            f"❌ <b>Ошибка создания плана</b>\n\n"
            f"Ошибка: {str(e)}\n\n"
            f"💡 Попробуйте еще раз или обработайте новое видео",
            parse_mode=ParseMode.HTML
        )


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик видео файлов"""
    message = update.message
    logger.info(f"Получено сообщение: video={message.video}, document={message.document}")
    
    # Проверяем наличие видео или документа
    file = None
    file_name = None
    
    if message.video:
        file = await context.bot.get_file(message.video.file_id)
        file_name = f"video_{message.message_id}.mp4"
        logger.info(f"Видео найдено: {file_name}, размер: {message.video.file_size}")
    elif message.document:
        # Проверяем MIME тип для документов
        mime_type = message.document.mime_type or ""
        file_name = message.document.file_name or ""
        
        # Проверяем расширение файла и MIME тип
        is_video = False
        if "video" in mime_type.lower():
            is_video = True
        elif file_name:
            # Проверяем расширение файла (MOV, MP4, AVI, MKV и т.д.)
            file_ext = file_name.lower().split('.')[-1] if '.' in file_name else ""
            if file_ext in ['mp4', 'mov', 'avi', 'mkv', 'm4v', '3gp', 'flv', 'wmv', 'webm']:
                is_video = True
        
        if is_video:
            file = await context.bot.get_file(message.document.file_id)
            # Сохраняем оригинальное расширение или используем .mp4 по умолчанию
            if not file_name:
                file_name = f"video_{message.message_id}.mp4"
            logger.info(f"Документ-видео найден: {file_name}, MIME: {mime_type}")
        else:
            await message.reply_text(
                f"❌ Файл не является видео.\n"
                f"Тип: {mime_type}\n"
                f"Имя файла: {file_name or 'не указано'}\n\n"
                f"📹 Поддерживаемые форматы: MP4, MOV, AVI, MKV, M4V"
            )
            return
    else:
        await message.reply_text("❌ Пожалуйста, отправьте видео файл (MP4, MOV, AVI)")
        return
    
    if not file:
        await message.reply_text("❌ Не удалось получить файл. Попробуйте снова.")
        return
    
    # Проверяем размер файла ПЕРЕД загрузкой (лимит Telegram Bot API: 20 MB для getFile)
    # Telegram ограничивает getFile до 20 MB, для больших файлов нужен другой подход
    file_size_bytes = None
    if message.video and message.video.file_size:
        file_size_bytes = message.video.file_size
    elif message.document and message.document.file_size:
        file_size_bytes = message.document.file_size
    
    if file_size_bytes:
        file_size_mb = file_size_bytes / 1024 / 1024
        
        # Оптимизация: ограничиваем размер файла для предотвращения перегрева
        if file_size_mb > 50:
            await message.reply_text(
                f"❌ <b>Файл слишком большой для обработки</b>\n\n"
                f"📦 Размер файла: {file_size_mb:.1f} MB\n"
                f"⚠️ Для предотвращения перегрева ПК ограничение: 50 MB\n\n"
                f"💡 <b>Рекомендации:</b>\n"
                f"• Сожмите видео (используйте меньшее разрешение)\n"
                f"• Укоротите видео (до 5-10 минут)\n"
                f"• Используйте формат MP4 с кодеком H.264\n\n"
                f"📹 Оптимальный размер: 10-30 MB",
                parse_mode=ParseMode.HTML
            )
            return
        
        # Telegram Bot API ограничение: до 20 MB для getFile
        if file_size_mb > 20:
            await message.reply_text(
                f"❌ <b>Файл слишком большой для загрузки</b>\n\n"
                f"📦 Размер файла: {file_size_mb:.1f} MB\n"
                f"⚠️ Telegram Bot API ограничивает загрузку до 20 MB\n\n"
                f"💡 <b>Решения:</b>\n"
                f"• Сожмите видео до размера меньше 20 MB\n"
                f"• Разделите видео на части\n"
                f"• Используйте видео меньшего размера\n\n"
                f"📹 Рекомендуемый размер: до 20 MB",
                parse_mode=ParseMode.HTML
            )
            return
    
    # Создаем пути
    input_path = DOWNLOAD_DIR / file_name
    output_stem = Path(file_name).stem
    output_path = OUTPUT_DIR / f"processed_{output_stem}.avi"
    
    try:
        # Отправляем начальное сообщение
        status_msg = await message.reply_text(
            "⏳ Загрузка видео...\n"
            "📥 Пожалуйста, подождите..."
        )
        
        # Скачиваем видео с увеличенным таймаутом
        try:
            # Используем увеличенный таймаут для больших файлов (до 200 МБ)
            await asyncio.wait_for(
                file.download_to_drive(input_path),
                timeout=600.0  # 10 минут для загрузки больших файлов (до 200 МБ)
            )
        except asyncio.TimeoutError:
            raise ValueError("Таймаут при загрузке видео. Файл слишком большой или медленное соединение.")
        except Exception as e:
            if "Timed out" in str(e) or "timeout" in str(e).lower():
                raise ValueError(f"Таймаут при загрузке видео. Попробуйте отправить файл меньшего размера или проверьте соединение.")
            raise
        
        # Видео загружено — показываем меню выбора детекторов
        await status_msg.edit_text("✅ Видео загружено!")
        
        # Сохраняем информацию о видео в user_data для последующей обработки
        context.user_data['pending_video'] = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'file_name': file_name
        }
        
        # Показываем меню выбора детекторов
        await show_detector_menu(message, context)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {e}", exc_info=True)
        error_msg = str(e)
        
        # Удаляем временные файлы
        try:
            if input_path.exists():
                input_path.unlink()
        except:
            pass
        
        await message.reply_text(
            f"❌ <b>Ошибка при обработке видео</b>\n\n"
            f"{error_msg}\n\n"
            f"💡 Попробуйте:\n"
            f"• Отправить видео меньшего размера\n"
            f"• Использовать формат MP4\n"
            f"• Проверить качество видео",
            parse_mode=ParseMode.HTML
        )


async def show_detector_menu(message: Message, context: ContextTypes.DEFAULT_TYPE):
    """Показывает меню выбора детекторов"""
    # Инициализируем настройки детекторов (по умолчанию все включены)
    if 'detector_settings' not in context.user_data:
        context.user_data['detector_settings'] = {
            'signs': True,
            'infrastructure': True,
            'barriers': True,
            'marking': True,
            'exits': True,
            'noise_strips': True
        }
    
    settings = context.user_data['detector_settings']
    
    # Создаем кнопки с галочками
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✅' if settings['signs'] else '☐'} Дорожные знаки",
                callback_data='toggle_signs'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['infrastructure'] else '☐'} Инфраструктура (авто, светофоры)",
                callback_data='toggle_infrastructure'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['barriers'] else '☐'} Барьеры и ограждения",
                callback_data='toggle_barriers'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['marking'] else '☐'} Дорожная разметка",
                callback_data='toggle_marking'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['exits'] else '☐'} Съезды",
                callback_data='toggle_exits'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['noise_strips'] else '☐'} Шумовые полосы",
                callback_data='toggle_noise_strips'
            )
        ],
        [
            InlineKeyboardButton(
                "🚀 Начать обработку",
                callback_data='start_processing'
            )
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await message.reply_text(
        "⚙️ <b>Настройка детекторов</b>\n\n"
        "Выберите, какие объекты нужно искать на видео:\n"
        "(нажмите на кнопку для вкл/выкл)\n\n"
        "💡 После выбора нажмите <b>Начать обработку</b>",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )


async def handle_detector_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок меню детекторов"""
    query = update.callback_query
    await query.answer()
    
    if 'detector_settings' not in context.user_data:
        context.user_data['detector_settings'] = {
            'signs': True,
            'infrastructure': True,
            'barriers': True,
            'marking': True,
            'exits': True,
            'noise_strips': True
        }
    
    settings = context.user_data['detector_settings']
    
    # Обрабатываем нажатие кнопки
    if query.data == 'start_processing':
        # Начинаем обработку с выбранными настройками
        await query.edit_message_text("🚀 Начинаю обработку видео с выбранными детекторами...")
        await process_video_with_settings(query.message, context)
        return
    
    # Переключение детекторов
    if query.data == 'toggle_signs':
        settings['signs'] = not settings['signs']
    elif query.data == 'toggle_infrastructure':
        settings['infrastructure'] = not settings['infrastructure']
    elif query.data == 'toggle_barriers':
        settings['barriers'] = not settings['barriers']
    elif query.data == 'toggle_marking':
        settings['marking'] = not settings['marking']
    elif query.data == 'toggle_exits':
        settings['exits'] = not settings['exits']
    elif query.data == 'toggle_noise_strips':
        settings['noise_strips'] = not settings['noise_strips']
    
    # Обновляем меню с новыми галочками
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✅' if settings['signs'] else '☐'} Дорожные знаки",
                callback_data='toggle_signs'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['infrastructure'] else '☐'} Инфраструктура (авто, светофоры)",
                callback_data='toggle_infrastructure'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['barriers'] else '☐'} Барьеры и ограждения",
                callback_data='toggle_barriers'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['marking'] else '☐'} Дорожная разметка",
                callback_data='toggle_marking'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['exits'] else '☐'} Съезды",
                callback_data='toggle_exits'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['noise_strips'] else '☐'} Шумовые полосы",
                callback_data='toggle_noise_strips'
            )
        ],
        [
            InlineKeyboardButton(
                "🚀 Начать обработку",
                callback_data='start_processing'
            )
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "⚙️ <b>Настройка детекторов</b>\n\n"
        "Выберите, какие объекты нужно искать на видео:\n"
        "(нажмите на кнопку для вкл/выкл)\n\n"
        "💡 После выбора нажмите <b>Начать обработку</b>",
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML
    )


async def process_video_with_settings(message: Message, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает видео с учетом выбранных настроек детекторов"""
    if 'pending_video' not in context.user_data:
        await message.reply_text("❌ Видео не найдено. Отправьте видео заново.")
        return
    
    video_info = context.user_data['pending_video']
    input_path = Path(video_info['input_path'])
    output_path = Path(video_info['output_path'])
    file_name = video_info['file_name']  # Извлекаем file_name из video_info
    
    status_msg = None
    try:
        # Извлекаем GPS координаты из видео (в отдельном потоке, чтобы не блокировать event loop)
        gps_coords = await asyncio.to_thread(extract_gps_from_video, input_path)
        if gps_coords:
            context.user_data['video_gps_coords'] = gps_coords
            logger.info(f"📍 GPS координаты сохранены: {gps_coords}")
        
        settings = context.user_data.get('detector_settings', {})
        enabled_detectors = [k for k, v in settings.items() if v]
        
        status_text = (
            f"✅ {'GPS найден' if gps_coords else 'GPS не найден'}\n"
            f"🚀 Начинаю обработку...\n\n"
            f"🔧 Активные детекторы: {len(enabled_detectors)}/6\n"
            f"📊 Прогресс: 0%\n"
            f"🎬 Загрузка моделей..."
        )
        for _attempt in range(3):
            try:
                status_msg = await message.reply_text(status_text)
                break
            except Exception as send_err:
                logger.warning(f"Попытка отправки статуса {_attempt+1}/3 не удалась: {send_err}")
                await asyncio.sleep(2)
        if status_msg is None:
            status_msg = await message.reply_text("🚀 Обработка...")
        
        # Инициализируем процессор
        proc = get_processor()
        
        # Обрабатываем видео с настройками
        stats, all_detections, processed_output_path = await proc.process_video_with_progress(
            str(input_path),
            str(output_path),
            status_msg,
            context,
            conf_threshold=0.1,
            detector_settings=settings  # Передаем настройки
        )
        output_path = Path(processed_output_path)
        
        # Сохраняем детекции для команды /plan
        context.user_data['last_detections'] = all_detections
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(str(input_path))
            _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
            _frames = _cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0
            _cap.release()
            _duration = _frames / _fps if _fps > 0 else 0.0
        except Exception:
            _fps, _duration = 30.0, 0.0
        context.user_data['last_video_info'] = {
            'filename': file_name,
            'duration': _duration,
            'fps': _fps
        }
        logger.info(f"✅ Сохранено {len(all_detections)} детекций для /plan")
        
        # Финальное сообщение
        await status_msg.edit_text(
            f"✅ <b>Обработка завершена!</b>\n\n"
            f"📊 Статистика:\n"
            f"🚦 Дорожных знаков: {stats['signs']} детекций ({stats['sign_frames']} кадров)\n"
            f"🏗️  Объектов инфраструктуры: {stats['infra']} детекций ({stats['infra_frames']} кадров)\n"
            f"🚧 Дорожных барьеров: {stats.get('barriers', 0)} детекций ({stats.get('barrier_frames', 0)} кадров)\n"
            f"🛣️  Кадров с разметкой: {stats['marking_frames']}\n\n"
            f"📤 Отправляю обработанное видео...",
            parse_mode=ParseMode.HTML
        )
        
        # Отправляем обработанное видео
        file_size = output_path.stat().st_size
        
        # Убеждаемся что файл существует и не пустой
        if file_size < 1024:
            raise ValueError(f"Обработанное видео слишком маленькое: {file_size} bytes")
        
        # Информируем о размере файла
        file_size_mb = file_size / 1024 / 1024
        logger.info(f"Размер обработанного видео: {file_size_mb:.2f} MB")
        
        # Для файлов больше 20MB предупреждаем пользователя
        if file_size_mb > 20:
            await status_msg.edit_text(
                f"⏳ <b>Отправляю большое видео...</b>\n\n"
                f"📦 Размер: {file_size_mb:.1f} MB\n"
                f"⏱️ Это может занять несколько минут...",
                parse_mode=ParseMode.HTML
            )
        
        # Открываем файл для отправки (правильное управление файлом)
        # Используем увеличенный таймаут для больших файлов
        try:
            with open(output_path, 'rb') as video_file:
                _ensure_external_telegram()
                # Telegram API ограничение: до 50MB для video, до 2GB для document
                # Для надежности используем document для файлов > 20MB
                is_mp4 = output_path.suffix.lower() == '.mp4'
                
                if file_size > 20 * 1024 * 1024:  # Если больше 20MB
                    logger.info(f"Отправка как документ (файл > 20MB)")
                    # Для больших файлов отправляем как документ с увеличенным таймаутом
                    upload_task = message.reply_document(
                        document=video_file,
                        filename=output_path.name,
                        caption="✅ Обработанное видео со всеми детекциями",
                        write_timeout=600.0,
                        read_timeout=600.0,
                        connect_timeout=30.0,
                    )
                elif is_mp4:
                    logger.info(f"Отправка как видео MP4 (файл < 20MB)")
                    # Для MP4 файлов отправляем как видео
                    upload_task = message.reply_video(
                        video=video_file,
                        filename=output_path.name,
                        caption="✅ Обработанное видео со всеми детекциями",
                        supports_streaming=True,
                        write_timeout=600.0,
                        read_timeout=600.0,
                        connect_timeout=30.0,
                    )
                else:
                    logger.info(f"Отправка как документ (не MP4)")
                    # Для не-MP4 файлов отправляем как документ
                    upload_task = message.reply_document(
                        document=video_file,
                        filename=output_path.name,
                        caption="✅ Обработанное видео со всеми детекциями",
                        write_timeout=600.0,
                        read_timeout=600.0,
                        connect_timeout=30.0,
                    )
                
                # Используем таймаут 20 минут для отправки (Telegram может быть медленным)
                await asyncio.wait_for(upload_task, timeout=1200.0)  # 20 минут
                
        except asyncio.TimeoutError:
            logger.error(f"Таймаут при отправке видео {file_size_mb:.2f} MB")
            raise ValueError(
                f"⏱️ Таймаут при отправке видео ({file_size_mb:.1f} MB).\n\n"
                f"📦 Файл слишком большой для отправки через Telegram.\n"
                f"💡 Попробуйте:\n"
                f"   • Отправить видео меньшего размера\n"
                f"   • Использовать более сжатый формат\n"
                f"   • Проверить интернет-соединение"
            )
        except Exception as e:
            error_str = str(e)
            if "Timed out" in error_str or "timeout" in error_str.lower():
                logger.error(f"Таймаут при отправке: {error_str}")
                raise ValueError(
                    f"⏱️ Таймаут при отправке видео ({file_size_mb:.1f} MB).\n\n"
                    f"📦 Файл слишком большой для Telegram API.\n"
                    f"💡 Попробуйте отправить видео меньшего размера."
                )
            # Если это другая ошибка - пробрасываем её дальше
            raise
        
        # Создаем и отправляем Excel файл с результатами
        excel_path = None
        if EXCEL_AVAILABLE and all_detections:
            try:
                await status_msg.edit_text(
                    f"✅ <b>Видео отправлено!</b>\n\n"
                    f"📊 Создаю Excel отчет...\n"
                    f"   Детекций: {len(all_detections)}",
                    parse_mode=ParseMode.HTML
                )
                
                # Создаем Excel файл
                excel_path = create_excel_report(
                    all_detections,
                    video_name=Path(file_name).stem
                )
                
                # Отправляем Excel файл
                excel_file_size = Path(excel_path).stat().st_size
                with open(excel_path, 'rb') as excel_file:
                    await message.reply_document(
                        document=excel_file,
                        filename=f"detections_{Path(file_name).stem}.xlsx",
                        caption=(
                            f"📊 <b>Excel отчет с детекциями</b>\n\n"
                            f"📈 Всего детекций: {len(all_detections)}\n"
                            f"🚦 Знаков: {stats['signs']}\n"
                            f"🏗️  Инфраструктуры: {stats['infra']}\n"
                            f"🚧 Барьеров: {stats.get('barriers', 0)}\n\n"
                            f"📁 Файл содержит координаты всех объектов"
                        ),
                        parse_mode=ParseMode.HTML,
                        write_timeout=120.0,
                        read_timeout=120.0,
                        connect_timeout=30.0,
                    )
                
                logger.info(f"Excel файл создан: {excel_path} ({excel_file_size / 1024:.1f} KB)")
                
            except Exception as e:
                logger.error(f"Ошибка при создании Excel: {e}", exc_info=True)
                await message.reply_text(
                    f"⚠️ Excel отчет не создан: {str(e)}"
                )
        
        # Удаляем временные файлы
        try:
            input_path.unlink()
            output_path.unlink()
            if excel_path and Path(excel_path).exists():
                # Excel файл не удаляем - он уже отправлен пользователю
                pass
        except:
            pass
        
        await status_msg.edit_text(
            f"✅ <b>Готово!</b>\n\n"
            f"📊 Статистика обработки:\n"
            f"🚦 Дорожных знаков: {stats['signs']} детекций ({stats['sign_frames']} кадров)\n"
            f"🏗️  Объектов инфраструктуры: {stats['infra']} детекций ({stats['infra_frames']} кадров)\n"
            f"🚧 Дорожных барьеров: {stats.get('barriers', 0)} детекций ({stats.get('barrier_frames', 0)} кадров)\n"
            f"🛣️  Кадров с разметкой: {stats['marking_frames']}\n\n"
            f"💡 Можете отправить еще одно видео для обработки!",
            parse_mode=ParseMode.HTML
        )
        
    except Exception as e:
        logger.error(f"Ошибка при обработке видео: {e}", exc_info=True)
        try:
            error_text = (
                f"❌ <b>Ошибка при обработке видео</b>\n\n"
                f"Ошибка: {str(e)}\n\n"
                f"Попробуйте отправить видео снова."
            )
            if status_msg is not None:
                await status_msg.edit_text(error_text, parse_mode=ParseMode.HTML)
            else:
                await message.reply_text(error_text, parse_mode=ParseMode.HTML)
        except Exception as notify_err:
            logger.error(f"Не удалось отправить сообщение об ошибке: {notify_err}")
        
        # Удаляем временные файлы при ошибке
        try:
            if input_path.exists():
                input_path.unlink()
            if output_path.exists():
                output_path.unlink()
        except:
            pass


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Update {update} вызвал ошибку {context.error}")


def main():
    """Запуск бота"""
    if not TELEGRAM_AVAILABLE:
        print("❌ python-telegram-bot не установлен!")
        print("💡 Установите: pip install python-telegram-bot")
        return
    
    print("=" * 80)
    print("🤖 ЗАПУСК TELEGRAM БОТА ДЛЯ ОБРАБОТКИ ВИДЕО".center(80))
    print("=" * 80)
    print(f"\n📋 Токен бота: {BOT_TOKEN[:20]}...")
    print(f"📁 Папка загрузки: {DOWNLOAD_DIR}")
    print(f"📁 Папка вывода: {OUTPUT_DIR}")
    print(f"\n⏳ Создание приложения...")
    
    # Создаем приложение с увеличенными таймаутами
    from telegram.request import HTTPXRequest
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=600.0,   # 10 минут — нужно для медленных соединений
        write_timeout=600.0,  # 10 минут — для загрузки видео/файлов
        pool_timeout=60.0,
    )
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .build()
    )
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("gps", set_gps))
    application.add_handler(CommandHandler("plan", create_plan))
    # Обработчик для кнопок меню детекторов
    application.add_handler(CallbackQueryHandler(handle_detector_callback))
    # Обработчик для видео (отправленных как видео)
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    # Обработчик для документов (видео файлов)
    application.add_handler(MessageHandler(filters.Document.ALL, handle_video))
    application.add_error_handler(error_handler)
    
    print("✅ Бот готов к работе!")
    print("\n🚀 Запускаю бота...")
    print("💡 Отправьте /start боту в Telegram")
    print("=" * 80 + "\n")
    
    # Запускаем бота
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
