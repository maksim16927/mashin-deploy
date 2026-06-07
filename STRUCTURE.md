# Структура проекта

## Основные директории

### `/src/` - Основные модули проекта
- `road_detector.py` - Детекторы дорожных объектов (знаки, разметка, светофоры)
- `gps_extractor.py` - Извлечение и обработка GPS координат
- `road_passport.py` - Управление паспортом дороги
- `detect_marking.py` - Детекция разметки
- `__init__.py` - Инициализация модулей

### `/scripts/` - Скрипты для запуска
**Детекция:**
- `detect.py` - Простая детекция
- `detect_all.py` - Детекция знаков и разметки
- `detect_infrastructure.py` - Комплексная детекция инфраструктуры
- `detect_marking.py` - Детекция разметки
- `detect_video.py` - Детекция на видео
- `detect_clean.py` - Чистая детекция
- `detect_coco_infrastructure.py` - Детекция COCO инфраструктуры
- `detect_full_system.py` - Полная система детекции

**Обучение:**
- `train_infrastructure.py` - Обучение модели инфраструктуры
- `train_marking_model.py` - Обучение модели разметки
- `train_marking_simple.py` - Простое обучение разметки
- `quick_train.py` - Быстрое обучение

**Загрузка данных:**
- `download_infrastructure_dataset.py` - Загрузка датасетов инфраструктуры
- `download_marking_model.py` - Загрузка модели разметки
- `download_public_dataset.py` - Загрузка публичных датасетов
- `download_roboflow_dataset.py` - Загрузка датасетов Roboflow

**Утилиты:**
- `process_video.py` - Обработка видео
- `quick_example.py` - Быстрый пример

**Shell-скрипты:**
- `process_all_videos.sh` - Обработка всех видео
- `process_full_system.sh` - Обработка полной системы

### `/tests/` - Тесты
- `test_all_models.py` - Тест всех моделей
- `test_pretrained_model.py` - Тест предобученной модели
- `check_system.py` - Проверка системы

### `/docs/` - Документация
- `README_INFRASTRUCTURE.md` - Документация по инфраструктуре
- `MODEL_INFO.md` - Информация о моделях
- `QUICK_START_INFRASTRUCTURE.txt` - Быстрый старт
- `RESULTS.txt` - Результаты тестирования

### `/models/` - Модели YOLO
- `yolov8s_35epochs_rtsd155.pt` - Модель дорожных знаков (155 классов)
- `yolov8n_road_marking.pt` - Модель дорожной разметки
- `yolov8n-seg.pt` - Сегментационная модель
- `yolov8n.pt` - Базовая YOLO модель
- `MARKING_MODEL_INFO.md` - Информация о модели разметки

### `/videos/` - Видео файлы
- `original/` - Оригинальные видео
- `detected/` - Обработанные видео с детекциями
- `test/` - Тестовые видео

### `/datasets/` - Датасеты для обучения
- `quick_infrastructure/` - Датасет инфраструктуры
- `road_marking_dataset/` - Датасет разметки
- `rtsd_yolo/` - Датасет RTSD в формате YOLO

### `/runs/` - Результаты обучения и детекции
- `detect/` - Результаты детекции
- `marking_train/` - Результаты обучения разметки

### `/tools/` - Утилиты
- Вспомогательные скрипты и инструменты

### `/video_detections/` - Детекции видео
- `result/` - Результаты детекций

## Корневая директория

- `README.md` - Главная документация проекта
- `requirements.txt` - Зависимости проекта

