# 🚗 Система комплексного распознавания дорожных объектов

Дипломный проект по компьютерному зрению для распознавания:
- 🚦 **Дорожных знаков** (155 классов RTSD)
- 🛣️ **Дорожной разметки** (полосы движения)
- 🚌 **Дорожной инфраструктуры** (автобусные остановки, П-образные опоры, столбы)
- 🔧 **Повреждений инфраструктуры** (ямы, поврежденные знаки)

## 📋 Содержание
- [Возможности](#-возможности)
- [Установка](#-установка)
- [Быстрый старт](#-быстрый-старт)
- [Датасеты](#-датасеты)
- [Обучение моделей](#-обучение-моделей)
- [Использование](#-использование)
- [Результаты](#-результаты)
- [Структура проекта](#-структура-проекта)

## 🎯 Возможности

### 1. Распознавание дорожных знаков
- ✅ 155 классов российских дорожных знаков (RTSD)
- ✅ Модель YOLOv8s обученная на 35 эпох
- ✅ Высокая точность распознавания

### 2. Детекция дорожной разметки
- ✅ Обнаружение полос движения
- ✅ Использование OpenCV (Canny + HoughLinesP)
- ✅ Работает в реальном времени

### 3. Распознавание инфраструктуры (NEW! 🎉)
- 🆕 Автобусные остановки
- 🆕 П-образные опоры (на платных участках)
- 🆕 Электрические столбы
- 🆕 Повреждения дорог (ямы, трещины)
- 🆕 Поврежденные знаки и инфраструктура

## 🔧 Установка

### Требования
- Python 3.8+
- PyTorch 2.0+
- CUDA/MPS (опционально, для GPU)

### Установка зависимостей

```bash
# Установка основных зависимостей
pip install -r requirements.txt

# Для работы с Roboflow (загрузка датасетов)
pip install roboflow
```

### Содержимое requirements.txt
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
roboflow>=1.1.0
```

## 🚀 Быстрый старт

### 1. Базовая детекция (знаки + разметка)

Обработать видео с детекцией знаков и разметки:

```bash
python3 scripts/detect_all.py "videos/original/video.mp4" -o "videos/detected/result.mp4"
```

### 2. Расширенная детекция (всё вместе)

После обучения модели инфраструктуры:

```bash
python3 scripts/detect_infrastructure.py "videos/original/video.mp4" \
    --infra-model "models/infrastructure_model.pt" \
    -o "videos/detected/full_result.mp4"
```

## 📊 Датасеты

### Дорожные знаки (уже готово ✅)
- **Источник**: RTSD (Russian Traffic Sign Dataset)
- **Модель**: `models/yolov8s_35epochs_rtsd155.pt`
- **Классы**: 155 типов российских дорожных знаков

### Дорожная инфраструктура (требуется загрузка)

#### Доступные датасеты на Roboflow Universe:

1. **Bus Stop Detection** (544 изображения)
   - Источник: The University of Hong Kong
   - Классы: Bus-Stop, shelter, sign
   - URL: https://universe.roboflow.com/the-university-of-hong-kong-zsdbj/bus-stop-detection

2. **Road Infrastructure** (914 изображений)
   - Источник: Road Infrastructure Project
   - Классы: straight, intersections
   - URL: https://universe.roboflow.com/road-infrastructure-0hg8j/road-infrastructure-predict

3. **Infrastructure Split** (2470 изображений)
   - Источник: Infrastructure Project
   - Классы: pothole, damaged-sign, electricity-pole-damage, fallen-trees, pipe-burst, trash, graffiti
   - URL: https://universe.roboflow.com/infrastructure-aqfrw/infrastructure-split-rebalanced

### Загрузка датасетов

#### Шаг 1: Получите API ключ Roboflow

1. Перейдите на https://app.roboflow.com/
2. Зарегистрируйтесь или войдите
3. Settings → API: https://app.roboflow.com/settings/api
4. Скопируйте Private API Key

#### Шаг 2: Загрузите датасеты

```bash
# Просмотр инструкций
python3 scripts/download_infrastructure_dataset.py --info

# Загрузка с API ключом
python3 scripts/download_infrastructure_dataset.py --api-key "ваш_ключ_здесь"
```

Датасеты будут загружены в папку `datasets/`

## 🎓 Обучение моделей

### Обучение модели инфраструктуры

После загрузки датасетов обучите модель:

```bash
# Базовое обучение (50 эпох)
python3 scripts/train_infrastructure.py

# С кастомными параметрами
python3 scripts/train_infrastructure.py \
    --dataset "datasets/Bus-Stop-Detection-1/data.yaml" \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 16
```

### Параметры обучения

```bash
python3 scripts/train_infrastructure.py \
    --dataset PATH_TO_YAML     # Путь к data.yaml
    --model yolov8n.pt         # Модель: n/s/m/l/x
    --epochs 50                # Количество эпох
    --img-size 640             # Размер изображения
    --batch 16                 # Размер батча
    --device mps               # Устройство: mps/cpu/cuda
```

### Ожидаемое время обучения

- **Apple M2 (MPS)**: ~2-3 часа для 50 эпох
- **NVIDIA GPU**: ~1-2 часа для 50 эпох
- **CPU**: ~8-12 часов для 50 эпох

## 💻 Использование

### Скрипты

| Скрипт | Описание | Использование |
|--------|----------|---------------|
| `detect_video.py` | Детекция только знаков | `python3 scripts/detect_video.py video.mp4` |
| `detect_all.py` | Знаки + разметка | `python3 scripts/detect_all.py video.mp4` |
| `detect_infrastructure.py` | Полная детекция | `python3 scripts/detect_infrastructure.py video.mp4 --infra-model model.pt` |
| `train_infrastructure.py` | Обучение модели | `python3 scripts/train_infrastructure.py` |
| `download_infrastructure_dataset.py` | Загрузка датасетов | `python3 scripts/download_infrastructure_dataset.py` |

### Примеры команд

#### Детекция только дорожных знаков
```bash
python3 scripts/detect_video.py "videos/original/video.mp4" \
    -o "videos/detected/signs_only.mp4" \
    --conf 0.3
```

#### Детекция знаков + разметка
```bash
python3 scripts/detect_all.py "videos/original/video.mp4" \
    -o "videos/detected/signs_and_marking.mp4" \
    --conf 0.25
```

#### Полная детекция (знаки + разметка + инфраструктура)
```bash
python3 scripts/detect_infrastructure.py "videos/original/video.mp4" \
    --sign-model "models/yolov8s_35epochs_rtsd155.pt" \
    --infra-model "models/infrastructure_model.pt" \
    --conf 0.25 \
    -o "videos/detected/full_detection.mp4"
```

#### Обучение на своем датасете
```bash
python3 scripts/train_infrastructure.py \
    --dataset "datasets/my_dataset/data.yaml" \
    --model yolov8s.pt \
    --epochs 100 \
    --batch 8
```

## 📈 Результаты

### Обработанные видео

Проект успешно обработал 3 видео:

| Видео | Кадров | Знаков | Разметка | Размер |
|-------|--------|--------|----------|--------|
| Video 1 | 815 | 183 (22.5%) | 100% | 7.0 MB |
| Video 2 | 309 | 104 (33.7%) | 100% | 7.7 MB |
| Video 3 | 4954 | — | 100% | 486 MB |

### Метрики модели (RTSD знаки)
- **Модель**: YOLOv8s
- **Эпох**: 35
- **Классов**: 155
- **Точность**: Высокая на российских знаках

### Цветовая схема детекций

- 🟢 **Зеленый** - Дорожные знаки
- 🔵 **Синий** - Дорожная разметка
- 🟡 **Желтый** - Автобусные остановки
- 🟣 **Фиолетовый** - П-образные опоры
- 🟠 **Оранжевый** - Столбы
- 🔴 **Красный** - Повреждения
- 🩷 **Розовый** - Другая инфраструктура

## 📁 Структура проекта

```
mashin/
├── videos/
│   ├── original/          # Исходные видео
│   └── detected/          # Обработанные видео
├── models/
│   ├── yolov8s_35epochs_rtsd155.pt     # Модель знаков
│   └── infrastructure_model_*.pt        # Модели инфраструктуры
├── datasets/              # Датасеты для обучения
├── scripts/
│   ├── detect_video.py                  # Детекция знаков
│   ├── detect_all.py                    # Знаки + разметка
│   ├── detect_infrastructure.py         # Полная детекция
│   ├── train_infrastructure.py          # Обучение
│   └── download_infrastructure_dataset.py  # Загрузка датасетов
├── runs/                  # Результаты обучения
├── requirements.txt       # Зависимости
└── README.md             # Документация
```

## 🎯 Дорожная карта

### Реализовано ✅
- [x] Распознавание 155 российских дорожных знаков
- [x] Детекция дорожной разметки
- [x] Обработка видео в реальном времени
- [x] Скрипты для загрузки датасетов
- [x] Скрипты для обучения моделей
- [x] Комплексная детекция всех объектов

### В разработке 🔄
- [ ] Обучение модели на датасетах автобусных остановок
- [ ] Распознавание П-образных опор
- [ ] Детекция повреждений дорог
- [ ] Веб-интерфейс для загрузки видео
- [ ] API для интеграции

### Планы 📝
- [ ] Мобильное приложение
- [ ] Распознавание в реальном времени с камеры
- [ ] Экспорт статистики в Excel/CSV
- [ ] Построение карт с координатами объектов

## 🤝 Вклад в проект

Проект разработан для дипломной работы по аспирантуре.

### Основные компоненты:
1. **YOLOv8** - Ultralytics
2. **OpenCV** - Обработка изображений
3. **PyTorch** - Глубокое обучение
4. **Roboflow** - Управление датасетами

## 📝 Лицензия

Проект создан в образовательных целях для дипломной работы.

## 🙏 Благодарности

- **RTSD Dataset** - Russian Traffic Sign Dataset
- **Roboflow Universe** - Датасеты дорожной инфраструктуры
- **Ultralytics** - YOLOv8 framework
- **OpenCV** - Computer Vision библиотека

## 📧 Контакты

Для вопросов по проекту обращайтесь к автору дипломной работы.

---

**Сделано с ❤️ для аспирантуры**

*Версия: 2.0 - Комплексная система детекции*
