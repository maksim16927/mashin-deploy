# Распознавание дорожной инфраструктуры

Система для распознавания дорожных знаков, разметки, светофоров и других элементов дорожной инфраструктуры из видео с привязкой к GPS координатам.

## Возможности

- 🚦 Распознавание дорожных знаков (155 классов RTSD - Russian Traffic Sign Dataset)
- 🛣️ Определение разметки и количества полос
- 🚥 Детекция светофоров и уличных фонарей
- 🚧 Обнаружение ограждений
- 📍 Привязка всех объектов к GPS координатам
- 📊 Создание паспорта дороги в различных форматах

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Taulan23/mashin.git
cd mashin
```

2. Создайте виртуальное окружение:
```bash
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### 1. Загрузка и обучение модели на RTSD датасете

```bash
python train_rtsd.py
```

Этот скрипт:
- Скачает RTSD датасет с Kaggle
- Конвертирует его в формат YOLO
- Обучит модель YOLOv8 на 155 классах дорожных знаков

**Примечание:** Для обучения требуется GPU (рекомендуется NVIDIA с CUDA или Apple M1/M2/M3 с Metal). На CPU обучение будет очень медленным.

### 2. Обработка видео

Откройте `road_analysis.ipynb` в Jupyter Notebook и запустите все ячейки.

Система автоматически:
- Найдет обученную модель
- Обработает видео `IMG_8810.MP4`
- Распознает все объекты дорожной инфраструктуры
- Создаст видео с визуализацией (`road_analysis_output.mp4`)
- Сгенерирует паспорт дороги (`road_passport.csv`, `road_passport.json`)

## Структура проекта

```
.
├── road_detector.py          # Детекторы объектов (знаки, разметка, светофоры)
├── gps_extractor.py          # Извлечение и интерполяция GPS координат
├── road_passport.py          # Управление паспортом дороги
├── rtsd_dataset.py           # Работа с RTSD датасетом
├── train_rtsd.py             # Обучение модели на RTSD
├── download_rtsd_direct.py   # Прямая загрузка датасета с Kaggle
├── road_analysis.ipynb       # Основной notebook для анализа
└── requirements.txt          # Зависимости
```

## Требования

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- PyTorch
- Pandas, NumPy, Matplotlib
- Kaggle API (для загрузки датасета)

## Датасет

Проект использует [RTSD (Russian Traffic Sign Dataset)](https://www.kaggle.com/datasets/watchman/rtsd-dataset) - датасет российских дорожных знаков с 155 классами по ГОСТ Р 52290-2004.

## Результаты

После обработки видео создаются:
- `road_passport.csv` - паспорт дороги в формате CSV
- `road_passport.json` - паспорт в формате JSON
- `road_analysis_output.mp4` - видео с визуализацией всех объектов
- `road_map.html` - интерактивная карта с объектами

## Лицензия

MIT

## Автор

Taulan23

