#!/usr/bin/env python3
"""
Скрипт для загрузки RTSD датасета с Kaggle
"""
import os
import zipfile
import shutil
from pathlib import Path

def download_rtsd_from_kaggle():
    """Загрузка RTSD датасета с Kaggle"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("=" * 60)
        print("Загрузка RTSD датасета с Kaggle")
        print("=" * 60)
        
        # Инициализация Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        dataset_name = "watchman/rtsd-dataset"
        download_path = "rtsd_dataset"
        
        # Создаем директорию для датасета
        Path(download_path).mkdir(exist_ok=True)
        
        print(f"Скачивание датасета {dataset_name}...")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        print(f"✓ Датасет загружен в {download_path}")
        return download_path
        
    except ImportError:
        print("⚠ Kaggle API не установлен")
        print("Установите: pip install kaggle")
        print("Или скачайте датасет вручную с: https://www.kaggle.com/datasets/watchman/rtsd-dataset")
        return None
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        print("\nАльтернативный способ:")
        print("1. Зайдите на: https://www.kaggle.com/datasets/watchman/rtsd-dataset")
        print("2. Скачайте датасет вручную")
        print("3. Распакуйте в директорию rtsd_dataset")
        return None

if __name__ == "__main__":
    dataset_path = download_rtsd_from_kaggle()
    if dataset_path:
        print(f"\n✓ Датасет готов: {dataset_path}")
    else:
        print("\n⚠ Загрузите датасет вручную и укажите путь к нему")


