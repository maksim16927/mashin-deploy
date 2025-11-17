#!/usr/bin/env python3
"""
Прямая загрузка RTSD датасета с Kaggle
"""
import os
import subprocess
import sys
from pathlib import Path
import zipfile
import shutil

def install_kaggle():
    """Установка Kaggle API"""
    try:
        # Проверяем, установлен ли kaggle, но не импортируем
        result = subprocess.run([sys.executable, "-c", "import kaggle"], 
                              capture_output=True, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return True
    except:
        pass
    
    print("Установка Kaggle API...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def is_valid_kaggle_credentials():
    """Проверка валидности Kaggle credentials"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return False
    try:
        import json
        with open(kaggle_json, 'r') as f:
            config = json.load(f)
            return 'username' in config and 'key' in config and config['username'] and config['key']
    except:
        return False

def setup_kaggle_credentials():
    """Настройка credentials для Kaggle"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        try:
            os.chmod(kaggle_json, 0o600)
            # Проверяем содержимое
            import json
            with open(kaggle_json, 'r') as f:
                config = json.load(f)
                if 'username' in config and 'key' in config:
                    return True
        except:
            pass
    
    print("=" * 70)
    print("НАСТРОЙКА KAGGLE API")
    print("=" * 70)
    print("Для скачивания датасета нужен Kaggle API токен:")
    print("1. Зайдите на https://www.kaggle.com/settings")
    print("2. Прокрутите до 'API' секции")  
    print("3. Нажмите 'Create New Token' - скачается kaggle.json")
    print()
    print("Или введите данные вручную:")
    print()
    
    try:
        username = input("Kaggle username (или Enter для пропуска): ").strip()
        key = input("Kaggle API key (или Enter для пропуска): ").strip()
        
        if username and key:
            kaggle_dir.mkdir(exist_ok=True, mode=0o700)
            with open(kaggle_json, 'w') as f:
                import json
                json.dump({"username": username, "key": key}, f)
            os.chmod(kaggle_json, 0o600)
            print("✓ Credentials сохранены")
            return True
        else:
            print("⚠ Credentials не введены")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n⚠ Прервано пользователем")
        return False

def download_rtsd_direct():
    """Прямая загрузка RTSD датасета"""
    print("=" * 70)
    print("ЗАГРУЗКА RTSD ДАТАСЕТА")
    print("=" * 70)
    
    dataset_path = "rtsd_dataset"
    
    # Проверяем, есть ли уже датасет
    if os.path.exists(dataset_path):
        files = list(Path(dataset_path).rglob('*.jpg')) + list(Path(dataset_path).rglob('*.png'))
        if files:
            print(f"✓ Датасет уже существует: {dataset_path}")
            print(f"  Найдено изображений: {len(files)}")
            return dataset_path
    
    # Устанавливаем Kaggle API
    if not install_kaggle():
        print("❌ Не удалось установить Kaggle API")
        return None
    
    # Настраиваем credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists() or not is_valid_kaggle_credentials():
        if not setup_kaggle_credentials():
            print("\n❌ Не удалось настроить Kaggle API")
            return None
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"❌ Ошибка аутентификации Kaggle API: {e}")
        return None
    
    # Скачиваем датасет
    print("\nСкачивание датасета watchman/rtsd-dataset...")
    print("Это может занять несколько минут...")
    
    try:
        Path(dataset_path).mkdir(exist_ok=True)
        api.dataset_download_files("watchman/rtsd-dataset", path=dataset_path, unzip=True)
        
        # Проверяем результат
        files = list(Path(dataset_path).rglob('*.jpg')) + list(Path(dataset_path).rglob('*.png'))
        if files:
            print(f"\n✓ Датасет успешно загружен!")
            print(f"  Путь: {dataset_path}")
            print(f"  Изображений: {len(files)}")
            return dataset_path
        else:
            print("⚠ Датасет загружен, но изображения не найдены")
            return dataset_path
            
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None

if __name__ == "__main__":
    dataset_path = download_rtsd_direct()
    if dataset_path:
        print(f"\n✓ Датасет готов: {dataset_path}")
        print("\nТеперь можно запустить обучение:")
        print("python train_rtsd.py")
    else:
        print("\n❌ Не удалось загрузить датасет")

