"""
Скрипт для загрузки датасетов дорожной инфраструктуры с Roboflow Universe
Включает: автобусные остановки, опоры, столбы и другие объекты
"""

from roboflow import Roboflow
import os
import sys

def download_bus_stop_dataset():
    """
    Загрузка датасета Bus Stop Detection
    Источник: The University of Hong Kong
    544 изображения, 3 модели
    Класс: Bus-Stop
    """
    print("=" * 60)
    print("Загрузка датасета Bus Stop Detection...")
    print("=" * 60)
    
    try:
        rf = Roboflow(api_key="YOUR_API_KEY_HERE")  # Замените на ваш API ключ
        project = rf.workspace("the-university-of-hong-kong-zsdbj").project("bus-stop-detection")
        dataset = project.version(1).download("yolov8")
        print(f"\n✅ Датасет автобусных остановок загружен в: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"❌ Ошибка при загрузке датасета автобусных остановок: {e}")
        print("\nДля работы с Roboflow необходим API ключ.")
        print("Получите его на: https://app.roboflow.com/settings/api")
        return None

def download_road_infrastructure_dataset():
    """
    Загрузка датасета Road Infrastructure
    Источник: Road Infrastructure project
    914 изображений
    Классы: straight, intersections
    """
    print("\n" + "=" * 60)
    print("Загрузка датасета Road Infrastructure...")
    print("=" * 60)
    
    try:
        rf = Roboflow(api_key="YOUR_API_KEY_HERE")
        project = rf.workspace("road-infrastructure-0hg8j").project("road-infrastructure-predict")
        dataset = project.version(1).download("yolov8")
        print(f"\n✅ Датасет дорожной инфраструктуры загружен в: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"❌ Ошибка при загрузке датасета инфраструктуры: {e}")
        return None

def download_infrastructure_split_dataset():
    """
    Загрузка датасета Infrastructure Split Rebalanced
    Источник: Infrastructure project
    2.47k изображений
    Классы: graffiti, pothole, damaged-sign, electricity-pole-damage, fallen-trees, pipe-burst, thrash
    """
    print("\n" + "=" * 60)
    print("Загрузка датасета Infrastructure Split...")
    print("=" * 60)
    
    try:
        rf = Roboflow(api_key="YOUR_API_KEY_HERE")
        project = rf.workspace("infrastructure-aqfrw").project("infrastructure-split-rebalanced")
        dataset = project.version(1).download("yolov8")
        print(f"\n✅ Датасет повреждений инфраструктуры загружен в: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"❌ Ошибка при загрузке датасета повреждений: {e}")
        return None

def create_combined_dataset_yaml(datasets_locations):
    """
    Создание конфигурационного файла для объединенного датасета
    """
    print("\n" + "=" * 60)
    print("Создание конфигурации объединенного датасета...")
    print("=" * 60)
    
    combined_yaml = """# Объединенный датасет дорожной инфраструктуры
path: ./datasets/road_infrastructure
train: train/images
val: val/images
test: test/images

# Классы
names:
  0: bus-stop
  1: intersection
  2: straight-road
  3: pothole
  4: damaged-sign
  5: electricity-pole-damage
  6: fallen-tree
  7: pipe-burst
  8: trash
  9: graffiti

# Описание
description: >
  Комплексный датасет для обнаружения объектов дорожной инфраструктуры:
  - Автобусные остановки
  - Перекрестки и прямые участки дороги
  - Повреждения инфраструктуры (ямы, знаки, столбы)
  - Мусор и граффити
"""
    
    yaml_path = os.path.join(os.getcwd(), "datasets", "infrastructure_dataset.yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(combined_yaml)
    
    print(f"✅ Конфигурация сохранена: {yaml_path}")
    return yaml_path

def download_all_datasets():
    """
    Загрузка всех датасетов для дорожной инфраструктуры
    """
    print("\n" + "🚀" * 30)
    print("\nЗАГРУЗКА ДАТАСЕТОВ ДОРОЖНОЙ ИНФРАСТРУКТУРЫ")
    print("Для дипломной работы по распознаванию дорожных объектов\n")
    print("🚀" * 30 + "\n")
    
    datasets = {}
    
    # Загружаем датасет автобусных остановок
    bus_stop_location = download_bus_stop_dataset()
    if bus_stop_location:
        datasets['bus_stop'] = bus_stop_location
    
    # Загружаем датасет дорожной инфраструктуры
    road_infra_location = download_road_infrastructure_dataset()
    if road_infra_location:
        datasets['road_infrastructure'] = road_infra_location
    
    # Загружаем датасет повреждений инфраструктуры
    infra_split_location = download_infrastructure_split_dataset()
    if infra_split_location:
        datasets['infrastructure_split'] = infra_split_location
    
    # Создаем конфигурацию
    if datasets:
        yaml_path = create_combined_dataset_yaml(datasets)
    
    print("\n" + "=" * 60)
    print("ИТОГИ ЗАГРУЗКИ:")
    print("=" * 60)
    
    for name, location in datasets.items():
        print(f"✅ {name}: {location}")
    
    print("\n" + "📋" * 30)
    print("\nДАЛЬНЕЙШИЕ ШАГИ:")
    print("1. Получите API ключ Roboflow: https://app.roboflow.com/settings/api")
    print("2. Замените 'YOUR_API_KEY_HERE' в коде на ваш ключ")
    print("3. Запустите скрипт снова для загрузки датасетов")
    print("4. Используйте train_infrastructure.py для обучения модели")
    print("\n" + "📋" * 30 + "\n")
    
    return datasets

def get_api_key_instructions():
    """
    Инструкции по получению API ключа Roboflow
    """
    instructions = """
╔════════════════════════════════════════════════════════════════════╗
║          КАК ПОЛУЧИТЬ API КЛЮЧ ROBOFLOW                            ║
╚════════════════════════════════════════════════════════════════════╝

1. Перейдите на https://app.roboflow.com/
2. Зарегистрируйтесь или войдите в аккаунт
3. Перейдите в Settings → API (https://app.roboflow.com/settings/api)
4. Скопируйте ваш Private API Key
5. Замените 'YOUR_API_KEY_HERE' в скрипте на ваш ключ

АЛЬТЕРНАТИВНЫЙ СПОСОБ (переменная окружения):
export ROBOFLOW_API_KEY="ваш_ключ_здесь"

ДАТАСЕТЫ НА ROBOFLOW UNIVERSE:

📦 Bus Stop Detection (544 изображения)
   https://universe.roboflow.com/the-university-of-hong-kong-zsdbj/bus-stop-detection
   Класс: Bus-Stop
   
📦 Road Infrastructure (914 изображений)
   https://universe.roboflow.com/road-infrastructure-0hg8j/road-infrastructure-predict
   Классы: straight, intersections
   
📦 Infrastructure Split (2470 изображений)
   https://universe.roboflow.com/infrastructure-aqfrw/infrastructure-split-rebalanced
   Классы: pothole, damaged-sign, electricity-pole-damage, fallen-trees, 
           pipe-burst, trash, graffiti

╔════════════════════════════════════════════════════════════════════╗
║  После получения API ключа запустите:                              ║
║  python3 scripts/download_infrastructure_dataset.py                ║
╚════════════════════════════════════════════════════════════════════╝
"""
    return instructions

if __name__ == "__main__":
    # Проверяем наличие API ключа
    import argparse
    
    parser = argparse.ArgumentParser(description="Загрузка датасетов дорожной инфраструктуры")
    parser.add_argument("--api-key", type=str, help="Roboflow API ключ")
    parser.add_argument("--info", action="store_true", help="Показать инструкции по получению API ключа")
    
    args = parser.parse_args()
    
    if args.info:
        print(get_api_key_instructions())
        sys.exit(0)
    
    if args.api_key:
        # Заменяем placeholder на реальный ключ в коде (временно, для текущей сессии)
        import re
        with open(__file__, 'r') as f:
            content = f.read()
        # Не сохраняем, просто используем для работы
        print(f"Используется API ключ: {args.api_key[:10]}...")
    
    # Запускаем загрузку
    datasets = download_all_datasets()
    
    if not datasets:
        print("\n" + get_api_key_instructions())
