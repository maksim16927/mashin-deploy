"""
Основные модули проекта для распознавания дорожной инфраструктуры
"""

# Убираем импорт несуществующего road_detector
# from .road_detector import ...

# Импортируем только существующие модули
try:
    from .gps_extractor import GPSExtractor
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False

try:
    from .road_passport import RoadPassport
    PASSPORT_AVAILABLE = True
except ImportError:
    PASSPORT_AVAILABLE = False

__all__ = []
if GPS_AVAILABLE:
    __all__.append('GPSExtractor')
if PASSPORT_AVAILABLE:
    __all__.append('RoadPassport')
