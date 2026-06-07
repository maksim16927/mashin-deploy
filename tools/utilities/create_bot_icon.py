#!/usr/bin/env python3
"""
Создание иконки для Telegram бота
Создает изображение с символом дорожной инфраструктуры
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_bot_icon():
    """Создает иконку для бота"""
    
    # Размеры иконки (512x512 для Telegram)
    size = 512
    icon = Image.new('RGB', (size, size), color='#1E88E5')  # Синий фон
    draw = ImageDraw.Draw(icon)
    
    # Рисуем дорогу (горизонтальная полоса)
    road_y = size // 2
    road_height = size // 4
    draw.rectangle([0, road_y - road_height//2, size, road_y + road_height//2], 
                   fill='#424242')  # Серый цвет дороги
    
    # Рисуем разметку (желтые линии)
    line_width = 8
    line_y = road_y
    # Центральная линия
    draw.rectangle([size//4, line_y - line_width//2, 3*size//4, line_y + line_width//2], 
                   fill='#FFD700')  # Желтый
    
    # Рисуем дорожный знак (треугольник)
    sign_size = size // 3
    sign_x = size // 2
    sign_y = size // 3
    
    # Треугольник знака
    sign_points = [
        (sign_x, sign_y - sign_size//2),  # Верх
        (sign_x - sign_size//2, sign_y + sign_size//2),  # Слева внизу
        (sign_x + sign_size//2, sign_y + sign_size//2),  # Справа внизу
    ]
    draw.polygon(sign_points, fill='#FF6B6B', outline='#FFFFFF', width=4)
    
    # Рисуем барьер (вертикальные линии по краям)
    barrier_width = 6
    barrier_height = size // 3
    barrier_y = road_y - barrier_height // 2
    
    # Левый барьер
    draw.rectangle([size//8, barrier_y, size//8 + barrier_width, barrier_y + barrier_height], 
                   fill='#FF8C00')  # Оранжевый
    
    # Правый барьер
    draw.rectangle([7*size//8 - barrier_width, barrier_y, 7*size//8, barrier_y + barrier_height], 
                   fill='#FF8C00')  # Оранжевый
    
    # Сохраняем иконку
    icon_path = "bot_icon.png"
    icon.save(icon_path, "PNG")
    print(f"✅ Иконка создана: {icon_path}")
    print(f"📐 Размер: {size}x{size} пикселей")
    print(f"💡 Загрузите файл bot_icon.png в настройки бота через @BotFather")
    
    return icon_path

if __name__ == "__main__":
    create_bot_icon()
