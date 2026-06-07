#!/usr/bin/env python3
"""
Запуск Telegram бота для обработки видео
"""

import importlib.util
from pathlib import Path

# Загружаем бот без импорта локального пакета telegram (чтобы не конфликтовать с python-telegram-bot)
bot_path = Path(__file__).parent / "src" / "telegram" / "bot.py"
spec = importlib.util.spec_from_file_location("mashin_telegram_bot", bot_path)
bot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot_module)
main = bot_module.main

if __name__ == "__main__":
    main()
