#!/usr/bin/env python3
"""
Запуск Max-бота (max.ru) для обработки видео
"""

import importlib.util
from pathlib import Path

bot_path = Path(__file__).parent / "src" / "max" / "bot.py"
spec = importlib.util.spec_from_file_location("mashin_max_bot", bot_path)
bot_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot_module)

if __name__ == "__main__":
    bot_module.main()
