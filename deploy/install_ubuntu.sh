#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Установка MAX-бота распознавания дорожной инфраструктуры на Ubuntu (только CPU)
# Запуск:  bash deploy/install_ubuntu.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"
echo "==> Каталог приложения: $APP_DIR"

# 1. Системные зависимости (ffmpeg + библиотеки для OpenCV на headless-сервере)
echo "==> Устанавливаю системные пакеты (apt)…"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    ffmpeg \
    libgl1 libglib2.0-0 \
    ca-certificates

# 2. Виртуальное окружение
if [ ! -d ".venv-ubuntu" ]; then
    echo "==> Создаю виртуальное окружение .venv-ubuntu…"
    python3 -m venv .venv-ubuntu
fi
# shellcheck disable=SC1091
source .venv-ubuntu/bin/activate

# 3. Python-зависимости (torch CPU-сборка ставится автоматически через ultralytics)
echo "==> Обновляю pip и ставлю зависимости…"
pip install --upgrade pip wheel
pip install -r requirements.txt

echo
echo "✅ Установка завершена."
echo "   Проверка импорта бота:"
python -c "import maxapi, torch, cv2, ultralytics; print('  maxapi/torch/cv2/ultralytics — OK')"
echo
echo "Запуск вручную:"
echo "   MAX_BOT_TOKEN=<токен> .venv-ubuntu/bin/python run_max_bot.py"
echo
echo "Либо как systemd-сервис — см. deploy/mashin-max-bot.service"
