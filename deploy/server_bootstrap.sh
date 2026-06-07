#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Полный bootstrap MAX-бота на чистом Ubuntu-сервере (только CPU).
# Запуск (в веб-консоли хостера, под root):
#   curl -fsSL https://raw.githubusercontent.com/maksim16927/mashin-deploy/main/deploy/server_bootstrap.sh | MAX_BOT_TOKEN=ВАШ_ТОКЕН bash
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="https://github.com/maksim16927/mashin-deploy.git"
REL="https://github.com/maksim16927/mashin-deploy/releases/download/models-v1"
APP_DIR="/opt/mashin"

if [ -z "${MAX_BOT_TOKEN:-}" ]; then
    echo "❌ Не задан MAX_BOT_TOKEN. Запустите так:"
    echo "   curl -fsSL .../server_bootstrap.sh | MAX_BOT_TOKEN=ВАШ_ТОКЕН bash"
    exit 1
fi

echo "==> [1/6] Системные пакеты…"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git python3 python3-venv python3-dev \
    ffmpeg libgl1 libglib2.0-0 ca-certificates curl

echo "==> [2/6] Код из GitHub в $APP_DIR…"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch --depth 1 origin main
    git -C "$APP_DIR" reset --hard origin/main
else
    git clone --depth 1 "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> [3/6] Скачиваю модели…"
mkdir -p models
for m in russian_signs_yolo12m.pt yolov8n.pt yolov8n_road_marking.pt; do
    if [ ! -s "models/$m" ]; then
        echo "    - $m"
        curl -fSL --retry 3 -o "models/$m" "$REL/$m"
    fi
done

echo "==> [4/6] Виртуальное окружение и зависимости…"
python3 -m venv .venv-ubuntu
./.venv-ubuntu/bin/pip install --upgrade pip wheel
./.venv-ubuntu/bin/pip install -r requirements.txt
./.venv-ubuntu/bin/python -c "import maxapi, torch, cv2, ultralytics; print('    deps OK')"

echo "==> [5/6] systemd-сервис…"
cat > /etc/systemd/system/mashin-max-bot.service <<UNIT
[Unit]
Description=MAX bot — распознавание дорожной инфраструктуры (CPU)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
Environment=BOT_DEVICE=cpu
Environment=MAX_BOT_TOKEN=$MAX_BOT_TOKEN
ExecStart=$APP_DIR/.venv-ubuntu/bin/python $APP_DIR/run_max_bot.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now mashin-max-bot

echo "==> [6/6] Готово. Статус:"
sleep 3
systemctl --no-pager status mashin-max-bot | head -12 || true
echo
echo "✅ Бот запущен. Логи:  journalctl -u mashin-max-bot -f"
