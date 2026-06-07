#!/bin/bash

# Скрипт для автоматической обработки всех видео

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║    🎬 АВТОМАТИЧЕСКАЯ ОБРАБОТКА ВСЕХ ВИДЕО                               ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📹 Найдены видео в videos/original/:"
ls -lh videos/original/

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "1️⃣  ДЕТЕКЦИЯ ИНФРАСТРУКТУРЫ (COCO: автобусы, машины, люди)"
echo "════════════════════════════════════════════════════════════════════════════"

# Обрабатываем каждое видео
for video in videos/original/*; do
    if [ -f "$video" ]; then
        filename=$(basename "$video")
        name="${filename%.*}"
        
        echo ""
        echo "⏳ Обработка: $filename"
        echo "────────────────────────────────────────────────────────────────────────────"
        
        python3 scripts/detect_coco_infrastructure.py "$video" \
            -o "videos/detected/${name}_coco_infra.mp4"
        
        if [ $? -eq 0 ]; then
            echo "✅ Готово: videos/detected/${name}_coco_infra.mp4"
        else
            echo "❌ Ошибка при обработке $filename"
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "2️⃣  ДЕТЕКЦИЯ ЗНАКОВ (RTSD: 155 классов)"
echo "════════════════════════════════════════════════════════════════════════════"

for video in videos/original/*; do
    if [ -f "$video" ]; then
        filename=$(basename "$video")
        name="${filename%.*}"
        
        echo ""
        echo "⏳ Обработка: $filename"
        echo "────────────────────────────────────────────────────────────────────────────"
        
        python3 scripts/detect_video.py "$video" \
            -o "videos/detected/${name}_signs.mp4"
        
        if [ $? -eq 0 ]; then
            echo "✅ Готово: videos/detected/${name}_signs.mp4"
        else
            echo "❌ Ошибка при обработке $filename"
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "3️⃣  ПОЛНАЯ ДЕТЕКЦИЯ (знаки + разметка)"
echo "════════════════════════════════════════════════════════════════════════════"

for video in videos/original/*; do
    if [ -f "$video" ]; then
        filename=$(basename "$video")
        name="${filename%.*}"
        
        echo ""
        echo "⏳ Обработка: $filename"
        echo "────────────────────────────────────────────────────────────────────────────"
        
        python3 scripts/detect_all.py "$video" \
            -o "videos/detected/${name}_full.mp4"
        
        if [ $? -eq 0 ]; then
            echo "✅ Готово: videos/detected/${name}_full.mp4"
        else
            echo "❌ Ошибка при обработке $filename"
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ ВСЕ ВИДЕО ОБРАБОТАНЫ!"
echo "════════════════════════════════════════════════════════════════════════════"

echo ""
echo "📊 Результаты:"
ls -lh videos/detected/

echo ""
echo "🎉 Готово! Откройте папку videos/detected/ для просмотра"
