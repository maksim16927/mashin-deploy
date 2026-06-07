#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║    🚀 ПОЛНАЯ ОБРАБОТКА ВСЕХ ВИДЕО - ВСЕ МОДЕЛИ ВМЕСТЕ                  ║"
echo "║    • Дорожные знаки (155 классов RTSD)                                  ║"
echo "║    • Инфраструктура (12 классов COCO)                                   ║"
echo "║    • Дорожная разметка (OpenCV)                                         ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📹 Видео в папке videos/original/:"
ls -lh videos/original/

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "🚀 ЗАПУСК ПОЛНОЙ СИСТЕМЫ ДЕТЕКЦИИ"
echo "════════════════════════════════════════════════════════════════════════════"

# Счетчик
count=0
total=$(ls -1 videos/original/* 2>/dev/null | wc -l)

# Обрабатываем каждое видео
for video in videos/original/*; do
    if [ -f "$video" ]; then
        count=$((count + 1))
        filename=$(basename "$video")
        name="${filename%.*}"
        
        echo ""
        echo "┌────────────────────────────────────────────────────────────────────────────┐"
        echo "│ 📹 Видео $count/$total: $filename"
        echo "└────────────────────────────────────────────────────────────────────────────┘"
        
        python3 scripts/detect_full_system.py "$video"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Готово: videos/detected/${name}_FULL_DETECTION.mp4"
        else
            echo ""
            echo "❌ Ошибка при обработке $filename"
        fi
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ ВСЕ ВИДЕО ОБРАБОТАНЫ!"
echo "════════════════════════════════════════════════════════════════════════════"

echo ""
echo "📊 Результаты в videos/detected/:"
ls -lh videos/detected/*FULL_DETECTION.mp4 2>/dev/null

echo ""
echo "🎉 Готово! Все модели работали вместе!"
echo ""
echo "💡 Откройте видео для просмотра:"
echo "   open videos/detected/*FULL_DETECTION.mp4"
