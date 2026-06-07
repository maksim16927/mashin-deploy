"""
Модуль создания ситуационного плана дороги
Генерирует схематичное изображение с расположением всех объектов
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon
import io
from PIL import Image


class SituationalPlanGenerator:
    """Генератор ситуационного плана дороги"""
    
    def __init__(self, plan_width: int = 1200, plan_height: int = 800):
        """
        Инициализация генератора
        
        Args:
            plan_width: Ширина плана в пикселях
            plan_height: Высота плана в пикселях
        """
        self.plan_width = plan_width
        self.plan_height = plan_height
        self.road_width = 300  # Ширина дороги на плане
        self.scale_factor = 1.0  # Масштаб (метры на пиксель)
        
    def create_plan(self, 
                   detections: List[Dict], 
                   video_info: Dict,
                   picket_range: Tuple[float, float]) -> np.ndarray:
        """
        Создание ситуационного плана
        
        Args:
            detections: Список всех детекций с пикетажем
            video_info: Информация о видео (fps, длительность и т.д.)
            picket_range: Диапазон пикетажа (начало, конец)
            
        Returns:
            np.ndarray: Изображение ситуационного плана
        """
        # Создаем фигуру matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(self.plan_width/100, self.plan_height/100), dpi=100)
        
        # Настройка осей
        ax.set_xlim(0, self.plan_width)
        ax.set_ylim(0, self.plan_height)
        ax.set_aspect('equal')
        
        # Вычисляем масштаб
        picket_start, picket_end = picket_range
        total_distance = picket_end - picket_start
        self.scale_factor = total_distance / (self.plan_height * 0.8)  # 80% высоты для дороги
        
        # Рисуем основу плана
        self._draw_road_base(ax, picket_start, picket_end)
        
        # Группируем детекции по типам
        grouped_detections = self._group_detections_by_type(detections)
        
        # Рисуем объекты по типам
        self._draw_signs(ax, grouped_detections.get('sign', []), picket_start)
        self._draw_barriers(ax, grouped_detections.get('barrier', []), picket_start)
        self._draw_exits(ax, grouped_detections.get('exit', []), picket_start)
        self._draw_noise_strips(ax, grouped_detections.get('noise_strip', []), picket_start)
        self._draw_gantries(ax, grouped_detections.get('gantry', []), picket_start)
        
        # Добавляем легенду и подписи
        self._add_legend(ax)
        self._add_picket_scale(ax, picket_start, picket_end)
        self._add_title(ax, video_info, picket_range)
        
        # Убираем оси
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Сохраняем во временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        plt.savefig(temp_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Загружаем как numpy array
        buf = cv2.imread(temp_path)
        
        # Удаляем временный файл
        import os
        os.unlink(temp_path)
        
        return buf
    
    def _draw_road_base(self, ax, picket_start: float, picket_end: float):
        """Рисование основы дороги"""
        # Центр дороги
        road_center_x = self.plan_width // 2
        road_top = self.plan_height * 0.9
        road_bottom = self.plan_height * 0.1
        
        # Основное полотно дороги (серый прямоугольник)
        road_rect = Rectangle(
            (road_center_x - self.road_width//2, road_bottom),
            self.road_width, road_top - road_bottom,
            facecolor='#808080', edgecolor='black', linewidth=2
        )
        ax.add_patch(road_rect)
        
        # Центральная разделительная полоса (желтая пунктирная)
        dash_length = 20
        gap_length = 15
        y_current = road_bottom
        
        while y_current < road_top:
            dash_rect = Rectangle(
                (road_center_x - 2, y_current),
                4, min(dash_length, road_top - y_current),
                facecolor='yellow', edgecolor='none'
            )
            ax.add_patch(dash_rect)
            y_current += dash_length + gap_length
        
        # Боковые линии (белые сплошные)
        # Левая линия
        left_line = Rectangle(
            (road_center_x - self.road_width//2 - 2, road_bottom),
            4, road_top - road_bottom,
            facecolor='white', edgecolor='none'
        )
        ax.add_patch(left_line)
        
        # Правая линия
        right_line = Rectangle(
            (road_center_x + self.road_width//2 - 2, road_bottom),
            4, road_top - road_bottom,
            facecolor='white', edgecolor='none'
        )
        ax.add_patch(right_line)
    
    def _group_detections_by_type(self, detections: List[Dict]) -> Dict[str, List[Dict]]:
        """Группировка детекций по типам"""
        grouped = {}
        
        for detection in detections:
            obj_type = detection.get('object_type', 'unknown')
            if obj_type not in grouped:
                grouped[obj_type] = []
            grouped[obj_type].append(detection)
        
        return grouped
    
    def _draw_signs(self, ax, signs: List[Dict], picket_start: float):
        """Рисование дорожных знаков"""
        road_center_x = self.plan_width // 2
        road_bottom = self.plan_height * 0.1
        road_height = self.plan_height * 0.8
        
        for sign in signs:
            picket = sign.get('picket', 0)
            if picket is None:
                continue
                
            # Вычисляем Y координату на основе пикетажа
            y_pos = road_bottom + (picket - picket_start) / self.scale_factor
            
            if road_bottom <= y_pos <= road_bottom + road_height:
                # Определяем сторону (левая/правая) на основе bbox
                bbox = sign.get('bbox', (0, 0, 0, 0))
                x_center = (bbox[0] + bbox[2]) / 2
                
                # Предполагаем, что центр кадра - это центр дороги
                if x_center < 320:  # Левая сторона (для кадра 640px)
                    x_pos = road_center_x - self.road_width//2 - 40
                else:  # Правая сторона
                    x_pos = road_center_x + self.road_width//2 + 20
                
                # Рисуем знак как треугольник
                triangle = np.array([
                    [x_pos, y_pos - 10],
                    [x_pos - 8, y_pos + 10],
                    [x_pos + 8, y_pos + 10]
                ])
                
                sign_patch = Polygon(triangle, facecolor='red', edgecolor='darkred', linewidth=1)
                ax.add_patch(sign_patch)
                
                # Подпись
                class_name = sign.get('class_name', 'SIGN')[:8]  # Обрезаем длинные названия
                ax.text(x_pos + 15, y_pos, class_name, fontsize=6, 
                       verticalalignment='center', color='darkred', weight='bold')
    
    def _draw_barriers(self, ax, barriers: List[Dict], picket_start: float):
        """Рисование барьеров"""
        road_center_x = self.plan_width // 2
        road_bottom = self.plan_height * 0.1
        road_height = self.plan_height * 0.8
        
        for barrier in barriers:
            picket = barrier.get('picket', 0)
            if picket is None:
                continue
                
            y_pos = road_bottom + (picket - picket_start) / self.scale_factor
            
            if road_bottom <= y_pos <= road_bottom + road_height:
                # Барьеры обычно по центру или краям дороги
                bbox = barrier.get('bbox', (0, 0, 0, 0))
                x_center = (bbox[0] + bbox[2]) / 2
                
                if x_center < 200:  # Левый край
                    x_pos = road_center_x - self.road_width//3
                elif x_center > 440:  # Правый край
                    x_pos = road_center_x + self.road_width//3
                else:  # Центр
                    x_pos = road_center_x
                
                # Рисуем барьер как прямоугольник
                barrier_rect = Rectangle(
                    (x_pos - 15, y_pos - 3),
                    30, 6,
                    facecolor='orange', edgecolor='darkorange', linewidth=1
                )
                ax.add_patch(barrier_rect)
                
                # Подпись
                ax.text(x_pos + 20, y_pos, 'BARRIER', fontsize=5, 
                       verticalalignment='center', color='darkorange', weight='bold')
    
    def _draw_exits(self, ax, exits: List[Dict], picket_start: float):
        """Рисование съездов"""
        road_center_x = self.plan_width // 2
        road_bottom = self.plan_height * 0.1
        road_height = self.plan_height * 0.8
        
        for exit_obj in exits:
            picket = exit_obj.get('picket', 0)
            if picket is None:
                continue
                
            y_pos = road_bottom + (picket - picket_start) / self.scale_factor
            
            if road_bottom <= y_pos <= road_bottom + road_height:
                # Определяем сторону съезда
                bbox = exit_obj.get('bbox', (0, 0, 0, 0))
                x_center = (bbox[0] + bbox[2]) / 2
                
                if x_center < 320:  # Левый съезд
                    x_start = road_center_x - self.road_width//2
                    x_end = road_center_x - self.road_width//2 - 60
                else:  # Правый съезд
                    x_start = road_center_x + self.road_width//2
                    x_end = road_center_x + self.road_width//2 + 60
                
                # Рисуем съезд как стрелку
                # Основная линия
                ax.plot([x_start, x_end], [y_pos, y_pos], 
                       color='blue', linewidth=3, alpha=0.8)
                
                # Стрелка
                arrow_size = 8
                if x_end > x_start:  # Правый съезд
                    arrow_x = [x_end - arrow_size, x_end, x_end - arrow_size]
                    arrow_y = [y_pos - arrow_size//2, y_pos, y_pos + arrow_size//2]
                else:  # Левый съезд
                    arrow_x = [x_end + arrow_size, x_end, x_end + arrow_size]
                    arrow_y = [y_pos - arrow_size//2, y_pos, y_pos + arrow_size//2]
                
                ax.plot(arrow_x, arrow_y, color='blue', linewidth=2)
                
                # Подпись
                ax.text(x_end, y_pos + 15, 'EXIT', fontsize=6, 
                       horizontalalignment='center', color='blue', weight='bold')
    
    def _draw_noise_strips(self, ax, noise_strips: List[Dict], picket_start: float):
        """Рисование шумовых полос"""
        road_center_x = self.plan_width // 2
        road_bottom = self.plan_height * 0.1
        road_height = self.plan_height * 0.8
        
        for strip in noise_strips:
            picket = strip.get('picket', 0)
            if picket is None:
                continue
                
            y_pos = road_bottom + (picket - picket_start) / self.scale_factor
            
            if road_bottom <= y_pos <= road_bottom + road_height:
                # Шумовые полосы идут поперек всей дороги
                strip_rect = Rectangle(
                    (road_center_x - self.road_width//2, y_pos - 2),
                    self.road_width, 4,
                    facecolor='brown', edgecolor='saddlebrown', linewidth=1, alpha=0.7
                )
                ax.add_patch(strip_rect)
                
                # Подпись (только для первой полосы, чтобы не загромождать)
                if strip == noise_strips[0]:
                    ax.text(road_center_x + self.road_width//2 + 10, y_pos, 
                           'NOISE STRIPS', fontsize=5, 
                           verticalalignment='center', color='saddlebrown', weight='bold')
    
    def _draw_gantries(self, ax, gantries: List[Dict], picket_start: float):
        """Рисование опор и порталов"""
        road_center_x = self.plan_width // 2
        road_bottom = self.plan_height * 0.1
        road_height = self.plan_height * 0.8
        
        for gantry in gantries:
            picket = gantry.get('picket', 0)
            if picket is None:
                continue
                
            y_pos = road_bottom + (picket - picket_start) / self.scale_factor
            
            if road_bottom <= y_pos <= road_bottom + road_height:
                # Рисуем П-образную опору
                # Левая стойка
                left_post = Rectangle(
                    (road_center_x - self.road_width//2 - 20, y_pos - 5),
                    8, 10,
                    facecolor='gray', edgecolor='darkgray', linewidth=1
                )
                ax.add_patch(left_post)
                
                # Правая стойка
                right_post = Rectangle(
                    (road_center_x + self.road_width//2 + 12, y_pos - 5),
                    8, 10,
                    facecolor='gray', edgecolor='darkgray', linewidth=1
                )
                ax.add_patch(right_post)
                
                # Перекладина
                beam = Rectangle(
                    (road_center_x - self.road_width//2 - 20, y_pos + 3),
                    self.road_width + 40, 4,
                    facecolor='gray', edgecolor='darkgray', linewidth=1
                )
                ax.add_patch(beam)
                
                # Подпись
                ax.text(road_center_x, y_pos + 15, 'GANTRY', fontsize=6, 
                       horizontalalignment='center', color='darkgray', weight='bold')
    
    def _add_legend(self, ax):
        """Добавление легенды"""
        legend_x = 50
        legend_y = self.plan_height - 50
        
        # Заголовок легенды (на английском)
        ax.text(legend_x, legend_y, 'LEGEND:', 
               fontsize=8, weight='bold', color='black')
        
        legend_items = [
            ('Road signs', 'red', 'triangle'),
            ('Barriers', 'orange', 'rectangle'),
            ('Exits', 'blue', 'arrow'),
            ('Noise strips', 'brown', 'rectangle'),
            ('Gantries', 'gray', 'rectangle')
        ]
        
        for i, (name, color, shape) in enumerate(legend_items):
            y_item = legend_y - 20 - i * 15
            
            # Символ
            if shape == 'triangle':
                triangle = np.array([
                    [legend_x + 10, y_item - 3],
                    [legend_x + 7, y_item + 3],
                    [legend_x + 13, y_item + 3]
                ])
                symbol = Polygon(triangle, facecolor=color, edgecolor='black')
                ax.add_patch(symbol)
            elif shape == 'rectangle':
                symbol = Rectangle((legend_x + 8, y_item - 2), 6, 4, 
                                 facecolor=color, edgecolor='black')
                ax.add_patch(symbol)
            elif shape == 'arrow':
                ax.plot([legend_x + 8, legend_x + 14], [y_item, y_item], 
                       color=color, linewidth=2)
                ax.plot([legend_x + 12, legend_x + 14, legend_x + 12], 
                       [y_item - 2, y_item, y_item + 2], color=color, linewidth=2)
            
            # Текст
            ax.text(legend_x + 20, y_item, name, fontsize=6, 
                   verticalalignment='center', color='black')
    
    def _add_picket_scale(self, ax, picket_start: float, picket_end: float):
        """Добавление шкалы пикетажа"""
        road_bottom = self.plan_height * 0.1
        road_height = self.plan_height * 0.8
        
        # Вертикальная шкала справа
        scale_x = self.plan_width - 80
        
        # Рисуем основную линию шкалы
        ax.plot([scale_x, scale_x], [road_bottom, road_bottom + road_height], 
               color='black', linewidth=2)
        
        # Добавляем деления каждые 100 метров
        current_picket = int(picket_start // 100) * 100  # Округляем до сотен
        
        while current_picket <= picket_end:
            if current_picket >= picket_start:
                y_pos = road_bottom + (current_picket - picket_start) / self.scale_factor
                
                if road_bottom <= y_pos <= road_bottom + road_height:
                    # Деление
                    ax.plot([scale_x - 5, scale_x + 5], [y_pos, y_pos], 
                           color='black', linewidth=1)
                    
                    # Подпись пикетажа (на английском для совместимости)
                    km = int(current_picket // 1000)
                    m = int(current_picket % 1000)
                    label = f"KM {km}+{m:03d}"
                    
                    ax.text(scale_x + 10, y_pos, label, fontsize=5, 
                           verticalalignment='center', color='black')
            
            current_picket += 100
        
        # Заголовок шкалы (на английском)
        ax.text(scale_x, road_bottom + road_height + 20, 'PICKET', 
               fontsize=7, horizontalalignment='center', weight='bold', color='black')
    
    def _add_title(self, ax, video_info: Dict, picket_range: Tuple[float, float]):
        """Добавление заголовка плана"""
        title_y = self.plan_height - 20
        
        # Основной заголовок (на английском)
        ax.text(self.plan_width // 2, title_y, 
               'ROAD SITUATIONAL PLAN', 
               fontsize=12, horizontalalignment='center', weight='bold', color='black')
        
        # Информация о видео (на английском)
        picket_start, picket_end = picket_range
        total_distance = picket_end - picket_start
        
        info_text = f"Section: KM {int(picket_start//1000)}+{int(picket_start%1000):03d} - KM {int(picket_end//1000)}+{int(picket_end%1000):03d} (~{total_distance:.0f}m)"
        
        ax.text(self.plan_width // 2, title_y - 15, info_text, 
               fontsize=8, horizontalalignment='center', color='black')


def create_situational_plan(detections: List[Dict], 
                          video_info: Dict,
                          output_path: Optional[str] = None) -> str:
    """
    Создание ситуационного плана из детекций
    
    Args:
        detections: Список всех детекций с пикетажем
        video_info: Информация о видео
        output_path: Путь для сохранения плана
        
    Returns:
        str: Путь к созданному файлу плана
    """
    if not detections:
        raise ValueError("Нет детекций для создания плана")
    
    # Определяем диапазон пикетажа
    pickets = [d.get('picket', 0) for d in detections if d.get('picket') is not None]
    if not pickets:
        raise ValueError("Нет данных о пикетаже")
    
    picket_start = min(pickets)
    picket_end = max(pickets)
    
    # Создаем генератор плана
    generator = SituationalPlanGenerator()
    
    # Генерируем план
    plan_image = generator.create_plan(detections, video_info, (picket_start, picket_end))
    
    # Сохраняем
    if output_path is None:
        from datetime import datetime
        from pathlib import Path
        import os
        
        # Получаем корневую директорию проекта
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent  # До mashin/
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "videos" / "situational_plans" / f"situational_plan_{timestamp}.png"
    
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Конвертируем BGR в RGB для правильного сохранения
    plan_rgb = cv2.cvtColor(plan_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(output_path), plan_rgb)
    
    return str(output_path)