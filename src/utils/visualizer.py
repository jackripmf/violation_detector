import cv2


class Visualizer:
    """Визуализация информации на кадрах"""

    def __init__(self):
        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 165, 255),
            "white": (255, 255, 255),
            "blue": (255, 0, 0),
            "purple": (255, 0, 255),
            "cyan": (255, 255, 0),
            "dark_green": (0, 100, 0),
            "dark_blue": (139, 0, 0),
            "dark_red": (0, 0, 139),
        }

    def _draw_movement_info(self, frame, movement_info):
        """Отрисовка информации о движении камеры"""
        rotation = movement_info.get("rotation", 0)
        translation = movement_info.get("translation", 0)
        reason = movement_info.get("reason", "Unknown")
        consecutive_frames = movement_info.get("consecutive_frames", 0)
        movement_duration = movement_info.get("movement_duration", 0)

        # Определение цвета и статуса движения
        if movement_info.get("filtered", False):
            if movement_info.get("movement_detected", False):
                move_color = self.colors["red"]
                move_status = "CAMERA MOVED!"
            else:
                if consecutive_frames > 0:
                    move_color = self.colors["orange"]
                    move_status = f"CAMERA MOVING...({consecutive_frames} frames)"
                else:
                    move_color = self.colors["green"]
                    move_status = "CAMERA STABLE"
        else:
            move_color = self.colors["yellow"]
            move_status = "MOVEMENT DETECTION..."

        # Статус движения
        cv2.putText(
            frame,
            f"MOVEMENT: {move_status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            move_color,
            2,
        )

        # Детальная информация о движении
        y_offset = 60
        info_lines = [
            f"Rotation: {rotation:+.1f}°",
            f"Translation: {translation:.3f}",
            f"Frames: {consecutive_frames}",
            f"Duration: {movement_duration:.1f}s",
        ]

        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                move_color,
                2,
            )
            y_offset += 25

        # Визуализация смещения
        if movement_info.get("movement_detected", False):
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            scale = 10
            translation_x = movement_info.get("translation_x", 0)
            translation_y = movement_info.get("translation_y", 0)
            end_x = int(center_x + translation_x * frame.shape[1] * scale)
            end_y = int(center_y + translation_y * frame.shape[0] * scale)
            cv2.arrowedLine(
                frame,
                (center_x, center_y),
                (end_x, end_y),
                self.colors["yellow"],
                3,
                tipLength=0.3,
            )


    def draw_detection_info(self, frame, result, total_duration, movement_info,
                            video_timestamp):
        """Отрисовка всей информации о детекции"""
        display_frame = frame.copy()

        # Получаем статистику
        stats = result.get("stats", {})

        # 1. Панель статистики в правом верхнем углу (ОБНОВЛЕНА)
        self._draw_stats_panel(display_frame, stats, movement_info)

        # 2. Движение камеры - ВЕРХНИЙ ЛЕВЫЙ УГОЛ
        self._draw_movement_info(display_frame, movement_info)

        # 3. Перекрытие камеры - ЦЕНТР СЛЕВА
        self._draw_obstruction_info(display_frame, result, total_duration)

        # 4. Время видео - ПРАВЫЙ НИЖНИЙ УГОЛ
        self._draw_video_time(display_frame, video_timestamp)

        # 5. Запрещенные объекты - bounding boxes
        if result.get("forbidden_objects"):
            for obj in result["forbidden_objects"]:
                self._draw_forbidden_object(display_frame, obj)

        # 6. DMS объекты - bounding boxes (НОВОЕ)
        if result.get("dms_objects"):
            for obj in result["dms_objects"]:
                self._draw_dms_object(display_frame, obj)

        # 7. YOLO объекты
        for obj in result["details"]["yolo"]["objects"]:
            self._draw_yolo_object(display_frame, obj)

        # 8. Метрики в нижней ЛЕВОЙ части
        self._draw_metrics(display_frame, result, display_frame.shape[0] - 150)

        # 9. Цветовые индикаторы статуса
        self._draw_status_indicator(display_frame, result, movement_info)

        return display_frame

    def _draw_stats_panel(self, frame, stats, movement_info):
        """Отрисовка панели статистики в правом верхнем углу"""
        right_x = frame.shape[1] - 320
        y_offset = 30
        line_height = 25

        # Увеличиваем высоту панели немного
        panel_width = 310
        panel_height = 200  # Увеличили на 1 строку
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (right_x - 10, y_offset - 10),
            (right_x + panel_width, y_offset + panel_height),
            (0, 0, 0),
            -1
        )

        # Наложение с прозрачностью
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Заголовок панели
        cv2.putText(
            frame,
            "VIOLATION STATISTICS",
            (right_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.colors["cyan"],
            2,
        )

        y_offset += line_height

        # 1. Статистика перекрытий
        obstruction_stats = stats.get("obstruction", {})
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Obstructions:",
            f"{obstruction_stats.get('total', 0)}",
            self.colors["orange"]
        )
        y_offset += line_height

        # Длительность текущего перекрытия
        if obstruction_stats.get("current_duration", 0) > 0:
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Current duration:",
                f"{obstruction_stats.get('current_duration', 0):.1f}s",
                self.colors["yellow"]
            )
            y_offset += line_height

        # 2. Статистика движений камеры
        movement_stats = stats.get("movement", {})
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Camera movements:",
            f"{movement_stats.get('total', 0)}",
            self.colors["purple"]
        )
        y_offset += line_height

        # Длительность текущего движения
        if movement_info.get("movement_detected", False):
            duration = movement_info.get("movement_duration", 0)
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Moving for:",
                f"{duration:.1f}s",
                self.colors["red"]
            )
            y_offset += line_height

        # 3. Статистика запрещенных объектов
        forbidden_stats = stats.get("forbidden_items", {})
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Forbidden items:",
            f"{forbidden_stats.get('total', 0)}",
            self.colors["red"]
        )
        y_offset += line_height

        # Текущие запрещенные объекты
        current_objects = forbidden_stats.get("current_objects", 0)
        if current_objects > 0:
            color = self.colors["red"] if current_objects > 0 else self.colors[
                "green"]
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Current objects:",
                f"{current_objects}",
                color
            )
            y_offset += line_height

        # 4. Статистика DMS - ПРОСТО СЧЕТЧИК (НОВОЕ!)
        dms_stats = stats.get("dms", {})
        dms_total = dms_stats.get("total", 0)
        dms_color = self.colors["purple"] if dms_total > 0 else self.colors[
            "white"]
        self._draw_stat_line(
            frame, right_x, y_offset,
            "DMS violations:",
            f"{dms_total}",
            dms_color
        )
        y_offset += line_height

        # Информация о кулдаунах
        active_cooldowns = forbidden_stats.get("active_cooldowns", {})
        if active_cooldowns:
            self._draw_stat_line(
                frame, right_x, y_offset,
                "Cooldowns active:",
                f"{len(active_cooldowns)}",
                self.colors["orange"]
            )
            y_offset += line_height

            # Показываем первые 2 кулдауна
            for i, (class_name, info) in enumerate(
                    list(active_cooldowns.items())[:2]):
                cooldown_left = info.get("cooldown_left", 0)
                self._draw_stat_line(
                    frame, right_x + 10, y_offset,
                    f"{class_name}:",
                    f"{cooldown_left:.0f}s",
                    self.colors["yellow"]
                )
                y_offset += line_height - 5

        # 5. Общая статистика
        self._draw_stat_line(
            frame, right_x, y_offset,
            "Total frames:",
            f"{stats.get('frame_count', 0)}",
            self.colors["white"]
        )
        y_offset += line_height

        self._draw_stat_line(
            frame, right_x, y_offset,
            "Total alerts:",
            f"{stats.get('alerts_count', 0)}",
            self.colors["yellow"]
        )

    def _draw_stat_line(self, frame, x, y, label, value, color):
        """Отрисовка одной строки статистики"""
        # Метка
        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.colors["white"],
            1,
        )

        # Значение
        value_x = x + 180
        cv2.putText(
            frame,
            value,
            (value_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _draw_obstruction_info(self, frame, result, total_duration):
        """Отрисовка информации о перекрытии"""
        y_offset = 180

        # Статус перекрытия
        status = "BLOCKED!" if result["obstructed"] else "NORMAL"
        color = self.colors["red"] if result["obstructed"] else self.colors[
            "green"]

        cv2.putText(
            frame,
            f"OBSTRUCTION: {status}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # Детали
        info_lines = [
            f"Confidence: {result['confidence']}",
            f"Detectors: {result['detectors_count']}/3",
            f"Duration: {total_duration:.1f}s",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (10, y_offset + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.colors["white"],
                1,
            )

    def _draw_video_time(self, frame, video_timestamp):
        """Отрисовка времени видео в правом нижнем углу"""
        # Форматируем время: MM:SS.ms
        minutes = int(video_timestamp // 60)
        seconds = int(video_timestamp % 60)
        milliseconds = int((video_timestamp * 1000) % 1000)

        time_text = f"Time: {minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        # Позиция в правом нижнем углу
        text_size = \
        cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x = frame.shape[1] - text_size[0] - 20
        y = frame.shape[0] - 50  # Выше FPS

        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - text_size[1] - 10),
            (x + text_size[0] + 10, y + 10),
            (0, 0, 0),
            -1
        )

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Текст времени
        cv2.putText(
            frame,
            time_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.colors["cyan"],
            2,
        )


    def _draw_yolo_object(self, frame, obj):
        """Отрисовка YOLO объекта"""
        x1, y1, x2, y2 = obj["bbox"]
        class_name = obj["class"]
        confidence = obj["confidence"]
        area_ratio = obj["area_ratio"]

        color = self.colors["red"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {confidence:.2f} ({area_ratio:.1%})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[
            0]

        cv2.rectangle(
            frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1),
            color, -1
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.colors["white"],
            1,
            cv2.LINE_AA,
        )

    def _draw_forbidden_object(self, frame, obj):
        """Отрисовка запрещенного объекта (упрощенная версия)"""
        if "bbox" not in obj or len(obj["bbox"]) != 4:
            return

        x1, y1, x2, y2 = obj["bbox"]
        class_name = obj.get("class", "Unknown")

        # Цвет для запрещенных объектов
        color = self.colors["purple"]

        # Более толстый bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Простая подпись (без фона для производительности)
        label = f"{class_name}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10 if y1 > 20 else y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_forbidden_stats(self, frame, stats):
        """Отрисовка статистики запрещенных объектов"""
        right_x = frame.shape[1] - 300
        y_offset = 30

        cv2.putText(
            frame,
            "FORBIDDEN ITEMS:",
            (right_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.colors["purple"],
            2,
        )

        cv2.putText(
            frame,
            f"Violations: {stats.get('total_violations', 0)}",
            (right_x, y_offset + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors["cyan"],
            1,
        )

        current_objects = stats.get('current_objects', 0)
        color = self.colors["red"] if current_objects > 0 else self.colors[
            "green"]

        cv2.putText(
            frame,
            f"Objects: {current_objects}",
            (right_x, y_offset + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
        )

    def _draw_metrics(self, frame, result, start_y):
        """Отрисовка метрик в нижней левой части"""
        metrics = result["details"]["cv"]["metrics"]

        y = start_y

        # Заголовок метрик
        cv2.putText(
            frame,
            "IMAGE METRICS:",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors["yellow"],
            1,
        )

        y += 25

        # Создаем компактную таблицу метрик
        metric_lines = [
            ("Brightness", f"{metrics['Brightness']:.1f}"),
            ("Contrast", f"{metrics['Contrast']:.1f}"),
            ("Sharpness", f"{metrics['Sharpness']:.1f}"),
        ]

        # Рисуем метрики в две колонки
        for i, (label, value) in enumerate(metric_lines):
            col = i % 2
            row = i // 2

            x = 10 + col * 150
            line_y = y + row * 25

            # Метка
            cv2.putText(
                frame,
                f"{label}:",
                (x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["white"],
                1,
            )

            # Значение
            cv2.putText(
                frame,
                value,
                (x + 100, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["yellow"],
                1,
            )

        # Дополнительные метрики (темные области, YOLO)
        dark_ratio = result["details"]["dark_area"]["metrics"]["dark_ratio"]
        yolo_ratio = result["details"]["yolo"]["metrics"][
            "large_objects_ratio"]

        cv2.putText(
            frame,
            f"Dark Areas: {dark_ratio:.3f}",
            (10, y + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.colors["white"],
            1,
        )

        cv2.putText(
            frame,
            f"YOLO Coverage: {yolo_ratio:.1%}",
            (10, y + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.colors["white"],
            1,
        )

    def _draw_status_indicator(self, frame, result, movement_info):
        """Отрисовка цветовых индикаторов статуса"""
        height, width = frame.shape[:2]

        # Индикатор в левом верхнем углу (5x5 пикселей)
        indicator_size = 15

        # 1. Индикатор перекрытия (красный/зеленый)
        obstruction_color = self.colors["red"] if result["obstructed"] else \
        self.colors["green"]
        cv2.rectangle(
            frame,
            (5, 5),
            (5 + indicator_size, 5 + indicator_size),
            obstruction_color,
            -1
        )

        # Подпись
        cv2.putText(
            frame,
            "O",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["white"],
            1
        )

        # 2. Индикатор движения (оранжевый/зеленый)
        if movement_info.get("movement_detected", False):
            movement_color = self.colors["red"]
        elif movement_info.get("consecutive_frames", 0) > 0:
            movement_color = self.colors["orange"]
        else:
            movement_color = self.colors["green"]

        cv2.rectangle(
            frame,
            (5, 25),
            (5 + indicator_size, 25 + indicator_size),
            movement_color,
            -1
        )

        # Подпись
        cv2.putText(
            frame,
            "M",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["white"],
            1
        )

        # 3. Индикатор запрещенных объектов (фиолетовый/серый)
        forbidden_objects = result.get("forbidden_objects", [])
        forbidden_color = self.colors["purple"] if forbidden_objects else (
        100, 100, 100)

        cv2.rectangle(
            frame,
            (5, 45),
            (5 + indicator_size, 45 + indicator_size),
            forbidden_color,
            -1
        )

        # Подпись
        cv2.putText(
            frame,
            "F",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["white"],
            1
        )

        # 4. Индикатор DMS (синий/серый)
        dms_violations = result.get("current_dms_violations", [])
        dms_color = self.colors["blue"] if dms_violations else (100, 100, 100)

        cv2.rectangle(
            frame,
            (5, 65),
            (5 + indicator_size, 65 + indicator_size),
            dms_color,
            -1
        )

        # Подпись
        cv2.putText(
            frame,
            "D",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["white"],
            1
        )

        # 5. Индикатор сигареты (оранжевый/серый)
        cigarette_detected = any(obj.get("class", "").lower() == "cigarette"
                                 for obj in result.get("dms_objects", []))
        cigarette_color = self.colors["orange"] if cigarette_detected else (
        100, 100, 100)

        cv2.rectangle(
            frame,
            (5, 85),
            (5 + indicator_size, 85 + indicator_size),
            cigarette_color,
            -1
        )

        # Подпись
        cv2.putText(
            frame,
            "C",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["white"],
            1
        )

        # 6. Индикатор телефона (фиолетовый/серый)
        phone_detected = any(obj.get("class", "").lower() == "phone"
                             for obj in result.get("dms_objects", []))
        phone_color = self.colors["purple"] if phone_detected else (
        100, 100, 100)

        cv2.rectangle(
            frame,
            (5, 105),
            (5 + indicator_size, 105 + indicator_size),
            phone_color,
            -1
        )

        # Подпись
        cv2.putText(
            frame,
            "P",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self.colors["white"],
            1
        )


    def _draw_dms_info(self, frame, dms_stats, dms_objects, violations):
        """Отрисовка информации DMS"""
        height, width = frame.shape[:2]

        # Панель DMS в левом верхнем углу (сдвигаем вправо, чтобы не перекрывать движение)
        panel_x = 300
        panel_y = 30
        panel_width = 280
        panel_height = 150

        # Полупрозрачный фон
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x - 10, panel_y - 10),
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Заголовок DMS
        cv2.putText(
            frame,
            "DMS STATUS",
            (panel_x, panel_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.colors["cyan"],
            2,
        )

        y_offset = panel_y + 30

        # Состояние глаз
        eye_state = dms_stats.get("eye_state", "unknown")
        eye_color = self.colors["green"] if eye_state == "open" else \
        self.colors["red"]
        cv2.putText(
            frame,
            f"Eyes: {eye_state.upper()}",
            (panel_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            eye_color,
            2,
        )

        # Длительность закрытых глаз
        if eye_state == "closed" and "eye_closed_duration" in dms_stats:
            closed_duration = dms_stats["eye_closed_duration"]
            cv2.putText(
                frame,
                f"Closed: {closed_duration:.1f}s",
                (panel_x + 120, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["yellow"],
                1,
            )

        y_offset += 25

        # Состояние ремня
        seatbelt_state = dms_stats.get("seatbelt_state", "unknown")
        seatbelt_color = self.colors["green"] if seatbelt_state == "on" else \
        self.colors["red"]
        cv2.putText(
            frame,
            f"Seatbelt: {seatbelt_state.upper()}",
            (panel_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            seatbelt_color,
            2,
        )

        y_offset += 25

        # Телефон
        phone_detected = dms_stats.get("phone_detected", False)
        phone_color = self.colors["red"] if phone_detected else self.colors[
            "green"]
        cv2.putText(
            frame,
            f"Phone: {'DETECTED' if phone_detected else 'OK'}",
            (panel_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            phone_color,
            2,
        )

        y_offset += 25

        # Сигарета
        cigarette_detected = dms_stats.get("cigarette_detected", False)
        cigarette_color = self.colors["red"] if cigarette_detected else \
        self.colors["green"]
        cv2.putText(
            frame,
            f"Cigarette: {'DETECTED' if cigarette_detected else 'OK'}",
            (panel_x, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            cigarette_color,
            2,
        )

        # Рисуем bounding boxes для DMS объектов
        for obj in dms_objects:
            self._draw_dms_object(frame, obj)

        # Отображаем активные нарушения
        if violations:
            violation_text = "VIOLATIONS: " + ", ".join(
                [v.get('type', 'unknown') for v in violations[:2]])
            if len(violations) > 2:
                violation_text += f" +{len(violations) - 2}"

            cv2.putText(
                frame,
                violation_text,
                (panel_x, panel_y + panel_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.colors["red"],
                2,
            )

    def _draw_dms_object(self, frame, obj):
        """Отрисовка DMS объекта"""
        if "bbox" not in obj or len(obj["bbox"]) != 4:
            return

        x1, y1, x2, y2 = obj["bbox"]
        class_name = obj.get("class", "Unknown")
        confidence = obj.get("confidence", 0)

        # Выбираем цвет в зависимости от класса
        if "eye" in class_name.lower():
            color = self.colors["cyan"] if "open" in class_name.lower() else \
            self.colors["red"]
            thickness = 2
        elif "seatbelt" in class_name.lower():
            color = self.colors["green"]
            thickness = 2
        elif "phone" in class_name.lower():
            color = self.colors["purple"]
            thickness = 3
        elif "cigarette" in class_name.lower():
            color = self.colors["orange"]
            thickness = 3
        else:
            color = self.colors["yellow"]
            thickness = 2

        # Рисуем bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Подпись
        label = f"{class_name} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[
            0]

        # Фон для подписи
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )

        # Текст подписи
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.colors["white"],
            1,
            cv2.LINE_AA,
        )