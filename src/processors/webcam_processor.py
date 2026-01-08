# webcam_processor.py - полный исправленный файл
import cv2
import time
import logging
from .video_processor import VideoViolationProcessor


class WebcamProcessor(VideoViolationProcessor):
    def __init__(self, save_dir="webcam_violations", camera_id=0,
                 enabled_detectors=["all"], disabled_detectors=[]):
        # Передаем настройки детекторов в родительский класс
        super().__init__(
            save_dir=save_dir,
            enabled_detectors=enabled_detectors,
            disabled_detectors=disabled_detectors
        )
        self.camera_id = camera_id
        self.cap = None
        self.real_processing_fps = None

    def start_camera(self, resolution=(1280, 720)):
        """Инициализация веб-камеры"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logging.error(f"Не удалось открыть камеру {self.camera_id}")
                return False
            # Разрешение
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            # Получение реального фпс
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            logging.info(f"Веб-камера {self.camera_id} запущена, FPS: {fps:.1f}")
            return True
        except Exception as e:
            logging.error(f"Ошибка запуска: {e}")
            return False

    def process_webcam(self, show_preview=True, max_duration=None):
        """Обработка видеопотока"""
        if not self.cap or not self.cap.isOpened():
            logging.error("Веб-камера не инициализирована")
            return

        # Получаем реальный FPS камеры
        camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if camera_fps <= 0:
            camera_fps = 30.0  # Значение по умолчанию

        start_time = time.time()
        frame_count = 0
        prev_frame_time = start_time

        # Для расчета FPS
        fps_update_interval = 0.5  # Обновлять FPS каждые 0.5 секунды
        last_fps_update = start_time
        fps_buffer = []
        current_fps = camera_fps

        try:
            while True:
                # Проверка максимальной длительности
                if max_duration and (time.time() - start_time) > max_duration:
                    logging.info("Достигнута максимальная длительность видео")
                    break

                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Не удалось получить кадр")
                    break

                current_time = time.time()
                # Используем реальное время вместо вычисления по FPS
                video_timestamp = current_time - start_time

                # Рассчитываем FPS текущего кадра
                frame_time = current_time - prev_frame_time
                prev_frame_time = current_time

                if frame_time > 0:
                    fps_buffer.append(1.0 / frame_time)

                # Обновляем отображаемый FPS каждые 1 секунду и используем больше кадров
                if current_time - last_fps_update > 1.0 and fps_buffer:
                    self.real_processing_fps = current_fps
                    # Ограничиваем буфер последними 30 кадрами
                    if len(fps_buffer) > 30:
                        fps_buffer = fps_buffer[-30:]

                    current_fps = sum(fps_buffer) / len(fps_buffer)
                    fps_buffer = []
                    last_fps_update = current_time

                    # Логируем для отладки
                    if abs(current_fps - camera_fps) > camera_fps * 0.3:  # Разница >30%
                        logging.warning(
                            f"FPS mismatch: real={current_fps:.1f}, expected={camera_fps:.1f}")

                # Обработка кадра
                result, total_duration, movement_info = self.process_frame(
                    frame, current_time, video_timestamp
                )

                if show_preview:
                    # Отображение кадра
                    display_frame = self.visualizer.draw_detection_info(
                        frame, result, total_duration, movement_info,
                        video_timestamp
                    )

                    # Добавляем FPS в САМЫЙ ПРАВЫЙ НИЖНИЙ УГОЛ
                    fps_text = f"FPS: {current_fps:.1f}"
                    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                    # Позиция в самом правом нижнем углу
                    fps_x = display_frame.shape[1] - text_size[0] - 20
                    fps_y = display_frame.shape[0] - 20

                    # Полупрозрачный фон
                    overlay = display_frame.copy()
                    cv2.rectangle(
                        overlay,
                        (fps_x - 10, fps_y - text_size[1] - 10),
                        (fps_x + text_size[0] + 10, fps_y + 10),
                        (0, 0, 0),
                        -1
                    )

                    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

                    # Текст FPS
                    cv2.putText(
                        display_frame,
                        fps_text,
                        (fps_x, fps_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    if self.preview_writer is None:
                        # Не передаем fps, пусть метод сам определит реальный FPS
                        self.start_preview_recording(display_frame)

                    # Сохранение превью
                    self.save_preview_frame(display_frame)

                    # Показываем превью
                    cv2.imshow("Webcam Violation Detector", display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logging.info("Обработка прервана пользователем")
                        break
                    elif key == ord("r"):
                        self.movement_detector.set_reference_frame(frame)
                        logging.info("Референсный кадр сброшен")
                    elif key == ord("p"):
                        # Пауза/продолжение
                        cv2.waitKey(0)

                frame_count += 1
                if frame_count % 100 == 0:
                    avg_fps = frame_count / (current_time - start_time)
                    logging.info(f"Обработано кадров: {frame_count}, Средний FPS: {avg_fps:.1f}")

        except KeyboardInterrupt:
            logging.info("Обработка прервана пользователем")
        except Exception as e:
            logging.error(f"Ошибка при обработке вебкамеры: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            if self.cap:
                self.cap.release()
            if show_preview:
                cv2.destroyAllWindows()

    def get_camera_info(self):
        """Получение информации о камере"""
        if not self.cap:
            return None
        return {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "camera_id": self.camera_id,
        }