# import os.path
import os
import sys

import cv2
import time
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from src.processors.video_processor import VideoViolationProcessor
from src.processors.webcam_processor import WebcamProcessor
from src.utils.utils import setup_logging, get_video_info


def process_video(input_video, save_dir="violations", show_preview=True,
                  enabled_detectors=["all"], disabled_detectors=[]):
    """Анализ видео файла"""
    logging.info(f"Запуск обработки видео: {input_video}")

    # Получаем информацию о видео
    video_info = get_video_info(input_video)
    if video_info is None:
        logging.error(f"Не удалось открыть видео: {input_video}")
        return

    logging.info(f"Информация: {video_info['width']}x{video_info['height']}, ")
    logging.info(f"{video_info['fps']:.1f} FPS, {video_info['total_frames']}")

    # Создаем процессор с указанием детекторов
    processor = VideoViolationProcessor(
        save_dir=save_dir,
        enabled_detectors=enabled_detectors,
        disabled_detectors=disabled_detectors
    )
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        logging.error("Ошибка открытия видео!")
        return

    logging.info("Начинаем обработку...")
    logging.info("Управление: 'q' - выход, 'r' - сброс референсного кадра ")

    start_time = time.time()
    frame_count = 0

    # Для расчета FPS
    fps_update_interval = 0.5
    last_fps_update = start_time
    fps_buffer = []
    current_fps = video_info["fps"]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Расчет FPS
            current_time = time.time()
            frame_time = current_time - last_fps_update

            if frame_time > 0:
                fps_buffer.append(1.0 / frame_time * (1000 / 1000))

            if current_time - last_fps_update > fps_update_interval and fps_buffer:
                current_fps = sum(fps_buffer) / len(fps_buffer)
                fps_buffer = []
                last_fps_update = current_time

            # Время в видео
            video_timestamp = frame_count / video_info["fps"]
            processing_time = current_time - start_time

            # Обрабатываем кадр
            result, duration, movement_info = processor.process_frame(
                frame, processing_time, video_timestamp
            )

            if show_preview:
                display_frame = processor.visualizer.draw_detection_info(
                    frame, result, duration, movement_info, video_timestamp
                )

                # Добавляем FPS в правый нижний угол
                fps_text = f"FPS: {current_fps:.1f}"
                text_size = \
                cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

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
                cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0,
                                display_frame)

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

                if processor.preview_writer is None:
                    processor.start_preview_recording(display_frame,
                                                      video_info["fps"])

                processor.save_preview_frame(display_frame)

                # Показываем превью
                cv2.imshow("Video Violation Detector", display_frame)

                # Клавиши управления
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    processor.movement_detector.set_reference_frame(frame)
                    logging.info("Сброс референсного кадра")

            frame_count += 1

            if frame_count % 100 == 0:
                progress = (frame_count / video_info["total_frames"]) * 100
                avg_fps = frame_count / (current_time - start_time)
                logging.info(
                    f"Обработано: {frame_count}/{video_info['total_frames']} ({progress:.1f}%) | FPS: {avg_fps:.1f}"
                )

    except KeyboardInterrupt:
        logging.info("Остановлено пользователем")
    except Exception as e:
        logging.exception(f"Ошибка: {e}")
    finally:
        processor.cleanup()
        cap.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    logging.info("Обработка завершена!")
    logging.info("=" * 40)
    logging.info(f"Время: {total_time:.1f} секунд")
    logging.info(f"Скорость: {frame_count / total_time:.1f} FPS")
    logging.info(f"Нарушений: {processor.stats['total_detections']}")
    logging.info(f"Сохранено: {processor.stats['saved_violations']}")
    logging.info(
        f"Движений камеры: {processor.stats['movement_violations_saved']}")
    logging.info(
        f"Запрещенных объектов: {processor.stats['forbidden_items_violations']}")
    logging.info(f"DMS нарушений: {processor.stats['dms_violations']}")
    logging.info(f"Сохранено DMS: {processor.stats['dms_violations_saved']}")
    logging.info(f"Состояние глаз: {processor.stats['eye_state']}")
    logging.info(f"Состояние ремня: {processor.stats['seatbelt_state']}")


def process_webcam(
        cam_id=0, save_dir="webcam_violations", show_preview=True,
        max_duration=None,
        enabled_detectors=["all"], disabled_detectors=[]  # НОВЫЕ ПАРАМЕТРЫ
):
    """Обработка потока с веб-камеры"""
    logging.info(f"Запуск обрабоки видеопотока с веб-камеры {cam_id}")

    # Инициализация процессора веб-камеры с указанием детекторов
    processor = WebcamProcessor(
        save_dir=save_dir,
        camera_id=cam_id,
        enabled_detectors=enabled_detectors,
        disabled_detectors=disabled_detectors
    )

    # Запуск камеры
    if not processor.start_camera():
        return None

    # Получение информации о камере
    if processor.start_camera():
        camera_info = processor.get_camera_info()
        logging.info(f"Theoretical camera FPS: {camera_info['fps']:.1f}")

    # ВЫЗОВ МЕТОДА ПРОЦЕССОРА ДЛЯ ОБРАБОТКИ
    processor.process_webcam(show_preview=show_preview,
                             max_duration=max_duration)

    stats = {
        "source": f"webcam_{cam_id}",
        "total_frames": processor.frame_count,
        "violations": {
            "obstruction": {
                "detected": processor.stats["total_detections"],
                "saved": processor.stats["saved_violations"],
                "count": processor.stats.get("obstruction_violations", 0)
            },
            "movement": {
                "detected": processor.stats["camera_movements"],
                "saved": processor.stats["movement_violations_saved"],
                "count": processor.stats.get("movement_violations", 0)
            },
            "forbidden_items": {
                "detected": processor.stats["forbidden_items_violations"],
                "saved": processor.stats["forbidden_items_saved"],
                "count": processor.stats["forbidden_items_violations"]
            }
        }
    }

    logging.info("===ОБРАБОТКА ВЕБ-КАМЕРЫ ЗАВЕРШЕНА===")
    logging.info(f"Всего кадров: {stats['total_frames']}")
    logging.info(f"Всего алертов: {processor.alert_count}")

    logging.info("\n=== ИТОГОВАЯ СТАТИСТИКА НАРУШЕНИЙ ===")
    logging.info(f"Перекрытия камеры:")
    logging.info(
        f"  - Обнаружено: {stats['violations']['obstruction']['detected']}")
    logging.info(
        f"  - Сохранено: {stats['violations']['obstruction']['saved']}")

    logging.info(f"Движения камеры:")
    logging.info(
        f"  - Обнаружено: {stats['violations']['movement']['detected']}")
    logging.info(f"  - Сохранено: {stats['violations']['movement']['saved']}")

    logging.info(f"Запрещенные объекты:")
    logging.info(
        f"  - Обнаружено: {stats['violations']['forbidden_items']['detected']}")
    logging.info(
        f"  - Сохранено: {stats['violations']['forbidden_items']['saved']}")
    logging.info(f"DMS нарушений: {processor.stats['dms_violations']}")
    logging.info(f"Сохранено DMS: {processor.stats['dms_violations_saved']}")
    logging.info(f"Состояние глаз: {processor.stats['eye_state']}")
    logging.info(f"Состояние ремня: {processor.stats['seatbelt_state']}")

    return stats


# main.py - заменяем текущий парсер аргументов
if __name__ == "__main__":
    import os
    import argparse

    setup_logging(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Детектор перекрытия и движений камеры"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="test_dir/test_video.mp4",
        help="Путь к видео-файлу или веб-камере",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="violations",
        help="Папка с нарушениями"
    )

    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Не показывать превью"
    )

    # НОВЫЕ АРГУМЕНТЫ ДЛЯ ДЕТЕКТОРОВ
    parser.add_argument(
        "--detectors",
        type=str,
        nargs="+",  # Можно передавать несколько значений
        default=["all"],  # По умолчанию все
        choices=[
            "all", "cv", "yolo", "movement", "forbidden", "dms",
            "cvdetector", "yolodetector", "movementdetector",
            "forbiddendetector", "dmsdetector", "dark"
        ],
        help="Детекторы для включения: all, cv, yolo, movement, forbidden, dms, dark"
    )

    parser.add_argument(
        "--disable-detectors",
        type=str,
        nargs="+",
        default=[],
        choices=["cv", "yolo", "movement", "forbidden", "dms", "dark"],
        help="Детекторы для отключения (имеет приоритет над --detectors)"
    )

    args = parser.parse_args()

    logging.info("=" * 40)
    logging.info("Детектор перекрытия и движений камеры")
    logging.info("=" * 40)

    # Обработка аргументов детекторов
    enabled_detectors = args.detectors
    disabled_detectors = args.disable_detectors

    # Логируем выбранные детекторы
    logging.info(f"Запрошенные детекторы: {enabled_detectors}")
    logging.info(f"Отключенные детекторы: {disabled_detectors}")

    if args.input:
        if args.input.isdigit():
            camera_id = int(args.input)
            # Передаем информацию о детекторах в процессор
            process_webcam(
                cam_id=camera_id,
                save_dir=args.output,
                show_preview=not args.no_preview,
                enabled_detectors=enabled_detectors,
                disabled_detectors=disabled_detectors
            )
        elif args.input.startswith("rtsp://"):
            process_webcam(
                cam_id=args.input,
                save_dir=args.output,
                show_preview=not args.no_preview,
                enabled_detectors=enabled_detectors,
                disabled_detectors=disabled_detectors
            )
        else:
            process_video(
                input_video=args.input,
                save_dir=args.output,
                show_preview=not args.no_preview,
                enabled_detectors=enabled_detectors,
                disabled_detectors=disabled_detectors
            )
    else:
        logging.error(
            """
        Укажите источник:

        путь_к_файлу - для обработки видео файла
        ID камеры - для использования веб-камеры
        Примеры:
        python main.py --input test_video.mp4
        python main.py --input 0
        python main.py --input 0 --detectors dms movement
        python main.py --input 0 --detectors all --disable-detectors dms
        """
        )