import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import time
import logging

# Включаем ВСЕ логи
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.processors.video_processor import VideoViolationProcessor

print("=== DEBUG FULL CHAIN ===")

# Создаем процессор с минимальными параметрами
processor = VideoViolationProcessor(save_dir="debug_violations")

# Упрощаем параметры для тестирования
processor.forbidden_items_detector.confidence_threshold = 0.3
processor.forbidden_items_detector.min_detection_duration = 2.0  # Всего 2 секунды
processor.forbidden_items_detector.violation_cooldown = 10.0  # 10 секунд кулдаун
processor.forbidden_items_detector.class_specific_cooldown = True

print(f"Confidence: {processor.forbidden_items_detector.confidence_threshold}")
print(
    f"Min duration: {processor.forbidden_items_detector.min_detection_duration}")
print(f"Cooldown: {processor.forbidden_items_detector.violation_cooldown}")

# Открываем веб-камеру
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("\nINSTRUCTIONS:")
print("1. Hold phone for 3 seconds")
print("2. Wait for first violation")
print("3. Then hold bottle for 3 seconds")
print("4. Check if second violation saves")
print("\nPress Enter to start...")
input()

frame_count = 0
start_time = time.time()
phone_violation = False
bottle_violation = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame_count += 1
        video_timestamp = frame_count / 30.0

        print(f"\n--- Frame {frame_count} ---")

        # Обрабатываем кадр
        result, total_duration, movement_info = processor.process_frame(
            frame, current_time, video_timestamp
        )

        # Проверяем запрещенные объекты
        if result.get("forbidden_violation", False):
            print("=== FORBIDDEN VIOLATION IN RESULT ===")

        forbidden_res = result.get("details", {}).get("forbidden", {})
        if forbidden_res:
            print(f"Detected objects: {len(forbidden_res.get('objects', []))}")
            print(
                f"Current violation: {forbidden_res.get('current_violation', False)}")

            if forbidden_res.get("current_violation", False):
                violation_info = forbidden_res.get("violation_info")
                if violation_info:
                    print(
                        f"Violation ID: {violation_info.get('violation_id')}")
                    objects = violation_info.get("objects", [])
                    print(
                        f"Violation objects: {[obj.get('class') for obj in objects]}")

                    # Проверяем какие объекты
                    for obj in objects:
                        if obj.get('class') == 'cell phone':
                            phone_violation = True
                            print("✓ Phone violation detected")
                        elif obj.get('class') == 'bottle':
                            bottle_violation = True
                            print("✓ Bottle violation detected")

        # Отображение
        display = frame.copy()

        # Рисуем bounding boxes
        for obj in result.get("forbidden_objects", []):
            if "bbox" in obj:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(display, obj.get("class", "Unknown"),
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 255), 1)

        cv2.imshow("Debug - Full Chain", display)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (
                phone_violation and bottle_violation):
            break

except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"\nError: {e}")
    import traceback

    traceback.print_exc()

cap.release()
cv2.destroyAllWindows()

print(f"\n=== DEBUG COMPLETE ===")
print(f"Frames: {frame_count}")
print(f"Time: {time.time() - start_time:.1f}s")
print(f"Phone violation: {phone_violation}")
print(f"Bottle violation: {bottle_violation}")
print(f"Check 'debug_violations/forbidden_items/' folder")