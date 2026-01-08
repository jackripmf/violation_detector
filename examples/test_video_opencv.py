# test_simple_video.py
import cv2
import os

# Тестовое видео
video_path = r"/WIN_20251104_17_33_47_Pro.mp4"

print(f"Testing video: {video_path}")
print(f"Exists: {os.path.exists(video_path)}")

# Прямой тест OpenCV
cap = cv2.VideoCapture(video_path)
print(f"OpenCV can open: {cap.isOpened()}")

if cap.isOpened():
    # Получаем информацию
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo info:")
    print(f"  FPS: {fps}")
    print(f"  Frames: {frame_count}")
    print(f"  Size: {width}x{height}")
    print(f"  Duration: {frame_count / fps:.1f} seconds")

    # Читаем первый кадр
    ret, frame = cap.read()
    print(f"\nCan read frame: {ret}")
    if ret:
        print(f"Frame shape: {frame.shape}")

        # Показываем кадр
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()
else:
    print("OpenCV cannot open video")