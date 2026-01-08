# import sys
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from main import process_video, process_webcam
#
# # Определяем путь к корню проекта (на 2 уровня выше examples)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)  # Поднимаемся на уровень выше examples
#
# # Добавляем корень проекта в Python path
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# # Устанавливаем рабочую директорию в корень проекта
# os.chdir(project_root)
#
# print(f"Project root: {project_root}")
# print(f"Working directory: {os.getcwd()}")
#
# if __name__== "__main__":
#     # process_webcam(
#     #     cam_id=0,
#     #     save_dir='my_violations',
#     #     show_preview=True
#     # )
#     process_video(
#         input_video='WIN_20251104_17_33_47_Pro.mp4',
#         save_dir='my_violations',
#         show_preview=True
#     )

import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Определяем путь к корню проекта
current_dir = os.path.dirname(os.path.abspath(__file__))  # examples
project_root = os.path.dirname(current_dir)  # корень проекта

# Добавляем корень проекта в Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# НЕ МЕНЯЕМ РАБОЧУЮ ДИРЕКТОРИЮ!
# os.chdir(project_root)  # ЗАКОММЕНТИРУЙТЕ ЭТУ СТРОКУ!

print(f"Project root: {project_root}")
print(f"Current directory (where script is): {current_dir}")
print(f"Working directory: {os.getcwd()}")

from main import process_video, process_webcam

if __name__ == "__main__":

    # Видео лежит в той же папке что и скрипт
    video_file = "../video_2025-12-14_15-12-17.mp4"
    video_path = os.path.join(current_dir, video_file)  # Абсолютный путь

    print(f"\nVideo path: {video_path}")
    print(f"Video exists: {os.path.exists(video_path)}")

    if os.path.exists(video_path):
        process_video(
            input_video=video_path,  # Абсолютный путь!
            save_dir='my_violations',
            show_preview=True
        )
    else:
        print(f"ERROR: Video not found at {video_path}")
        print(f"Files in {current_dir}:")
        for f in os.listdir(current_dir):
            print(f"  - {f}")