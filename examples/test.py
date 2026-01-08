import cv2

# import cv2, os
#
# video_path = 'test_video.mp4'          # либо путь, который вы точно знаете
# abs_path = os.path.abspath(video_path)
#
# print('Путь:', abs_path)
# print('Существует ли файл?', os.path.exists(abs_path))
#
# cap = cv2.VideoCapture(abs_path)
# print('cap.isOpened() →', cap.isOpened())
# print('cap.get(CAP_PROP_FPS) →', cap.get(cv2.CAP_PROP_FPS))
# cap.release()

# import torch
#
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('CUDA доступна, используем GPU')
# else:
#     device = torch.device('cpu')
#     print('CUDA не найдена, будем использовать CPU')

# parser.add_argument('--gpu', action='store_true', help='Включить работу на GPU (если доступен CUDA)')
# device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
# logging.info(f'Используемое устройство: {device}')
# check_gpu.py
# import os
# import torch
# from ultralytics import YOLO  # Убедитесь, что ultralytics установлена
#
# # 1. Пытаемся «объявить» видимый GPU (нет никакого эффекта, если CUDA не включена)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# print("=== Тест CUDA‑поддержки ===")
# print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
# print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
# try:
#     print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
# except Exception as e:
#     print("torch.cuda.current_device() бросила:", e)
#
# # 2. Загружаем YOLO‑модель, явно задавая CPU
# print("\n=== Загрузка YOLO на CPU ===")
# try:
#     model = YOLO('yolov8n.pt', device='cpu')   # <- явно CPU
#     print("Модель загружена: ", model)
# except Exception as e:
#     print("Ошибка при загрузке YOLO:", e)
#
# # 3. Проверка, что модель не попала на GPU
# print("\n=== Проверка назначения модели ===")
# if torch.cuda.is_available():
#     # Если вдруг CUDA была бы включена, это бы выдалось 0
#     print("device_count:", torch.cuda.device_count())
# else:
#     print("Поскольку CUDA не включена, модель остаётся на CPU.")
# cv2.CAP_FFMPEG
# cv2.CAP_DSHOW