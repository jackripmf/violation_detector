"""Файл: tests/conftest.py
Тип: файл автотестов.
Назначение: содержит тестовые сценарии и проверяет устойчивость контракта поведения.
Связи: взаимодействует с рабочими модулями через публичные API и тестовые заглушки.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""

import os
import sys
import pytest
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


@pytest.fixture
def sample_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def bright_frame():
    return np.full((720, 1280, 3), 255, dtype=np.uint8)


@pytest.fixture
def dark_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def test_image_path(tmp_path):
    img_path = tmp_path / "test_image.jpg"
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return str(img_path)


@pytest.fixture
def test_video_path(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1280, 720))

    for i in range(10):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return str(video_path)
