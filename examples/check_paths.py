# check_paths.py
import os
import sys


def check_dms_model_path():
    """Проверка всех возможных путей к модели DMS"""

    print("Текущая рабочая директория:", os.getcwd())
    print("\nПроверяем возможные пути к dms_model.pt:")

    possible_paths = [
        # Относительный путь от текущего файла
        os.path.join("src", "utils", "models", "dms_model.pt"),
        # Абсолютный путь
        os.path.join(os.getcwd(), "src", "utils", "models", "dms_model.pt"),
        # Другие возможные пути
        "dms_model.pt",
        os.path.join("utils", "models", "dms_model.pt"),
        os.path.join("models", "dms_model.pt"),
    ]

    found = False
    for path in possible_paths:
        exists = os.path.exists(path)
        print(f"{path}: {'НАЙДЕН' if exists else 'не найден'}")
        if exists:
            found = True
            abs_path = os.path.abspath(path)
            print(f"  Абсолютный путь: {abs_path}")

    if not found:
        print("\nФайл dms_model.pt не найден!")
        print(
            "\nСоздайте папку src/utils/models/ и поместите туда файл dms_model.pt")
        print("Или укажите полный путь к модели при создании детектора:")
        print('DMSDetector(model_path="ПОЛНЫЙ_ПУТЬ/К/dms_model.pt")')


def check_project_structure():
    """Проверка структуры проекта"""
    print("\n=== Структура проекта ===")

    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for file in files:
            if file.endswith(".pt"):
                print(f"{subindent}{file}")


if __name__ == "__main__":
    check_dms_model_path()
    check_project_structure()