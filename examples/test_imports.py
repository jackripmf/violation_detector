# test_imports.py (поместите в папку examples и запустите)
import sys
import os

# Настраиваем пути как в вашем основном скрипте
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# Пробуем импортировать модули
modules_to_test = [
    "src.utils.utils",
    "src.processors.video_processor",
    "src.detectors.dms_detector",
    "src.detectors.movement_detector"
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"✓ {module_name} - OK")
    except ImportError as e:
        print(f"✗ {module_name} - ERROR: {e}")
    except Exception as e:
        print(f"✗ {module_name} - OTHER ERROR: {e}")