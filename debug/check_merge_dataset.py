"""Файл: debug/check_merge_dataset.py
Тип: отладочный скрипт.
Назначение: используется для локальной диагностики и проверки отдельных подсистем.
Связи: обычно запускается вручную и читает данные из основных модулей проекта.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""
import yaml
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/home/kajitsu/violation_detector-main/edu/datasets/merged_coco_dms")
DATA_YAML = ROOT / "data.yaml"

def load_class_names(data_yaml: Path):
    """Функция: load_class_names()
Назначение: формирует или получает данные для следующего шага обработки.
Параметры функции:
- `data_yaml` (`Path`): рабочие данные `data_yaml`, используемые на текущем этапе обработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    nc = data.get("nc")

    if isinstance(names, dict):
        # конвертируем dict -> list
        max_id = max(int(k) for k in names.keys())
        names_list = [""] * (max_id + 1)
        for k, v in names.items():
            names_list[int(k)] = str(v)
        names = names_list
    elif isinstance(names, list):
        names = [str(x) for x in names]

    if names is None and nc is not None:
        names = [f"class_{i}" for i in range(int(nc))]
    if nc is None and names is not None:
        nc = len(names)

    return names, int(nc) if nc is not None else None

def count_yolo_split(split_dir: Path, names, nc):
    """Функция: count_yolo_split()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `split_dir` (`Path`): путь/имя файла, используемый текущим шагом чтения или сохранения данных.
- `names` (`Any`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `nc` (`Any`): рабочие данные `nc`, используемые на текущем этапе обработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    labels_dir = split_dir / "labels"
    images_dir = split_dir / "images"

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in img_exts] \
             if images_dir.exists() else []
    total_images = len(images)

    obj_per_class = Counter()
    imgset_per_class = defaultdict(set)
    labeled_stems = set()

    if not labels_dir.exists():
        return total_images, total_images, [], {}

    for p in labels_dir.glob("*.txt"):
        stem = p.stem
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue

        labeled_stems.add(stem)
        found = set()
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            obj_per_class[cls] += 1
            found.add(cls)

        for cls in found:
            imgset_per_class[cls].add(stem)

    if total_images > 0:
        image_stems = {p.stem for p in images}
        images_without_any_ann = len(image_stems - labeled_stems)
    else:
        images_without_any_ann = 0

    rows = []
    max_cls = (nc - 1) if nc is not None else (max(obj_per_class.keys()) if obj_per_class else -1)
    for cls in range(max_cls + 1):
        name = names[cls] if names and cls < len(names) else f"class_{cls}"
        img_count = len(imgset_per_class.get(cls, set()))
        obj_count = obj_per_class.get(cls, 0)
        pct = (img_count / total_images * 100) if total_images else 0.0
        rows.append((img_count, obj_count, pct, cls, name))

    rows.sort(reverse=True, key=lambda x: x[0])
    return total_images, images_without_any_ann, rows, obj_per_class

def print_like_coco(title, total_images, images_without_any_ann, rows):
    """Функция: print_like_coco()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `title` (`Any`): текстовый/категориальный признак, используемый в логике ветвления и выводе.
- `total_images` (`Any`): кадр/изображение, которое передается на обработку текущему этапу.
- `images_without_any_ann` (`Any`): кадр/изображение, которое передается на обработку текущему этапу.
- `rows` (`Any`): рабочие данные `rows`, используемые на текущем этапе обработки.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    print(f"\n=== {title} ===")
    print(f"Total images: {total_images}")
    print(f"Images without any annotations: {images_without_any_ann}")
    print("Img Count | Obj Count |  % imgs | cat_id | name")
    print("-" * 60)
    for img_count, obj_count, pct, cls, name in rows:
        print(f"{img_count:9d} | {obj_count:9d} | {pct:6.2f}% | {cls:6d} | {name}")

def main():
    """Функция: main()
Назначение: служит точкой входа и запускает основной сценарий выполнения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    names, nc = load_class_names(DATA_YAML)
    print(f"Loaded {nc} classes from {DATA_YAML}")

    for split_name in ["train", "val"]:
        split_dir = ROOT / split_name
        if not split_dir.exists():
            continue
        total_images, images_wo, rows, _ = count_yolo_split(split_dir, names, nc)
        print_like_coco(f"{split_name.upper()}", total_images, images_wo, rows)

if __name__ == "__main__":
    main()