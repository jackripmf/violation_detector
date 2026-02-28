"""Файл: debug/inspect_yolo_classes_in_model.py
Тип: отладочный скрипт.
Назначение: используется для локальной диагностики и проверки отдельных подсистем.
Связи: обычно запускается вручную и читает данные из основных модулей проекта.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""
import argparse
from collections import defaultdict

def as_int_keyed_dict(names_obj):
    """Функция: as_int_keyed_dict()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `names_obj` (`Any`): данные обнаруженного объекта или его служебного представления.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    if names_obj is None:
        return {}

    if isinstance(names_obj, dict):
        out = {}
        for k, v in names_obj.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return dict(sorted(out.items(), key=lambda x: x[0]))

    if isinstance(names_obj, (list, tuple)):
        return {int(i): str(v) for i, v in enumerate(names_obj)}

    return {}

def main():
    """Функция: main()
Назначение: служит точкой входа и запускает основной сценарий выполнения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="src/utils/models/best_auto.pt", help="Path to .pt model")
    ap.add_argument("--find", default="", help="Comma-separated substrings to search in class names (case-insensitive)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[ERROR] ultralytics import failed:", e)
        print("Install/verify: pip install ultralytics")
        raise

    print("=== Loading model:", args.model)
    model = YOLO(args.model)

    names_raw = getattr(model, "names", None)
    names = as_int_keyed_dict(names_raw)

    nc = None

    try:
        nc = getattr(getattr(model, "model", None), "nc", None)
    except Exception:
        pass
    if nc is None and names:
        nc = len(names)

    print("\n=== Model info ===")
    print("nc:", nc)
    print("type(model.names):", type(names_raw).__name__)
    print("len(model.names):", len(names))

    if not names:
        print("\n[WARN] model.names is empty/unreadable. This is unusual for detect models.")
        return

    print("\n=== Class list: id -> name ===")
    for cid, cname in names.items():
        print(f"{cid:>3d}: {cname}")

    def norm(s: str) -> str:
        """Функция: norm()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `s` (`str`): рабочие данные `s`, используемые на текущем этапе обработки.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        return (s or "").strip().lower().replace("-", "_").replace(" ", "_")

    dup = defaultdict(list)
    for cid, cname in names.items():
        dup[norm(cname)].append(cid)

    dups = {k: v for k, v in dup.items() if len(v) > 1}
    if dups:
        print("\n=== Duplicate/alias-like names (normalized) ===")
        for k, ids in sorted(dups.items(), key=lambda x: (-len(x[1]), x[0])):
            originals = sorted({names[i] for i in ids})
            print(f"{k}: ids={sorted(ids)} originals={originals}")
    else:
        print("\n=== No duplicate names detected (by normalization) ===")

    if args.find.strip():
        needles = [x.strip().lower() for x in args.find.split(",") if x.strip()]
        print("\n=== Find results ===")
        for needle in needles:
            hits = [(cid, cname) for cid, cname in names.items() if needle in cname.lower()]
            print(f"\n-- '{needle}' hits ({len(hits)}):")
            for cid, cname in hits:
                print(f"   {cid:>3d}: {cname}")

if __name__ == "__main__":
    main()
