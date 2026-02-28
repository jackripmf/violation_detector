"""Файл: debug/debug_preview_boxes.py
Тип: отладочный скрипт.
Назначение: используется для локальной диагностики и проверки отдельных подсистем.
Связи: обычно запускается вручную и читает данные из основных модулей проекта.
Импортируемые внутренние модули:
- Явные внутренние зависимости отсутствуют или модуль работает автономно."""
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO


def color_for_class(cls_id: int) -> tuple[int, int, int]:
    """Функция: color_for_class()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `cls_id` (`int`): идентификатор/индекс для адресации и сопоставления сущностей.
Возвращаемое значение: tuple[int, int, int]: результат шага обработки, который используется следующим этапом пайплайна."""
    rng = np.random.RandomState(cls_id * 9973 + 42)
    b = int(rng.randint(30, 256))
    g = int(rng.randint(30, 256))
    r = int(rng.randint(30, 256))
    return (b, g, r)


def main():
    """Функция: main()
Назначение: служит точкой входа и запускает основной сценарий выполнения.
Параметры функции:
- Внешние параметры отсутствуют.
Возвращаемое значение: Any: результат шага обработки, который используется следующим этапом пайплайна."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLO .pt model")
    ap.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    ap.add_argument("--device", default="", help="e.g. cuda:0 or cpu (default: auto)")
    ap.add_argument("--imgsz", type=int, default=704, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.1, help="Draw boxes with conf >= this")
    ap.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold")
    ap.add_argument("--max-det", type=int, default=200, help="Max detections per frame")
    ap.add_argument("--show-fps", action="store_true", help="Overlay FPS")
    args = ap.parse_args()

    # source: camera index or path
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # names can be dict or list; handle both
    names = getattr(model, "names", {})
    def class_name(cid: int) -> str:
        """Функция: class_name()
Назначение: реализует отдельный шаг логики текущего модуля.
Параметры функции:
- `cid` (`int`): рабочие данные `cid`, используемые на текущем этапе обработки.
Возвращаемое значение: str: результат шага обработки, который используется следующим этапом пайплайна."""
        if isinstance(names, dict):
            return str(names.get(cid, f"Unknown_{cid}"))
        if isinstance(names, (list, tuple)):
            return str(names[cid]) if 0 <= cid < len(names) else f"Unknown_{cid}"
        return f"Unknown_{cid}"

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    win = "YOLO debug preview (q/esc to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed (end of stream?)")
                break

            # Ultralytics inference
            # Results.boxes has .xyxy .conf .cls :contentReference[oaicite:1]{index=1}
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=0.001,          # берем низкий порог на инференсе
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False
            )

            r0 = results[0]
            boxes = getattr(r0, "boxes", None)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)

                for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, clss):
                    if float(c) < args.conf:
                        continue

                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    color = color_for_class(int(cid))

                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)

                    label = f"{cid}:{class_name(cid)} {float(c):.2f}"
                    # фон под текст
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    y_text = max(0, y1i - 8)
                    cv2.rectangle(frame, (x1i, y_text - th - 6), (x1i + tw + 4, y_text + 4), color, -1)
                    cv2.putText(frame, label, (x1i + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # FPS
            now = time.time()
            dt = now - last_t
            last_t = now
            fps = (0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))) if fps > 0 else (1.0 / max(dt, 1e-6))

            if args.show_fps:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()