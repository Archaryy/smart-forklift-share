"""
Find frames where the model is unsure about people, so you can label them.

A frame is saved when, at a low confidence threshold, it contains
  (a) a person detection with confidence between LOW and HIGH (uncertain), or
  (b) a person box that sits mostly inside a truck box (the red-on-red case)
      and is not confidently detected.
Saved next to each image is a YOLO-format .txt of the model's predictions, so
you can upload images + labels to Roboflow and just correct them.

Usage:
    python mine_hard_frames.py videos/test_site2.MOV [more videos...]
        [--model runs/forklift/v3_small_person/weights/best.pt]
        [--every 10] [--min-gap-sec 2] [--max 80] [--imgsz 960]
Output:  hard_frames/<video name>/<video>_f<frame>.jpg  (+ .txt)
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

CLS_FORKLIFT, CLS_PERSON, CLS_TRUCK = 0, 1, 2
LOW, HIGH = 0.15, 0.55          # "uncertain" person confidence band


def inside_fraction(inner, outer):
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="runs/v2_with_truck-2/weights/best.pt")
    ap.add_argument("--every", type=int, default=10, help="test every Nth frame")
    ap.add_argument("--min-gap-sec", type=float, default=2.0,
                    help="min seconds between saved frames (avoids near-duplicates)")
    ap.add_argument("--max", type=int, default=80, help="max frames saved per video")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    model = YOLO(args.model)

    for vpath in args.videos:
        vp = Path(vpath)
        out = Path("hard_frames") / vp.stem
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"Cannot open {vp}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        gap = int(args.min_gap_sec * fps)
        saved, last_saved, idx = 0, -10**9, -1

        while saved < args.max:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % args.every or idx - last_saved < gap:
                continue

            r = model.predict(frame, conf=LOW, imgsz=args.imgsz, verbose=False)[0]
            people, trucks, rows = [], [], []
            h, w = frame.shape[:2]
            for b in r.boxes:
                cls, c = int(b.cls[0]), float(b.conf[0])
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if cls == CLS_PERSON:
                    people.append(((x1, y1, x2, y2), c))
                elif cls == CLS_TRUCK:
                    trucks.append((x1, y1, x2, y2))
                rows.append(f"{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                            f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")

            uncertain = any(LOW <= c < HIGH for _, c in people)
            red_on_truck = any(c < 0.7 and inside_fraction(pb, tb) > 0.6
                               for pb, c in people for tb in trucks)
            if not (uncertain or red_on_truck):
                continue

            name = f"{vp.stem}_f{idx:06d}"
            cv2.imwrite(str(out / f"{name}.jpg"), frame)
            (out / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            saved, last_saved = saved + 1, idx

        cap.release()
        print(f"{vp.name}: saved {saved} frames to {out}")

    print("\nNext: upload hard_frames/<video>/ to Roboflow (images + .txt), "
          "correct the boxes, export, then python add_site_data.py <export folder>")


if __name__ == "__main__":
    main()
