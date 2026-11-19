"""
Site Footage Frame Extractor
─────────────────────────────
Extracts diverse, useful frames from your site videos for labeling.

Saves frames into 3 folders so labeling is faster:
  site_frames/
    likely_forklift/   model already sees a forklift here → just verify & correct boxes
    likely_truck/      model fires but it looks like a truck → label as truck or leave empty
    background/        no detections → pure background negatives (no labeling needed)

After extraction:
  1. Upload likely_forklift/ and likely_truck/ to Roboflow
  2. Label forklifts, persons, trucks
  3. Export as YOLOv8 format
  4. Run  python add_site_data.py  to merge into your dataset
  5. Retrain with  python train.py
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────
MODEL_PATH       = "runs/forklift/combined_yolo26n/weights/best.pt"
VIDEO_DIR        = Path("videos")
OUT_DIR          = Path("site_frames")

CLASS_FORKLIFT   = 0
CLASS_PERSON     = 1

CONF             = 0.30    # low – catch uncertain detections too
IMGSZ            = 640

# How many frames to extract per video per category
FRAMES_PER_CAT   = 80      # adjust up if you want more data

# Minimum pixel distance a new frame's detection must differ from last saved
# (prevents saving near-identical frames)
MIN_DIVERSITY_PX = 60

# Sample 1 frame every N (spread coverage across the full video)
SAMPLE_EVERY_N   = 20

# Truck heuristics (same as test_video.py)
MAX_ASPECT       = 1.6
MAX_AREA_FRAC    = 0.30

VIDEO_EXTS       = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def box_center(box):
    return ((box[0]+box[2])//2, (box[1]+box[3])//2)

def center_moved_enough(new_box, last_centers, thresh):
    cx, cy = box_center(new_box)
    for lx, ly in last_centers:
        if abs(cx-lx) < thresh and abs(cy-ly) < thresh:
            return False
    return True

def looks_like_truck(box, fw, fh):
    x1,y1,x2,y2 = box
    w = x2-x1; h = y2-y1
    return (w/max(h,1)) > MAX_ASPECT or (w*h)/(fw*fh) > MAX_AREA_FRAC

def annotate(frame, boxes, labels, colors):
    out = frame.copy()
    for box, lbl, col in zip(boxes, labels, colors):
        x1,y1,x2,y2 = box
        cv2.rectangle(out, (x1,y1),(x2,y2), col, 2)
        cv2.putText(out, lbl, (x1, max(y1-6,14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return out


# ────────────────────────────────────────────────────────
# Main extractor
# ────────────────────────────────────────────────────────

def extract():
    dirs = {
        "likely_forklift": OUT_DIR / "likely_forklift",
        "likely_truck":    OUT_DIR / "likely_truck",
        "background":      OUT_DIR / "background",
    }
    preview_dir = OUT_DIR / "previews"   # annotated copies for human reference only
    preview_dir.mkdir(parents=True, exist_ok=True)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    model  = YOLO(MODEL_PATH)
    videos = sorted(p for p in VIDEO_DIR.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS)

    if not videos:
        print(f"No videos found in {VIDEO_DIR.resolve()}")
        return

    total = {k: 0 for k in dirs}

    for vid in videos:
        print(f"\n── {vid.name}")
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print("   Cannot open – skip")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30

        # Per-category diversity tracking
        last_centers = {k: [] for k in dirs}
        counts       = {k: 0  for k in dirs}

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if frame_idx % SAMPLE_EVERY_N != 0:
                continue

            # Stop this video once all categories are saturated
            if all(counts[k] >= FRAMES_PER_CAT for k in dirs):
                break

            fh, fw = frame.shape[:2]
            results = model.predict(source=frame, conf=CONF,
                                    imgsz=IMGSZ, verbose=False)

            forklift_boxes = []
            truck_boxes    = []
            person_boxes   = []

            for box in results[0].boxes:
                cls  = int(box.cls[0])
                xyxy = list(map(int, box.xyxy[0].tolist()))
                if cls == CLASS_FORKLIFT:
                    if looks_like_truck(xyxy, fw, fh):
                        truck_boxes.append(xyxy)
                    else:
                        forklift_boxes.append(xyxy)
                elif cls == CLASS_PERSON:
                    person_boxes.append(xyxy)

            ts = int(frame_idx / fps)

            # ── Category: likely_forklift ──────────────
            if forklift_boxes and counts["likely_forklift"] < FRAMES_PER_CAT:
                ref = forklift_boxes[0]
                if center_moved_enough(ref, last_centers["likely_forklift"],
                                       MIN_DIVERSITY_PX):
                    boxes  = forklift_boxes + person_boxes
                    labels = (["forklift"]*len(forklift_boxes) +
                               ["person"]*len(person_boxes))
                    colors = ([(0,220,80)]*len(forklift_boxes) +
                               [(200,200,200)]*len(person_boxes))

                    fname = f"{vid.stem}_{ts:04d}s_f{frame_idx:06d}.jpg"
                    cv2.imwrite(str(dirs["likely_forklift"] / fname), frame)  # clean
                    cv2.imwrite(str(preview_dir / fname),                      # preview
                                annotate(frame, boxes, labels, colors))
                    last_centers["likely_forklift"].append(box_center(ref))
                    counts["likely_forklift"] += 1
                    total["likely_forklift"]  += 1

            # ── Category: likely_truck ─────────────────
            elif truck_boxes and counts["likely_truck"] < FRAMES_PER_CAT:
                ref = truck_boxes[0]
                if center_moved_enough(ref, last_centers["likely_truck"],
                                       MIN_DIVERSITY_PX):
                    fname = f"{vid.stem}_{ts:04d}s_f{frame_idx:06d}.jpg"
                    cv2.imwrite(str(dirs["likely_truck"] / fname), frame)      # clean
                    cv2.imwrite(str(preview_dir / fname),                       # preview
                                annotate(frame, truck_boxes,
                                         ["TRUCK?"]*len(truck_boxes),
                                         [(0,0,220)]*len(truck_boxes)))
                    last_centers["likely_truck"].append(box_center(ref))
                    counts["likely_truck"] += 1
                    total["likely_truck"]  += 1

            # ── Category: background ───────────────────
            elif (not forklift_boxes and not truck_boxes and
                  counts["background"] < FRAMES_PER_CAT):
                fname = f"{vid.stem}_{ts:04d}s_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(dirs["background"] / fname), frame)
                last_centers["background"].append((fw//2, fh//2))
                counts["background"] += 1
                total["background"]  += 1

        cap.release()
        print(f"   forklift:{counts['likely_forklift']}  "
              f"truck:{counts['likely_truck']}  "
              f"bg:{counts['background']}")

    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"  likely_forklift : {total['likely_forklift']} frames")
    print(f"  likely_truck    : {total['likely_truck']} frames")
    print(f"  background      : {total['background']} frames")
    print(f"  Output : {OUT_DIR.resolve()}")
    print(f"{'='*55}")
    print()
    print("NEXT STEPS:")
    print()
    print("  1. Go to https://roboflow.com  (free account)")
    print("  2. Create a new project → Object Detection")
    print("  3. Upload  site_frames/likely_forklift/  images")
    print("  4. Upload  site_frames/likely_truck/     images")
    print("     (skip background/ – those are already label-free negatives)")
    print()
    print("  5. In Roboflow, draw boxes and label:")
    print("       forklift  → forklifts")
    print("       person    → operators / bystanders")
    print("       truck     → trucks / trailers")
    print("     For truck images where there's NO forklift: leave empty")
    print()
    print("  6. Export as  YOLOv8 PyTorch  format → download zip")
    print("  7. Run:  python add_site_data.py  <path-to-unzipped-export>")


if __name__ == "__main__":
    extract()
