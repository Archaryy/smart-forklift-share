"""
Training script – Warehouse Forklift Safety Model
───────────────────────────────────────────────────
Classes:
  0: forklift
  1: person
  2: truck     ← NEW after hard-negative mining round

Augmentations are tuned for CCTV / warehouse conditions:
  - Lighting variation  (hsv_v, brightness)
  - Shadows / contrast  (clahe equivalently via hsv)
  - Camera angles       (degrees, perspective)
  - Scale range         (compact forklifts vs large trucks)
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# ── Dataset ──────────────────────────────────────────
DATA_YAML = Path("combined_dataset/data.yaml")

# ── Model ────────────────────────────────────────────
MODEL_NAME = "yolo26n.pt"      # swap to yolo26s.pt for more capacity

# ── Training hyperparams ─────────────────────────────
EPOCHS     = 150               # more epochs for 3 classes
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS    = 2
PROJECT    = "runs/forklift"
RUN_NAME   = sys.argv[1] if len(sys.argv) > 1 else "v2_with_truck"   # python train.py <run_name>

# ── Augmentation – CCTV-aware ─────────────────────────
# These are passed directly to model.train()
AUGMENTATION = dict(
    # ── Colour / lighting ──────────────────────
    hsv_h      = 0.015,   # small hue jitter (warehouse lights consistent)
    hsv_s      = 0.5,     # saturation – handles overcast vs sunny
    hsv_v      = 0.5,     # brightness  – handles deep shadows, sunlight glare
    # ── Geometry ───────────────────────────────
    degrees    = 5.0,     # mild rotation (camera is mostly fixed)
    translate  = 0.1,
    scale      = 0.6,     # important: scale 0.4–1.6× catches far/near forklifts
    shear      = 2.0,
    perspective= 0.0003,  # subtle perspective warp for CCTV angle
    flipud     = 0.0,     # forklifts don't appear upside-down
    fliplr     = 0.5,     # left-right flip is fine
    # ── Mosaic / mixup ─────────────────────────
    mosaic     = 1.0,     # mosaic helps with multi-object scenes
    mixup      = 0.1,
    copy_paste = 0.1,     # copy-paste forklift onto truck backgrounds
    # ── Additional ─────────────────────────────
    erasing    = 0.3,     # random erasing simulates partial occlusion
    auto_augment = "randaugment",  # auto augmentation policy
)


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML.resolve()}\n"
            "Run merge_dataset.py first."
        )

    for split in ("train", "val", "test"):
        img  = DATA_YAML.parent / "images" / split
        lbl  = DATA_YAML.parent / "labels" / split
        if not img.exists() or not lbl.exists():
            raise FileNotFoundError(
                f"Missing {split} split: {img} / {lbl}"
            )

    model = YOLO(MODEL_NAME)

    model.train(
        data       = str(DATA_YAML.resolve()),
        epochs     = EPOCHS,
        imgsz      = IMAGE_SIZE,
        batch      = BATCH_SIZE,
        device     = 0,
        workers    = WORKERS,
        patience   = 25,          # more patience for harder task
        project    = PROJECT,
        name       = RUN_NAME,
        exist_ok   = False,
        pretrained = True,
        optimizer  = "auto",
        lr0        = 0.01,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3.0,
        box        = 7.5,
        cls        = 0.5,
        dfl        = 1.5,
        plots      = True,
        save       = True,
        amp        = True,
        cache      = False,
        seed       = 42,
        deterministic = True,
        close_mosaic  = 15,
        label_smoothing = 0.1,    # helps with boundary ambiguity
        **AUGMENTATION,
    )

    print("\nTraining complete.")
    print(f"Best model: {PROJECT}/{RUN_NAME}/weights/best.pt")
    print("\nUpdate MODEL_PATH in test_video.py to point to the new best.pt")


if __name__ == "__main__":
    main()
