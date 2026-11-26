"""
Fine-tune the current model for small / far-away pedestrians.

What it does differently from train.py
  * starts from your trained best.pt (not from yolo26n.pt)
  * imgsz 960 so far-away people keep more pixels
  * oversamples YOUR site frames (site_*) SITE_REPEAT times, and uses only a
    random subset of the generic Kaggle/Ultralytics images, so site data
    carries real weight instead of being <5% of every epoch
  * more hue jitter so the model cannot rely on "red = truck"
  * low learning rate, few epochs

Usage:
    python finetune.py [run_name] [path/to/start_weights.pt]
Output weights:
    runs/forklift/<run_name>/weights/best.pt      (absolute project path, so
    Ultralytics does not add runs/detect/ in front)
"""
import random
import sys
from pathlib import Path

from ultralytics import YOLO

# ── Settings ─────────────────────────────────────────
START_WEIGHTS = "runs/v2_with_truck-2/weights/best.pt"
DATASET       = Path("combined_dataset")
RUN_NAME      = "v3_small_person"

SITE_REPEAT   = 5      # each site_ frame appears this many times per epoch
GENERIC_N     = 2500   # random generic train images (Kaggle + Ultralytics)
VAL_N         = 400    # random val images (generic; site frames are all in train)
# Site videos to keep OUT of training and use as a site-domain val set.
# e.g. ["test_site2"] - empty = train on every site frame.
HOLDOUT_SITE  = ["site_hold_"]   # 25 own-annotated frames kept out of training (valid+test of the zip)

EPOCHS        = 25
IMGSZ         = 960
BATCH         = 8      # lower to 4 if you run out of GPU memory
WORKERS       = 2      # Windows: more workers can trigger the paging-file error
SEED          = 42

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def images(split):
    return sorted(p for p in (DATASET / "images" / split).iterdir()
                  if p.suffix.lower() in IMG_EXT)


def write_list(path, imgs):
    path.write_text("\n".join(str(p.resolve()).replace("\\", "/") for p in imgs) + "\n")


def build_lists():
    random.seed(SEED)
    train = images("train")
    site    = [p for p in train if p.name.startswith("site_")]
    generic = [p for p in train if not p.name.startswith("site_")]

    held = [p for p in site if any(h in p.name for h in HOLDOUT_SITE)]
    site = [p for p in site if p not in set(held)]

    generic = random.sample(generic, min(GENERIC_N, len(generic)))
    train_list = generic + site * SITE_REPEAT
    random.shuffle(train_list)

    val = images("val")
    val_list = random.sample(val, min(VAL_N, len(val))) + held

    write_list(DATASET / "finetune_train.txt", train_list)
    write_list(DATASET / "finetune_val.txt", val_list)

    yaml_path = DATASET / "finetune.yaml"
    yaml_path.write_text(
        f"path: {DATASET.resolve().as_posix()}\n"
        "train: finetune_train.txt\n"
        "val: finetune_val.txt\n"
        "names:\n  0: forklift\n  1: person\n  2: truck\n"
    )
    print(f"train list: {len(generic)} generic + {len(site)} site x{SITE_REPEAT} "
          f"= {len(train_list)} per epoch")
    print(f"val list  : {len(val_list)} ({len(held)} held-out site frames)")
    return yaml_path


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else RUN_NAME
    weights  = sys.argv[2] if len(sys.argv) > 2 else START_WEIGHTS
    if not Path(weights).exists():
        raise FileNotFoundError(f"Start weights not found: {weights}")

    yaml_path = build_lists()
    model = YOLO(weights)
    model.train(
        data=str(yaml_path.resolve()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        patience=10,
        project=str(Path("runs/forklift").resolve()),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        lr0=0.002,
        lrf=0.1,
        warmup_epochs=1.0,
        close_mosaic=5,
        cache=False,
        amp=True,
        seed=SEED,
        # augmentation: lighting + colour independence, keep scale range wide
        # so both near and far people are seen
        hsv_h=0.03, hsv_s=0.6, hsv_v=0.5,
        degrees=5.0, translate=0.1, scale=0.6, shear=2.0,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.05, copy_paste=0.1, erasing=0.3,
        plots=True, save=True,
    )
    print(f"\nDone. Weights: runs/forklift/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
