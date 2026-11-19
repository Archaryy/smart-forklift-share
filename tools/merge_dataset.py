
from pathlib import Path
from collections import Counter
import hashlib
import shutil

# --------------------------------------------------
# PATHS
# --------------------------------------------------

KAGGLE_ROOT = Path("original_dataset")
ULTRA_ROOT = Path("ultralytics_forklift/ultralytics_forklift_data")
OUTPUT_ROOT = Path("combined_dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Test gets priority when removing exact duplicate images.
SPLITS = {
    "test": [
        ("kaggle", "Test"),
        ("ultra", "test"),
    ],
    "val": [
        ("kaggle", "Valid"),
    ],
    "train": [
        ("kaggle", "Train"),
        ("ultra", "train"),
    ],
}


def image_hash(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()



def convert_labels(label_path, dataset_type):
    converted = []

    # Missing label = no annotations
    if not label_path.exists():
        return converted

    for line_number, line in enumerate(
        label_path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid annotation in {label_path}, line {line_number}"
            )

        class_id = int(parts[0])

        if dataset_type == "kaggle":
            # Kaggle: 0 = forklift, 1 = not_forklift
            if class_id == 0:
                converted.append("0 " + " ".join(parts[1:]))

        elif dataset_type == "ultra":
            # Ultralytics: 0 = Person, 1 = Forklift
            if class_id == 0:
                converted.append("1 " + " ".join(parts[1:]))
            elif class_id == 1:
                converted.append("0 " + " ".join(parts[1:]))
            else:
                raise ValueError(
                    f"Unexpected Ultralytics class {class_id}: {label_path}"
                )

    return converted


def get_source_paths(dataset_type, split):
    if dataset_type == "kaggle":
        root = KAGGLE_ROOT / split
    else:
        root = ULTRA_ROOT

    return root / "images", root / "labels"


def main():
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"{OUTPUT_ROOT} already exists. Rename or remove it first."
        )

    for split in SPLITS:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True)

    seen_hashes = {}
    counts = Counter()
    class_counts = Counter()
    skipped_duplicates = []

    # Process test first, then validation, then training.
    for output_split, sources in SPLITS.items():
        for dataset_type, source_split in sources:
            image_dir, label_dir = get_source_paths(
                dataset_type, source_split
            )

            if not image_dir.exists() or not label_dir.exists():
                raise FileNotFoundError(
                    f"Missing source folders: {image_dir} / {label_dir}"
                )

            images = sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in images:
                digest = image_hash(image_path)

                if digest in seen_hashes:
                    skipped_duplicates.append(
                        (image_path, seen_hashes[digest])
                    )
                    continue

                label_path = label_dir / f"{image_path.stem}.txt"

                converted = convert_labels(label_path, dataset_type)

                # Prefix names to avoid filename collisions.
                new_name = (
                    f"{dataset_type}_{source_split}_{image_path.name}"
                )

                output_image = (
                    OUTPUT_ROOT / "images" / output_split / new_name
                )
                output_label = (
                    OUTPUT_ROOT / "labels" / output_split /
                    f"{Path(new_name).stem}.txt"
                )

                shutil.copy2(image_path, output_image)
                output_label.write_text(
                    "\n".join(converted) + ("\n" if converted else "")
                )

                seen_hashes[digest] = f"{output_split}/{new_name}"
                counts[output_split] += 1

                for line in converted:
                    class_counts[(output_split, int(line.split()[0]))] += 1

    # Write Ultralytics dataset configuration.
    yaml_content = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: forklift
  1: person
"""

    (OUTPUT_ROOT / "data.yaml").write_text(yaml_content)

    print("\n========== DATASET CREATED ==========")

    for split in ("train", "val", "test"):
        print(f"\n{split.upper()}")
        print(f"Images: {counts[split]}")
        print(f"Forklift boxes: {class_counts[(split, 0)]}")
        print(f"Person boxes: {class_counts[(split, 1)]}")

    print(f"\nExact duplicate images skipped: {len(skipped_duplicates)}")
    print(f"Output: {OUTPUT_ROOT.resolve()}")

    if skipped_duplicates:
        print("\nFirst 10 duplicate examples:")
        for duplicate, kept in skipped_duplicates[:10]:
            print(f"Skipped: {duplicate} | Kept: {kept}")


if __name__ == "__main__":
    main()
