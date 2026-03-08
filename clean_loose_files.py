"""
Remove .xml files and move loose images from train/ valid/ test/ roots
into the correct Level1-Level4 subfolders (using the trained model to classify).
Only processes files directly in train/, valid/, test/ — not inside Level* or Unlabeled.
"""
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_SAVE_DIR = ROOT / "saved_model"
IMG_SIZE = (224, 224)
CLASSES = ["Level1", "Level2", "Level3", "Level4"]
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_model_and_classes():
    path = MODEL_SAVE_DIR / "acne_severity_model.keras"
    if not path.exists():
        path = MODEL_SAVE_DIR / "best_model.keras"
    if not path.exists():
        return None, None
    model = keras.models.load_model(str(path))
    if (MODEL_SAVE_DIR / "class_names.json").exists():
        import json
        with open(MODEL_SAVE_DIR / "class_names.json") as f:
            classes = json.load(f).get("classes", CLASSES)
    else:
        classes = CLASSES
    return model, classes


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def next_index(split_dir, level):
    folder = split_dir / level
    if not folder.is_dir():
        return 1
    max_n = 0
    for f in folder.iterdir():
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.match(r"Level[1-4]_(\d+)", f.stem)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def main():
    # 1) Delete .xml in train/ valid/ test/ roots (direct children only)
    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.is_dir():
            continue
        for f in split_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".xml":
                f.unlink()
                print(f"Removed: {split}/{f.name}")
    print()

    model, class_list = load_model_and_classes()
    if model is None:
        print("No model. Classifying skipped for loose images; .xml were still removed.")
        return
    print("Model loaded. Moving loose images into level folders.\n")

    # 2) Move loose images (direct children of split dir) into Level folders
    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.is_dir():
            continue
        loose = [f for f in split_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not loose:
            continue
        indices = {c: next_index(split_dir, c) for c in CLASSES}
        moved = {c: 0 for c in CLASSES}
        for path in loose:
            x = preprocess(path)
            probs = model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            level = class_list[idx]
            new_name = f"{level}_{indices[level]}{path.suffix}"
            dest = split_dir / level / new_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            indices[level] += 1
            moved[level] += 1
        print(f"{split}/ (root): moved {len(loose)} images -> Level1:{moved['Level1']} Level2:{moved['Level2']} Level3:{moved['Level3']} Level4:{moved['Level4']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
