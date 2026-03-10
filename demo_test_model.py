# test model on image(s)
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_model"
TEST_DIR = ROOT / "test"
IMG_SIZE = (224, 224)
CLASSES = ["Level1", "Level2", "Level3", "Level4"]


def load_model():
    p = MODEL_DIR / "acne_severity_model.keras"
    if not p.exists():
        p = MODEL_DIR / "best_model.keras"
    if not p.exists():
        return None, CLASSES
    m = keras.models.load_model(str(p))
    classes = CLASSES
    if (MODEL_DIR / "class_names.json").exists():
        with open(MODEL_DIR / "class_names.json") as f:
            classes = json.load(f).get("classes", CLASSES)
    return m, classes


def prep(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    return x[np.newaxis, ...]


def main():
    model, cls = load_model()
    if model is None:
        print("Run train_model.py first")
        sys.exit(1)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_file():
            print("File not found")
            sys.exit(1)
        probs = model.predict(prep(path), verbose=0)[0]
        i = int(np.argmax(probs))
        print(f"{path.name} -> {cls[i]} ({probs[i]:.0%})")
        return
    if not TEST_DIR.is_dir():
        print("Use: demo_test_model.py path/to/image.jpg")
        sys.exit(1)
    print("One image per class:\n")
    for level in CLASSES:
        folder = TEST_DIR / level
        imgs = list(folder.glob("*.jpg"))[:1] or list(folder.glob("*.png"))[:1]
        if imgs:
            probs = model.predict(prep(imgs[0]), verbose=0)[0]
            i = int(np.argmax(probs))
            print(f"{imgs[0].name} -> {cls[i]} ({probs[i]:.0%})")
    print("\ndone")


if __name__ == "__main__":
    main()
