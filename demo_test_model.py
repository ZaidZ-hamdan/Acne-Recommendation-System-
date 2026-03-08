"""
Demo: Load the trained acne severity model and run prediction on sample or custom images.
Usage:
  python demo_test_model.py                    # test one image from each class in test/
  python demo_test_model.py path/to/image.jpg # test a single image
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ---------- Config (must match train_model.py) ----------
DATA_ROOT = Path(__file__).resolve().parent
MODEL_SAVE_DIR = DATA_ROOT / "saved_model"
TEST_DIR = DATA_ROOT / "test"
IMG_SIZE = (224, 224)
CLASSES = ["Level1", "Level2", "Level3", "Level4"]


def load_model_and_classes():
    """Load saved model and class names. Returns (model, class_list) or (None, None)."""
    model_path = MODEL_SAVE_DIR / "acne_severity_model.keras"
    if not model_path.exists():
        model_path = MODEL_SAVE_DIR / "best_model.keras"
    if not model_path.exists():
        return None, None

    model = keras.models.load_model(str(model_path))
    class_path = MODEL_SAVE_DIR / "class_names.json"
    if class_path.exists():
        with open(class_path) as f:
            data = json.load(f)
            class_list = data.get("classes", CLASSES)
    else:
        class_list = CLASSES
    return model, class_list


def preprocess_image(image_path):
    """Load image, resize to 224x224, normalize to [0,1]. Returns shape (1, 224, 224, 3)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # batch dim


def predict_and_print(model, class_list, image_path, show_top=3):
    """Run prediction and print result."""
    x = preprocess_image(image_path)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_list[pred_idx]
    conf = float(probs[pred_idx])

    print(f"  Image: {image_path.name}")
    print(f"  Predicted: {pred_label}  (confidence: {conf:.2%})")
    print(f"  All classes: ", end="")
    for i, c in enumerate(class_list):
        print(f"{c}={probs[i]:.2%}", end="  ")
    print()
    return pred_label, conf


def main():
    model, class_list = load_model_and_classes()
    if model is None:
        print("No trained model found.")
        print("Run training first:  python train_model.py")
        print("Then run this demo:   python demo_test_model.py")
        sys.exit(1)

    print("Model loaded successfully.")
    print("Classes:", class_list)
    print("-" * 50)

    # Single image from command line
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if not img_path.is_file():
            print(f"File not found: {img_path}")
            sys.exit(1)
        print("Prediction for provided image:")
        predict_and_print(model, class_list, img_path)
        return

    # Demo: one image from each test class
    if not TEST_DIR.is_dir():
        print("No test folder found. Run with: python demo_test_model.py path/to/image.jpg")
        sys.exit(1)

    print("Demo: one image from each severity level (test set)\n")
    for level in CLASSES:
        folder = TEST_DIR / level
        if not folder.is_dir():
            continue
        images = list(folder.glob("*.jpg"))[:1]
        if not images:
            images = list(folder.glob("*.jpeg"))[:1]
        if not images:
            images = list(folder.glob("*.png"))[:1]
        if images:
            predict_and_print(model, class_list, images[0])
            print()
    print("Demo done.")


if __name__ == "__main__":
    main()
