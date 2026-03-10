# train acne model - MobileNetV2, 4 classes
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
VALID_DIR = ROOT / "valid"
TEST_DIR = ROOT / "test"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "saved_model"
IMG_SIZE = (224, 224)
BATCH = 32
EP1, EP2 = 20, 70
PATIENCE = 28
CLASSES = ["Level1", "Level2", "Level3", "Level4"]
N_CLASS = len(CLASSES)


def get_data():
    train_gen = ImageDataGenerator(
        rescale=1.0/255, rotation_range=25, width_shift_range=0.15, height_shift_range=0.15,
        horizontal_flip=True, zoom_range=0.2, brightness_range=[0.8, 1.2], shear_range=5, fill_mode="nearest"
    )
    val_gen = ImageDataGenerator(rescale=1.0/255)
    train_flow = train_gen.flow_from_directory(str(TRAIN_DIR), target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", classes=CLASSES, shuffle=True, seed=42)
    val_flow = val_gen.flow_from_directory(str(VALID_DIR), target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", classes=CLASSES, shuffle=False, seed=42)
    return train_flow, val_flow


def build_model(freeze=True, unfreeze_frac=0.15):
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet", pooling=None, alpha=1.0)
    base.trainable = not freeze
    if not freeze:
        n = len(base.layers)
        for layer in base.layers[: int(n * (1 - unfreeze_frac))]:
            layer.trainable = False
    reg = keras.regularizers.l2(1e-4)
    inp = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inp)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(N_CLASS, activation="softmax", kernel_regularizer=reg)(x)
    return keras.Model(inp, out)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    train_flow, val_flow = get_data()
    print("classes:", train_flow.class_indices)

    w = compute_class_weight("balanced", classes=np.unique(train_flow.classes), y=train_flow.classes)
    cw = dict(zip(np.unique(train_flow.classes), w))
    for i in [2, 3]:
        if i in cw:
            cw[i] = float(cw[i]) * 1.4
    print("class weights:", cw)

    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.02)
    model = build_model(freeze=True)
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss=loss_fn, metrics=["accuracy"])
    model.summary()

    cb1 = [EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
           ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)]
    print("phase 1: head only")
    h1 = model.fit(train_flow, epochs=EP1, validation_data=val_flow, callbacks=cb1, class_weight=cw, verbose=1)

    base = model.layers[1]
    base.trainable = True
    for layer in base.layers[: int(len(base.layers) * 0.85)]:
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(3e-5), loss=loss_fn, metrics=["accuracy"])
    cb2 = [EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1),
           ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
           ModelCheckpoint(str(MODEL_DIR / "best_model.keras"), monitor="val_accuracy", save_best_only=True, verbose=1)]
    print("phase 2: fine-tune")
    h2 = model.fit(train_flow, epochs=EP2, validation_data=val_flow, callbacks=cb2, class_weight=cw, verbose=1)

    model.save(str(MODEL_DIR / "acne_severity_model.keras"))
    with open(MODEL_DIR / "class_names.json", "w") as f:
        json.dump({"classes": CLASSES, "indices": train_flow.class_indices}, f, indent=2)

    val_flow.reset()
    pred = np.argmax(model.predict(val_flow, verbose=1), axis=1)
    true = val_flow.classes
    report = classification_report(true, pred, target_names=CLASSES, digits=4)
    print("\n", report)
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    hist = {"accuracy": h1.history["accuracy"]+h2.history["accuracy"], "val_accuracy": h1.history["val_accuracy"]+h2.history["val_accuracy"],
            "loss": h1.history["loss"]+h2.history["loss"], "val_loss": h1.history["val_loss"]+h2.history["val_loss"]}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="val"); ax[0].set_title("accuracy"); ax[0].legend()
    ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="val"); ax[1].set_title("loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "training_history.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(true, pred), annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix"); plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    if TEST_DIR.is_dir():
        test_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
            str(TEST_DIR), target_size=IMG_SIZE, batch_size=BATCH, class_mode="categorical",
            classes=CLASSES, shuffle=False, seed=42)
        if test_gen.samples > 0:
            loss, acc = model.evaluate(test_gen, verbose=1)
            print("test accuracy:", acc)
            with open(OUTPUT_DIR / "test_metrics.txt", "w") as f:
                f.write(f"Test accuracy: {acc:.4f}\nTest loss: {loss:.4f}\n")
    print("done")


if __name__ == "__main__":
    main()
