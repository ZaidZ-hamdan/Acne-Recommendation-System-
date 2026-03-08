"""
Train acne severity CNN using transfer learning (MobileNetV2).
Fits ~2.8k images: lighter backbone + two-phase training (head first, then fine-tune).
Saves: model, class names, history plot, confusion matrix.
"""
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

# ---------- Config ----------
DATA_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"
OUTPUT_DIR = DATA_ROOT / "output"
MODEL_SAVE_DIR = DATA_ROOT / "saved_model"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 70
PATIENCE = 28
CLASSES = ["Level1", "Level2", "Level3", "Level4"]
NUM_CLASSES = len(CLASSES)


def get_data_generators():
    """Build train and validation generators; only use Level1-Level4."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        shear_range=5,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True,
        seed=42,
    )
    valid_generator = val_test_datagen.flow_from_directory(
        str(VALID_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
        seed=42,
    )
    return train_generator, valid_generator


def build_model(freeze_backbone=True, unfreeze_top_fraction=0.15):
    """MobileNetV2 + regularized head. Less fine-tuning to reduce overfitting."""
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
        alpha=1.0,
    )
    base.trainable = not freeze_backbone
    if not freeze_backbone:
        n = len(base.layers)
        for layer in base.layers[: int(n * (1 - unfreeze_top_fraction))]:
            layer.trainable = False

    reg = keras.regularizers.l2(1e-4)
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=reg)(x)
    return keras.Model(inputs, outputs)


def plot_history(history, save_path):
    """Save accuracy and loss curves. history is dict with accuracy, val_accuracy, loss, val_loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["accuracy"], label="Train")
    axes[0].plot(history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["loss"], label="Train")
    axes[1].plot(history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved history plot to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Compute and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Validation)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def main():
    print("Data root:", DATA_ROOT)
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")
    if not VALID_DIR.is_dir():
        raise FileNotFoundError(f"Valid directory not found: {VALID_DIR}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_SAVE_DIR.mkdir(exist_ok=True)

    train_gen, valid_gen = get_data_generators()
    print("Class indices:", train_gen.class_indices)

    cls_vals = np.unique(train_gen.classes)
    class_weights = compute_class_weight(
        "balanced", classes=cls_vals, y=train_gen.classes
    )
    class_weight_dict = dict(zip(cls_vals, class_weights))
    for idx in [2, 3]:
        if idx in class_weight_dict:
            class_weight_dict[idx] = float(class_weight_dict[idx]) * 1.4
    print("Class weights (Level3/4 boosted 1.4x):", class_weight_dict)

    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.02)
    model = build_model(freeze_backbone=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    model.summary()
    print("Phase 1: training head only (backbone frozen)")
    cb1 = [
        EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    ]
    hist1 = model.fit(
        train_gen, epochs=EPOCHS_PHASE1,
        validation_data=valid_gen, callbacks=cb1,
        class_weight=class_weight_dict, verbose=1,
    )

    # Phase 2: unfreeze only top 15% of backbone (less overfitting)
    base = model.layers[1]
    base.trainable = True
    n = len(base.layers)
    for layer in base.layers[: int(n * 0.85)]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    print("Phase 2: fine-tuning top layers of backbone")
    cb2 = [
        EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(MODEL_SAVE_DIR / "best_model.keras"), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]
    hist2 = model.fit(
        train_gen, epochs=EPOCHS_PHASE2,
        validation_data=valid_gen, callbacks=cb2,
        class_weight=class_weight_dict, verbose=1,
    )
    history = {
        "accuracy": hist1.history["accuracy"] + hist2.history["accuracy"],
        "val_accuracy": hist1.history["val_accuracy"] + hist2.history["val_accuracy"],
        "loss": hist1.history["loss"] + hist2.history["loss"],
        "val_loss": hist1.history["val_loss"] + hist2.history["val_loss"],
    }

    # Save final model (best weights already restored by EarlyStopping)
    model.save(str(MODEL_SAVE_DIR / "acne_severity_model.keras"))
    print(f"Model saved to {MODEL_SAVE_DIR / 'acne_severity_model.keras'}")

    # Save class names and indices for Streamlit
    class_indices = train_gen.class_indices
    with open(MODEL_SAVE_DIR / "class_names.json", "w") as f:
        json.dump({"classes": CLASSES, "indices": class_indices}, f, indent=2)
    print(f"Class names saved to {MODEL_SAVE_DIR / 'class_names.json'}")

    # Validation metrics and plots
    valid_gen.reset()
    y_pred_proba = model.predict(valid_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = valid_gen.classes

    report = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4
    )
    print("\nClassification Report (Validation):\n", report)
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    plot_history(history, OUTPUT_DIR / "training_history.png")
    plot_confusion_matrix(y_true, y_pred, CLASSES, OUTPUT_DIR / "confusion_matrix.png")

    # Optional: evaluate on test set if present
    if TEST_DIR.is_dir():
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        test_gen = test_datagen.flow_from_directory(
            str(TEST_DIR),
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            classes=CLASSES,
            shuffle=False,
            seed=42,
        )
        if test_gen.samples > 0:
            test_loss, test_acc = model.evaluate(test_gen, verbose=1)
            print(f"\nTest accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
            with open(OUTPUT_DIR / "test_metrics.txt", "w") as f:
                f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n")

    print("\nTraining complete. You can proceed to Streamlit.")


if __name__ == "__main__":
    main()
