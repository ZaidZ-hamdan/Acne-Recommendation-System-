# generate figures for the graduation report
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
TRAIN_DIR = ROOT / "train"
VALID_DIR = ROOT / "valid"
TEST_DIR = ROOT / "test"
CLASSES = ["Level1", "Level2", "Level3", "Level4"]
OUTPUT.mkdir(exist_ok=True)


def count_per_class(split_dir):
    counts = {}
    for c in CLASSES:
        folder = split_dir / c
        if folder.is_dir():
            n = len([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            counts[c] = n
        else:
            counts[c] = 0
    return counts


def fig_dataset_distribution():
    train = count_per_class(TRAIN_DIR)
    valid = count_per_class(VALID_DIR)
    test = count_per_class(TEST_DIR)
    x = np.arange(len(CLASSES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, [train[c] for c in CLASSES], w, label="Train", color="steelblue")
    ax.bar(x, [valid[c] for c in CLASSES], w, label="Validation", color="darkorange")
    ax.bar(x + w, [test[c] for c in CLASSES], w, label="Test", color="forestgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Number of images")
    ax.set_title("Dataset distribution by severity level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig_dataset_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_dataset_distribution.png")


def fig_class_balance_train():
    train = count_per_class(TRAIN_DIR)
    total = sum(train.values())
    if total == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#f44336"]
    wedges, texts, autotexts = ax.pie([train[c] for c in CLASSES], labels=CLASSES, autopct="%1.1f%%", colors=colors, startangle=90)
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Training set class balance")
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig_class_balance_train.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_class_balance_train.png")


def fig_metrics_from_report():
    report_path = OUTPUT / "classification_report.txt"
    if not report_path.exists():
        return
    text = report_path.read_text()
    prec, rec, f1 = [], [], []
    for c in CLASSES:
        m = re.search(rf"{c}\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", text)
        if m:
            prec.append(float(m.group(1)))
            rec.append(float(m.group(2)))
            f1.append(float(m.group(3)))
        else:
            prec.append(0); rec.append(0); f1.append(0)
    if not prec:
        return
    x = np.arange(len(CLASSES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, prec, w, label="Precision", color="steelblue")
    ax.bar(x, rec, w, label="Recall", color="darkorange")
    ax.bar(x + w, f1, w, label="F1-score", color="forestgreen")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation metrics per severity class")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig_metrics_per_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_metrics_per_class.png")


def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color="lightblue"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, wrap=True)

    box(0.5, 7, 2, 1.2, "User\nuploads image", "lavender")
    box(3, 7, 2, 1.2, "Preprocess\n224×224, normalize", "lightyellow")
    box(5.5, 7, 2, 1.2, "MobileNetV2\nCNN", "lightblue")
    box(8, 7, 1.5, 1.2, "Severity\nLevel1–4", "lightgreen")
    ax.annotate("", xy=(2.6, 7.6), xytext=(2.5, 7.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(5.4, 7.6), xytext=(3.2, 7.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.9, 7.6), xytext=(7.6, 7.6), arrowprops=dict(arrowstyle="->"))

    box(0.5, 4, 2, 1.2, "Severity\nresult", "lightgreen")
    box(3, 4, 2.5, 1.2, "Chatbot (LLM)\nNebius API", "mistyrose")
    box(6.2, 4, 2.3, 1.2, "Recommendations\nmorning / night", "honeydew")
    ax.annotate("", xy=(2.6, 4.6), xytext=(2.5, 4.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.1, 4.6), xytext=(5.6, 4.6), arrowprops=dict(arrowstyle="->"))

    box(0.5, 1, 2, 1, "Survey\n(skin type, etc.)", "aliceblue")
    ax.annotate("", xy=(2.6, 1.5), xytext=(1.5, 1.5), arrowprops=dict(arrowstyle="->"))
    ax.text(5, 5.5, "Streamlit UI", fontsize=11, ha="center", style="italic")
    ax.set_title("System architecture: Acne detection and skincare recommendation", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig_system_architecture.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_system_architecture.png")


def fig_training_pipeline():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis("off")
    box = lambda x, y, w, h, t, c="lightblue": ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.02", facecolor=c, edgecolor="black", linewidth=1)) or ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=9)
    box(0.3, 1.5, 1.4, 1, "Train/Valid\nimages", "lavender")
    box(2, 1.5, 1.6, 1, "Augment\n(resize, flip, etc.)", "lightyellow")
    box(3.9, 1.5, 1.8, 1, "Phase 1:\nHead only", "lightblue")
    box(6, 1.5, 1.8, 1, "Phase 2:\nFine-tune backbone", "lightgreen")
    box(8, 1.5, 0.8, 1, "Model\nsave", "mistyrose")
    for (x1, x2) in [(1.7, 2), (3.6, 3.9), (5.7, 6), (7.8, 8)]:
        ax.annotate("", xy=(x2, 2), xytext=(x1, 2), arrowprops=dict(arrowstyle="->"))
    ax.text(4.5, 0.5, "Two-phase training pipeline", fontsize=11, ha="center")
    ax.set_title("CNN training methodology", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig_training_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig_training_pipeline.png")


def main():
    fig_dataset_distribution()
    fig_class_balance_train()
    fig_metrics_from_report()
    fig_system_architecture()
    fig_training_pipeline()
    print("All figures saved to output/")


if __name__ == "__main__":
    main()
