"""
Remove .xml from Level1-Level4. Rename images so each is named after its folder: Level1_1.jpg, Level1_2.jpg, ... (folder name + number).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ["train", "valid", "test"]
LEVELS = ["Level1", "Level2", "Level3", "Level4"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def main():
    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.is_dir():
            continue
        for level in LEVELS:
            folder = split_dir / level
            if not folder.is_dir():
                continue
            # Remove .xml files
            for f in folder.glob("*.xml"):
                f.unlink()
                print(f"Removed: {f}")
            # Collect image files
            images = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS])
            if not images:
                continue
            # Two-pass rename to avoid overwriting: first to temp, then to final
            for i, f in enumerate(images):
                f.rename(folder / f"__tmp_{i}{f.suffix}")
            tmp_files = sorted(folder.glob("__tmp_*"), key=lambda p: int(p.stem.split("_")[-1]))
            for i, f in enumerate(tmp_files):
                ext = f.suffix
                f.rename(folder / f"{level}_{i+1}{ext}")
            print(f"{split}/{level}: renamed {len(images)} images to {level}_1 .. {level}_{len(images)}")
    print("Done.")

if __name__ == "__main__":
    main()
