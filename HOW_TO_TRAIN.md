# How to Train the Acne Severity Model

The model and training script are ready. Run them on your machine (Python required).

**Detected on this PC:** Python 3.14.3 at `%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe` (see `python_version_info.txt`).  
**Note:** TensorFlow does not support Python 3.14. For training, use **Python 3.9, 3.10, or 3.11**. Install 3.11 from [python.org](https://www.python.org/downloads/) and use it in a venv or via `py -3.11` if you have multiple versions.

## 1. Install Python (3.9–3.11 for TensorFlow)

- Install **Python 3.8, 3.9, 3.10, or 3.11** from [python.org](https://www.python.org/downloads/).
- During setup, check **"Add Python to PATH"**.

## 2. Open Terminal in the CRS Folder

- In File Explorer, go to `c:\Users\Zaidz\Downloads\CRS`.
- In the address bar type `cmd` and press Enter (opens Command Prompt here),  
  **or** right‑click → "Open in Terminal" / "Open PowerShell window here".

## 3. Create a Virtual Environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

(This installs TensorFlow, Keras, Pillow, etc. It may take a few minutes.)

## 5. Run Training

```bash
python train_model.py
```

- Training uses **EfficientNetB0** (transfer learning) on your **train** and **valid** folders (Level1–Level4 only; Unlabeled is ignored).
- **Outputs:**
  - **saved_model/acne_severity_model.keras** — trained model (for Streamlit).
  - **saved_model/best_model.keras** — best checkpoint by validation accuracy.
  - **saved_model/class_names.json** — class order for the app.
  - **output/training_history.png** — accuracy and loss curves.
  - **output/confusion_matrix.png** — validation confusion matrix.
  - **output/classification_report.txt** — precision, recall, F1.
- If a **test** folder exists with Level1–Level4, the script also reports test accuracy.

## 6. After Training

- Use **saved_model/acne_severity_model.keras** and **saved_model/class_names.json** in your Streamlit app for predictions.
- Use the **output/** plots and report in your project report or presentation.

## Troubleshooting

- **"Python was not found"** — Reinstall Python and tick "Add Python to PATH", or use the full path to `python.exe` (e.g. `C:\Users\Zaidz\AppData\Local\Programs\Python\Python311\python.exe train_model.py`).
- **Out of memory (GPU/CPU)** — In `train_model.py`, change `BATCH_SIZE = 32` to `16` or `8`.
- **Training very slow** — Normal on CPU (EfficientNetB0). Using a GPU with TensorFlow‑GPU will speed it up.
