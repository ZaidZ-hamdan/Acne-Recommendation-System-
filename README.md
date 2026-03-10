# Acne AI Assistant

AI-based acne severity detection and personalized skincare routine recommendation system. A graduation project that combines a CNN classifier (MobileNetV2), a Streamlit web app, an LLM chatbot, and rule-based recommendations.

## Features

- **Acne severity classification** — Upload a face image and get a prediction (Level 1–4).
- **Chatbot** — Ask about skin or acne; the bot can use your detection result as context (Nebius LLM API).
- **Recommendations** — Morning/night skincare routines after you chat or complete a short survey.
- **Educational use only** — Not a substitute for medical advice.

## Requirements

- Python 3.10 or 3.11 (TensorFlow does not support 3.12+ in older versions).
- Dependencies in `requirements.txt`.

## Setup

1. Clone or download this repo and open a terminal in the project folder.

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. For the chatbot, create a `.env` file in the project root with:
   ```
   LLM_BASE_URL=https://api.studio.nebius.com/v1/
   LLM_API_KEY=your_api_key_here
   LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
   ```
   Replace `your_api_key_here` with your Nebius API key. See `.env.example` if available.

## Dataset

Place your data in this structure:

- `train/` — subfolders `Level1/`, `Level2/`, `Level3/`, `Level4/` with training images.
- `valid/` — same structure for validation.
- `test/` — same structure for testing (optional).

Images can be `.jpg` or `.png`. The folder name is the label.

## Training the model

From the project folder:

```bash
py -3.11 model/train_model.py
```

Or use the batch file (Windows) if it points to this path:

```bash
run_training.bat
```

Trained model and class names are saved in `saved_model/`. Plots and reports go to `output/`. To regenerate report figures: `py -3.11 model/generate_figures.py`.

## Running the app

```bash
streamlit run app.py
```

Then open the URL shown (usually http://localhost:8501). Use **Acne Detection** to upload and predict, **Chatbot** to talk to the assistant, and **Recommendations** after chatting or doing the survey.

## Project structure

- `app.py` — Streamlit application (run from project root).
- `model/train_model.py` — CNN training (MobileNetV2, two-phase training).
- `model/generate_figures.py` — Generate charts for the report (run after training).
- `saved_model/` — Trained model and `class_names.json`.
- `output/` — Training curves, confusion matrix, classification report.
- `train/`, `valid/`, `test/` — Dataset folders (not in repo if excluded by `.gitignore`).

## Disclaimer

This project is for **educational purposes only**. It does not provide medical advice. Users with skin concerns should consult a dermatologist.
