# AI-Based Acne Detection & Personalized Skincare Routine Recommendation System

**Graduation Project — Summary from Project Document**

---

## 1. Project Overview

Build an **AI-powered dermatology assistant** that:
- Analyzes facial acne from uploaded images (CNN).
- Lets users specify skin type and symptoms.
- Classifies **acne severity** (and/or type if labels allow).
- Uses a **chatbot** to collect extra info.
- Generates a **personalized skincare routine** (morning, night, warnings, when to see a doctor).

---

## 2. Main Objectives

| Objective | Description |
|----------|-------------|
| CNN model | Classify acne severity/type from facial images |
| Transfer learning | Use pre-trained CNN (EfficientNetB0 / MobileNetV2 / ResNet50) for good accuracy with limited time |
| Streamlit app | Upload images, show results, run chatbot, show recommendations |
| Chatbot | Intent-based questionnaire (no LLM): skin type, symptoms, duration, etc. |
| Recommendation engine | Rule-based morning + night routine, warnings, dermatologist referral when needed |
| Evaluation | Accuracy, precision, recall, F1, confusion matrix, train/val curves |

---

## 3. Dataset (Your CRS Folder)

- **Source:** Roboflow Universe (Acne/Severity).
- **Content:** Facial acne images, **severity labels**.
- **Your structure:** `train` / `valid` / `test`, each with **Level1, Level2, Level3, Level4** (+ Unlabeled).
- **Mapping for presentation:** Level1 = mildest → Level4 = most severe (can map to Clear/Mild/Moderate/Severe if needed).
- **Preprocessing (from doc):** Resize 224×224, normalize [0,1], augmentation (rotation, flip, zoom, brightness). Train/val/test split already done.

---

## 4. System Architecture

1. **User interface (Streamlit):** Upload image, input symptoms, view prediction and routine.
2. **Preprocessing:** Resize, normalize before CNN.
3. **CNN module:** Trained model → severity (and optionally type) class.
4. **Chatbot / NLP:** Intent-based Q&A to collect skin type, symptoms, etc.
5. **Recommendation engine:** Rules from prediction + user answers → morning/night routine + warnings.
6. **Reporting:** Confidence score, result, safety warnings.

---

## 5. CNN Model (from doc)

- **Approach:** Transfer learning, fine-tune on acne dataset.
- **Options:** EfficientNetB0 (accuracy), MobileNetV2 (lightweight), ResNet50 (medical imaging).
- **Input:** 224×224, normalized.
- **Loss:** Categorical crossentropy (multi-class).
- **Optimizer:** Adam. Batch size 16–32. Epochs 10–30.
- **Regularization:** Dropout. Unfreeze top layers for fine-tuning.
- **Output:** Severity (e.g. Level1–Level4 or Clear/Mild/Moderate/Severe).

---

## 6. Streamlit Pages (Suggested)

| Page | Purpose |
|------|---------|
| **Home** | Intro, how to use, disclaimer |
| **Acne Detection** | Upload image, preview, run prediction, show confidence |
| **Chatbot** | Collect symptoms and skin info via questions |
| **Recommendation** | Show personalized morning + night routine |
| **About** | Dataset info, model info, evaluation (metrics, curves) |

---

## 7. Chatbot (Intent-Based)

- **No LLM:** Predefined intents and responses.
- **Example questions:** Skin type (Oily/Dry/Combination/Sensitive/Normal), how long acne, painful/inflamed, redness, scars, current products, hormonal (jawline, monthly).
- **Goal:** Feed answers into recommendation engine.

---

## 8. Recommendation Engine

- **Input:** CNN prediction + chatbot answers.
- **Output:**
  - **Morning:** Cleanser → Serum/Treatment → Moisturizer → Sunscreen
  - **Night:** Cleanser → Treatment → Moisturizer
  - **Weekly:** Exfoliation, masks, hydration
  - **Warnings:** e.g. avoid retinol + acids together, patch test.
  - **Doctor:** Suggest dermatologist if severe/cystic/painful.
- **Logic examples:**
  - Mild + oily → e.g. salicylic acid cleanser, light moisturizer.
  - Moderate + redness → e.g. niacinamide, gentle exfoliation.
  - Severe/cystic → recommend doctor, avoid strong self-treatment.
  - Sensitive skin → avoid strong benzoyl peroxide, fragrance.

---

## 9. Evaluation Metrics

- Accuracy, Precision, Recall, F1-score  
- Confusion matrix  
- Training vs validation accuracy and loss curves  

**Target:** ~80–95% accuracy depending on data.

---

## 10. Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV / PIL  
- NumPy, Pandas  
- Matplotlib  
- Streamlit  
- Scikit-learn  

---

## 11. Ethical Disclaimer (from doc)

Educational and research use only. Not a substitute for professional medical advice. Users should see a certified dermatologist for severe acne, infections, allergies, or abnormal skin changes.

---

## 12. Your Next Steps (Checklist)

- [ ] Preprocess dataset (224×224, normalize, augmentation pipeline).
- [ ] Train CNN with transfer learning (choose one: EfficientNetB0 / MobileNetV2 / ResNet50).
- [ ] Evaluate: metrics + confusion matrix + accuracy/loss plots.
- [ ] Build Streamlit: Home, Acne Detection, Chatbot, Recommendation, About.
- [ ] Implement intent-based chatbot (skin type, symptoms).
- [ ] Implement rule-based recommendation engine (routines + warnings).
- [ ] Add disclaimer on Home and when recommending doctor.
- [ ] Prepare final report and presentation.

---

*Summary extracted from the graduation project document. Use this as the single reference for scope when building the system.*
