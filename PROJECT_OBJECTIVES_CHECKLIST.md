# Project Objectives — Status Checklist

From the graduation project document: **AI-Based Acne Detection & Personalized Skincare Routine Recommendation System**

---

## 1. Build a CNN model to classify acne severity or acne type using facial images

| Status | Notes |
|--------|--------|
| ✅ Done | **EfficientNetB0** transfer-learning model in `train_model.py`. Classifies **acne severity** into Level1, Level2, Level3, Level4 from facial images (224×224). |

**Deliverable:** `saved_model/acne_severity_model.keras` (after you run training).

---

## 2. Use transfer learning to achieve high accuracy with limited training time

| Status | Notes |
|--------|--------|
| ✅ Done | EfficientNetB0 pre-trained on ImageNet; top layers fine-tuned on your dataset. Early stopping and ReduceLROnPlateau keep training time reasonable. |

---

## 3. Develop a Streamlit web application for uploading images and displaying results

| Status | Notes |
|--------|--------|
| ⏳ Next | Streamlit app not built yet. Plan: Home, Acne Detection (upload → prediction + confidence), Chatbot, Recommendation, About. |

**Next step:** Build the Streamlit app that loads the saved model and shows upload + results.

---

## 4. Create a chatbot-style questionnaire to collect symptoms and skin characteristics

| Status | Notes |
|--------|--------|
| ⏳ Pending | Intent-based chatbot (no LLM): skin type, duration of acne, pain/inflammation, redness, scars, current products, hormonal symptoms. To be added in Streamlit. |

---

## 5. Generate personalized skincare routines (morning and night routines)

| Status | Notes |
|--------|--------|
| ⏳ Pending | Rule-based recommendation engine: morning routine (cleanser → serum → moisturizer → sunscreen), night routine (cleanser → treatment → moisturizer), plus weekly tips. Uses CNN prediction + chatbot answers. To be added in Streamlit. |

---

## 6. Provide safe recommendations, warnings, and dermatologist referral suggestions when necessary

| Status | Notes |
|--------|--------|
| ⏳ Pending | In recommendation module: warnings (e.g. avoid mixing retinol + acids, patch test), and suggest dermatologist visit for severe/cystic/painful acne. To be added in Streamlit. |

---

## 7. Evaluate the model using standard performance metrics and visualize results

| Status | Notes |
|--------|--------|
| ✅ Done (in script) | `train_model.py` computes and saves: **accuracy**, **precision**, **recall**, **F1** (classification report), **confusion matrix** (plot), and **training vs validation accuracy/loss** curves. Outputs in `output/`. |

**Deliverables (after training):** `output/classification_report.txt`, `output/confusion_matrix.png`, `output/training_history.png`, and optional `output/test_metrics.txt`.

---

## Summary

| Objective | Status |
|-----------|--------|
| 1. CNN for acne severity/type | ✅ Model built |
| 2. Transfer learning | ✅ Implemented |
| 3. Streamlit app (upload + results) | ⏳ To do |
| 4. Chatbot questionnaire | ⏳ To do |
| 5. Morning/night routines | ⏳ To do |
| 6. Warnings + dermatologist referral | ⏳ To do |
| 7. Metrics + visualizations | ✅ In training script |

**Immediate next step:** Run training (see `HOW_TO_TRAIN.md`), then build the Streamlit app (objectives 3–6).
