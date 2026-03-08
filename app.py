"""
Acne AI Assistant – Streamlit app.
Pages: Home, Acne Detection, Chatbot (LLM), Recommendations, About.
Chatbot uses Nebius LLM API (OpenAI-compatible). Config from .env.
"""
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATA_ROOT = Path(__file__).resolve().parent
MODEL_SAVE_DIR = DATA_ROOT / "saved_model"
OUTPUT_DIR = DATA_ROOT / "output"
IMG_SIZE = (224, 224)
CLASSES = ["Level1", "Level2", "Level3", "Level4"]

# Nebius LLM API (from .env or environment)
LLM_BASE_URL = (os.environ.get("LLM_BASE_URL") or "").strip()
LLM_API_KEY = (os.environ.get("LLM_API_KEY") or "").strip()
LLM_MODEL = (os.environ.get("LLM_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct").strip()

st.set_page_config(page_title="Acne AI Assistant", page_icon="🧴", layout="wide")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None


@st.cache_resource
def load_model():
    """Load trained acne severity model and class names."""
    path = MODEL_SAVE_DIR / "acne_severity_model.keras"
    if not path.exists():
        path = MODEL_SAVE_DIR / "best_model.keras"
    if not path.exists():
        return None, CLASSES
    model = keras.models.load_model(str(path))
    if (MODEL_SAVE_DIR / "class_names.json").exists():
        with open(MODEL_SAVE_DIR / "class_names.json") as f:
            data = json.load(f)
            classes = data.get("classes", CLASSES)
    else:
        classes = CLASSES
    return model, classes


def get_llm_client():
    """Return OpenAI-compatible client for Nebius API, or None if not configured."""
    if not LLM_API_KEY or not LLM_BASE_URL:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    except Exception:
        return None


def llm_chat(messages, model=LLM_MODEL):
    """Send messages to LLM and return assistant reply. Returns None on failure."""
    client = get_llm_client()
    if not client:
        return None
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            max_tokens=512,
        )
        if r.choices and len(r.choices) > 0:
            return r.choices[0].message.content
    except Exception as e:
        st.error(f"LLM error: {e}")
    return None


# ---------- Sidebar ----------
st.sidebar.markdown(
    "<style>"
    "section[data-testid='stSidebar'] .stButton > button { "
    "  border: none; box-shadow: none; outline: none; background: transparent; "
    "  transition: transform 0.2s ease; text-align: left; "
    "}"
    "section[data-testid='stSidebar'] .stButton > button:hover { transform: scale(1.05); }"
    "</style>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<p style='text-align: center; font-size: 1.5rem; font-weight: bold; margin: 0 0 1rem 0;'>Acne AI Assistant</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("")
PAGES = ["Home", "Acne Detection", "Chatbot", "Recommendations", "About"]
for p in PAGES:
    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()
st.sidebar.markdown("")
st.sidebar.caption("Educational use only. Not medical advice.")
page = st.session_state.page

# ---------- Home ----------
if page == "Home":
    st.title("Acne Detection & Skincare Recommendations")
    st.markdown("Upload a face image to get an acne severity result, chat about your skin, and receive a personalized skincare routine.")
    st.markdown("---")
    st.subheader("How to use")
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("**Step 1 — Acne Detection**")
        st.markdown("Go to **Acne Detection** in the sidebar. Upload a clear photo of your face (jpg or png). Click **Predict** to get your severity level (Level 1 = mild, Level 4 = severe). The result is saved for your recommendations.")
    with step2:
        st.markdown("**Step 2 — Chatbot**")
        st.markdown("Open **Chatbot** and ask anything about skin or acne. The assistant can answer questions and give general tips. Your answers help tailor the routine in the next step.")
    with step3:
        st.markdown("**Step 3 — Recommendations**")
        st.markdown("Go to **Recommendations** to see your morning and night skincare routine. Suggestions are based on your detection result and are meant as a starting point—adjust with a dermatologist if needed.")
    st.markdown("---")
    st.warning("For education only. Not medical advice. See a dermatologist for serious skin issues.")

# ---------- Acne Detection ----------
elif page == "Acne Detection":
    st.title("Acne Detection")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    model, class_list = load_model()
    if model is None:
        st.error("No model found. Run: py -3.11 train_model.py")
    elif uploaded:
        img = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, use_container_width=True)
        with col2:
            if st.button("Predict"):
                x = img.convert("RGB").resize(IMG_SIZE)
                x = np.array(x, dtype=np.float32) / 255.0
                x = x[np.newaxis, ...]
                probs = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(probs))
                severity = class_list[idx]
                conf = float(probs[idx])
                st.session_state.prediction = (severity, conf)
                uploaded.seek(0)
                st.session_state.uploaded_image_bytes = uploaded.read()
                st.success(f"**{severity}** ({conf:.0%})")
                for c, p in zip(class_list, probs):
                    st.progress(float(p), text=f"{c}: {p:.0%}")
        if st.session_state.prediction:
            st.info(f"Saved severity: {st.session_state.prediction[0]}. You can talk about it on the Chatbot page.")

# ---------- Chatbot (LLM) ----------
elif page == "Chatbot":
    st.title("Skin & Chat")
    st.caption("Ask about skin, acne, or skincare. Replies use the LLM (Nebius).")

    if not LLM_API_KEY or not LLM_BASE_URL:
        st.warning(
            "LLM not configured. Set **LLM_BASE_URL**, **LLM_API_KEY** (and optionally **LLM_MODEL**) in a `.env` file or environment, then restart the app."
        )
        st.code(
            "LLM_BASE_URL=https://api.studio.nebius.com/v1/\n"
            "LLM_API_KEY=your_key_here\n"
            "LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct",
            language="bash",
        )
    else:
        pred = st.session_state.prediction
        pred_text = ""
        if pred:
            severity, conf = pred
            pred_text = f" The user has uploaded a photo that was classified as **{severity}** acne severity (confidence: {conf:.0%}). Use this when giving personalized advice and when they ask about their image or results."
        system_content = (
            "You are a helpful assistant for skin and acne questions. "
            "Give short, practical advice. Remind users this is not medical advice and to see a dermatologist when needed."
            + pred_text
        )
        if not st.session_state.chat_messages:
            if pred:
                sev = pred[0]
                st.session_state.chat_messages = [{"role": "assistant", "content": f"Hi. I see your photo was classified as **{sev}**. Ask me about your skin or acne and I'll keep that in mind."}]
            else:
                st.session_state.chat_messages = [{"role": "assistant", "content": "Hi. Ask me about skin, acne, or skincare. If you upload a photo on Acne Detection, I can give advice based on your result."}]

        if st.session_state.uploaded_image_bytes and pred:
            st.caption("Your uploaded photo and result are shared with the assistant below.")
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(st.session_state.uploaded_image_bytes, caption="Your photo", use_container_width=True)
            with col_info:
                st.markdown(f"**Detection result:** {pred[0]} ({pred[1]:.0%} confidence)")
            st.markdown("---")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about skin or acne..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build API messages: system + history + new user message
            api_messages = [{"role": "system", "content": system_content}]
            for m in st.session_state.chat_messages:
                api_messages.append({"role": m["role"], "content": m["content"]})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = llm_chat(api_messages)
                if reply:
                    st.markdown(reply)
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                else:
                    st.error("Could not get a reply. Check API key and network.")
                    st.session_state.chat_messages.pop()  # remove the user message so they can retry

        if st.button("Clear chat"):
            if pred:
                st.session_state.chat_messages = [
                    {"role": "assistant", "content": f"Chat cleared. I still have your **{pred[0]}** result in mind. Ask me anything."}
                ]
            else:
                st.session_state.chat_messages = [
                    {"role": "assistant", "content": "Chat cleared. Ask me anything about skin or acne."}
                ]
            st.rerun()

# ---------- Recommendations ----------
elif page == "Recommendations":
    st.title("Your Routine")
    pred = st.session_state.prediction
    ans = st.session_state.answers
    has_chatted = any(m.get("role") == "user" for m in st.session_state.chat_messages)
    survey_done = bool(ans and any(v for v in ans.values() if v))

    if not has_chatted and not survey_done:
        st.info("To see your recommendations, **chat with the assistant** on the Chatbot page, or **complete the short survey** below.")
        st.markdown("---")
        st.subheader("Quick survey (optional)")
        with st.form("survey_form"):
            skin = st.selectbox("Skin type", ["", "Oily", "Dry", "Combination", "Sensitive", "Normal"], key="survey_skin")
            duration = st.selectbox("How long have you had acne?", ["", "Weeks", "Months", "Years", "On and off"], key="survey_duration")
            painful = st.radio("Painful or inflamed?", ["", "Yes", "No"], horizontal=True, key="survey_painful")
            submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.answers = {
                "skin_type": skin or None,
                "duration": duration or None,
                "painful": painful or None,
            }
            if any(st.session_state.answers.values()):
                st.success("Thanks. Your recommendations are below.")
                st.rerun()
            else:
                st.warning("Please answer at least one question.")
        st.markdown("---")
        st.caption("After you chat or submit the survey, your routine will appear here.")
    else:
        ans = st.session_state.answers
        severity = pred[0] if pred else None
        level = int(severity.replace("Level", "")) if severity and "Level" in severity else None

        if severity:
            st.subheader(f"Severity: {severity}")
        if level and level >= 4:
            st.error("Please see a dermatologist for severe acne.")
        st.subheader("Morning")
        st.write(
            "1. Gentle cleanser (salicylic acid 2% if oily). 2. Serum (niacinamide if redness). "
            "3. Moisturizer. 4. Sunscreen SPF 30+."
        )
        st.subheader("Night")
        st.write("1. Cleanser. 2. Treatment (e.g. benzoyl peroxide or salicylic, low %). 3. Moisturizer.")
        st.subheader("Weekly")
        st.write("Exfoliate 1–2x (gentle). Avoid if very inflamed.")
        st.warning("Don't mix retinol and acids. Patch test. See a doctor if painful or cystic.")

# ---------- About ----------
else:
    st.title("About")
    st.write("Dataset: Roboflow acne severity (Level1–Level4). Model: MobileNetV2 + head, 224×224.")
    st.write("Chatbot: LLM via Nebius API (OpenAI-compatible).")
    if (OUTPUT_DIR / "classification_report.txt").exists():
        st.code((OUTPUT_DIR / "classification_report.txt").read_text())
    if (OUTPUT_DIR / "test_metrics.txt").exists():
        st.code((OUTPUT_DIR / "test_metrics.txt").read_text())
    st.caption("Educational use only.")
