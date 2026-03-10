# acne app - streamlit, detection + chatbot + recommendations
import json
import os
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
ROOT = Path(__file__).resolve().parent
env_path = ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(str(env_path))
except ImportError:
    pass
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v:
                    os.environ[k] = v
MODEL_DIR = ROOT / "saved_model"
OUT_DIR = ROOT / "output"
IMG_SIZE = (224, 224)
CLASSES = ["Level1", "Level2", "Level3", "Level4"]
LLM_URL = (os.environ.get("LLM_BASE_URL") or "").strip()
LLM_KEY = (os.environ.get("LLM_API_KEY") or "").strip()
LLM_MODEL = (os.environ.get("LLM_MODEL") or "meta-llama/Meta-Llama-3.1-8B-Instruct").strip()

st.set_page_config(page_title="Acne AI Assistant", page_icon="🧴", layout="wide")
for k, v in [("prediction", None), ("answers", {}), ("chat_messages", []), ("page", "Home"), ("uploaded_image_bytes", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource
def load_model():
    p = MODEL_DIR / "acne_severity_model.keras"
    if not p.exists():
        p = MODEL_DIR / "best_model.keras"
    if not p.exists():
        return None, CLASSES
    m = keras.models.load_model(str(p))
    cls = CLASSES
    if (MODEL_DIR / "class_names.json").exists():
        with open(MODEL_DIR / "class_names.json") as f:
            cls = json.load(f).get("classes", CLASSES)
    return m, cls


def chat_llm(messages):
    if not LLM_KEY or not LLM_URL:
        return None
    try:
        from openai import OpenAI
        c = OpenAI(api_key=LLM_KEY, base_url=LLM_URL)
        r = c.chat.completions.create(model=LLM_MODEL, messages=[{"role": m["role"], "content": m["content"]} for m in messages], max_tokens=512)
        return r.choices[0].message.content if r.choices else None
    except Exception as e:
        st.error(str(e))
        return None


st.sidebar.markdown("<style>section[data-testid='stSidebar'] .stButton > button { border:none; box-shadow:none; outline:none; background:transparent; transition:transform 0.2s; text-align:left; } section[data-testid='stSidebar'] .stButton > button:hover { transform:scale(1.05); }</style>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center; font-size:1.5rem; font-weight:bold; margin:0 0 1rem 0;'>Acne AI Assistant</p>", unsafe_allow_html=True)
for p in ["Home", "Acne Detection", "Chatbot", "Recommendations"]:
    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.page = p
        st.rerun()
st.sidebar.caption("Educational use only.")
page = st.session_state.page

if page == "Home":
    st.title("Acne Detection & Skincare Recommendations")
    st.write("Upload a face image, get severity, chat, then get a routine.")
    st.markdown("---")
    st.subheader("How to use")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Step 1 — Acne Detection**")
        st.write("Upload a face photo and click Predict. You get Level 1–4.")
    with c2:
        st.markdown("**Step 2 — Chatbot**")
        st.write("Ask about skin or acne. The bot uses your result if you did detection.")
    with c3:
        st.markdown("**Step 3 — Recommendations**")
        st.write("See morning/night routine. Chat or do the short survey first.")
    st.markdown("---")
    st.warning("Education only. Not medical advice.")

elif page == "Acne Detection":
    st.title("Acne Detection")
    up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    model, cls = load_model()
    if model is None:
        st.error("No model. Run: py -3.11 train_model.py")
    elif up:
        img = Image.open(up)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, use_container_width=True)
        with col2:
            if st.button("Predict"):
                x = np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32) / 255.0
                x = x[np.newaxis, ...]
                probs = model.predict(x, verbose=0)[0]
                i = int(np.argmax(probs))
                sev, conf = cls[i], float(probs[i])
                st.session_state.prediction = (sev, conf)
                up.seek(0)
                st.session_state.uploaded_image_bytes = up.read()
                st.success(f"**{sev}** ({conf:.0%})")
                for c, p in zip(cls, probs):
                    st.progress(float(p), text=f"{c}: {p:.0%}")
        if st.session_state.prediction:
            st.info("Result saved. Use Chatbot to discuss.")

elif page == "Chatbot":
    st.title("Skin & Chat")
    if not LLM_KEY or not LLM_URL:
        st.warning("Set LLM_BASE_URL and LLM_API_KEY in .env")
    else:
        pred = st.session_state.prediction
        sys_extra = f" User's photo was classified as **{pred[0]}** ({pred[1]:.0%})." if pred else ""
        sys_msg = "You help with skin/acne. Short advice. Not medical advice." + sys_extra
        if not st.session_state.chat_messages:
            first = f"Hi. Your photo: **{pred[0]}**. Ask me anything." if pred else "Hi. Ask about skin or acne."
            st.session_state.chat_messages = [{"role": "assistant", "content": first}]
        if st.session_state.uploaded_image_bytes and pred:
            a, b = st.columns([1, 2])
            with a:
                st.image(st.session_state.uploaded_image_bytes, caption="Your photo", use_container_width=True)
            with b:
                st.markdown(f"**Result:** {pred[0]} ({pred[1]:.0%})")
            st.markdown("---")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Ask..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            msgs = [{"role": "system", "content": sys_msg}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
            with st.chat_message("assistant"):
                with st.spinner("..."):
                    reply = chat_llm(msgs)
                if reply:
                    st.markdown(reply)
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                else:
                    st.error("No reply.")
                    st.session_state.chat_messages.pop()
        if st.button("Clear chat"):
            st.session_state.chat_messages = [{"role": "assistant", "content": "Cleared. Ask anything."}]
            st.rerun()

elif page == "Recommendations":
    st.title("Your Routine")
    pred, ans = st.session_state.prediction, st.session_state.answers
    has_chat = any(m.get("role") == "user" for m in st.session_state.chat_messages)
    has_survey = bool(ans and any(v for v in ans.values() if v))
    if not has_chat and not has_survey:
        st.info("Chat or do the survey below first.")
        st.subheader("Quick survey")
        with st.form("survey"):
            skin = st.selectbox("Skin type", ["", "Oily", "Dry", "Combination", "Sensitive", "Normal"])
            duration = st.selectbox("How long with acne?", ["", "Weeks", "Months", "Years", "On and off"])
            painful = st.radio("Painful?", ["", "Yes", "No"], horizontal=True)
            if st.form_submit_button("Submit"):
                st.session_state.answers = {"skin_type": skin or None, "duration": duration or None, "painful": painful or None}
                if any(st.session_state.answers.values()):
                    st.success("Done."); st.rerun()
        st.caption("Then your routine will show here.")
    else:
        sev = pred[0] if pred else None
        lvl = int(sev.replace("Level", "")) if sev and "Level" in sev else None
        if sev:
            st.subheader(f"Severity: {sev}")
        if lvl and lvl >= 4:
            st.error("See a dermatologist.")
        st.subheader("Morning")
        st.write("Cleanser, serum (niacinamide if redness), moisturizer, sunscreen SPF 30+.")
        st.subheader("Night")
        st.write("Cleanser, treatment (benzoyl peroxide or salicylic), moisturizer.")
        st.subheader("Weekly")
        st.write("Exfoliate 1–2x. Avoid if very inflamed.")
        st.warning("Don't mix retinol and acids. Patch test.")

else:
    st.session_state.page = "Home"
    st.rerun()
