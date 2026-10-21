# ============= STEP 1: LOAD MODULES ===============
from typing import List, Literal, Optional

import streamlit as st
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# Keep this SAME as MODEL_NAME in app.py
MODEL_NAME = "gemini-3.5-flash"

# ============= STEP 2: PAGE SETUP ===============
st.set_page_config(page_title="Symptom Checker", page_icon="🤒", layout="wide")
st.title("🤒 Symptom Checker")
st.caption("Describe what you are feeling and get simple, general information about "
           "common possible causes, self-care tips, and when to see a doctor.")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #001f3f; }
    [data-testid="stSidebar"] * { color: #ffffff; }
    [data-testid="stSidebar"] input { background-color: #0a2d5c; color: #ffffff; }
    [data-testid="stSidebar"] hr { border-color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("API Key")
    google_api_key = st.text_input(
        "Google API Key", type="password",
        value=st.session_state.get("google_api_key", ""),
        help="Free key at aistudio.google.com/app/apikey")
    st.divider()
    st.markdown("⚠️ This tool gives general information only. It cannot diagnose "
                "you and is **not** a replacement for a doctor.")

# ============= STEP 3: SAFETY CHECK (runs BEFORE the AI) ===============
EMERGENCY_WORDS = [
    "chest pain", "chest tightness", "can't breathe", "cannot breathe",
    "difficulty breathing", "trouble breathing", "shortness of breath",
    "unconscious", "fainted", "passed out", "seizure", "stroke",
    "face drooping", "slurred speech", "paralysis", "severe bleeding",
    "vomiting blood", "coughing blood", "swelling of face", "swelling of throat",
    "overdose", "poison",
]
SELF_HARM_WORDS = ["suicid", "kill myself", "want to die", "end my life", "self harm", "self-harm"]

# ============= STEP 4: OUTPUT STRUCTURE ===============
class PossibleCause(BaseModel):
    name: str
    why_it_fits: str
    likelihood: Literal["Common", "Possible", "Less likely"]


class SymptomAnalysis(BaseModel):
    urgency: Literal["Emergency", "See a doctor soon", "Self-care is usually fine"]
    urgency_reason: str
    possible_causes: List[PossibleCause]
    self_care: List[str]
    warning_signs: List[str]
    specialist: Optional[str] = None


SYSTEM_RULES = """You are a careful health-information assistant. You are NOT a doctor.
A user describes symptoms. Give general educational information.

Rules:
1. List 2-4 COMMON possible causes only (for example cold, dehydration, acidity,
   tension headache). Never state a diagnosis as certain. Use simple language.
2. Choose urgency conservatively. If unsure between two levels, pick the higher one.
   Anything that could be serious means "See a doctor soon" or "Emergency".
3. Self-care: only general safe tips (rest, fluids, light food, sleep, avoiding triggers).
   Do NOT name prescription medicines and do NOT give any medicine dosage.
   You may say a pharmacist or doctor can advise on over-the-counter options.
4. warning_signs: 3-5 specific signs that mean the person should see a doctor promptly.
5. specialist: the type of doctor to consult if symptoms persist (or null).
6. Take age, sex, duration and severity into account. Be calm and reassuring."""

# ============= STEP 5: USER INPUT ===============
symptoms = st.text_area("📝 Describe your symptoms", height=120,
                        placeholder="e.g. headache since morning, mild fever, body ache, sore throat")

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=1, max_value=110, value=25)
with c2:
    sex = st.selectbox("Sex", ["Female", "Male", "Other / prefer not to say"])
with c3:
    duration = st.selectbox("Since when?", ["Less than 1 day", "1-3 days", "4-7 days",
                                            "1-2 weeks", "More than 2 weeks"])

severity = st.slider("How severe is it? (1 = very mild, 10 = unbearable)", 1, 10, 3)
other_info = st.text_input("Anything else we should know? (optional)",
                           placeholder="e.g. pregnant, diabetic, allergic to something, taking any medicine")

use_report = False
if "vectorstore" in st.session_state:
    use_report = st.checkbox("Also consider my uploaded medical report", value=False)

# ============= STEP 6: RUN =============
if st.button("🔍 Check my symptoms", use_container_width=True):
    text = symptoms.lower()

    if not symptoms.strip():
        st.warning("Please describe your symptoms first.")
    elif any(w in text for w in SELF_HARM_WORDS):
        st.error("💙 You are not alone, and help is available right now. Please talk to someone "
                 "you trust, or call **Tele-MANAS at 14416** (India, free, 24x7). If you are in "
                 "immediate danger, call **112**.")
    elif any(w in text for w in EMERGENCY_WORDS):
        st.error("🚨 **These symptoms can be serious.** Please do not wait. Go to the nearest "
                 "emergency department or call **112** (India emergency number) right away.")
    elif not google_api_key:
        st.warning("Please enter your Google API key in the sidebar.")
    else:
        try:
            report_context = ""
            if use_report:
                docs = st.session_state["vectorstore"].as_retriever(
                    search_kwargs={"k": 4}).invoke(symptoms)
                report_context = "\n\nRelevant parts of the user's medical report:\n" + \
                                 "\n\n".join(d.page_content for d in docs)

            prompt = (
                f"{SYSTEM_RULES}\n\n"
                f"Age: {age}\nSex: {sex}\nDuration: {duration}\nSeverity: {severity}/10\n"
                f"Other info: {other_info or 'none'}\n"
                f"Symptoms: {symptoms}{report_context}"
            )

            model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=google_api_key,
                                           temperature=0.2)
            with st.spinner("Analysing your symptoms..."):
                result = model.with_structured_output(SymptomAnalysis).invoke(prompt)

            # ---------- SHOW RESULT ----------
            if result.urgency == "Emergency":
                st.error(f"🚨 **{result.urgency}**: {result.urgency_reason}  \n"
                         "Please go to the nearest emergency department or call **112**.")
            elif result.urgency == "See a doctor soon":
                st.warning(f"🩺 **{result.urgency}**: {result.urgency_reason}")
            else:
                st.success(f"🌿 **{result.urgency}**: {result.urgency_reason}")

            st.markdown("### 🔎 Possible common causes")
            for c in result.possible_causes:
                with st.container(border=True):
                    st.subheader(c.name)
                    st.caption(f"Likelihood: {c.likelihood}")
                    st.write(c.why_it_fits)

            left, right = st.columns(2)
            with left:
                st.markdown("### 🌿 Self-care tips")
                for tip in result.self_care:
                    st.write(f"• {tip}")
            with right:
                st.markdown("### ⚠️ See a doctor if you notice")
                for sign in result.warning_signs:
                    st.write(f"• {sign}")

            if result.specialist:
                st.info(f"👨‍⚕️ If symptoms continue, consider consulting a **{result.specialist}**.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.caption("⚠️ General information only, not a diagnosis or medical advice. If you feel unwell or "
           "unsure, please consult a qualified doctor.")
