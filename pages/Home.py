# ============= HOME / DASHBOARD PAGE ===============
import streamlit as st

st.set_page_config(page_title="Chat with Medical Reports", page_icon="🩺", layout="wide")

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

# ---------- Greeting (name is kept only for this session) ----------
saved_name = st.session_state.get("saved_name", "")
st.title(f"👋 Welcome{', ' + saved_name if saved_name else ''}!")
st.caption("Your AI health companion: understand your reports, check symptoms, "
           "plan your diet and find doctors, all in one place.")

name = st.text_input("What should we call you? (optional)", value=saved_name)
st.session_state["saved_name"] = name

st.divider()

# ---------- Report status ----------
loaded = st.session_state.get("guidance_for")
if loaded:
    st.success(f"📄 Report loaded: **{loaded}**. Doctor search and report-based diet plans are ready.")
else:
    st.info("📄 No report uploaded yet. Start with **Report Reader** to unlock "
            "doctor search and report-based diet plans.")

# ---------- Feature cards ----------
st.subheader("What would you like to do?")

FEATURES = [
    ("🩺", "Report Reader",
     "Upload a lab report or prescription (PDF, Word, image, Excel...) and chat with it. "
     "Get plain-language explanations of your values.",
     "pages/Report_reader.py"),
    ("🤒", "Symptom Checker",
     "Describe what you are feeling and get common possible causes, self-care tips, "
     "and clear signs of when to see a doctor.",
     "pages/2_Symptom_Checker.py"),
    ("🥗", "Diet Plan",
     "Get a diet chart based on your medical report, or based on your age, height, "
     "weight, routine and food preferences.",
     "pages/3_Diet_Plan.py"),
    ("🏥", "Find Doctors",
     "An AI agent reads your report, picks the right specialty, and shows matching "
     "doctors and hospitals near you.",
     "pages/1_Find_Doctors.py"),
]

cols = st.columns(2)
for i, (icon, title, desc, path) in enumerate(FEATURES):
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(desc)
            st.page_link(path, label=f"Open {title}", icon="➡️")

st.divider()

# ---------- How it works ----------
st.subheader("How it works")
s1, s2, s3, s4 = st.columns(4)
s1.markdown("**1. Upload**\n\nAdd your report in Report Reader.")
s2.markdown("**2. Understand**\n\nAsk questions, get plain-language answers (RAG).")
s3.markdown("**3. Get guidance**\n\nDiet plan, symptom info, specialist suggestions.")
s4.markdown("**4. Find care**\n\nAn AI agent finds doctors near you.")

st.caption("Built with Generative AI (Gemini), RAG (FAISS + embeddings), LangChain and Agentic AI.")
st.warning("⚠️ This app gives general information only. It is not a medical diagnosis or "
           "treatment. In an emergency, call **112** or go to the nearest hospital.")
