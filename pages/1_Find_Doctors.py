# ============= STEP 1: LOAD MODULES ===============
from typing import List, Optional
from urllib.parse import quote_plus

import streamlit as st
from pydantic import BaseModel
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Keep this SAME as MODEL_NAME in app.py
MODEL_NAME = "gemini-3.5-flash"

# ============= STEP 2: PAGE SETUP ===============
st.set_page_config(page_title="Find Doctors", page_icon="🏥", layout="wide")
st.title("🏥 Find Doctors & Hospitals Near You")
st.caption("An AI agent reads your report, decides the right specialty, and finds matching doctors.")

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

# Page needs a report uploaded on the main page first
if "vectorstore" not in st.session_state:
    st.warning("Please upload your report on the main page first.")
    st.stop()

vectorstore = st.session_state["vectorstore"]
google_api_key = st.session_state["google_api_key"]

with st.sidebar:
    st.header("API Keys")
    tavily_key = st.text_input("Tavily API Key", type="password",
                               help="Free key at tavily.com")
    st.divider()
    st.markdown("⚠️ Profiles are built from web search results and may be "
                "incomplete or outdated. Always confirm with the doctor/hospital.")

st.info("🩺 **Suggested specialists (from your report)**\n\n" + st.session_state["guidance"])

# ============= STEP 3: USER PREFERENCES ===============
col1, col2 = st.columns(2)
with col1:
    location = st.text_input("📍 Your city / area", placeholder="e.g. Rohini, Delhi")
    looking_for = st.radio("Looking for", ["Doctors", "Hospitals / clinics", "Both"], horizontal=True)
    specialty = st.text_input("Specialty (optional)",
                              placeholder="Leave blank to let the AI decide from your report")
with col2:
    prefs = st.multiselect("Preferences", [
        "Government", "Private", "Budget-friendly", "Female doctor",
        "Open on weekends", "Online consultation", "Multi-speciality hospital", "Highly rated"])
    other = st.text_input("Anything else? (optional)", placeholder="e.g. Hindi-speaking, near metro")
    num_results = st.slider("How many results?", 3, 8, 5)

# ============= STEP 4: DATA MODEL FOR PROFILES ===============
class DoctorProfile(BaseModel):
    name: str
    specialty: Optional[str] = None
    hospital_or_clinic: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    rating: Optional[str] = None
    experience: Optional[str] = None
    languages: Optional[str] = None
    fees: Optional[str] = None
    why_suitable: str
    source_url: str


class DoctorList(BaseModel):
    doctors: List[DoctorProfile]


def to_text(content):
    """Gemini sometimes returns a list of parts instead of a plain string."""
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


# ============= STEP 5: AGENT TOOLS ===============
@tool
def read_report(question: str) -> str:
    """Reads the user's uploaded medical report and returns the parts relevant to the
    question. Use this first to understand which type of specialist is needed."""
    docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)
    return "\n\n".join(d.page_content for d in docs)


@tool
def search_doctors(query: str) -> str:
    """Searches the web for doctors, clinics or hospitals. Always include the
    specialty and the city in the query."""
    client = TavilyClient(api_key=tavily_key)
    results = client.search(query, max_results=6)
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content']}" for r in results["results"])


SYSTEM_PROMPT = """You are a healthcare guidance assistant. You help a user decide
which doctors or hospitals to approach after getting a medical report.

Steps:
1. Use read_report to understand what the report is about.
2. Decide the 1-2 most relevant specialties (never diagnose).
3. Use search_doctors at least twice with different queries (for example one for
   specialist doctors and one for hospitals/clinics), always including the city and
   the user's preferences in the query.
4. Finish with a SHORT explanation (3-4 lines) of why this specialty is relevant,
   in simple language. Do not list the doctors, they will be shown separately.

Rules:
- Never diagnose or suggest medicines.
- If the report looks urgent, advise going to an emergency department immediately."""

EXTRACT_PROMPT = """From the web search results below, extract up to {n} doctors or
hospitals/clinics suitable for this user.

User location: {location}
Specialty needed: {specialty}
Looking for: {looking_for}
Preferences: {prefs}

STRICT RULES:
- Use ONLY information that is explicitly present in the search results.
- If a detail (phone, rating, fees, address, experience, languages) is not in the
  results, leave it empty. NEVER invent or guess any detail.
- source_url must be a URL that appears in the results.
- why_suitable: one short line linking the result to the specialty and the user's preferences.
- Skip results that are articles, directories without a specific doctor/hospital, or irrelevant.

SEARCH RESULTS:
{results}"""

# ============= STEP 6: RUN THE AGENT ===============
if st.button("🔍 Find the right doctors for me", use_container_width=True):
    if not location or not tavily_key:
        st.warning("Please enter your location and Tavily API key.")
    else:
        try:
            model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=google_api_key,
                                           temperature=0.2)
            agent = create_agent(model=model, tools=[read_report, search_doctors],
                                 system_prompt=SYSTEM_PROMPT)

            pref_text = ", ".join(prefs + ([other] if other else [])) or "none"
            task = (f"My location is {location}. I am looking for: {looking_for}. "
                    f"Specialty: {specialty or 'decide from my report'}. "
                    f"Preferences: {pref_text}. Read my report and find suitable options.")

            with st.spinner("Agent is reading your report and searching..."):
                result = agent.invoke({"messages": [{"role": "user", "content": task}]})

            # Real search results fetched by the agent (grounding for the profiles)
            raw = "\n\n=====\n\n".join(
                to_text(m.content) for m in result["messages"]
                if m.type == "tool" and m.name == "search_doctors"
            )

            doctors = []
            if raw:
                with st.spinner("Building doctor profiles..."):
                    extractor = model.with_structured_output(DoctorList)
                    doctors = extractor.invoke(EXTRACT_PROMPT.format(
                        n=num_results, location=location,
                        specialty=specialty or "as decided from the report",
                        looking_for=looking_for, prefs=pref_text, results=raw,
                    )).doctors

            steps = [f"Called **{c['name']}** with {c['args']}"
                     for m in result["messages"] if m.type == "ai" and m.tool_calls
                     for c in m.tool_calls]

            # Save so results survive Streamlit reruns
            st.session_state["doctor_results"] = {
                "explanation": to_text(result["messages"][-1].content),
                "doctors": doctors, "steps": steps, "location": location,
            }
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ============= STEP 7: SHOW PROFILES ===============
saved = st.session_state.get("doctor_results")
if saved:
    st.markdown("### 🧠 Why this specialty?")
    st.markdown(saved["explanation"])

    st.markdown("### 👨‍⚕️ Doctors & hospitals for you")
    if not saved["doctors"]:
        st.warning("No suitable profiles found. Try a broader area or fewer preferences.")

    for d in saved["doctors"]:
        with st.container(border=True):
            left, right = st.columns([1, 6])
            with left:
                st.markdown("# 🧑‍⚕️")
            with right:
                st.subheader(d.name)
                sub = " • ".join(x for x in [d.specialty, d.hospital_or_clinic] if x)
                if sub:
                    st.caption(sub)
                if d.address:
                    st.write(f"📍 {d.address}")
                details = []
                if d.phone:
                    details.append(f"📞 {d.phone}")
                if d.rating:
                    details.append(f"⭐ {d.rating}")
                if d.experience:
                    details.append(f"🎓 {d.experience}")
                if d.languages:
                    details.append(f"🗣️ {d.languages}")
                if d.fees:
                    details.append(f"💰 {d.fees}")
                if details:
                    st.write("   |   ".join(details))
                st.write(f"✅ **Why suitable:** {d.why_suitable}")

                b1, b2, _ = st.columns([1, 1, 3])
                with b1:
                    st.link_button("🔗 View details", d.source_url)
                with b2:
                    maps_q = quote_plus(f"{d.name} {d.address or saved['location']}")
                    st.link_button("🗺️ Open in Maps",
                                   f"https://www.google.com/maps/search/?api=1&query={maps_q}")

    with st.expander("🔧 See what the agent did"):
        for s in saved["steps"]:
            st.write(s)

st.caption("⚠️ This is guidance only, not medical advice. Always confirm with the doctor or hospital directly.")
