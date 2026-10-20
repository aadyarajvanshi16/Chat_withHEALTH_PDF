import streamlit as st
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Find Doctors", page_icon="🏥", layout="wide")
st.title("🏥 Find Doctors & Hospitals Near You")

if "vectorstore" not in st.session_state:
    st.warning("Please upload your report on the main page first.")
    st.stop()

vectorstore = st.session_state["vectorstore"]
google_api_key = st.session_state["google_api_key"]

with st.sidebar:
    tavily_key = st.text_input("Tavily API Key", type="password",
                               help="Free key at tavily.com")

st.info("🩺 **Suggested specialists**\n\n" + st.session_state["guidance"])

location = st.text_input("Your city / area", placeholder="e.g. Rohini, Delhi")
preference = st.text_input("Any preference? (optional)",
                           placeholder="e.g. government hospital, budget-friendly, female doctor")


# ---------------- TOOLS ----------------
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
    results = client.search(query, max_results=5)
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content']}" for r in results["results"])


SYSTEM_PROMPT = """You are a healthcare guidance assistant. You help a user decide
which doctors or hospitals to approach after getting a medical report.

Steps:
1. Use read_report to understand what the report is about.
2. Decide which 1-2 specialties are most relevant (never diagnose).
3. Use search_doctors to find doctors/hospitals of that specialty in the user's city.
4. Give a short, clear answer: why this specialty, then a list of 3-5 options
   (name, what they offer, link).

Rules:
- Never diagnose or suggest medicines.
- Say that search results may be outdated and the user should call to confirm
  availability and fees.
- If the report shows anything that looks urgent, advise visiting an emergency
  department or calling a local emergency number immediately."""


# ---------------- RUN THE AGENT ----------------
if st.button("🔍 Find the right doctors for me", use_container_width=True):
    if not location or not tavily_key:
        st.warning("Please enter your location and Tavily API key.")
    else:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                       google_api_key=google_api_key, temperature=0.2)
        agent = create_agent(model=model, tools=[read_report, search_doctors],
                             system_prompt=SYSTEM_PROMPT)

        task = f"My location is {location}. Preferences: {preference or 'none'}. " \
               f"Read my report and find suitable doctors/hospitals near me."

        with st.spinner("Agent is reading your report and searching..."):
            result = agent.invoke({"messages": [{"role": "user", "content": task}]})

        st.markdown(result["messages"][-1].content)

        with st.expander("🔧 See what the agent did"):
            for m in result["messages"]:
                if m.type == "ai" and m.tool_calls:
                    for c in m.tool_calls:
                        st.write(f"Called **{c['name']}** with {c['args']}")

st.caption("⚠️ This is guidance only, not medical advice. Always confirm with the doctor or hospital directly.")
