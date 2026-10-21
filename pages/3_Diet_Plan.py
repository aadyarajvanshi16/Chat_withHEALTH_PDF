# ============= STEP 1: LOAD MODULES ===============
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# Keep this SAME as MODEL_NAME in app.py
MODEL_NAME = "gemini-3.5-flash"

# ============= STEP 2: PAGE SETUP ===============
st.set_page_config(page_title="Diet Plan", page_icon="🥗", layout="wide")
st.title("🥗 Personalised Diet Plan")
st.caption("Get a simple diet chart based on your medical report, or on your personal details.")

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
    st.markdown("⚠️ General healthy-eating guidance only. Not a replacement for a "
                "doctor or a registered dietitian.")

# ============= STEP 3: OUTPUT STRUCTURE ===============
class Meal(BaseModel):
    meal: str
    items: str
    approx_calories: Optional[int] = None


class DayPlan(BaseModel):
    day: str
    meals: List[Meal]


class DietPlan(BaseModel):
    summary: str
    days: List[DayPlan]
    foods_to_include: List[str]
    foods_to_limit: List[str]
    hydration_and_lifestyle: List[str]
    when_to_see_dietitian: str


RULES = """You are a careful nutrition-guidance assistant. You are NOT a doctor or a
registered dietitian. Create a practical, easy-to-follow diet chart.

Rules:
1. Use simple, home-style foods. Default to Indian-friendly meals unless the user
   asks for another cuisine.
2. Respect the diet type and allergies/dislikes STRICTLY. Never include a food the
   user said they avoid or are allergic to.
3. No crash diets, fasting, detox plans, or extreme restriction. No supplements,
   medicines, or doses.
4. Keep each day balanced (protein, fibre, vegetables, whole grains, healthy fats).
5. If a medical condition, pregnancy, or serious abnormal values are involved, keep the
   advice general and say a doctor or dietitian should confirm the plan.
6. Give approximate calories per meal as whole numbers.
7. Use day names like "Day 1", "Day 2". Use the requested number of meals per day."""

# ============= STEP 4: COMMON PREFERENCES ===============
mode = st.radio(
    "How should we build your diet chart?",
    ["📄 Based on my medical report", "🧍 Based on my personal details"],
    horizontal=True,
)

st.markdown("#### Food preferences")
p1, p2, p3 = st.columns(3)
with p1:
    diet_type = st.selectbox("Diet type", ["Vegetarian", "Non-vegetarian", "Eggetarian",
                                           "Vegan", "Jain"])
with p2:
    meals_per_day = st.selectbox("Meals per day", [3, 4, 5], index=1)
with p3:
    plan_days = st.selectbox("Plan length", [1, 3, 7], index=1,
                             format_func=lambda d: f"{d} day" + ("s" if d > 1 else ""))
cuisine = st.text_input("Cuisine / region", value="North Indian")
avoid = st.text_input("Allergies or foods to avoid (optional)",
                      placeholder="e.g. peanuts, mushrooms, lactose intolerant")

# ============= STEP 5: MODE-SPECIFIC INPUTS ===============
meta = {}
extra_text = ""
can_run = True

if mode.startswith("📄"):
    if "vectorstore" not in st.session_state:
        st.warning("Please upload your report in **Report Reader** first, or switch to "
                   "'Based on my personal details'.")
        can_run = False
    else:
        st.info("The plan will focus on the findings in your uploaded report "
                f"(**{st.session_state.get('guidance_for', 'report')}**).")
else:
    st.markdown("#### Your details")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        age = st.number_input("Age", min_value=12, max_value=100, value=25)
    with d2:
        sex = st.selectbox("Gender", ["Female", "Male", "Other / prefer not to say"])
    with d3:
        height = st.number_input("Height (cm)", min_value=100, max_value=230, value=165)
    with d4:
        weight = st.number_input("Weight (kg)", min_value=25.0, max_value=250.0,
                                 value=60.0, step=0.5)

    a1, a2 = st.columns(2)
    with a1:
        activity = st.selectbox("Activity level", [
            "Sedentary (desk job, little exercise)",
            "Lightly active (1-3 days/week)",
            "Moderately active (3-5 days/week)",
            "Very active (6-7 days/week)"])
    with a2:
        goal = st.selectbox("Goal", ["Maintain weight", "Lose weight gradually",
                                     "Gain weight gradually"])

    routine = st.text_area("Your daily routine (optional)", height=80,
                           placeholder="e.g. wake up 7am, office 10-7, gym in the evening, "
                                       "late dinner, travel a lot")
    conditions = st.text_input("Medical conditions, pregnancy or medicines (optional)",
                               placeholder="e.g. diabetes, PCOS, thyroid, pregnant")

    # ---- Calculations done in Python (not by the AI) ----
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"

    sex_const = {"Male": 5, "Female": -161}.get(sex, -78)
    floor = {"Male": 1500, "Female": 1200}.get(sex, 1350)
    bmr = 10 * weight + 6.25 * height - 5 * age + sex_const      # Mifflin-St Jeor
    factors = {"Sedentary": 1.2, "Lightly": 1.375, "Moderately": 1.55, "Very": 1.725}
    tdee = bmr * factors[activity.split(" ")[0]]

    # Safety limits on the goal
    note = ""
    if age < 18 and goal != "Maintain weight":
        goal, note = "Maintain weight", "For under 18, only a healthy maintenance plan is given."
    if bmi < 18.5 and goal == "Lose weight gradually":
        goal, note = "Maintain weight", "Your BMI is in the underweight range, so weight loss is not suggested."
    adjust = {"Maintain weight": 0, "Lose weight gradually": -400, "Gain weight gradually": 300}[goal]
    target = max(tdee + adjust, floor)

    if note:
        st.warning(note)
    m1, m2, m3 = st.columns(3)
    m1.metric("BMI", f"{bmi:.1f}", bmi_cat)
    m2.metric("Maintenance calories", f"{round(tdee)} kcal")
    m3.metric("Your daily target", f"{round(target)} kcal")

    meta = {"BMI": f"{bmi:.1f} ({bmi_cat})", "Maintenance": f"{round(tdee)} kcal",
            "Target": f"{round(target)} kcal"}
    extra_text = (
        f"User profile: age {age}, gender {sex}, height {height} cm, weight {weight} kg, "
        f"BMI {bmi:.1f} ({bmi_cat}), activity: {activity}, goal: {goal}.\n"
        f"Daily calorie target (already calculated, follow it, each day within about 10%): "
        f"{round(target)} kcal.\n"
        f"Daily routine: {routine or 'not given'}.\n"
        f"Medical conditions / pregnancy / medicines: {conditions or 'none mentioned'}."
    )

# ============= STEP 6: GENERATE ===============
def plan_to_text(plan):
    lines = [plan.summary, ""]
    for d in plan.days:
        lines.append(d.day.upper())
        for m in d.meals:
            kcal = f" (~{m.approx_calories} kcal)" if m.approx_calories else ""
            lines.append(f"  {m.meal}{kcal}: {m.items}")
        lines.append("")
    lines += ["Foods to include: " + ", ".join(plan.foods_to_include),
              "Foods to limit: " + ", ".join(plan.foods_to_limit), "",
              "Tips: " + "; ".join(plan.hydration_and_lifestyle), "",
              plan.when_to_see_dietitian,
              "", "General guidance only, not medical advice."]
    return "\n".join(lines)


if st.button("✨ Create my diet chart", use_container_width=True, disabled=not can_run):
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar.")
    else:
        try:
            prefs = (f"Diet type: {diet_type}\nCuisine: {cuisine}\n"
                     f"Allergies / foods to avoid: {avoid or 'none'}\n"
                     f"Meals per day: {meals_per_day}\nNumber of days: {plan_days}")

            if mode.startswith("📄"):
                docs = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6}).invoke(
                    "abnormal values high low deficiency cholesterol sugar hemoglobin "
                    "vitamin thyroid kidney liver uric acid")
                report_text = "\n\n".join(d.page_content for d in docs)
                task = ("TASK: Create a diet plan based ONLY on the findings in the report "
                        "excerpts below. In 'summary' (3-4 simple lines) explain which report "
                        "findings the plan is helping with.\n\n"
                        f"{prefs}\n\nREPORT EXCERPTS:\n{report_text}")
            else:
                task = ("TASK: Create a diet plan for this person. In 'summary' (3-4 simple "
                        "lines) explain the approach.\n\n"
                        f"{prefs}\n\n{extra_text}")

            model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=google_api_key,
                                           temperature=0.4)
            with st.spinner("Creating your diet chart..."):
                plan = model.with_structured_output(DietPlan).invoke(RULES + "\n\n" + task)

            st.session_state["diet_plan"] = plan      # saved so it survives reruns
            st.session_state["diet_meta"] = meta
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ============= STEP 7: SHOW THE PLAN ===============
plan = st.session_state.get("diet_plan")
if plan:
    st.divider()
    st.markdown("## 📋 Your diet chart")
    st.info(plan.summary)

    tabs = st.tabs([d.day for d in plan.days])
    for tab, d in zip(tabs, plan.days):
        with tab:
            for m in d.meals:
                with st.container(border=True):
                    kcal = f"  ·  ~{m.approx_calories} kcal" if m.approx_calories else ""
                    st.markdown(f"**{m.meal}**{kcal}")
                    st.write(m.items)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Foods to include")
        for f in plan.foods_to_include:
            st.write(f"• {f}")
    with c2:
        st.markdown("### ⚠️ Foods to limit")
        for f in plan.foods_to_limit:
            st.write(f"• {f}")

    st.markdown("### 💧 Hydration & lifestyle tips")
    for t in plan.hydration_and_lifestyle:
        st.write(f"• {t}")

    st.warning(f"👩‍⚕️ {plan.when_to_see_dietitian}")
    st.download_button("⬇️ Download diet chart (.txt)", plan_to_text(plan),
                       file_name="my_diet_chart.txt", use_container_width=True)

st.caption("⚠️ General healthy-eating guidance only, not medical advice. Please consult a doctor "
           "or registered dietitian, especially with a medical condition or pregnancy.")
