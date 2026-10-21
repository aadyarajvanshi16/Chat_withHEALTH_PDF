import streamlit as st

pg = st.navigation([
    st.Page("pages/Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/Report_reader.py", title="Report Reader", icon="🩺"),
    st.Page("pages/2_Symptom_Checker.py", title="Symptom Checker", icon="🤒"),
    st.Page("pages/3_Diet_Plan.py", title="Diet Plan", icon="🥗"),
    st.Page("pages/1_Find_Doctors.py", title="Find Doctors", icon="🏥"),
])
pg.run()
