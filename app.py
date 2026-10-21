import streamlit as st

pg = st.navigation([
    st.Page("pages/report_reader.py", title="Report Reader", icon="🩺", default=True),
    st.Page("pages/1_Find_Doctors.py", title="Find Doctors", icon="🏥"),
    st.Page("pages/2_Symptom_Checker.py", title="Symptom Checker", icon="🤒"),
])
pg.run()
