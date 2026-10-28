"""Golden Fighter AI - Retention Intelligence dashboard.

    .\\.venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py

Offline by design: it reads the committed artifacts and never calls the Claude API. Its only actions are free
and safe: a break-the-data demo on a copy, and a Flow run without AI that writes to artifacts/_no_llm/.
"""
from pathlib import Path

import streamlit as st

from golden_fighter.dashboard.ui import NO_AI_NOTE

st.set_page_config(page_title="Golden Fighter AI - Retention Intelligence", page_icon=":material/fitness_center:",
                   layout="wide")

SCREENS = Path(__file__).parent / "screens"  # not "pages/": that name makes Streamlit add its own menu
navigation = st.navigation({
    "Results": [
        st.Page(SCREENS / "overview.py", title="Overview", icon=":material/home:", default=True),
        st.Page(SCREENS / "insights.py", title="Churn insights", icon=":material/insights:"),
        st.Page(SCREENS / "model.py", title="Model performance", icon=":material/monitoring:"),
        st.Page(SCREENS / "predict.py", title="Predict churn", icon=":material/person_search:"),
    ],
    "How it works": [
        st.Page(SCREENS / "flow.py", title="Flow runs & costs", icon=":material/account_tree:"),
        st.Page(SCREENS / "gate_demo.py", title="Break-the-data demo", icon=":material/gpp_bad:"),
    ],
})
with st.sidebar:
    st.caption(f"**Offline mode.** {NO_AI_NOTE}")
    st.caption("Public Kaggle data (gym members, anonymised). No private or CRM data.")
navigation.run()
