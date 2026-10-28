"""Golden Fighter AI - Retention Intelligence dashboard (Hebrew, right to left).

    .\\.venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py

Offline by design: it reads the committed artifacts and never calls the Claude API. Its only actions are free
and safe: a break-the-data demo on a copy, and a Flow run without AI that writes to artifacts/_no_llm/.
"""
from pathlib import Path

import streamlit as st

from golden_fighter.dashboard.ui import NO_AI_NOTE, apply_rtl

st.set_page_config(page_title="Golden Fighter AI - חיזוי נטישת מתאמנים", page_icon=":material/fitness_center:",
                   layout="wide")
apply_rtl()

SCREENS = Path(__file__).parent / "screens"  # not "pages/": that name makes Streamlit add its own menu
navigation = st.navigation({
    "תוצאות": [
        st.Page(SCREENS / "overview.py", title="סקירה כללית", icon=":material/home:", default=True),
        st.Page(SCREENS / "insights.py", title="תובנות על עזיבה", icon=":material/insights:"),
        st.Page(SCREENS / "model.py", title="ביצועי המודל", icon=":material/monitoring:"),
        st.Page(SCREENS / "predict.py", title="חיזוי עזיבה", icon=":material/person_search:"),
    ],
    "איך זה עובד": [
        st.Page(SCREENS / "flow.py", title="ריצות Flow ועלויות", icon=":material/account_tree:"),
        st.Page(SCREENS / "gate_demo.py", title="הדגמת שבירת נתונים", icon=":material/gpp_bad:"),
    ],
})
with st.sidebar:
    st.caption(f"**מצב לא מקוון.** {NO_AI_NOTE}")
    st.caption("נתונים ציבוריים מ-Kaggle (מתאמני חדר כושר, אנונימיים). בלי נתונים פרטיים ובלי CRM.")
navigation.run()
