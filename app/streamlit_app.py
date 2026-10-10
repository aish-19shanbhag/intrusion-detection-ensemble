"""Intrusion detection demo: classify network connections and see how each model voted.

Run:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASS_DESCRIPTIONS, DATA_DIR  # noqa: E402
from src.predict import load_model, predict_frame  # noqa: E402

st.set_page_config(page_title="Intrusion Detection Using Ensemble Techniques", layout="wide")
st.title("🛡️ Intrusion Detection Using Ensemble Techniques")
st.caption("Naive Bayes, Decision Tree and XGBoost vote on every connection; the weighted majority wins.")


@st.cache_resource
def model():
    return load_model()


pipeline, class_names = model()
sample = pd.read_csv(DATA_DIR / "sample_connections.csv")

source = st.radio("Data", ["Use sample connections from KDDTest", "Upload a CSV"], horizontal=True)
if source == "Upload a CSV":
    file = st.file_uploader("CSV with the 41 NSL-KDD feature columns", type="csv")
    if file is None:
        st.stop()
    data = pd.read_csv(file)
else:
    data = sample

result = predict_frame(data, pipeline, class_names)
if "attack_cat" in data.columns:
    result.insert(0, "actual", data["attack_cat"])
    correct = (result["actual"] == result["prediction"]).mean()
    st.metric("Correct on these rows", f"{correct:.0%}")

left, right = st.columns([2, 1])
with left:
    st.subheader("Predictions and individual votes")
    st.dataframe(result, width="stretch")
with right:
    st.subheader("Predicted categories")
    st.bar_chart(result["prediction"].value_counts())
    st.markdown("\n".join(f"- **{k}**: {v}" for k, v in CLASS_DESCRIPTIONS.items()))
