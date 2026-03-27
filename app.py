import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer import load_signal, prepare_windows, summarize_predictions
from preprocessing import load_preprocessing_artifacts


st.set_page_config(page_title="Bearing RUL Dashboard", layout="wide")

st.title("Bearing Remaining Useful Life Prediction")

MODEL_PATH = Path("models/cnn_rul_model.keras")
ARTIFACT_PATH = Path("models/preprocessing_artifacts.pkl")


@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_artifacts():
    return load_preprocessing_artifacts(ARTIFACT_PATH)


model = load_model_cached()
artifacts = load_artifacts()

uploaded_file = st.file_uploader("Upload IMS vibration file")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    signal = load_signal(tmp_path)

    # Raw waveform
    st.subheader("Raw Vibration Waveform")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(signal)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    X = prepare_windows(
        signal,
        artifacts["window_size"],
        artifacts["stride"],
        artifacts["signal_mean"],
        artifacts["signal_std"],
    )

    # Sliding windows
    st.subheader("Sliding Windows used by CNN")

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    for i in range(min(5, len(X))):
        ax2.plot(X[i].flatten(), alpha=0.7)
    st.pyplot(fig2)

    # CNN prediction
    predictions_scaled = model.predict(X, verbose=0).flatten()
    predictions = artifacts["scaler_y"].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Prediction distribution
    st.subheader("Prediction Distribution")

    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    ax3.hist(predictions, bins=30)
    st.pyplot(fig3)

    # Aggregate results
    summary = summarize_predictions(predictions, artifacts["avg_interval_hours"])

    st.subheader("RUL Prediction Range")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean RUL (hours)", f"{summary['mean_hours']:.2f}")
    col2.metric("Min RUL (hours)", f"{summary['min_hours']:.2f}")
    col3.metric("Max RUL (hours)", f"{summary['max_hours']:.2f}")

    # Machine health status
    st.subheader("Machine Health Status")

    mean_rul = summary["mean_hours"]

    if mean_rul > 50:
        st.success("Healthy")
    elif mean_rul > 20:
        st.warning("Maintenance Recommended")
    else:
        st.error("Failure Soon")
