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


def determine_health_status(summary, kurtosis, crest_factor):
    mean_rul = summary["mean_hours"]
    min_rul = summary["min_hours"]
    max_rul = summary["max_hours"]
    rul_spread = max_rul - min_rul

    if min_rul <= 20 or kurtosis >= 8 or crest_factor >= 7:
        return "error", "Failure Soon"

    if mean_rul <= 50 or min_rul <= 50 or kurtosis >= 6 or crest_factor >= 5 or rul_spread >= 120:
        return "warning", "Maintenance Recommended"

    return "success", "Healthy"


model = load_model_cached()
artifacts = load_artifacts()

uploaded_file = st.file_uploader("Upload IMS vibration file")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    signal = load_signal(tmp_path)

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

    st.subheader("Sliding Windows used by CNN")
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    for i in range(min(5, len(X))):
        ax2.plot(X[i].flatten(), alpha=0.7)
    st.pyplot(fig2)

    predictions_scaled = model.predict(X, verbose=0).flatten()
    predictions = artifacts["scaler_y"].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    st.subheader("Signal Health Indicators")
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    kurtosis = np.mean((signal - np.mean(signal)) ** 4) / (np.std(signal) ** 4)
    crest_factor = peak / rms

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="RMS Vibration", value=f"{rms:.4f}")
    col2.metric(label="Peak Amplitude", value=f"{peak:.4f}")
    col3.metric(label="Kurtosis", value=f"{kurtosis:.2f}")
    col4.metric(label="Crest Factor", value=f"{crest_factor:.2f}")

   # Basic interpretation
    if crest_factor > 7 or kurtosis > 6:
        st.error("⚠️ High impulsive vibration detected — possible bearing fault")
    elif crest_factor > 5:
        st.warning("⚠️ Elevated vibration shocks — maintenance recommended")
    else:
        st.success("✓ Vibration levels appear normal")
    st.markdown(
        """
**Indicator Meaning**

- **RMS Vibration** -> overall vibration energy of the bearing
- **Peak Amplitude** -> maximum shock or impact detected
- **Kurtosis** -> impulsiveness of vibration (high values may indicate faults)
- **Crest Factor** -> ratio of peak vibration to RMS vibration
"""
    )

    summary = summarize_predictions(predictions, artifacts["avg_interval_hours"])

    st.subheader("RUL Prediction Range")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean RUL (hours)", f"{summary['mean_hours']:.2f}")
    col2.metric("Min RUL (hours)", f"{summary['min_hours']:.2f}")
    col3.metric("Max RUL (hours)", f"{summary['max_hours']:.2f}")

    st.subheader("Machine Health Status")
    health_level, health_label = determine_health_status(summary, kurtosis, crest_factor)
    getattr(st, health_level)(health_label)
