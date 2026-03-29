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


st.set_page_config(
    page_title="Bearing RUL Studio",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("models/cnn_rul_model.keras")
ARTIFACT_PATH = Path("models/preprocessing_artifacts.pkl")


@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_artifacts():
    return load_preprocessing_artifacts(ARTIFACT_PATH)


def determine_health_status(summary, kurtosis, crest_factor):
    median_rul = summary["median_hours"]
    lower_rul = summary["lower_hours"]
    std_rul = summary["std_hours"]

    if lower_rul <= 20 or kurtosis >= 8 or crest_factor >= 7:
        return "error", "Failure Soon"

    if median_rul <= 50 or lower_rul <= 50 or kurtosis >= 6 or crest_factor >= 5 or std_rul >= 60:
        return "warning", "Maintenance Recommended"

    return "success", "Healthy"


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5efe3;
            --panel: rgba(255, 252, 246, 0.84);
            --panel-strong: rgba(255, 248, 238, 0.96);
            --text: #1f2933;
            --muted: #5f6c76;
            --accent: #c96f3b;
            --accent-dark: #8d4e2c;
            --teal: #1c7c7d;
            --ok: #1f7a4d;
            --warn: #b66a1e;
            --danger: #a33636;
            --border: rgba(132, 99, 67, 0.18);
            --shadow: 0 18px 40px rgba(96, 72, 48, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201, 111, 59, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(28, 124, 125, 0.16), transparent 24%),
                linear-gradient(180deg, #f7f2e8 0%, #f1eadf 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1220px;
        }

        h1, h2, h3 {
            color: var(--text);
            font-family: Georgia, "Times New Roman", serif;
            letter-spacing: -0.02em;
        }

        p, li, label, div[data-testid="stMarkdownContainer"] {
            color: var(--text);
        }

        div[data-testid="stFileUploader"] > section,
        div[data-testid="stMetric"],
        div[data-testid="stExpander"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        div[data-testid="stMetric"] {
            padding: 0.75rem 0.9rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.75rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-family: Georgia, "Times New Roman", serif;
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(255, 248, 238, 0.96), rgba(247, 239, 227, 0.82)),
                linear-gradient(135deg, #c96f3b, #1c7c7d);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .hero-kicker {
            color: var(--accent-dark);
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .hero-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2.55rem;
            line-height: 1.05;
            color: var(--text);
            margin: 0;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
            max-width: 760px;
            margin-top: 0.8rem;
        }

        .info-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 1.15rem 1.2rem;
            min-height: 100%;
        }

        .info-card h3 {
            margin-top: 0;
            margin-bottom: 0.7rem;
            font-size: 1.05rem;
        }

        .info-card p,
        .info-card li {
            color: var(--muted);
            line-height: 1.65;
        }

        .status-chip {
            display: inline-block;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-size: 0.78rem;
            margin-bottom: 0.7rem;
        }

        .status-chip.success {
            background: rgba(31, 122, 77, 0.14);
            color: var(--ok);
        }

        .status-chip.warning {
            background: rgba(182, 106, 30, 0.14);
            color: var(--warn);
        }

        .status-chip.error {
            background: rgba(163, 54, 54, 0.14);
            color: var(--danger);
        }

        .section-label {
            color: var(--accent-dark);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .chart-card {
            background: var(--panel-strong);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 0.8rem 0.8rem 0.2rem;
        }

        .upload-hint {
            color: var(--muted);
            margin-top: 0.45rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Condition Monitoring Dashboard</div>
            <h1 class="hero-title">Bearing RUL Studio</h1>
            <div class="hero-copy">
                Upload an IMS vibration file to inspect waveform behavior, compare CNN window slices,
                and review remaining useful life estimates with confidence bounds and health indicators.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_intro():
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-label">How To Use</div>
                <h3>Start with one vibration capture</h3>
                <p>
                    The app normalizes the uploaded signal, creates sliding windows for the CNN,
                    predicts RUL per window, and summarizes the result in hours using the saved
                    preprocessing artifacts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-label">What You Will See</div>
                <h3>Signal quality, uncertainty, and health state</h3>
                <p>
                    Review RMS, peak amplitude, kurtosis, crest factor, the median RUL estimate,
                    and interval bounds to understand both predicted life and confidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", color="#1f2933", pad=12)
    ax.set_xlabel(xlabel, color="#5f6c76")
    ax.set_ylabel(ylabel, color="#5f6c76")
    ax.set_facecolor("#fffaf3")
    ax.grid(True, alpha=0.22, color="#9b8a75", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#d8c7b3")
    ax.tick_params(colors="#5f6c76")


def render_status_summary(health_level, health_label, summary, kurtosis, crest_factor):
    st.subheader("Machine Health Status")
    st.markdown(
        f"""
        <div class="info-card">
            <div class="status-chip {health_level}">{health_label}</div>
            <h3>Health Assessment Overview</h3>
            <p>
                The current decision combines lower-bound RUL, prediction spread, kurtosis, and crest factor.
                Median RUL is <strong>{summary['median_hours']:.2f}</strong> hours, lower bound is
                <strong>{summary['lower_hours']:.2f}</strong> hours, kurtosis is
                <strong>{kurtosis:.2f}</strong>, and crest factor is <strong>{crest_factor:.2f}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()
render_header()
render_intro()

model = load_model_cached()
artifacts = load_artifacts()

st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload IMS vibration file",
    type=None,
    help="Select one IMS vibration text file to run inference.",
)
st.markdown(
    '<div class="upload-hint">Best results come from files that match the same bearing column and preprocessing setup used during training.</div>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info("Upload a vibration file to generate waveform plots, health indicators, and RUL estimates.")
else:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    signal = load_signal(tmp_path)
    X = prepare_windows(
        signal,
        artifacts["window_size"],
        artifacts["stride"],
        artifacts["signal_mean"],
        artifacts["signal_std"],
    )

    predictions_scaled = model.predict(X, verbose=0).flatten()
    predictions = artifacts["scaler_y"].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    summary = summarize_predictions(predictions, artifacts["avg_interval_hours"])

    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    signal_std = np.std(signal)
    kurtosis = np.mean((signal - np.mean(signal)) ** 4) / ((signal_std ** 4) + 1e-8)
    crest_factor = peak / (rms + 1e-8)
    health_level, health_label = determine_health_status(summary, kurtosis, crest_factor)

    top_left, top_right = st.columns([1.35, 0.65], gap="large")

    with top_left:
        st.subheader("Signal Overview")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8.8, 3.8))
        ax.plot(signal, color="#1c7c7d", linewidth=1.1)
        style_axis(ax, "Raw Vibration Waveform", "Sample", "Amplitude")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        render_status_summary(health_level, health_label, summary, kurtosis, crest_factor)

    st.subheader("Health Indicators")
    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("RMS Vibration", f"{rms:.4f}")
    metric_cols[1].metric("Peak Amplitude", f"{peak:.4f}")
    metric_cols[2].metric("Kurtosis", f"{kurtosis:.2f}")
    metric_cols[3].metric("Crest Factor", f"{crest_factor:.2f}")

    rul_cols = st.columns(4, gap="medium")
    rul_cols[0].metric("Predicted RUL", f"{summary['median_hours']:.2f} h")
    rul_cols[1].metric("Lower Bound", f"{summary['lower_hours']:.2f} h")
    rul_cols[2].metric("Upper Bound", f"{summary['upper_hours']:.2f} h")
    rul_cols[3].metric("Std Deviation", f"{summary['std_hours']:.2f} h")

    lower_panel, right_panel = st.columns([1.1, 0.9], gap="large")

    with lower_panel:
        st.subheader("Windowed Signal View")
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8.8, 3.8))
        for i in range(min(5, len(X))):
            ax2.plot(X[i].flatten(), alpha=0.74, linewidth=1.0)
        style_axis(ax2, "Normalized Sliding Windows", "Window Sample", "Normalized Amplitude")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_panel:
        st.subheader("Interpretation Guide")
        st.markdown(
            """
            <div class="info-card">
                <div class="section-label">Reading The Output</div>
                <h3>Quick interpretation</h3>
                <p><strong>RMS Vibration</strong> reflects the overall vibration energy in the bearing.</p>
                <p><strong>Peak Amplitude</strong> highlights strong impacts or shocks in the signal.</p>
                <p><strong>Kurtosis</strong> increases when the waveform becomes more impulsive.</p>
                <p><strong>Crest Factor</strong> compares the strongest peak against the average vibration level.</p>
                <p><strong>RUL Bounds</strong> summarize uncertainty after trimming extreme window predictions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

