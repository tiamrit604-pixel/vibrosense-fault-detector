"""
VibroSense — Engine Fault Detection
Streamlit App (single file, ready for Streamlit Cloud deployment)
"""

import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import correlate, welch
from scipy.stats import skew, kurtosis
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibroSense — Engine Fault Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main { background-color: #0a0c10; }

.stApp {
    background-color: #0a0c10;
    color: #e8eaf0;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #111318;
    border: 1px solid #22262f;
    border-radius: 10px;
    padding: 16px !important;
}

div[data-testid="metric-container"] label {
    color: #5a6070 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #00e5ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 24px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #22262f;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: #111318;
    border: 1.5px dashed #22262f;
    border-radius: 12px;
    padding: 8px;
}

/* Buttons */
div.stButton > button {
    background: #00e5ff;
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: none;
    border-radius: 6px;
    width: 100%;
}

div.stButton > button:hover {
    background: #33eeff;
    transform: translateY(-1px);
}

/* Headers */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

.fault-badge {
    display: inline-block;
    background: rgba(255,77,109,0.12);
    border: 1.5px solid #ff4d6d;
    color: #ff4d6d;
    font-family: 'Space Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    padding: 12px 28px;
    border-radius: 10px;
    letter-spacing: -1px;
    margin: 8px 0;
}

.ok-badge {
    display: inline-block;
    background: rgba(184,255,87,0.10);
    border: 1.5px solid #b8ff57;
    color: #b8ff57;
    font-family: 'Space Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    padding: 12px 28px;
    border-radius: 10px;
    letter-spacing: -1px;
    margin: 8px 0;
}

.info-row {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #5a6070;
    background: #111318;
    border: 1px solid #22262f;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 16px;
}

.info-row span { color: #e8eaf0; }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #00e5ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #22262f;
}

hr { border-color: #22262f; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
FS = 1000  # sampling frequency in Hz

FAULT_DESCRIPTIONS = {
    "NoFault":  "Engine is operating normally. No fault detected.",
    "Fault1":   "Fault 1 — Abnormal vibration pattern in low-frequency band (0–20 Hz).",
    "Fault2":   "Fault 2 — Elevated spectral energy in mid-frequency range.",
    "Fault3":   "Fault 3 — High kurtosis indicating impulsive events (bearing defect signature).",
    "Fault4":   "Fault 4 — Significant crest factor anomaly.",
    "Fault5":   "Fault 5 — Dominant frequency shift detected.",
    "Fault6":   "Fault 6 — High RMS with abnormal spectral distribution.",
    "Fault7":   "Fault 7 — Imbalance signature in 50–100 Hz band.",
    "Fault8":   "Fault 8 — Multi-band energy anomaly.",
    "Fault9":   "Fault 9 — High-frequency vibration anomaly (100–200 Hz band).",
}

# ─────────────────────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model_artifacts.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

artifacts = load_model()

# ─────────────────────────────────────────────────────────────
# Signal processing functions (mirror notebook exactly)
# ─────────────────────────────────────────────────────────────
def parse_file(content: str) -> pd.DataFrame:
    """Parse accelerometer file. Auto-detects header or falls back to skiprows=20."""
    lines = content.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "TimeStamp" in line or "timestamp" in line.lower():
            start = i
            break

    if start is not None:
        body = "\n".join(lines[start + 1:])
        df = pd.read_csv(io.StringIO(body), sep=";", header=None)
    else:
        body = "\n".join(lines[20:])
        df = pd.read_csv(io.StringIO(body), sep=";", header=None)

    df = df.dropna()

    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["Time", "Z", "X", "Y"]
    elif df.shape[1] == 2:
        df.columns = ["Time", "Signal"]
        df["X"] = df["Signal"]
        df["Y"] = df["Signal"]
        df["Z"] = df["Signal"]
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}. Expected 2 or 4.")

    for col in ["X", "Y", "Z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df


def extract_features(signal: np.ndarray, fs: int = FS):
    """Extract 14 time + frequency domain features."""
    rms = np.sqrt(np.mean(signal ** 2))
    var = np.var(signal)
    std = np.std(signal)
    peak = np.max(np.abs(signal))
    ptp = np.ptp(signal)
    sk = skew(signal)
    kurt_val = kurtosis(signal)
    crest = peak / rms if rms != 0 else 0

    f, pxx = welch(signal, fs=fs, nperseg=256)
    dominant_freq = float(f[np.argmax(pxx)])
    spectral_centroid = float(np.sum(f * pxx) / np.sum(pxx)) if np.sum(pxx) > 0 else 0
    band1 = float(np.sum(pxx[(f >= 0)   & (f < 20)]))
    band2 = float(np.sum(pxx[(f >= 20)  & (f < 50)]))
    band3 = float(np.sum(pxx[(f >= 50)  & (f < 100)]))
    band4 = float(np.sum(pxx[(f >= 100) & (f < 200)]))

    features = [rms, var, std, peak, ptp, sk, kurt_val, crest,
                dominant_freq, spectral_centroid, band1, band2, band3, band4]
    return features, f, pxx


def run_prediction(df: pd.DataFrame, artifacts: dict):
    """Full prediction pipeline. Returns dict of results."""
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    reference_signal = artifacts["reference_signal"]

    x = df["X"].values.astype(float)
    y = df["Y"].values.astype(float)
    z = df["Z"].values.astype(float)

    mag = np.sqrt(x**2 + y**2 + z**2)
    original_length = len(mag)

    # Cross-correlation alignment
    corr = correlate(mag, reference_signal, mode="full")
    shift = int(corr.argmax()) - len(reference_signal)
    shift_time = round(shift / FS, 3)
    if shift > 0:
        mag = mag[shift:]

    # Remove first 5 seconds (startup transient)
    mag = mag[5 * FS:]

    # Segment into 1-second windows
    window = FS
    segments = [mag[i * window:(i + 1) * window] for i in range(len(mag) // window)]

    if len(segments) == 0:
        raise ValueError("File is too short — need at least 6 seconds of data after alignment.")

    # Feature extraction
    feature_list, dom_freqs = [], []
    mid_f, mid_pxx = None, None
    mid_idx = len(segments) // 2

    for i, seg in enumerate(segments):
        feats, f_arr, pxx_arr = extract_features(seg)
        feature_list.append(feats)
        dom_freqs.append(feats[8])
        if i == mid_idx:
            mid_f, mid_pxx = f_arr, pxx_arr

    X_new = scaler.transform(np.array(feature_list))

    # Predict
    preds = model.predict(X_new)
    counter = Counter(preds)
    final_pred = counter.most_common(1)[0][0]
    confidence = counter.most_common(1)[0][1] / len(preds)

    # Signal stats
    processed = mag[:len(segments) * window]
    stats = {
        "RMS":          round(float(np.sqrt(np.mean(processed**2))), 4),
        "Peak":         round(float(np.max(np.abs(processed))), 4),
        "Std Dev":      round(float(np.std(processed)), 4),
        "Skewness":     round(float(skew(processed)), 4),
        "Kurtosis":     round(float(kurtosis(processed)), 4),
        "Dom. Freq (Hz)": round(float(np.mean(dom_freqs)), 2),
    }

    return {
        "predicted_fault":        final_pred,
        "confidence_pct":         round(confidence * 100, 2),
        "segments_analyzed":      len(segments),
        "original_length":        original_length,
        "shift_samples":          shift,
        "shift_seconds":          shift_time,
        "segment_distribution":   dict(counter),
        "signal_stats":           stats,
        "psd_freqs":              mid_f,
        "psd_power":              mid_pxx,
        "magnitude":              processed,
        "dom_freqs":              dom_freqs,
    }

# ─────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#111318",
    plot_bgcolor="#111318",
    font=dict(family="Space Mono, monospace", color="#5a6070", size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#22262f", linecolor="#22262f", zerolinecolor="#22262f"),
    yaxis=dict(gridcolor="#22262f", linecolor="#22262f", zerolinecolor="#22262f"),
)

def plot_time_series(mag: np.ndarray) -> go.Figure:
    step = max(1, len(mag) // 3000)
    t = np.arange(len(mag))[::step] / FS
    m = mag[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=m, mode="lines",
        line=dict(color="rgba(0,229,255,0.85)", width=1),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.05)",
        name="Magnitude"
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        height=260,
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (g)",
        showlegend=False,
    )
    return fig


def plot_psd(f: np.ndarray, pxx: np.ndarray) -> go.Figure:
    mask = f <= 200
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f[mask], y=pxx[mask], mode="lines",
        line=dict(color="rgba(184,255,87,0.85)", width=1.5),
        fill="tozeroy", fillcolor="rgba(184,255,87,0.05)",
        name="PSD"
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        height=260,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        showlegend=False,
    )
    return fig


def plot_segment_dist(dist: dict, total: int) -> go.Figure:
    COLORS = ["#00e5ff","#ff4d6d","#b8ff57","#ff9f43","#a29bfe",
              "#fd79a8","#00cec9","#e17055","#74b9ff","#55efc4"]
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [COLORS[i % len(COLORS)] for i in range(len(items))]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v} ({round(v/total*100)}%)" for v in values],
        textposition="outside",
        textfont=dict(family="Space Mono", size=10, color="#e8eaf0"),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        height=max(160, len(items) * 40 + 60),
        xaxis_title="Segments",
        showlegend=False,
    )
    fig.update_yaxes(
        gridcolor="#22262f",
        linecolor="#22262f",
        tickfont=dict(family="Space Mono", size=11, color="#e8eaf0"),
    )
    return fig


def plot_confidence_gauge(pct: float, is_fault: bool) -> go.Figure:
    color = "#ff4d6d" if is_fault else "#b8ff57"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(family="Space Mono", size=36, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#5a6070", size=10), tickcolor="#22262f"),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#0a0c10",
            bordercolor="#22262f",
            steps=[
                dict(range=[0,   50], color="#1a0c10"),
                dict(range=[50,  75], color="#111318"),
                dict(range=[75, 100], color="#0d1a0a"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=pct),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#111318",
        font=dict(family="Space Mono", color="#e8eaf0"),
        height=200,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 VibroSense")
    st.markdown("**Engine Fault Detection**")
    st.markdown("---")

    st.markdown("### How it works")
    st.markdown("""
1. Upload an accelerometer `.txt` or `.csv` file
2. Signal is parsed → magnitude computed
3. Cross-correlation alignment applied
4. First 5s removed (startup transient)
5. Signal segmented into 1s windows
6. 14 features extracted per segment
7. SVM model classifies each segment
8. Majority vote → final prediction
    """)

    st.markdown("---")
    st.markdown("### Fault Classes")
    for fault, desc in FAULT_DESCRIPTIONS.items():
        color = "#b8ff57" if fault == "NoFault" else "#ff4d6d"
        st.markdown(f"<span style='color:{color};font-family:Space Mono;font-size:12px'>{fault}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model")
    if artifacts:
        st.success("✅ Model loaded")
        st.caption("SVM · RBF kernel · 14 features")
    else:
        st.error("❌ model_artifacts.pkl not found")
        st.caption("Run `python train_and_export.py` first")

    st.markdown("---")
    st.caption("Sampling rate: 1000 Hz")
    st.caption("Window: 1 second (1000 samples)")
    st.caption("Min file length: 6 seconds")

# ─────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:42px;letter-spacing:-2px;margin-bottom:4px'>
Engine <span style='color:#00e5ff'>Fault</span> Detector
</h1>
<p style='color:#5a6070;font-family:Space Mono,monospace;font-size:13px;margin-bottom:32px'>
Upload accelerometer data → get instant fault classification with signal analysis
</p>
""", unsafe_allow_html=True)

# ── Model guard ────────────────────────────────────────────────
if artifacts is None:
    st.error("""
    **Model not found.**  
    `model_artifacts.pkl` is missing. Follow these steps:

    ```bash
    # 1. Put your Analysis Files/ folder in the project directory
    # 2. Run the training script
    python train_and_export.py
    ```

    Then restart this app.
    """)
    st.stop()

# ── File upload ────────────────────────────────────────────────
st.markdown('<div class="section-label">Input — Upload Accelerometer File</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="Drop your file here",
    type=["txt", "csv"],
    help="Semicolon-delimited file with X, Y, Z accelerometer columns. Any filename, any length.",
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style='text-align:center;padding:40px;color:#5a6070;font-family:Space Mono,monospace;font-size:13px;
    background:#111318;border:1.5px dashed #22262f;border-radius:12px;margin-top:16px'>
        📡 &nbsp; Drag & drop or click Browse to upload your accelerometer file
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Process ────────────────────────────────────────────────────
with st.spinner("Processing signal…"):
    try:
        content = uploaded.read().decode("utf-8", errors="replace")
        df = parse_file(content)
        result = run_prediction(df, artifacts)
    except Exception as e:
        st.error(f"**Error processing file:** {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────
pred = result["predicted_fault"]
conf = result["confidence_pct"]
is_fault = "nofault" not in pred.lower()

st.markdown("---")

# File info bar
st.markdown(f"""
<div class="info-row">
📄 File: <span>{uploaded.name}</span>
&emsp;·&emsp;
Original length: <span>{result['original_length']:,} samples ({result['original_length']//FS}s)</span>
&emsp;·&emsp;
Segments analyzed: <span>{result['segments_analyzed']}</span>
&emsp;·&emsp;
Alignment shift: <span>{result['shift_samples']} samples ({result['shift_seconds']}s)</span>
</div>
""", unsafe_allow_html=True)

# ── Top row: Prediction + Confidence + Stats ───────────────────
col_pred, col_gauge, col_stats = st.columns([1.2, 1, 1.8])

with col_pred:
    st.markdown('<div class="section-label">Predicted Condition</div>', unsafe_allow_html=True)
    badge_class = "fault-badge" if is_fault else "ok-badge"
    st.markdown(f'<div class="{badge_class}">{pred}</div>', unsafe_allow_html=True)

    desc = FAULT_DESCRIPTIONS.get(pred, FAULT_DESCRIPTIONS.get(
        next((k for k in FAULT_DESCRIPTIONS if pred.startswith(k)), "Fault1"), "Unknown fault condition."
    ))
    st.markdown(f"<p style='color:#5a6070;font-size:13px;margin-top:8px'>{desc}</p>", unsafe_allow_html=True)

with col_gauge:
    st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_confidence_gauge(conf, is_fault), use_container_width=True)

with col_stats:
    st.markdown('<div class="section-label">Signal Statistics</div>', unsafe_allow_html=True)
    stats = result["signal_stats"]
    keys = list(stats.keys())
    r1, r2 = keys[:3], keys[3:]
    cols1 = st.columns(3)
    for i, k in enumerate(r1):
        cols1[i].metric(k, stats[k])
    cols2 = st.columns(3)
    for i, k in enumerate(r2):
        cols2[i].metric(k, stats[k])

st.markdown("---")

# ── Charts row ─────────────────────────────────────────────────
col_ts, col_psd = st.columns(2)

with col_ts:
    st.markdown('<div class="section-label">Magnitude Time Series</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_time_series(result["magnitude"]), use_container_width=True)

with col_psd:
    st.markdown('<div class="section-label">Power Spectral Density (Middle Segment)</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_psd(result["psd_freqs"], result["psd_power"]), use_container_width=True)

# ── Segment distribution ───────────────────────────────────────
st.markdown('<div class="section-label">Per-Segment Prediction Distribution</div>', unsafe_allow_html=True)
st.plotly_chart(
    plot_segment_dist(result["segment_distribution"], result["segments_analyzed"]),
    use_container_width=True
)

# ── Dominant frequency per segment ────────────────────────────
st.markdown('<div class="section-label">Dominant Frequency Per Segment</div>', unsafe_allow_html=True)
dom_fig = go.Figure()
dom_fig.add_trace(go.Scatter(
    x=list(range(len(result["dom_freqs"]))),
    y=result["dom_freqs"],
    mode="lines+markers",
    line=dict(color="rgba(255,159,67,0.8)", width=1.5),
    marker=dict(size=4, color="#ff9f43"),
    name="Dominant Freq",
))
dom_fig.update_layout(
    **DARK_LAYOUT,
    height=220,
    xaxis_title="Segment Index",
    yaxis_title="Frequency (Hz)",
    showlegend=False,
)
st.plotly_chart(dom_fig, use_container_width=True)

# ── Raw results expander ───────────────────────────────────────
with st.expander("📋 Raw Prediction Data (JSON)"):
    import json
    display = {k: v for k, v in result.items()
               if k not in ("magnitude", "psd_freqs", "psd_power", "dom_freqs")}
    st.json(display)
