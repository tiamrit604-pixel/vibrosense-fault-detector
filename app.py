import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import joblib
import time
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vibrational Fault Diagnosis System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@300;400;600&display=swap');

:root {
    --ink:        #1a1a1a;
    --ink-light:  #444444;
    --ink-faint:  #888888;
    --rule:       #c8c8c8;
    --rule-light: #e4e4e4;
    --bg:         #fafaf7;
    --bg-white:   #ffffff;
    --accent:     #1a3a6b;
    --healthy:    #1a4a2a;
    --fault:      #6b1a1a;
    --highlight:  #fffbe6;
}
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: var(--bg);
    color: var(--ink);
}
.stApp { background-color: var(--bg); }

.journal-header {
    border-top: 3px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    padding: 1.4rem 0 1rem 0;
    margin-bottom: 0.5rem;
}
.journal-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 3px;
    color: var(--ink-faint);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.journal-title {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.15;
    margin: 0;
}
.journal-subtitle {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 1.05rem;
    font-style: italic;
    color: var(--ink-light);
    margin-top: 4px;
}
.journal-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    color: var(--ink-faint);
    letter-spacing: 0.4px;
    margin-top: 10px;
    border-top: 1px solid var(--rule-light);
    padding-top: 8px;
}
.section-rule {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.61rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--ink-faint);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 5px;
    margin: 2.2rem 0 1.1rem 0;
}
.sec-num { color: var(--accent); margin-right: 8px; }
.metric-panel {
    background: var(--bg-white);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--ink);
    padding: 1rem 1.2rem;
    height: 100%;
}
.metric-panel.blue  { border-top-color: var(--accent); }
.metric-panel.green { border-top-color: var(--healthy); }
.metric-panel.red   { border-top-color: var(--fault); }
.metric-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.61rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 6px;
}
.metric-val {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1;
}
.metric-unit {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.78rem;
    color: var(--ink-faint);
    margin-left: 3px;
}
.metric-sub {
    font-size: 0.74rem;
    color: var(--ink-faint);
    margin-top: 5px;
    font-style: italic;
}
.result-box {
    background: var(--bg-white);
    border: 1px solid var(--rule);
    border-left: 5px solid var(--healthy);
    padding: 1.4rem 1.8rem;
    margin: 0.8rem 0;
}
.result-box.fault-box { border-left-color: var(--fault); }
.result-verdict {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--healthy);
    line-height: 1;
}
.result-verdict.fault-v { color: var(--fault); }
.result-rlabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.61rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-faint);
    margin-bottom: 4px;
}
.conf-track { background: var(--rule-light); height: 4px; margin-top: 10px; width: 100%; }
.conf-fill-g { height: 4px; background: var(--healthy); }
.conf-fill-r { height: 4px; background: var(--fault); }
.vote-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    line-height: 2;
    color: var(--ink-light);
}
.fig-caption {
    font-family: 'Crimson Pro', Georgia, serif;
    font-size: 0.84rem;
    font-style: italic;
    color: var(--ink-light);
    border-top: 1px solid var(--rule-light);
    padding-top: 6px;
    margin-top: 4px;
    margin-bottom: 1rem;
}
.fig-label {
    font-style: normal;
    font-weight: 600;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.82rem;
}
.notice-box {
    background: var(--highlight);
    border: 1px solid #e0d080;
    border-left: 4px solid #c8a800;
    padding: 0.65rem 1rem;
    font-size: 0.81rem;
    color: #5a4800;
    margin: 0.6rem 0;
    font-style: italic;
}
.empty-state {
    text-align: center;
    padding: 3.5rem 0;
    border: 1px solid var(--rule);
    background: var(--bg-white);
    margin-top: 1rem;
}
div[data-testid="stFileUploader"] {
    background: var(--bg-white);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--accent);
    border-radius: 0;
    padding: 1rem;
}
.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    letter-spacing: 1px;
}
.stProgress > div > div { background: var(--accent); }
div[data-testid="stExpander"] { border: 1px solid var(--rule); border-radius: 0; }
footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────────────────────
SAMPLING_RATE = 1000

plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size"         : 8.5,
    "axes.linewidth"    : 0.8,
    "axes.edgecolor"    : "#444",
    "axes.labelcolor"   : "#1a1a1a",
    "axes.labelsize"    : 8.5,
    "axes.titlesize"    : 9,
    "axes.titleweight"  : "bold",
    "axes.grid"         : True,
    "grid.color"        : "#dddddd",
    "grid.linewidth"    : 0.45,
    "grid.linestyle"    : "--",
    "xtick.color"       : "#444",
    "ytick.color"       : "#444",
    "xtick.labelsize"   : 7.5,
    "ytick.labelsize"   : 7.5,
    "xtick.direction"   : "in",
    "ytick.direction"   : "in",
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#fafaf7",
    "legend.fontsize"   : 7.5,
    "legend.framealpha" : 1.0,
    "legend.edgecolor"  : "#cccccc",
    "legend.fancybox"   : False,
    "lines.linewidth"   : 0.9,
    "savefig.dpi"       : 180,
    "savefig.facecolor" : "white",
})

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("fault_classifier.pkl")
    model  = bundle["model"]
    # ── Single source of truth: build class_names from model.classes_ only ──
    # This is safe even when classes are non-contiguous (e.g. Fault 2 missing)
    class_names = {
        int(cls): ("IDLE" if int(cls) == 0 else f"Fault {int(cls)}")
        for cls in model.classes_
    }
    return model, bundle["scaler"], bundle["window_size"], class_names

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_txt_file(file_bytes):
    lines = file_bytes.decode("utf-8").splitlines()
    start = next((i+1 for i, l in enumerate(lines) if "TimeStamp" in l), None)
    if start is None:
        raise ValueError("Header row 'TimeStamp' not found.")
    data = []
    for line in lines[start:]:
        p = line.strip().split(";")
        if len(p) < 4:
            continue
        try:
            data.append([float(p[1]), float(p[2]), float(p[3])])
        except:
            continue
    if not data:
        raise ValueError("No valid data rows found.")
    return np.array(data)


def compute_magnitude(s):
    return np.sqrt(s[:,0]**2 + s[:,1]**2 + s[:,2]**2)


def detect_stable_region(signal, w_sec=0.5, buf_start=10, buf_end=5, min_sec=10):
    mag    = compute_magnitude(signal)
    mag_ac = np.abs(mag - np.mean(mag))
    win    = int(w_sec * SAMPLING_RATE)
    n      = len(mag_ac) // win
    energy = np.array([np.sqrt(np.mean(mag_ac[i*win:(i+1)*win]**2)) for i in range(n)])
    emin, emax = energy.min(), energy.max()
    erange = emax - emin
    if erange < 0.05 * emax:
        s = int(buf_start * SAMPLING_RATE)
        e = len(signal) - int(buf_end * SAMPLING_RATE)
        return signal[max(0,s):min(len(signal),e)], "full", max(0,s), min(len(signal),e)
    thr    = emin + 0.30 * erange
    active = energy > thr
    if not np.any(active):
        return signal, "fallback", 0, len(signal)
    fa = np.argmax(active)
    la = len(active) - 1 - np.argmax(active[::-1])
    s  = max(0, fa*win + int(buf_start*SAMPLING_RATE))
    e  = min(len(signal), (la+1)*win - int(buf_end*SAMPLING_RATE))
    if e - s < int(min_sec * SAMPLING_RATE):
        return signal, "fallback", 0, len(signal)
    return signal[s:e], "detected", s, e


def extract_features(w):
    mag  = compute_magnitude(w)
    rms  = np.sqrt(np.mean(mag**2))
    mabs = np.mean(np.abs(mag))
    peak = np.max(np.abs(mag))
    feats = [
        rms, np.var(mag), np.std(mag), peak, np.ptp(mag),
        skew(mag), kurtosis(mag),
        peak/rms  if rms  != 0 else 0,
        rms/mabs  if mabs != 0 else 0,
        peak/mabs if mabs != 0 else 0,
    ]
    f, pxx = welch(mag, fs=SAMPLING_RATE, nperseg=256)
    sc  = np.sum(f*pxx)/np.sum(pxx) if np.sum(pxx) != 0 else 0
    feats += [
        f[np.argmax(pxx)], sc,
        np.sum(((f-sc)**2)*pxx)/np.sum(pxx) if np.sum(pxx) != 0 else 0,
        np.sum(pxx[(f>=0)   & (f<20)]),
        np.sum(pxx[(f>=20)  & (f<50)]),
        np.sum(pxx[(f>=50)  & (f<100)]),
        np.sum(pxx[(f>=100) & (f<200)]),
        np.sum(pxx),
    ]
    return feats


def extract_windows(signal, ws):
    return [signal[i*ws:(i+1)*ws] for i in range(len(signal)//ws)]


FEATURE_NAMES = [
    "RMS","Variance","Std Dev","Peak","Peak-to-Peak",
    "Skewness","Kurtosis","Crest Factor","Shape Factor","Impulse Factor",
    "Dominant Freq","Spectral Centroid","Spectral Variance",
    "Band 0–20 Hz","Band 20–50 Hz","Band 50–100 Hz","Band 100–200 Hz",
    "Total Spectral Energy",
]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def fig_signal_overview(raw, stable, s_idx, e_idx):
    fig = plt.figure(figsize=(13, 7.2), facecolor="white")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.58, wspace=0.32,
                             left=0.07, right=0.97, top=0.94, bottom=0.08)
    raw_mag = compute_magnitude(raw)
    stb_mag = compute_magnitude(stable)
    t_raw   = np.arange(len(raw_mag)) / SAMPLING_RATE
    t_stb   = np.arange(len(stb_mag)) / SAMPLING_RATE

    ax = fig.add_subplot(gs[0, :])
    ax.plot(t_raw, raw_mag, color="#555", lw=0.45, alpha=0.85, label="Magnitude |a(t)|")
    ax.axvspan(s_idx/SAMPLING_RATE, e_idx/SAMPLING_RATE, color="#1a3a6b", alpha=0.09, label="Stable region")
    ax.axvline(s_idx/SAMPLING_RATE, color="#1a3a6b", lw=1.0, ls="--", alpha=0.65)
    ax.axvline(e_idx/SAMPLING_RATE, color="#1a3a6b", lw=1.0, ls="--", alpha=0.65)
    ax.set_xlabel("Time  (s)"); ax.set_ylabel("Magnitude  (g)")
    ax.set_title("(a)  Full Signal Magnitude — Stable Region Demarcated", loc="left")
    ax.legend(loc="upper right")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_stb, stable[:,0], color="#1a3a6b", lw=0.55)
    ax2.set_xlabel("Time  (s)"); ax2.set_ylabel("Acceleration  (g)")
    ax2.set_title("(b)  Stable Region — Ch_Z", loc="left")
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_stb, stable[:,1], color="#5a2a00", lw=0.55)
    ax3.set_xlabel("Time  (s)"); ax3.set_ylabel("Acceleration  (g)")
    ax3.set_title("(c)  Stable Region — Ch_X", loc="left")
    ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t_stb, stb_mag, color="#1a4a2a", lw=0.6)
    ax4.fill_between(t_stb, stb_mag, alpha=0.12, color="#1a4a2a")
    ax4.set_xlabel("Time  (s)"); ax4.set_ylabel("Magnitude  (g)")
    ax4.set_title("(d)  Stable Region — Vector Magnitude  √(Z²+X²+Y²)", loc="left")
    ax4.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax5 = fig.add_subplot(gs[2, 1])
    f_p, pxx = welch(stb_mag, fs=SAMPLING_RATE, nperseg=512)
    ax5.semilogy(f_p, pxx, color="#3a006b", lw=0.75)
    ax5.fill_between(f_p, pxx, alpha=0.10, color="#3a006b")
    ax5.set_xlabel("Frequency  (Hz)"); ax5.set_ylabel("PSD  (g²/Hz)")
    ax5.set_title("(e)  Power Spectral Density — Welch Method", loc="left")
    ax5.set_xlim(0, 500)
    ax5.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    return fig


def fig_predictions(preds, confs, class_names, ws):
    n = len(preds)
    t = np.arange(n) * ws / SAMPLING_RATE
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), facecolor="white",
                              gridspec_kw={"hspace": 0.5, "height_ratios": [1.1, 1]})
    colors = ["#1a4a2a" if p == 0 else "#8b1a1a" for p in preds]

    ax1 = axes[0]
    ax1.bar(t, [1]*n, width=ws/SAMPLING_RATE*0.82, color=colors, alpha=0.80, align="edge")
    ax1.set_xlim(0, t[-1] + ws/SAMPLING_RATE)
    ax1.set_ylim(0, 1.5); ax1.set_yticks([])
    ax1.set_xlabel("Time within stable region  (s)")
    ax1.set_title("(f)  Per-Window Classification — Green: IDLE, Red: Fault", loc="left")
    ax1.legend(handles=[Patch(facecolor="#1a4a2a", label="IDLE — Healthy"),
                         Patch(facecolor="#8b1a1a", label="Fault condition")], loc="upper right")
    step = max(1, n // 12)
    for i in range(0, n, step):
        ax1.text(t[i] + ws/SAMPLING_RATE*0.42, 1.08,
                 class_names.get(int(preds[i]), str(preds[i])),
                 ha="center", va="bottom", fontsize=6, rotation=45, color="#1a1a1a")

    ax2 = axes[1]
    ax2.bar(t, confs*100, width=ws/SAMPLING_RATE*0.82, color=colors, alpha=0.72, align="edge")
    ax2.axhline(50, color="#888", lw=0.8, ls=":", label="50% threshold")
    ax2.axhline(confs.mean()*100, color="#1a1a1a", lw=1.0, ls="--",
                label=f"Mean = {confs.mean()*100:.1f}%")
    ax2.set_xlim(0, t[-1] + ws/SAMPLING_RATE)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Time within stable region  (s)")
    ax2.set_ylabel("Confidence  (%)")
    ax2.set_title("(g)  Posterior Probability of Predicted Class per Window", loc="left")
    ax2.legend(loc="lower right")
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    return fig


def fig_heatmap(probs, model_classes, class_names):
    # probs columns correspond exactly to model_classes — no reordering needed
    row_labels = [class_names[int(cls)] for cls in model_classes]
    data       = probs * 100   # (n_windows, n_classes)
    n_classes  = len(model_classes)

    fig, ax = plt.subplots(figsize=(13, max(3.2, n_classes*0.52+1.5)), facecolor="white")
    im = ax.imshow(data.T, aspect="auto", cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Window index"); ax.set_ylabel("Class")
    ax.set_title("(h)  SVM Posterior Class Probabilities — All Windows (%)", loc="left")

    nw = data.shape[0]
    for c in range(n_classes):
        for w in range(nw):
            v = data[w, c]
            if v > 28:
                ax.text(w, c, f"{v:.0f}", ha="center", va="center",
                        fontsize=5.5, color="white" if v > 62 else "#1a1a1a")

    cbar = plt.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    cbar.set_label("Probability (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(0, nw, max(1, nw//15)))
    plt.tight_layout()
    return fig


def fig_features(X, names):
    key_idx  = [0, 6, 7, 10, 15, 17]
    key_lbls = [names[i] for i in key_idx]
    fig, axes = plt.subplots(1, 6, figsize=(13, 3.3), facecolor="white")
    fig.suptitle("(i)  Feature Distribution Across Extracted Windows",
                 fontsize=9, fontweight="bold", x=0.01, ha="left", y=0.98)
    for ax, idx, lbl in zip(axes, key_idx, key_lbls):
        ax.boxplot(X[:, idx], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#d8e4f0", linewidth=0.7),
                   medianprops=dict(color="#1a3a6b", linewidth=1.4),
                   whiskerprops=dict(linewidth=0.7),
                   capprops=dict(linewidth=0.7),
                   flierprops=dict(marker="o", markersize=2.5,
                                   markerfacecolor="#8b1a1a", alpha=0.55))
        ax.set_xticks([]); ax.set_title(lbl, fontsize=7.5, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, scaler, window_size, class_names = load_model()
    model_classes = model.classes_   # the ONLY reference used for indexing probs
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="journal-header">
    <div class="journal-kicker">Engine Health Monitoring · SVM-Based Fault Diagnosis · Accelerometer Data</div>
    <div class="journal-title">Vibrational Fault Diagnosis System</div>
    <div class="journal-subtitle">
        Automated classification of engine fault conditions from raw accelerometer recordings
        using Support Vector Machine with radial basis function kernel
    </div>
    <div class="journal-meta">
        Classifier: SVM (RBF) &nbsp;·&nbsp; Feature set: 18 time- and frequency-domain descriptors &nbsp;·&nbsp;
        Evaluation: Leave-One-File-Out Cross-Validation &nbsp;·&nbsp; LOFO-CV accuracy: 77.8% &nbsp;·&nbsp;
        Classes: {", ".join(class_names.values())} &nbsp;·&nbsp; Sampling rate: 1000 Hz
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# §1 INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-rule"><span class="sec-num">§1</span>Input — Upload Vibration Recording</div>',
            unsafe_allow_html=True)

col_up, col_spec = st.columns([2, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Select .txt vibration file (AX-3D accelerometer, semicolon-delimited)",
        type=["txt"],
        help="Format: TimeStamp;Ch_Z(g);Ch_X(g);Ch_Y(g)"
    )
with col_spec:
    st.markdown(f"""
    <div class="metric-panel blue">
        <div class="metric-key">Model Specification</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                    line-height:2; margin-top:8px; color:#444;">
            Kernel &nbsp;&nbsp;&nbsp;&nbsp;: RBF<br>
            Classes &nbsp;&nbsp;&nbsp;: {len(class_names)}<br>
            Features &nbsp;&nbsp;: {len(FEATURE_NAMES)}<br>
            Window &nbsp;&nbsp;&nbsp;&nbsp;: {window_size} samples<br>
            Scaler &nbsp;&nbsp;&nbsp;&nbsp;: StandardScaler<br>
            Decision &nbsp;&nbsp;: Majority vote
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file:

    prog = st.progress(0, text="Reading file…")

    try:
        raw = load_txt_file(uploaded_file.read())
    except Exception as e:
        st.error(f"File read error: {e}"); st.stop()

    raw_dur = len(raw) / SAMPLING_RATE
    prog.progress(20, text="Detecting stable region…")

    stable, mode, s_idx, e_idx = detect_stable_region(raw)
    stb_dur = len(stable) / SAMPLING_RATE
    removed = raw_dur - stb_dur

    prog.progress(40, text="Segmenting into windows…")
    windows = extract_windows(stable, window_size)
    n_win   = len(windows)
    if n_win == 0:
        st.error("Stable region too short to extract any complete windows."); st.stop()

    prog.progress(60, text="Extracting features…")
    X_win = np.array([extract_features(w) for w in windows])
    X_sc  = scaler.transform(X_win)

    prog.progress(80, text="Running SVM classifier…")

    # ── Core prediction — model.classes_ is the ONLY index reference ──────
    preds = model.predict(X_sc)
    probs = model.predict_proba(X_sc)   # shape: (n_windows, len(model_classes))
    # probs[:, i]  ←→  model_classes[i]   ALWAYS.  No reordering. No guessing.
    confs = probs.max(axis=1)

    votes      = Counter(preds)
    final_pred = int(votes.most_common(1)[0][0])
    final_name = class_names[final_pred]
    final_conf = float(confs.mean())
    is_idle    = (final_pred == 0)

    prog.progress(100, text="Analysis complete."); time.sleep(0.4); prog.empty()

    # ── §2 Signal summary ─────────────────────────────────────────────────
    st.markdown('<div class="section-rule"><span class="sec-num">§2</span>Signal Analysis Summary</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-panel">
            <div class="metric-key">Original Duration</div>
            <div class="metric-val">{raw_dur:.1f}<span class="metric-unit">s</span></div>
            <div class="metric-sub">{len(raw):,} samples @ 1000 Hz</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-panel blue">
            <div class="metric-key">Stable Region</div>
            <div class="metric-val">{stb_dur:.1f}<span class="metric-unit">s</span></div>
            <div class="metric-sub">{len(stable):,} samples retained</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-panel">
            <div class="metric-key">Transient Removed</div>
            <div class="metric-val">{removed:.1f}<span class="metric-unit">s</span></div>
            <div class="metric-sub">Start + stop buffers</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-panel">
            <div class="metric-key">Windows Extracted</div>
            <div class="metric-val">{n_win}<span class="metric-unit"> win</span></div>
            <div class="metric-sub">{window_size} samples · {window_size//SAMPLING_RATE} s each</div>
        </div>""", unsafe_allow_html=True)

    if mode == "fallback":
        st.markdown('<div class="notice-box">⚑ Note — Stable region detection used fallback mode. Full signal retained with buffer trimming only.</div>',
                    unsafe_allow_html=True)

    # ── §3 Result ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-rule"><span class="sec-num">§3</span>Classification Result</div>',
                unsafe_allow_html=True)

    box_cls  = "" if is_idle else "fault-box"
    ver_cls  = "" if is_idle else "fault-v"
    fill_cls = "g" if is_idle else "r"
    fill_el  = f'<div class="conf-fill-{fill_cls}" style="width:{final_conf*100:.1f}%"></div>'

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown(f"""
        <div class="result-box {box_cls}">
            <div class="result-rlabel">Majority-Vote Prediction &nbsp;({n_win} windows)</div>
            <div class="result-verdict {ver_cls}">{"✓" if is_idle else "✗"} &nbsp;{final_name}</div>
            <div style="font-size:0.82rem; color:#666; margin-top:6px; font-style:italic;">
                Mean posterior probability: &nbsp;<strong>{final_conf*100:.1f}%</strong>
            </div>
            <div class="conf-track">{fill_el}</div>
        </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="result-box" style="border-left-color:#c8c8c8;">', unsafe_allow_html=True)
        st.markdown('<div class="result-rlabel">Vote Distribution Across Windows</div>', unsafe_allow_html=True)
        for lbl, cnt in sorted(votes.items()):
            pct   = cnt / n_win * 100
            bar   = "█" * int(pct / 4)
            name  = class_names[int(lbl)]
            color = "#1a4a2a" if int(lbl) == 0 else "#8b1a1a"
            st.markdown(
                f'<div class="vote-row"><span style="color:{color}">{name:>8s}</span>'
                f'  <span style="color:#bbb">{bar:<25s}</span>'
                f'  {cnt:>3d}/{n_win} &nbsp;({pct:.0f}%)</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── §4 Signal visualisation ───────────────────────────────────────────
    st.markdown('<div class="section-rule"><span class="sec-num">§4</span>Signal Visualisation</div>',
                unsafe_allow_html=True)
    f1 = fig_signal_overview(raw, stable, s_idx, e_idx)
    st.pyplot(f1, use_container_width=True); plt.close(f1)
    st.markdown("""<div class="fig-caption">
        <span class="fig-label">Figure 1.</span>
        Signal overview. (a) Full raw vibration magnitude with stable region bounded by dashed vertical lines
        and shaded overlay. (b–c) Per-axis acceleration traces in the stable region (Ch_Z and Ch_X).
        (d) Vector magnitude in the stable region. (e) Power spectral density estimated via Welch's method (nperseg = 512).
    </div>""", unsafe_allow_html=True)

    # ── §5 Per-window classification ──────────────────────────────────────
    st.markdown('<div class="section-rule"><span class="sec-num">§5</span>Per-Window Classification</div>',
                unsafe_allow_html=True)
    f2 = fig_predictions(preds, confs, class_names, window_size)
    st.pyplot(f2, use_container_width=True); plt.close(f2)
    st.markdown("""<div class="fig-caption">
        <span class="fig-label">Figure 2.</span>
        Per-window classification results. (f) Predicted class label per window; green = IDLE, red = fault.
        (g) SVM posterior probability of the predicted class; dashed line = mean confidence.
    </div>""", unsafe_allow_html=True)

    # ── §6 Probability heatmap ────────────────────────────────────────────
    st.markdown('<div class="section-rule"><span class="sec-num">§6</span>Class Probability Heatmap</div>',
                unsafe_allow_html=True)
    f3 = fig_heatmap(probs, model_classes, class_names)
    st.pyplot(f3, use_container_width=True); plt.close(f3)
    st.markdown("""<div class="fig-caption">
        <span class="fig-label">Figure 3.</span>
        SVM posterior probability heatmap. Each column = one window; each row = one class.
        Cell annotations shown where probability exceeds 28%.
    </div>""", unsafe_allow_html=True)

    # ── §7 Feature distributions ──────────────────────────────────────────
    st.markdown('<div class="section-rule"><span class="sec-num">§7</span>Extracted Feature Distributions</div>',
                unsafe_allow_html=True)
    f4 = fig_features(X_win, FEATURE_NAMES)
    st.pyplot(f4, use_container_width=True); plt.close(f4)
    st.markdown("""<div class="fig-caption">
        <span class="fig-label">Figure 4.</span>
        Box plots of six diagnostic features across all windows.
        Centre line: median. Box: IQR. Whiskers: 1.5× IQR. Circles: outliers.
    </div>""", unsafe_allow_html=True)

    # ── §8 Tabulated results ──────────────────────────────────────────────
    st.markdown('<div class="section-rule"><span class="sec-num">§8</span>Tabulated Results</div>',
                unsafe_allow_html=True)

    with st.expander("Table 1 — Per-window prediction detail"):
        rows = []
        for i, (pred, conf, prob_row) in enumerate(zip(preds, confs, probs)):
            row = {
                "Window"       : i + 1,
                "t_start (s)"  : f"{i * window_size / SAMPLING_RATE:.2f}",
                "Prediction"   : class_names[int(pred)],
                "Confidence %" : f"{conf * 100:.1f}",
            }
            # prob_row[col_idx] maps to model_classes[col_idx] — guaranteed by sklearn
            for col_idx, cls in enumerate(model_classes):
                row[class_names[int(cls)]] = f"{prob_row[col_idx] * 100:.1f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Table 2 — Extracted feature matrix"):
        df = pd.DataFrame(X_win, columns=FEATURE_NAMES)
        df.insert(0, "Window", range(1, n_win + 1))
        st.dataframe(df.style.format("{:.4f}", subset=FEATURE_NAMES),
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="empty-state">
        <div style="font-family:'Crimson Pro',Georgia,serif; font-size:1.1rem; color:#888; font-style:italic;">
            Upload a vibration recording above to begin analysis.
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem;
                    color:#bbb; margin-top:10px; letter-spacing:1.5px;">
            ACCEPTED FORMAT &nbsp;·&nbsp; .TXT &nbsp;·&nbsp; SEMICOLON-DELIMITED &nbsp;·&nbsp; AX-3D SENSOR
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-rule" style="margin-top:2rem"><span class="sec-num">§A</span>Methodology Overview</div>',
                unsafe_allow_html=True)

    ma, mb, mc = st.columns(3)
    with ma:
        st.markdown("""<div class="metric-panel">
            <div class="metric-key">I. Pre-processing</div>
            <div style="font-size:0.8rem; line-height:2; margin-top:8px; color:#444;">
                1. Parse raw .txt recording<br>
                2. Compute vibration magnitude<br>
                3. Energy-based transient detection<br>
                4. Remove 10 s after engine start<br>
                5. Remove 5 s before engine stop
            </div>
        </div>""", unsafe_allow_html=True)
    with mb:
        st.markdown("""<div class="metric-panel blue">
            <div class="metric-key">II. Feature Extraction</div>
            <div style="font-size:0.8rem; line-height:2; margin-top:8px; color:#444;">
                Window: 1000 samples (1 s)<br>
                18 descriptors per window:<br>
                &nbsp;— Time domain: 10 features<br>
                &nbsp;— Frequency domain: 8 features<br>
                Welch PSD · 4 frequency bands<br>
                Scaling: StandardScaler
            </div>
        </div>""", unsafe_allow_html=True)
    with mc:
        st.markdown("""<div class="metric-panel">
            <div class="metric-key">III. Classification</div>
            <div style="font-size:0.8rem; line-height:2; margin-top:8px; color:#444;">
                Algorithm: SVM (scikit-learn)<br>
                Kernel: RBF<br>
                Output: class + probability<br>
                Decision: majority vote<br>
                Evaluation: LOFO-CV<br>
                Reported accuracy: 77.8%
            </div>
        </div>""", unsafe_allow_html=True)
