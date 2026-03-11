# 📡 VibroSense — Engine Fault Detector

A Streamlit web app that classifies engine faults from 3-axis accelerometer data using a trained SVM model.

**Upload one file → get instant fault classification + signal charts.**

---

## 🚀 Deploy in 4 Steps

### Step 1 — Train your model locally

Make sure you have the `Analysis Files/` folder with your training `.txt` files.

```bash
pip install -r requirements.txt
python train_and_export.py
```

This creates `model_artifacts.pkl` (~a few MB).

---

### Step 2 — Create a GitHub repository

1. Go to [github.com](https://github.com) → **New repository**
2. Name it something like `vibrosense-fault-detector`
3. Set it to **Public** (required for free Streamlit Cloud)
4. Click **Create repository**

Then push your files:

```bash
git init
git add app.py requirements.txt train_and_export.py model_artifacts.pkl .gitignore README.md
git commit -m "Initial commit — VibroSense fault detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vibrosense-fault-detector.git
git push -u origin main
```

> ⚠️ **Important:** `model_artifacts.pkl` must be committed. It's the trained model the app loads.

---

### Step 3 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/vibrosense-fault-detector`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**

Streamlit Cloud will install `requirements.txt` automatically and your app will be live in ~2 minutes.

---

### Step 4 — Share your app URL

Your app will be at:
```
https://YOUR_USERNAME-vibrosense-fault-detector-app-XXXXXX.streamlit.app
```

---

## 📁 File Structure

```
vibrosense-fault-detector/
├── app.py                  ← Streamlit app (UI + prediction pipeline)
├── train_and_export.py     ← Run locally to train & save model
├── model_artifacts.pkl     ← Trained model (commit this!)
├── requirements.txt        ← Python dependencies for Streamlit Cloud
├── .gitignore
└── README.md
```

---

## 🔬 How the Prediction Works

| Step | What happens |
|------|-------------|
| Parse file | Auto-detects TimeStamp header or falls back to skiprows=20 |
| Magnitude | `√(X² + Y² + Z²)` computed from 3 axes |
| Alignment | Cross-correlation shift vs NoFault1 reference signal |
| Transient removal | First 5 seconds discarded |
| Segmentation | 1-second (1000-sample) windows |
| Features | 14 per segment: RMS, Variance, Std, Peak, PTP, Skewness, Kurtosis, Crest Factor, Dominant Freq, Spectral Centroid, 4 band power values |
| Scaling | StandardScaler (fitted on training data) |
| Classification | SVM (RBF kernel, GridSearchCV-tuned) |
| Final output | Majority vote across all segments → fault label + confidence |

---

## 📊 What the App Displays

- **Predicted fault** (color-coded: green = healthy, red = fault)
- **Confidence gauge** (% of segments voting for predicted class)
- **6 signal statistics** — RMS, Peak, Std, Skewness, Kurtosis, Dominant Frequency
- **Magnitude time series** — full processed signal
- **Power Spectral Density** — Welch method, 0–200 Hz
- **Segment distribution** — bar chart of how each segment voted
- **Dominant frequency per segment** — trend over time
- **Raw JSON** — expandable full results

---

## 📄 Input File Format

| Property | Requirement |
|----------|-------------|
| Extension | `.txt` or `.csv` |
| Delimiter | Semicolon (`;`) |
| Header | `TimeStamp` row (auto-detected) or fixed 20-row header |
| Columns | `Time ; Z ; X ; Y` |
| Filename | Any — not used for prediction |
| Length | Minimum 6 seconds |
| Sampling rate | 1000 Hz |

---

## 🏷️ Fault Classes

| Label | Meaning |
|-------|---------|
| NoFault | Engine healthy |
| Fault1–Fault9 | 9 distinct fault conditions |

---

## Updating the Model

If you retrain with new data:
```bash
python train_and_export.py     # produces new model_artifacts.pkl
git add model_artifacts.pkl
git commit -m "Retrain model with new data"
git push
```
Streamlit Cloud redeploys automatically on push.
