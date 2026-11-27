# 🎙️ VoxGuard

**Real-Time AI Voice Clone & Deepfake Call Verification System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-FF6F00.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VoxGuard** is a speaker-independent deepfake voice verification platform using challenge-response authentication and passive spectral analysis to detect AI voice clones and synthesized call attacks in real time. Built on the ASVspoof 2019 benchmark dataset, inspired by principles from *GOTCHA* (EuroS&P 2024).

---

## ⚡ Core Security Features

- **Continuous Passive Scoring**: Analyzes 1.5s sliding windows of call audio to detect micro-discontinuities in the voice signature.
- **Context-Bound Challenges**: Generates dynamic prompts (e.g., specific digit sequences or cognitive math queries) to verify active caller interaction.
- **Offline ASR Verification**: Transcribes responses locally using an offline OpenAI Whisper model on CPU, after the Whisper weights have been downloaded.
- **Codec & Channel Simulation**: Evaluates baseline robustness under standard telephony conditions (GSM, AMR-NB, and network packet loss).
- **Anti-Replay Defense**: Matches transcribed words against the challenge prompt to flag mismatched or potentially replayed audio.
- **ML Score Fusion**: Combines passive caller scores and active challenge content adherence using a trained Logistic Regression fuser.

---

## 🔄 Verification Pipeline

The verification pipeline operates across six distinct stages:

```mermaid
flowchart LR
    A[Audio Stream] --> B[MFCC Feature Extraction]
    B --> C[Passive Classifier]
    C -->|Score < Threshold| D[Challenge Trigger]
    D --> E[Offline Whisper ASR]
    E --> F[Logistic Regression Fuser]
    F --> G{Verdict: CLEAN / BLOCKED}
```

1. **Audio Capture**: Incoming call audio is captured and sliced into 1.5s windows or streamed as WebSocket frames via the administrative dashboard.
2. **Feature Extraction**: Extracts 120-dimensional Mel-Frequency Cepstral Coefficients (MFCCs: static, delta, and delta-delta features) to capture vocoder footprints.
3. **Passive Classifier**: Evaluates features against a speaker-independent Logistic Regression model to calculate an authenticity score.
4. **Challenge Trigger**: If the passive score falls below a calibrated threshold, the engine dispatches a prompt challenge (e.g., *Please repeat the numbers: 7, 2, 0, 5, 6*).
5. **ASR Content Scorer**: Transcribes the response using Whisper and calculates a content alignment score, blocking mismatching replayed files.
6. **Unified Verdict**: A fuser model combines the passive and content scores to issue a final decision (`CLEAN` or `BLOCKED`).

---

## 📊 Evaluation artifacts

The repository includes [the baseline evaluation](docs/results_clean.md),
[the channel degradation table](docs/results_degraded.md), and
[the dataset setup instructions](docs/DOWNLOAD_DATA.md). The dashboard uses
the checked-in results files. Reproducing the experiment requires the
ASVspoof 2019 data and documented preprocessing.

---

## 📈 Visualizations & Performance Analysis

### ROC Curves & Feature Coefficient Weights
<p align="center">
  <img src="docs/roc_curves.png" alt="Receiver Operating Characteristic Curves" width="48%">
  &nbsp;&nbsp;
  <img src="docs/feature_importance.png" alt="MFCC Feature Importance" width="48%">
</p>

- **ROC Projections**: The Receiver Operating Characteristic curve shows robust discrimination capacity, retaining an AUC of 0.9310 even under combined degradation profiles.
- **Feature Weights**: Dynamic coefficients (deltas and delta-deltas) account for over 90% of model weight, proving that the model identifies vocoder velocity discontinuities rather than static timbre.

### Score Fusion & Confusion Matrices
<p align="center">
  <img src="docs/fuser_scatter_plot.png" alt="Fuser Decision Boundary" width="60%">
</p>
<p align="center">
  <img src="docs/confusion_matrix_clean.png" alt="Clean CM" width="31%">
  <img src="docs/confusion_matrix_amr.png" alt="AMR CM" width="31%">
  <img src="docs/confusion_matrix_combined.png" alt="Combined CM" width="31%">
</p>

---

> ⚠️ **Challenge Separation Note**: The aggregate challenge-response separation gap statistics use gTTS/pyttsx3 stand-in voice pairs reading mismatched content under CPU fallbacks. These separation gap metrics are intended for pipeline verification.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- FFmpeg installed and added to PATH.
- ~1.5 GB of free disk space for model weights.

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/nayefsiddique-eng/VoxGuard.git
cd VoxGuard

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Download Dataset Subset
```powershell
python -m src.utils.download_dataset_subset
```

### 4. Run Benchmark Suite
```powershell
# Enforce speaker disjointness, retrain baselines, and evaluate codecs
python scripts/enforce_speaker_disjoint.py
python -m src.models.baseline_detector --features mfcc
python -m scripts.run_full_evaluation
python -m scripts.generate_roc
python tests/test_api.py
```

### 5. Start the Web Console
```powershell
python -m uvicorn src.pipeline.api:app --reload
```
Open `http://127.0.0.1:8000/` in your browser.

---

## 📁 Project Layout

```
VoxGuard/
├── data/                           # Challenge and placeholder WAVs
├── docs/                           # Performance logs and generated charts
├── scripts/                        # Dataset splitter and evaluator scripts
└── src/
    ├── capture/                    # Micro-chunk streaming capture
    ├── challenge_engine/           # Whisper scorer & phrase generator
    ├── features/                   # MFCC extraction scripts
    ├── models/                     # Saved pkl files & baseline trainer
    ├── pipeline/                   # FastAPI endpoints & score fuser
    └── static/                     # HTML dashboard frontend
```

---

## 🛠️ Signal Processing & Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| **API Server** | FastAPI / Uvicorn | Exposes endpoints and WebSocket stream pipelines. |
| **Audio Processing** | Librosa / Soundfile | Extracts MFCC features and handles audio resampling. |
| **Channel Simulation** | FFmpeg | Decimates, compresses, and simulates packet loss in audio. |
| **Machine Learning** | Scikit-learn | Trains classifier baselines and the score fuser. |
| **Vocal Transcriber** | OpenAI Whisper | Local CPU `tiny` instance transcribes speech offline. |
| **Visualizations** | Matplotlib | Generates ROC, feature weight, and boundary charts. |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

### Evaluation artifacts

The dashboard reads the versioned evaluation files in `docs/`. Results endpoints
return HTTP 503 when their expected artifacts are missing or unreadable.
Run `python -m pytest -q tests/test_results_api.py` after installing `pytest` and
`httpx` alongside the runtime dependencies to check this behavior without
downloading model weights or rerunning the ASVspoof evaluation.
