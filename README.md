# VoxGuard

### *Real-Time AI Voice Clone and Deepfake Call Verification System*

📦 [![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) ⚡ [![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688)](https://fastapi.tiangolo.com/) 📄 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT) 🔬 [![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange)](#)

---

![VoxGuard System Architecture Diagram](docs/architecture_diagram.png)

---

## 🎯 Project Overview

VoxGuard is a real-time call verification prototype designed to detect neural voice clones (deepfakes) in live communication streams. Inspired by the **GOTCHA** (EuroS&P 2024) challenge-response framework, the system combines continuous passive voice analysis with spontaneous vocal challenges (repeating numbers, whispering, cognitive math questions) to authenticate callers and block replay attacks.

This project was developed as a university coursework project demonstrating IoT-focused communication security.

---

## 🚀 Key Features

* **Passive Real-Time Verification**: Continuous extraction of 120-dimensional MFCC dynamic features analyzed via a speaker-independent classifier.
* **Dynamic Challenge-Response**: Automatically triggers spontaneous prompts upon detecting anomalies, requesting user verbal responses.
* **Offline STT Verification**: Transcribes responses completely offline using a local OpenAI Whisper model, checking semantic context alignment.
* **Robustness Evaluation Matrix**: Benchmarked under telephony codecs (AMR-NB, GSM), codec decimation, and network packet loss.
* **Interactive Live Dashboard**: A dark-theme, glassmorphic UI displaying presets, streaming logs, and real-time WebSocket mic inputs.

---

## 💻 Live Dashboard Demo

The interactive Web UI allows you to test preset ASVspoof logical access audio frames under simulated degradation profiles, or stream your microphone live:

*(Dashboard screenshot placeholder — refer to local workspace dashboard preview)*

---

## 🧠 System Architecture

The verification pipeline operates in three distinct phases:
1. **Passive Feature Extraction**: Mel-Frequency Cepstral Coefficients (MFCC static, delta velocity, and delta-delta acceleration) are extracted from 1.5s windows.
2. **Context-Bound Challenge**: Whisper transcribes the response offline, matching speech content against the issued challenge.
3. **Score Fusion**: A Logistic Regression fuser combines passive caller authenticity and challenge alignment scores to output a unified validation verdict.

---

## 📊 Research Results & Benchmarks

### A. Clean Voice Spoof Detection
Baseline classifiers were evaluated on 800 speech files from the ASVspoof 2019 Logical Access (LA) database under strictly speaker-disjoint splits:

| Model | Dev Accuracy | Dev AUC | Dev EER | Eval Accuracy | Eval AUC | Eval EER |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **86.7%** | **0.9384** | **15.62%** | **80.9%** | **0.9870** | **5.92%** |
| Multi-Layer Perceptron (MLP) | 84.9% | 0.9329 | 13.75% | 77.3% | 0.9400 | 12.65% |
| Random Forest | 85.8% | 0.9077 | 16.02% | 79.4% | 0.9565 | 11.85% |

### B. Calibrated Degradation Benchmarks
Evaluated on the Eval split using condition-specific threshold calibration swept on the held-out Dev split:

| Channel Condition | Calibrated Threshold | Accuracy | EER | AUC | Accuracy Drop | EER Increase |
|---|---|---|---|---|---|---|
| **Clean Baseline** | 0.7943 | **80.9%** | **5.92%** | **0.9870** | - | - |
| AMR-NB Codec | 0.9791 | 72.3% | 12.65% | 0.9645 | 8.5% | +6.73% |
| Opus Codec | 0.1700 | 84.4% | 12.65% | 0.9660 | -3.5% | +6.73% |
| **GSM Codec** | 0.1863 | **76.6%** | **18.18%** | **0.9420** | **4.3%** | **+12.25%** |
| Low Loss (5% Loss / 10ms Jitter) | 0.0557 | 59.6% | 6.73% | 0.9835 | 21.3% | +0.80% |
| High Loss (15% Loss / 30ms Jitter) | 0.1241 | 78.7% | 11.45% | 0.9705 | 2.1% | +5.52% |
| **Combined Severe Telephony** | 0.9038 | **80.9%** | **12.25%** | **0.9310** | **0.0%** | **+6.33%** |

### C. Visualizations
- **Feature Importance**: Dynamic delta/delta-delta envelopes capture temporal vocoder artifacts, contributing over 90% of model weight:
  ![Feature Importance Plot](docs/feature_importance.png)
- **Confusion Matrices**: Evaluating block rates under Clean, AMR-NB, and Combined degradation:
  <p align="center">
    <img src="docs/confusion_matrix_clean.png" width="31%">
    <img src="docs/confusion_matrix_amr.png" width="31%">
    <img src="docs/confusion_matrix_combined.png" width="31%">
  </p>

---

> [!WARNING]
> **DYNAMIC CHALLENGE SEPARATION STATISTICS DISCLOSURE**
> - The aggregate challenge-response separation gap statistics (Non-Verbal 23.8%, Prosody 23.7%) currently utilize `gTTS` and `pyttsx3` synthetic stand-ins.
> - Under Windows CPU configurations, these stand-ins read **mismatched content** (the human and "clone" speak unrelated sentences rather than the actual issued challenge phrase).
> - These statistics **should NOT be presented as a validated scientific finding** during your viva. They represent descriptive pipeline proofs.
> - **Planned Next Step**: Record real human reference audio and integrate local matching-content voice cloning models (e.g. Coqui TTS) to recompute authentic speaker-cloned separation gaps.

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install requirements inside a virtual environment:
```powershell
# Clone repo
git clone https://github.com/nayefsiddique-eng/VoxGuard.git
cd VoxGuard

# Create venv and activate
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Download
Download the ASVspoof 2019 logical access subset by executing the utility script:
```powershell
python -m src.utils.download_dataset_subset
```
*(Refer to [docs/DOWNLOAD_DATA.md](docs/DOWNLOAD_DATA.md) for data split specs).*

### 3. Run Pipeline Benchmarks
```powershell
# Enforce speaker-disjoint partitions
python scripts/enforce_speaker_disjoint.py

# Retrain baseline model
python -m src.models.baseline_detector --features mfcc

# Run degradation matrices
python -m scripts.run_full_evaluation

# Test local Whisper replay blocks
python -m scripts.test_replay_attack
```

### 4. Run the Web Dashboard
```powershell
python -m uvicorn src.pipeline.api:app --reload
```
Open `http://127.0.0.1:8000/` in your browser.

---

## 📂 Project Structure

```
VoxGuard/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── challenges/                 # Paired dynamic challenge WAVs
│   └── placeholder/                # Synthetic baseline test waves
├── docs/
│   ├── BASE_STATE.md
│   ├── DEMO_SCRIPT.md
│   ├── DOWNLOAD_DATA.md
│   ├── RESULTS_SUMMARY.md
│   ├── results_clean.md
│   ├── results_degraded.md
│   ├── architecture_diagram.png
│   ├── feature_importance.png
│   ├── confusion_matrix_clean.png
│   ├── confusion_matrix_amr.png
│   └── confusion_matrix_combined.png
├── notebooks/
├── scripts/
│   ├── enforce_speaker_disjoint.py  # Fixes speaker leakage
│   ├── generate_fuser_plot.py       # Plots decision boundaries
│   ├── generate_plots.py            # Generates Matplotlib images
│   └── run_full_evaluation.py       # Degradation evaluator
└── src/
    ├── capture/
    │   └── streaming_capture.py     # Captures mic audio chunks
    ├── challenge_engine/
    │   ├── generate_cloned_responses.py # Offline challenge synth
    │   ├── phrase_challenge.py      # Category phrase pools
    │   └── response_scorer.py       # Offline Whisper verify
    ├── features/
    │   ├── extract_features.py      # MFCC caching extractor
    │   └── load_dataset.py          # Data ingestion parser
    ├── models/
    │   ├── baseline_detector.py     # Classifiers training
    │   ├── detector.pkl             # Trained detector model
    │   └── fuser.pkl                # Score fusion fuser model
    ├── pipeline/
    │   ├── api.py                   # FastAPI server endpoints
    │   ├── degrade_audio.py         # FFmpeg channel simulators
    │   └── fusion.py                # Fuser math algorithms
    └── static/
        └── index.html               # Frontend dashboard
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Core Languages** | Python, HTML5, Javascript |
| **Frameworks & API** | FastAPI, Uvicorn, WebSockets |
| **Audio Processing** | Librosa, Soundfile, FFmpeg |
| **Machine Learning** | Scikit-learn (Logistic Regression, Random Forest) |
| **Speech Verification** | OpenAI Whisper (Local tiny CPU instance) |
| **Visualizations** | Matplotlib |

---

## 🎓 Research Context

VoxGuard represents an academic implementation of voice clone authentication. The pipeline architecture aligns with concepts presented in:
* **GOTCHA**: *GOTCHA: Yoking Adversarial Speech Generators to Authenticate Speakers* (Mittal et al., EuroS&P 2024).
* **PITCH**: Dynamic pitch and prosody shifting metrics in communication channel security.

---

## 🗺️ Roadmap & Future Work

* [ ] Collect real human verification recordings to replace synthetic stand-ins.
* [ ] Integrate fully localized voice cloning synthesis (e.g. Coqui TTS).
* [ ] Perform low-level optimization for real-time mobile websocket streams.
* [ ] Port model weights for native on-device edge verification.

---

## 📄 License & Author

* **License**: MIT License
* **Author**: nayefsiddique-eng
* **Context**: 7th Semester Major Project, IoT Department, University Coursework.
