# VoxGuard

### *Real-Time AI Voice Clone and Deepfake Call Verification System*

📦 [![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) ⚡ [![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688)](https://fastapi.tiangolo.com/) 🔥 [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/) 📄 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE) 🔬 [![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange)](#)

---

<p align="center">
  <img src="./docs/architecture_diagram.png" alt="VoxGuard System Architecture Diagram" width="100%">
</p>

---

## 📖 Table of Contents
1. [Contextual Verification & Threat Model](#-1-contextual-verification--threat-model)
2. [Core Architectural Countermeasures](#%EF%B8%8F-2-core-architectural-countermeasures)
3. [Interactive Web Console & Interface Schematics](#-3-interactive-web-console--interface-schematics)
4. [Verification Pipeline & Subsystem Mechanics](#%EF%B8%8F-4-verification-pipeline--subsystem-mechanics)
5. [Empirical Performance Benchmarks](#-5-empirical-performance-benchmarks)
6. [Feature Space & Decision Boundary Analysis](#-6-feature-space--decision-boundary-analysis)
7. [Installation & Local Deployment Guide](#%EF%B8%8F-7-installation-&-local-deployment-guide)
8. [Repository Filepath Schematics](#-8-repository-filepath-schematics)
9. [Deep Learning & Signal Processing Stack](#%F0%9F%8E%9B%EF%B8%8F-9-deep-learning--signal-processing-stack)
10. [Research Context & Related Work](#-10-research-context--related-work)
11. [Limitations & Future Roadmap](#-11-limitations--future-roadmap)
12. [License](#-12-mit-license)

---

## 📡 1. Contextual Verification & Threat Model

VoxGuard is an end-to-end communication security framework designed to mitigate the threat of live-call voice deepfakes. As generative AI models make high-fidelity voice cloning accessible, impersonation fraud (such as CEO fraud and distress scams) is transitioning from emails to real-time telephony channels. VoxGuard provides a defense-in-depth architecture that intercepts suspicious calls and verifies speaker authenticity.

The system is academically positioned as an adaptation of the **GOTCHA** (EuroS&P 2024) challenge-response paradigm, porting video authentication logic into the voice and audio domain. By using continuous passive call analysis in tandem with spontaneous, context-bound voice prompts, VoxGuard forces adversarial deepfake generators to synthesize specific utterances on the fly. 

Unlike traditional, laboratory-based detectors that assume clean, studio-recorded audio inputs, VoxGuard evaluates detection bounds under realistic cellular channel compressions. The framework incorporates simulated telephone line compression and network packet loss, presenting EER benchmarks that reflect active mobile deployments.

---

## 🛡️ 2. Core Architectural Countermeasures

* **Passive Real-Time Authenticity Scoring**: Analyzes 1.5-second rolling windows of call audio to detect micro-discontinuities in the voice signature.
* **Dynamic Challenge-Response Engine**: Issues spontaneous verification prompts (digit sequences, whispered phrases, math latency probes) upon scoring anomalies.
* **Offline Whisper-Based Content Verification**: Runs local Automatic Speech Recognition (ASR) to transcribe and verify user responses without network dependency.
* **Telephony Codec & Degradation Simulator**: Evaluates baseline robustness under AMR-NB, GSM, and Opus codec compressions combined with simulated packet loss.
* **Replay-Attack Defense**: Blocks pre-recorded human playback bypasses by validating that the transcribed text matches the issued challenge context.
* **ML-Based Score Fusion**: Employs a trained Logistic Regression fuser that combines passive call authenticity and challenge content scores.
* **Glassmorphic Interactive Dashboard**: A unified dark-mode dashboard providing preset simulators, WebSocket streams, and live logs.

---

## 💻 3. Interactive Web Console & Interface Schematics

The frontend dashboard provides a live visualization of the verification pipeline:
* **Interactive Presets**: Allows users to select clean or spoofed audio frames and apply simulated GSM, AMR-NB, or packet loss degradation on the fly.
* **Mic & WebSockets Stream**: Features a **"Stream Live Mic"** pipeline that slices microphone audio and transmits chunks to the server every 500ms over WebSockets.
* **Step-by-Step Flow Trackers**: Visually guides the user through the monitor state (Passive Scan ➔ Challenge Active ➔ Fusion Verdict).
* **Live Developer Logs**: Renders real-time backend debugging logs in a console output box for live viva inspection.

*(Dashboard screenshots coming soon)*

For a complete step-by-step walkthrough, refer to the [Live Viva Demonstration Script](./docs/DEMO_SCRIPT.md).

---

## ⚙️ 4. Verification Pipeline & Subsystem Mechanics

The verification pipeline operates across modular subsystems in the source repository:

* **Audio Capture & Chunking**: The audio stream is sliced into 1.5-second windows (or received as raw WebSocket frames). This is implemented in [streaming_capture.py](./src/capture/streaming_capture.py).
* **Spectral Feature Extraction**: Extracts 120-dimensional Mel-Frequency Cepstral Coefficients (MFCCs: static, delta, and delta-delta features) and scores authenticity. This is implemented in [extract_features.py](./src/features/extract_features.py) and evaluated in [baseline_detector.py](./src/models/baseline_detector.py).
* **Prompt Challenge Dispatch**: Selects random verification prompts and handles local synthesis. This is implemented in [phrase_challenge.py](./src/challenge_engine/phrase_challenge.py) and [generate_cloned_responses.py](./src/challenge_engine/generate_cloned_responses.py).
* **Context Verification (ASR)**: Transcribes the caller's response offline using local CPU weights, checking semantic context alignment. This is implemented in [response_scorer.py](./src/challenge_engine/response_scorer.py).
* **ML Score Fusion**: Combines passive caller scores and challenge alignment metrics using a Logistic Regression model. This is implemented in [fusion.py](./src/pipeline/fusion.py).
* **API Endpoints Coordination**: Exposes FastAPI endpoints coordinating streaming frames and returns verification logs. This is implemented in [api.py](./src/pipeline/api.py).

---

## 📊 5. Empirical Performance Benchmarks

### A. Clean Voice Spoof Detection
Baseline classifiers were evaluated on 800 speech files from the ASVspoof 2019 Logical Access (LA) database under strictly speaker-disjoint splits:

| Model | Dev Accuracy | Dev AUC | Dev EER | Eval Accuracy | Eval AUC | Eval EER |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 86.7% | 0.9384 | 15.62% | **80.9%** | **0.9870** | **5.92%** |
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

---

## 📈 6. Feature Space & Decision Boundary Analysis

### A. Spectral Coefficent Weight Distributions
<p align="center">
  <img src="./docs/feature_importance.png" alt="Feature Importance Plot" width="85%">
</p>

*Interpretation*: Static MFCCs (which capture speaker timbre) are easily spoofed by voice cloning models. VoxGuard relies instead on dynamic Delta/Delta-Delta coefficients. This indicates the model detects temporal vocoding artifacts (abrupt formant velocity shifts) to separate real human audio from clones.

### B. Confusion Matrix Projections
<p align="center">
  <img src="./docs/confusion_matrix_clean.png" width="32%">
  <img src="./docs/confusion_matrix_amr.png" width="32%">
  <img src="./docs/confusion_matrix_combined.png" width="32%">
</p>

*Interpretation*: Under clean conditions, the model has 0% false block rates for humans. AMR-NB compression shifts the score distribution, raising false blocks slightly, while combined lossy telephony causes minor leakage of spoof calls.

### C. ML Fuser Boundary Separations
<p align="center">
  <img src="./docs/fuser_scatter_plot.png" alt="ML Fuser Score Decision Boundary" width="85%">
</p>

*Interpretation*: The multi-dimensional scatter plot projects the decision boundary learned by our Logistic Regression score fuser. Authentic callers congregate in the upper-right quadrant (high passive score, high challenge adherence). Spoof attempts attempting replay bypasses fall along the bottom axis (low challenge alignment), showing clear linear classification boundaries.

---

> [!WARNING]
> **DYNAMIC CHALLENGE SEPARATION STATISTICS DISCLOSURE**
> - The aggregate challenge-response separation gap statistics (Non-Verbal 23.8%, Prosody 23.7%) currently utilize `gTTS` and `pyttsx3` synthetic stand-ins.
> - Under Windows CPU configurations, these stand-ins read **mismatched content** (the human and "clone" speak unrelated sentences rather than the actual issued challenge phrase).
> - These statistics **should NOT be presented as a validated scientific finding** during your viva. They represent descriptive pipeline proofs.
> - **Planned Next Step**: Record real human reference audio and integrate local matching-content voice cloning models (e.g. Coqui TTS) to recompute authentic speaker-cloned separation gaps.

---

## 🛠️ 7. Installation & Local Deployment Guide

### A. Prerequisites
- Python 3.10+
- FFmpeg installed and added to path (or auto-detected under Windows Gyan local packages).
- ~1.5 GB of free disk space (for dataset files and the local Whisper model weight downloads).

### B. Setup Execution
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

### C. Dataset Download
Download the ASVspoof 2019 logical access subset by executing the utility script:
```powershell
python -m src.utils.download_dataset_subset
```
*(Refer to [docs/DOWNLOAD_DATA.md](docs/DOWNLOAD_DATA.md) for data split specs).*

### D. Pipeline Verification Suite
```powershell
# Enforce speaker-disjoint partitions
python scripts/enforce_speaker_disjoint.py

# Retrain baseline model
python -m src.models.baseline_detector --features mfcc

# Run degradation matrices
python -m scripts.run_full_evaluation

# Test local Whisper replay blocks
python -m scripts.test_replay_attack

# Run integration tests
python tests/test_api.py
```

### E. Run the Web Dashboard
```powershell
python -m uvicorn src.pipeline.api:app --reload
```
Open `http://127.0.0.1:8000/` in your browser.

---

## 📂 8. Repository Filepath Schematics

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
│   ├── confusion_matrix_combined.png
│   └── fuser_scatter_plot.png
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

## 🎛️ 9. Deep Learning & Signal Processing Stack

| Component | Technology | Purpose |
|---|---|---|
| **API Backend** | FastAPI / Uvicorn | Coordinates streaming frames, websockets, and endpoints. |
| **Audio Processing** | Librosa / Soundfile | Handles MFCC extraction, resampling, and decimation. |
| **Codecs & Loss** | FFmpeg | Simulates telephony channel degradation (AMR, GSM, loss, jitter). |
| **Classifiers** | Scikit-learn | Trains Logistic Regression and Random Forest pipelines. |
| **Vocal Transcriber** | OpenAI Whisper | Local CPU instance transcribes response audio offline. |
| **Visualizations** | Matplotlib | Generates confusion matrices and explainability plots. |
| **Frontend UI** | HTML5 / CSS3 / WebSockets | Glassmorphic dashboard rendering stream logs and metrics. |

---

## 🎓 10. Research Context & Related Work

VoxGuard represents an academic implementation of voice clone authentication. The pipeline architecture aligns with concepts presented in:
* **GOTCHA**: *GOTCHA: Yoking Adversarial Speech Generators to Authenticate Speakers* (Mittal et al., EuroS&P 2024). Adapts adversarial speech generator yoking to live audio streams.
* **PITCH**: Dynamic pitch and prosody shifting metrics in communication channel security.
* **ASVspoof**: Logical access (LA) spoof benchmarks measuring vocoder footprints.

---

## 🔮 11. Limitations & Future Roadmap

* **Challenge-Response Validation**: Current separation gap statistics use synthetic stand-ins; real human voice testing is required to validate final EER metrics.
* **Local Voice Cloning Integration**: Integrate local voice synthesis engines (e.g., Coqui TTS) to generate phrase-matched cloned responses for custom speech prompts.
* **Streaming Latency Optimization**: Optimize PyTorch/Whisper CPU memory usage to decrease initial cold-boot loading latencies.
* **Telephony Deployment**: Interface with open telephony platforms (like Asterisk/VoIP gateways) to evaluate performance over actual carrier lines.

---

## 📄 12. MIT License

VoxGuard is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
