# VoxGuard: Real-Time AI Voice Clone and Deepfake Call Verification System

### *Continuous Active Challenge-Response Authentication for Telephone Channel Security*

📦 [![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) ⚡ [![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688)](https://fastapi.tiangolo.com/) 🔥 [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/) 📄 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE) 🔬 [![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange)](#)

<p align="center">
  <img src="./docs/architecture_diagram.png" alt="VoxGuard: Real-Time AI Voice Clone and Deepfake Call Verification System Architecture" width="750">
  <br>
  <i>Figure 1: Modular dataflow pipeline of the VoxGuard system, tracking call frames from continuous passive scans to dynamic context-bound challenges.</i>
</p>

---

## 📖 Table of Contents
1. [Contextual Threat Model & Overview](#-1-contextual-threat-model--overview)
2. [Core Security Countermeasures](#-2-core-security-countermeasures)
3. [Interactive Web Console & User Interface](#-3-interactive-web-console--user-interface)
4. [Modular Verification Pipeline Subsystems](#-4-modular-verification-pipeline-subsystems)
5. [Empirical Evaluation Benchmarks](#-5-empirical-evaluation-benchmarks)
6. [Feature Space & Boundary Visualizations](#-6-feature-space--boundary-visualizations)
7. [Compilation & Deployment Guide](#-7-compilation--deployment-guide)
8. [Repository Filepath Schematics](#-8-repository-filepath-schematics)
9. [Signal Processing & Deep Learning Stack](#-9-signal-processing--deep-learning-stack)
10. [Research Context & Academic Citations](#-10-research-context--academic-citations)
11. [Limitations & Future Roadmap](#-11-limitations--future-roadmap)
12. [License](#-12-license)

---

## 📡 1. Contextual Threat Model & Overview

The **VoxGuard: Real-Time AI Voice Clone and Deepfake Call Verification System** is designed to mitigate the threat of live-call voice deepfakes. As generative AI speech models lower the barrier to high-fidelity voice cloning, impersonation fraud (including executive spoofing and distress-call scams) is transitioning from asynchronous formats to real-time voice communications. The system coordinates passive and active countermeasures to establish authentic context-binding on active telephone links.

Architecturally, the project adapts the video-centric **GOTCHA** (EuroS&P 2024) challenge-response paradigm to the acoustic voice domain. By employing continuous passive analysis alongside spontaneous, context-bound voice prompts, the framework forces adversarial generators to synthesize specific utterances on the fly. This introduces computational latency and content-mismatch failures that expose synthetic streams.

To address the limitations of typical lab-based detectors, this system evaluates EER bounds under realistic cell-channel compression. The framework incorporates simulated telephone line compression and network packet loss, presenting EER benchmarks that reflect active mobile deployments.

---

## 🛡️ 2. Core Security Countermeasures

| Countermeasure Subsystem | Technical Implementation |
|---|---|
| **Passive Real-Time Scoring** | Extracts 120-dimensional MFCC dynamic features analyzed via a speaker-independent Logistic Regression classifier to calculate authenticity scores. |
| **Dynamic Challenge-Response** | Generates dynamic verification prompts (digit sequences, whispered phrases, math latency probes) when the rolling passive score flags anomalies. |
| **Offline STT Verification** | Runs a local OpenAI Whisper model on CPU to transcribe user responses, ensuring zero reliance on external network APIs that could introduce proxy leakage. |
| **Degradation Simulator** | Evaluates baseline robustness under AMR-NB, GSM, and Opus codec compressions combined with simulated packet loss. |
| **Replay-Attack Defense** | Blocks pre-recorded human playback bypasses by validating that the transcribed text matches the issued challenge context. |
| **ML-Based Score Fusion** | Employs a trained Logistic Regression fuser that combines passive call authenticity and challenge content scores. |
| **Glassmorphic Interactive Dashboard** | A unified dark-mode dashboard providing preset simulators, WebSocket streams, and live logs. |

---

## 💻 3. Interactive Web Console & User Interface

The web interface acts as an administrative console for testing simulated and live calls:
- **ASVspoof Presets**: Users can click preset clean or spoofed audio files from the ASVspoof dataset to observe classifier behaviors.
- **WebSocket Streaming**: Transmits microphone chunks to the server every 500ms for continuous score updates.
- **State Trackers**: Guides the user visually through the pipeline phases (Passive Scan ➔ Challenge Active ➔ Fusion Verdict) as warnings trigger.

> [!NOTE]  
> **Headless Sandbox Environment Restriction**  
> Because the local execution sandbox operates without display drivers and lacks selenium or playwright dependencies, generating live browser screenshots dynamically is not possible. 
> 
> *Screenshots coming soon.*

For full execution details, refer to the [Live Viva Demonstration Script](./docs/DEMO_SCRIPT.md).

---

## ⚙️ 4. Modular Verification Pipeline Subsystems

The **VoxGuard: Real-Time AI Voice Clone and Deepfake Call Verification System** coordinates its verification pipeline across modular components in the source repository:

* **Audio Capture & Chunking**: Slices microphone input into 1.5-second windows or streams raw WebSocket frames. This is implemented in [streaming_capture.py](./src/capture/streaming_capture.py).
* **Spectral Feature Extraction**: Extracts MFCC static, velocity (delta), and acceleration (delta-delta) features, executing predictions using the saved model. This is implemented in [extract_features.py](./src/features/extract_features.py) and evaluated in [baseline_detector.py](./src/models/baseline_detector.py).
* **Prompt Challenge Dispatch**: Generates dynamic phrases and handles offline SAPI5/gTTS fallback voice synthesis. This is implemented in [phrase_challenge.py](./src/challenge_engine/phrase_challenge.py) and [generate_cloned_responses.py](./src/challenge_engine/generate_cloned_responses.py).
* **Context Verification (ASR)**: Transcribes incoming user responses locally on CPU using Whisper to check content alignment. This is implemented in [response_scorer.py](./src/challenge_engine/response_scorer.py).
* **ML Score Fusion**: Combines passive and active metrics using a Logistic Regression model. This is implemented in [fusion.py](./src/pipeline/fusion.py).
* **API Endpoints Coordination**: Serves static pages and exposes FastAPI endpoints for file uploads and WebSockets. This is implemented in [api.py](./src/pipeline/api.py).

---

## 📊 5. Empirical Evaluation Benchmarks

### A. Clean Voice Spoof Detection
Classifiers were evaluated on 800 speech files from the ASVspoof 2019 Logical Access (LA) database under strictly speaker-disjoint splits:

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

## 📈 6. Feature Space & Boundary Visualizations

### A. Spectral Coefficent Weight Distributions
<p align="center">
  <img src="./docs/feature_importance.png" alt="VoxGuard Feature Coefficient Magnitudes" width="680">
</p>

*Interpretation*: Static MFCCs (which capture speaker timbre) are easily spoofed by voice cloning models. VoxGuard relies instead on dynamic Delta/Delta-Delta coefficients. This indicates the model detects temporal vocoding artifacts (abrupt formant velocity shifts) to separate real human audio from clones.

### B. Confusion Matrix Projections
<p align="center">
  <img src="./docs/confusion_matrix_clean.png" alt="Clean CM" width="230">
  <img src="./docs/confusion_matrix_amr.png" alt="AMR CM" width="230">
  <img src="./docs/confusion_matrix_combined.png" alt="Combined CM" width="230">
</p>

*Interpretation*: Under clean conditions, the model has 0% false block rates for humans. AMR-NB compression shifts the score distribution, raising false blocks slightly, while combined lossy telephony causes minor leakage of spoof calls.

### C. Receiver Operating Characteristic (ROC) Curves
<p align="center">
  <img src="./docs/roc_curves.png" alt="VoxGuard ML Pipeline ROC Curves" width="680">
</p>

*Interpretation*: The ROC curves demonstrate the discrimination capacity of the classification pipeline under varying degrees of channel noise. The Area Under the Curve (AUC) remains high at 0.9870 in clean conditions, dropping gracefully to 0.9310 under severe, degraded combined channel loss and jitter profiles.

### D. ML Fuser Boundary Separations
<p align="center">
  <img src="./docs/fuser_scatter_plot.png" alt="ML Fuser Score Decision Boundary" width="680">
</p>

*Interpretation*: The scatter plot projects the decision boundary learned by our Logistic Regression score fuser. Authentic callers congregate in the upper-right quadrant (high passive score, high challenge adherence). Spoof attempts attempting replay bypasses fall along the bottom axis (low challenge alignment), showing clear linear classification boundaries.

---

> [!WARNING]
> **DYNAMIC CHALLENGE SEPARATION STATISTICS DISCLOSURE**
> - The aggregate challenge-response separation gap statistics (Non-Verbal 23.8%, Prosody 23.7%) currently utilize `gTTS` and `pyttsx3` synthetic stand-ins.
> - Under Windows CPU configurations, these stand-ins read **mismatched content** (the human and "clone" speak unrelated sentences rather than the actual issued challenge phrase).
> - These statistics **should NOT be presented as a validated scientific finding** during your viva. They represent descriptive pipeline proofs.
> - **Planned Next Step**: Record real human reference audio and integrate local matching-content voice cloning models (e.g. Coqui TTS) to recompute authentic speaker-cloned separation gaps.

---

## 🛠️ 7. Compilation & Deployment Guide

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

# Generate ROC Curves
python -m scripts.generate_roc

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
│   ├── fuser_scatter_plot.png
│   └── roc_curves.png
├── notebooks/
├── scripts/
│   ├── enforce_speaker_disjoint.py  # Fixes speaker leakage
│   ├── generate_fuser_plot.py       # Plots decision boundaries
│   ├── generate_plots.py            # Generates Matplotlib images
│   ├── generate_roc.py              # Plots ROC/AUC performance
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

## 🎛️ 9. Signal Processing & Deep Learning Stack

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

## 🎓 10. Research Context & Academic Citations

The **VoxGuard: Real-Time AI Voice Clone and Deepfake Call Verification System** is positioned as an academic implementation of dynamic speaker authentication. The pipeline architecture aligns with concepts presented in:
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

## 📄 12. License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
