# VoxGuard: Real-Time AI Voice Clone and Deepfake Call Verification System

VoxGuard is a real-time call verification system inspired by challenge-response security models (like the EuroS&P 2024 GOTCHA paper). It passively monitors live call audio streams to detect synthetic or cloned voices (deepfakes) and issues spontaneous vocal challenges (repeating random numbers, whispering, laughing) when a call footprint is flagged as suspicious. 

---

## 🏗️ Architecture Overview

The system consists of a FastAPI backend and a dark-mode, glassmorphic HTML5 frontend dashboard:

```
[ Incoming Live Call Stream ]
              │
              ▼
    ┌───────────────────┐      Passive Verification
    │  Passive Monitor  ├──────────────────────────────┐
    │  (Mel-Freq MFCC)  │                              │
    └─────────┬─────────┘                              ▼
              │                             ┌───────────────────┐
              │ If score < 0.50 (Suspicious)│  ML Score Fuser   │──► [ Authenticated/Blocked ]
              ▼                             │   (Logistic Reg)  │
    ┌───────────────────┐                   └───────────────────┘
    │  Challenge Engine │                              ▲
    │ (Offline Whisper) │                              │
    └─────────┬─────────┘                              │
              │                                        │ Fused Score
              ▼                                        │
    [ Verification Prompt ] ───────────────────────────┘
     (e.g., "Answer: 15 + 27")
```

1. **Passive Verification**: Extracts 120-dimensional MFCC dynamic features (means, standard deviations, deltas, and delta-deltas) from 1.5s rolling audio windows. Passes them to a speaker-independent Logistic Regression classifier to calculate the likelihood of being bonafide (human).
2. **Dynamic Challenge Engine**: If the passive score flags a potential spoof, the engine issues a spontaneous vocal challenge (e.g. whispering a specific sentence or answering a cognitive latency math question).
3. **Offline Whisper Verification**: Transcribes the response audio locally using an offline `openai-whisper` (tiny) model to verify content matches the issued challenge, neutralizing replay bypass attacks.
4. **ML Score Fusion**: A trained Logistic Regression fuser model combines the passive call score and the challenge adherence score, learning optimal decision boundaries to block call evasion or injection attempts.

---

## 📊 Key Results Summary

* **Clean Voice Spoof Classification**: Our speaker-independent Logistic Regression classifier achieved an Equal Error Rate of **5.92% EER** and **0.9870 AUC-ROC** on the ASVspoof 2019 LA evaluation set. See [results_clean.md](file:///docs/results_clean.md).
* **Telephony & Network Degradation Gaps**: We benchmarked performance under simulated cellular lines (AMR-NB, GSM) and network packet loss. In lossy channels, EER increases to **12.65%** (AMR/Opus) and **18.18%** (GSM). Threshold calibration on held-out Dev sets successfully restored accuracy (AMR accuracy rose to 72.3%). See [results_degraded.md](file:///docs/results_degraded.md).
* **Feature Importance**: Coefficient analysis reveals that **Delta-Delta (acceleration)** and **Delta (velocity)** spectral features contribute the most to classifier decisions (combining for ~90% of model weight), proving deepfake detectors rely on temporal vocoder transitions. Visualized at [feature_importance.png](file:///docs/feature_importance.png).
* **Challenge Separation Gaps**: Tests demonstrate a stable ~22% to 31% separation gap between synthetic stand-in voices. Note: These metrics are currently flagged as descriptive pending real human audio testing. See the [Challenge-Response section of RESULTS_SUMMARY.md](file:///docs/RESULTS_SUMMARY.md#3-dynamic-challenge-response-class-separation-gaps).

---

## 🚀 Setup & Execution

### 1. Prerequisites
- Python 3.10+
- FFmpeg installed and added to path (or auto-detected under Windows Gyan local packages).

### 2. Installation
Clone the repository and install dependencies inside a virtual environment:
```powershell
# Clone repo
git clone https://github.com/nayefsiddique-eng/VoxGuard.git
cd VoxGuard

# Create venv and activate
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```
*(Dependencies: fastapi, uvicorn, soundfile, librosa, scikit-learn, joblib, speechrecognition, openai-whisper, matplotlib, gTTS, pyttsx3)*

### 3. Run Pipeline Benchmarks & Tests
```powershell
# Enforce strictly speaker-independent splits (disjoint speakers)
python scripts/enforce_speaker_disjoint.py

# Retrain baseline classifiers
python -m src.models.baseline_detector --features mfcc

# Run degradation evaluation matrix
python -m scripts.run_full_evaluation

# Test local Whisper replay attack checks
python -m scripts.test_replay_attack
```

### 4. Run the Web Dashboard
1. Spin up the FastAPI server:
   ```powershell
   python -m uvicorn src.pipeline.api:app --reload
   ```
2. Open `http://127.0.0.1:8000/` in your browser.
3. Use the presets to simulate bonafide or voice clone calls under varying codec conditions, or click **"Stream Live Mic"** to verify your own voice in real time over WebSockets!
