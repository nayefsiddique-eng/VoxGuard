# VoxGuard: Real-Time AI Voice Clone and Deepfake Call Verification System

VoxGuard is a real-time call verification system inspired by challenge-response security models (like the EuroS&P 2024 GOTCHA paper). It passively monitors live call audio streams to detect synthetic or cloned voices (deepfakes) and issues spontaneous vocal challenges (repeating random numbers, whispering, laughing) when a call footprint is flagged as suspicious.

![VoxGuard System Architecture Diagram](file:///docs/architecture_diagram.png)

---

## 🏗️ Architecture Overview

The system consists of a FastAPI backend and a dark-mode, glassmorphic HTML5 frontend dashboard:

1. **Passive Verification**: Extracts 120-dimensional Mel-Frequency Cepstral Coefficients (MFCCs: static means, standard deviations, deltas, and delta-deltas) from 1.5s rolling audio windows. Passes them to a speaker-independent Logistic Regression classifier to calculate the likelihood of being bonafide (human).
2. **Dynamic Challenge Engine**: If the passive score flags a potential spoof, the engine issues a spontaneous vocal challenge (e.g. whispering a specific sentence or answering a cognitive latency math question).
3. **Offline Whisper Verification**: Transcribes the response audio locally using an offline `openai-whisper` (tiny) model to verify content matches the issued challenge, neutralizing replay bypass attacks.
4. **ML Score Fusion**: A trained Logistic Regression fuser model combines the passive call score and the challenge adherence score, learning optimal decision boundaries to block call evasion or injection attempts.

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

# Generate Matplotlib plots (confusion matrices & architecture)
python -m scripts.generate_plots
```

### 4. Run the Web Dashboard
1. Spin up the FastAPI server:
   ```powershell
   python -m uvicorn src.pipeline.api:app --reload
   ```
2. Open `http://127.0.0.1:8000/` in your browser.
3. Use the presets to simulate bonafide or voice clone calls under varying codec conditions, or click **"Stream Live Mic"** to verify your own voice in real time over WebSockets!

---

## 📊 Evaluation & Research Results

### A. Clean Voice Spoof Classification
Our speaker-independent Logistic Regression classifier achieved an Equal Error Rate of **5.92% EER** and **0.9870 AUC-ROC** on the ASVspoof 2019 LA evaluation set. See [results_clean.md](file:///docs/results_clean.md).

### B. Calibrated Channel Degradation Matrix
Using **per-condition threshold calibration** swept on the held-out Dev split, accuracy drops were mitigated. The **GSM Codec** remains the single most severe degradation condition, raising the Equal Error Rate to **18.18%**:

| Channel Condition | Calibrated Threshold | Accuracy | EER | AUC | Accuracy Drop | EER Increase |
|---|---|---|---|---|---|---|
| **Clean Baseline** | 0.7943 | **80.9%** | **5.92%** | **0.9870** | - | - |
| AMR-NB Codec | 0.9791 | 72.3% | 12.65% | 0.9645 | 8.5% | +6.73% |
| Opus Codec | 0.1700 | 84.4% | 12.65% | 0.9660 | -3.5% | +6.73% |
| **GSM Codec** | 0.1863 | **76.6%** | **18.18%** | **0.9420** | **4.3%** | **+12.25%** |
| Low Loss (5% Loss / 10ms) | 0.0557 | 59.6% | 6.73% | 0.9835 | 21.3% | +0.80% |
| High Loss (15% Loss / 30ms) | 0.1241 | 78.7% | 11.45% | 0.9705 | 2.1% | +5.52% |
| **Combined Severe Telephony** | 0.9038 | **80.9%** | **12.25%** | **0.9310** | **0.0%** | **+6.33%** |

See [results_degraded.md](file:///docs/results_degraded.md).

### C. Visualizations
- **Feature Importance**: Coefficient analysis shows **Delta-Delta (acceleration)** coefficients dominate the weights, capturing temporal vocoder micro-discontinuities in deepfakes:
  ![Feature Importance Plot](file:///docs/feature_importance.png)
- **Matplotlib Confusion Matrices**: Visualizes block rates across Clean, AMR, and Combined channels:
  ![Confusion Matrices](file:///docs/confusion_matrix_clean.png)

---

## ⚠️ Known Limitations & Future Work

> [!WARNING]
> **DYNAMIC CHALLENGE SEPARATION STATISTICS WARNING**
> - The aggregate challenge-response separation gap statistics (Non-Verbal 23.8%, Prosody Shift 23.7%, Latency Probe 21.1%, Phrase Repetition 19.3%) currently use `gTTS` and `pyttsx3` stand-ins.
> - Under early CPU fallback conditions, these files read **mismatched content** (the human and "clone" speak unrelated sentences rather than the actual issued challenge phrase).
> - These statistics **should NOT be presented as a validated scientific finding** during your viva presentation. They are descriptive pipeline proofs.
> - **Planned Next Step**: Record real human reference audio and integrate local matching-content voice cloning models (e.g. Coqui TTS) to recompute authentic speaker-cloned separation gaps.
