# VoxGuard

### *Real-Time AI Voice Clone & Call Verification System*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status: Prototype](https://img.shields.io/badge/Status-Prototype-orange.svg)](#)

VoxGuard is a real-time call verification framework designed to detect AI voice clones (deepfakes) and block replay attacks. By combining continuous passive classification with dynamic vocal challenges, the system binds the caller's context to verify authenticity on active audio links.

---

<p align="center">
  <img src="docs/architecture_diagram.png" alt="VoxGuard Pipeline Architecture" width="720">
</p>

---

## ⚙️ How It Works

VoxGuard implements a multi-layered verification pipeline:

1. **Continuous Passive Scan**: Slices incoming call audio into 1.5s windows, extracts MFCC features, and calculates an authenticity score using a speaker-independent Logistic Regression model.
2. **Context-Bound Challenge**: If the passive score drops below safety thresholds, the system issues a spontaneous challenge prompt (such as repeating a sequence of digits or answering a cognitive math question).
3. **Local Offline Transcription**: Transcribes the response offline using a local OpenAI Whisper model. It verifies that the spoken content matches the expected challenge answer, blocking pre-recorded replay attacks.
4. **ML Score Fusion**: A fuser model combines the passive scoring metrics and challenge content adherence score to render a final call verdict (CLEAN or BLOCKED).

---

## 📊 Empirical Benchmarks

### Spoof Detection Classifiers
Evaluated on the ASVspoof 2019 logical access evaluation set:

| Model | Dev Accuracy | Dev EER | Eval Accuracy | Eval EER | Eval AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Saved Baseline)** | **86.7%** | **15.62%** | **80.9%** | **5.92%** | **0.9870** |
| Multi-Layer Perceptron (MLP) | 84.9% | 13.75% | 77.3% | 12.65% | 0.9400 |
| Random Forest | 85.8% | 16.02% | 79.4% | 11.85% | 0.9565 |

### Telephony Channel Robustness
Performance under simulated cellular compression and network packet loss:

| Channel Condition | Calibrated Threshold | Accuracy | EER | AUC |
|---|---|---|---|---|
| **Clean Baseline** | 0.7943 | 80.9% | 5.92% | 0.9870 |
| AMR-NB Telephony Codec | 0.9791 | 72.3% | 12.65% | 0.9645 |
| GSM Cellular Codec | 0.1863 | 76.6% | 18.18% | 0.9420 |
| Low Loss (5% Loss, 10ms Jitter) | 0.0557 | 59.6% | 6.73% | 0.9835 |
| High Loss (15% Loss, 30ms Jitter) | 0.1241 | 78.7% | 11.45% | 0.9705 |
| **Combined Severe Telephony** | 0.9038 | 80.9% | 12.25% | 0.9310 |

---

## 📈 Visualizations

### ROC Curve & Feature Importance
<p align="center">
  <img src="docs/roc_curves.png" alt="ROC Performance Curves" width="45%">
  &nbsp;&nbsp;
  <img src="docs/feature_importance.png" alt="Feature Importance Weights" width="45%">
</p>

### Confusion Matrices
<p align="center">
  <img src="docs/confusion_matrix_clean.png" alt="Clean CM" width="30%">
  <img src="docs/confusion_matrix_amr.png" alt="AMR CM" width="30%">
  <img src="docs/confusion_matrix_combined.png" alt="Combined CM" width="30%">
</p>

---

## 🛠️ Getting Started

### 1. Installation & Environment Setup
Ensure you have Python 3.10+ and FFmpeg installed on your system.
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

### 2. Download Dataset
```powershell
python -m src.utils.download_dataset_subset
```

### 3. Run Benchmark Scripts
```powershell
# Run the pipeline tests and retrain baselines
python scripts/enforce_speaker_disjoint.py
python -m src.models.baseline_detector --features mfcc
python -m scripts.run_full_evaluation
python -m scripts.generate_roc
python tests/test_api.py
```

### 4. Run the Web Dashboard
```powershell
python -m uvicorn src.pipeline.api:app --reload
```
Open your browser and navigate to `http://127.0.0.1:8000/`.

---

## 📂 Project Structure

```
VoxGuard/
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                           # Challenge and placeholder WAVs
├── docs/                           # Performance logs and generated charts
├── scripts/                        # Dataset splitter and evaluator scripts
├── src/
│   ├── capture/                    # Micro-chunk streaming capture
│   ├── challenge_engine/           # Whisper scorer & phrase generator
│   ├── features/                   # MFCC extraction scripts
│   ├── models/                     # Saved pkl files & baseline trainer
│   ├── pipeline/                   # FastAPI endpoints & score fuser
│   └── static/                     # HTML dashboard frontend
└── tests/                          # API smoke tests
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
