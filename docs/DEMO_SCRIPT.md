# VoxGuard: Viva Live Demonstration Script

This step-by-step walkthrough is designed for rehearsing your major project viva presentation. It guides you through presenting the operational pipeline to the examiners, focusing strictly on parts of the system backed by valid, real data.

---

> [!IMPORTANT]
> **CRITICAL WARNING FOR VIVA PRESENTATION (SCIENTIFIC HONESTY)**
> * **Do NOT Cite the Challenge Separation Gap Stats**: The dynamic challenge-response separation gap table in [RESULTS_SUMMARY.md](../docs/RESULTS_SUMMARY.md) is currently flagged as **INVALID - DO NOT CITE**. This is because the challenge database currently uses synthetic stand-ins (`gTTS` US and `pyttsx3` Microsoft David). 
> * **Mismatched Content Disclosure**: Under the fallback schema, these stand-ins read mismatched, unrelated sentences rather than actual phrase-matched speech cloned from your own voice.
> * **Examiner Defense Strategy**: If asked, state clearly: *"The current separation statistics are descriptive placeholders. The next planned engineering step is to record real human reference audio and integrate local voice cloning (e.g. Coqui TTS) to recompute phrase-matched separation bounds. We showcase the challenge logic live to demonstrate the context-binding verification mechanisms, but do not present the statistics as validated findings."*

---

## 🚀 Pre-Demo Checklist

1. Make sure Python virtual environment is active.
2. Launch the backend API:
   ```powershell
   python -m uvicorn src.pipeline.api:app --reload
   ```
3. Open `http://127.0.0.1:8000/` in a web browser.
4. Open the **Live Call Monitor** tab.

---

## 🎙️ Step-by-Step Viva Demonstration Walkthrough

### **SCENARIO 1: Clean Human Call (The Happy Path)**
* **Narration**: *"First, we demonstrate how a genuine human caller interacts with VoxGuard. We stream an authentic speech file from the ASVspoof 2019 logical access evaluation set."*
* **Action**: Click the **"Real Human Audio"** preset button on the dashboard.
* **Observe UI**:
  - Step 1 (Passive Monitor) lights up **Green** (`success`).
  - The circular gauge moves to **100%** (or near `99.88%`).
  - Status pill displays **AUTHENTIC**.
  - Step 2 and Step 3 remain bypassed since the caller is verified clean.
* **Narration**: *"The passive monitor extracts 120-dimensional Mel-Frequency Cepstral Coefficients (MFCCs). Because the signal contains natural human features, the classifier scores it at 99.88% authenticity, allowing the call to proceed without interruption."*

---

### **SCENARIO 2: Voice Clone Call (Passive Triggering)**
* **Narration**: *"Now, we simulate an incoming deepfake call attempting to bypass verification using a pre-recorded neural voice clone."*
* **Action**: Click the **"Voice Clone Spoof"** preset button.
* **Observe UI**:
  - Step 1 (Passive Monitor) flashes **Red** (`suspicious`).
  - Circular gauge drops to **0%** (or near `0.05%`).
  - Status pill shifts to **SUSPICIOUS**.
  - Step 2 (Dynamic Challenge) activates and prints a prompt: *"Please repeat the numbers: 8, 3, 9, 1, 4."*
* **Narration**: *"The passive monitor instantly detects synthetic spectral transitions (velocities/deltas). The score drops to 0.05%, triggering an alarm. The system dynamically issues a spontaneous vocal challenge to bind the caller's context."*

---

### **SCENARIO 3: Replay Attack Bypass Attempt (Blocked!)**
* **Narration**: *"An attacker attempting to bypass this challenge might try to replay a pre-recorded audio snippet of a real human. We will simulate this by submitting a human audio file saying a different sentence to bypass our challenge."*
* **Action**:
  1. Click **"Record Response"** (or let it load the test scenario from preset).
  2. The attacker submits a file where the human says numbers instead of answering the math question, or vice versa (mismatched).
  3. The local Whisper model transcribes the speech on our CPU.
* **Observe Console Logs**:
  - The console outputs: `Loading local Whisper 'tiny' model onto CPU...`
  - The transcriber output shows: `Actual Transcribed Text: 'please repeat the numbers 8, 3, 9, 1, 4.'`
  - The challenge expecting: *15 plus 27* (expects `42`).
  - **Verdict**: Step 3 (Fusion Verdict) flashes **Red** (`blocked`). Fused score drops to **6%**, and the console flags: `[REPLAY ALARM] Replay attack flagged: transcribed content mismatched challenge context!`
* **Narration**: *"Although the replayed audio consists of a genuine human voice (which would fool the passive detector on its own), our local offline Whisper engine transcribes the audio. Because the transcribed words do not contain the answer '42', the system detects the content alignment mismatch, drops the Content Adherence score to 0%, and blocks the call."*

---

### **SCENARIO 4: Robustness & Explainability Showcase**
* **Narration**: *"Finally, we show the examiner our research benchmarks and interpret the underlying model features."*
* **Action**: Click the **"Results & Benchmarks"** tab at the top.
* **Observe UI**:
  - Point to the **Calibrated Channel Benchmarks** table. Explain that threshold calibration on the Dev split successfully restored AMR narrowband accuracy from 21.3% up to 72.3% in lossy conditions.
  - Switch the **Confusion Matrix Toggle** (Clean ➔ AMR-NB ➔ Combined) to show how True Spoofs are blocked under varying noise levels.
  - Point to the **Feature Importances Chart**: Explain that **Delta-Delta (acceleration)** coefficients have the highest weights, proving that the model relies on capturing temporal micro-discontinuities in cloned speech envelopes rather than static vocal timbre.
