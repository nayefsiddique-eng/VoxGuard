# Empirical Research Results Summary (Validated & Localized)
**Project Title**: Real-Time AI Voice Clone and Deepfake Call Verification System
**Department**: IoT / Communication Engineering Major Project

This report documents the final empirical performance metrics of our challenge-response call verification system, trained and evaluated on **800 speech files** from the **ASVspoof 2019 Logical Access (LA)** database under strictly **speaker-disjoint** partitions (zero speaker leakage across splits).

---

## 1. Clean Audio Spoof Detection Benchmarks
Classifiers were trained on 120-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) including delta and double-delta derivatives. Evaluated on the strictly speaker-independent Eval split:

| Model | Dev Accuracy | Dev AUC | Dev EER | Eval Accuracy | Eval AUC | Eval EER |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 86.7% | 0.9384 | 15.62% | **80.9%** | **0.9870** | **5.92%** |
| Multi-Layer Perceptron (MLP) | 84.9% | 0.9329 | 13.75% | 77.3% | 0.9400 | 12.65% |
| Random Forest | 85.8% | 0.9077 | 16.02% | 79.4% | 0.9565 | 11.85% |

> [!NOTE]
> **Logistic Regression** generalized best under speaker-independent validation, achieving an Equal Error Rate of **5.92%**. It was saved as the primary voice authenticity classifier (`detector.pkl`).

---

## 2. Calibrated Channel Degradation Benchmarks
Evaluated on the Eval split using **per-condition threshold calibration** swept on the held-out Dev split:

| Channel Condition | Calibrated Threshold | Accuracy | EER | AUC | Accuracy Drop | EER Increase |
|---|---|---|---|---|---|---|
| **Clean Baseline** | 0.7943 | **80.9%** | **5.92%** | **0.9870** | - | - |
| AMR-NB Codec | 0.9791 | 72.3% | 12.65% | 0.9645 | 8.5% | +6.73% |
| Opus Codec | 0.1700 | 84.4% | 12.65% | 0.9660 | -3.5% | +6.73% |
| **GSM Codec (Worst Single)** | 0.1863 | **76.6%** | **18.18%** | **0.9420** | **4.3%** | **+12.25%** |
| Low Loss (5% Loss / 10ms Jitter) | 0.0557 | 59.6% | 6.73% | 0.9835 | 21.3% | +0.80% |
| High Loss (15% Loss / 30ms Jitter) | 0.1241 | 78.7% | 11.45% | 0.9705 | 2.1% | +5.52% |
| **Combined Severe Telephony** | 0.9038 | **80.9%** | **12.25%** | **0.9310** | **0.0%** | **+6.33%** |

---

## 3. Dynamic Challenge-Response Class Separation Gaps

> [!WARNING]
> **PRELIMINARY DATA ONLY - DO NOT CITE AS VALIDATED VIVA FINDINGS**
> The table below is based on synthetic stand-in voices (gTTS as human, pyttsx3 as clone) rather than real human voice recordings paired with custom voice clones. It is presented solely for descriptive pipeline verification and should not be cited as a validated baseline statistical result.

**Metric Definition**: The "Separation Gap" represents the statistical difference (mean ± standard deviation) computed across **N=10 distinct pairs** per category.

| Challenge Type | Human Score (Mean ± SD) | Cloned Score (Mean ± SD) | **Separation Gap (Mean ± SD)** |
|---|---|---|---|
| **Non-Verbal Sound (Laugh)** | **42.8% ± 3.3%** | **19.0% ± 3.4%** | **23.8% ± 5.8%** |
| Prosody Shift (Whisper) | 38.6% ± 1.4% | 14.9% ± 4.2% | **23.7% ± 4.4%** |
| Latency Probe (Cognitive Math) | 45.3% ± 3.8% | 24.2% ± 3.9% | **21.1% ± 4.3%** |
| Phrase Repetition (Digits) | 40.2% ± 2.0% | 20.9% ± 3.3% | **19.3% ± 4.1%** |

---

## 4. Data Validity Changelog
To ensure academic rigor, this changelog documents the three stages this challenge-response separation metric went through during project development:

1. **Stage 1 (N=1 Placeholder)**: Initially tested using a single placeholder pair of unrelated sine tones. This resulted in an artificially inflated `100%` separation gap that did not reflect acoustic speech characteristics.
2. **Stage 2 (N=10 Mismatched-Content)**: Expanded to 10 pairs per category. However, the human and clone files were mapped to unrelated ASVspoof clean files reading generic sentences. The content did not match the issued challenge phrase, rendering the verification logic invalid for live call defense.
3. **Stage 3 (N=10 Phrase-Matched Stand-Ins)**: Corrected the code to ensure both files spoken match the issued challenge exactly. Due to the lack of real human recordings and local cloning models during early CPU iterations, gTTS (Google TTS) was used as the human stand-in, and pyttsx3 (SAPI5) as the clone. Because the passive spoof detector flagged the gTTS audio's synthetic envelope, the human score dropped from 90% to 42%, narrowing the separation gap.
4. **Planned Stage 4 (Future Work)**: Collect real human voice recordings from the user and synthesize cloned responses using local matching-content voice cloning models (e.g., Coqui TTS) to recompute authentic, validated speaker-cloned separation gaps.

---

## 5. Non-Monotonic Channel Mismatch Phenomenon (AMR-NB Case Study)
Under a single global decision threshold of `0.7943`, we observe a counter-intuitive behavior: evaluating the baseline classifier under the AMR-NB codec alone yields a lower overall accuracy (`39.7%`) than Combined Severe Telephony degradation (`72.3%`) which stacks AMR-NB with 15% packet loss and 30ms jitter.

### Analysis of Score Distributions
This phenomenon is documented in the generated score distributions plot: [score_distributions.png](file:///c:/Users/Admin%20pc/Desktop/voice%20detection/voice-deepfake-verify/docs/score_distributions.png).
- **AMR-NB Codec Only**: Applying bandpass filtering (300Hz-3400Hz) and bit quantization preserves the smooth voice contour envelopes. However, the compression shifts the extracted MFCC speaker coefficients of spoofed (fake) files upwards toward the authentic region. Since the evaluation split is heavily unbalanced (125 fakes vs. 16 reals), this upward score drift causes 85 out of 125 spoof files to leak past the `0.7943` threshold (False Positives), dropping overall accuracy to `39.7%`.
- **Combined Telephony (AMR + Loss + Jitter)**: Stacking packet drops (15%) and jitter delays introduces sharp silence gaps and block discontinuities in the audio waveform. This severe distortion breaks the acoustic footprint, causing the classifier's output probabilities to fall dramatically. Consequently, the spoof scores shift back down below the `0.7943` decision boundary, correctly blocking 87 out of 125 fakes (True Negatives) and raising the overall accuracy to `72.3%`.

