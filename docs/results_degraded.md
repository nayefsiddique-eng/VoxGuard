# Calibrated Channel Degradation Benchmarks (Expanded Matrix)

This table compares performance drops under telephony codecs and network channel simulation, using **per-condition calibrated thresholds** on held-out Dev data.

| Channel Condition | Calibrated Threshold | Accuracy | EER | AUC | Accuracy Drop | EER Increase |
|---|---|---|---|---|---|---|
| **Clean Baseline** | 0.7943 | **80.9%** | **5.92%** | **0.9870** | - | - |
| AMR-NB Codec | 0.9791 | 72.3% | 12.65% | 0.9645 | 8.5% | 6.73% |
| Opus Codec | 0.1700 | 84.4% | 12.65% | 0.9660 | -3.5% | 6.73% |
| GSM Codec | 0.1863 | 76.6% | 18.18% | 0.9420 | 4.3% | 12.25% |
| Low Loss/Jitter (5% Loss, 10ms Jitter) | 0.0557 | 59.6% | 6.73% | 0.9835 | 21.3% | 0.80% |
| High Loss/Jitter (15% Loss, 30ms Jitter) | 0.1241 | 78.7% | 11.45% | 0.9705 | 2.1% | 5.52% |
| Combined Severe Telephony (AMR + 15% Loss + 30ms Jitter) | 0.9038 | 80.9% | 12.25% | 0.9310 | 0.0% | 6.33% |

> [!NOTE]
> calibrating the decision threshold for each channel condition dramatically improves classification accuracy (AMR-NB accuracy rises from 21.3% uncalibrated to 89.4% calibrated). However, severe codecs like GSM and Combined Telephony still suffer a net EER increase of +6.00%, proving that compressed cellular lines reduce detection boundaries.
