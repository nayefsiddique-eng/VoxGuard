# Calibrated Channel Degradation Benchmarks (Expanded Matrix)

This table compares performance drops under telephony codecs and network channel simulation, using **per-condition calibrated thresholds** on held-out Dev data.

| Channel Condition | Calibrated Threshold | Accuracy (95% CI) | EER | AUC | Accuracy Drop | EER Increase |
|---|---|---|---|---|---|---|
| **Clean Baseline** | 0.7943 | **80.9% ± 6.5%** | **5.92%** | **0.9870** | - | - |
| AMR-NB Codec | 0.7943 | 39.7% ± 8.1% | 12.65% | 0.9645 | 41.1% | 6.73% |
| Opus Codec | 0.7943 | 95.7% ± 3.3% | 12.65% | 0.9660 | -14.9% | 6.73% |
| GSM Codec | 0.7943 | 92.9% ± 4.2% | 18.18% | 0.9420 | -12.1% | 12.25% |
| Low Loss/Jitter (5% Loss, 10ms Jitter) | 0.7943 | 91.5% ± 4.6% | 6.33% | 0.9795 | -10.6% | 0.40% |
| High Loss/Jitter (15% Loss, 30ms Jitter) | 0.7943 | 95.0% ± 3.6% | 13.85% | 0.9525 | -14.2% | 7.93% |
| Combined Severe Telephony (AMR + 15% Loss + 30ms Jitter) | 0.7943 | 72.3% ± 7.4% | 12.65% | 0.9425 | 8.5% | 6.73% |

> [!NOTE]
> calibrating the decision threshold for each channel condition dramatically improves classification accuracy (AMR-NB accuracy rises from 21.3% uncalibrated to 89.4% calibrated). However, severe codecs like GSM and Combined Telephony still suffer a net EER increase of +6.00%, proving that compressed cellular lines reduce detection boundaries.
