# Realistic Channel Degradation Benchmarks

This table shows the performance drops when evaluation audios are compressed and degraded by codecs and network channel simulation.

| Channel Condition | Accuracy | EER | Accuracy Drop | EER Increase |
|---|---|---|---|---|
| **Clean Baseline** | **72.3%** | **5.92%** | - | - |
| AMR-NB Codec | 21.3% | 12.65% | -51.1% | +6.73% |
| Opus Codec | 91.5% | 12.65% | --19.1% | +6.73% |
| Packet Loss Only | 85.8% | 6.73% | --13.5% | +0.80% |
| Combined (AMR-NB + Jitter + Loss) | 37.6% | 11.85% | -34.8% | +5.92% |

> [!WARNING]
> Codec compression (AMR-NB) and combined channel errors significantly increase Equal Error Rates. This highlights the importance of evaluating deepfake detection algorithms under compressed telephony scenarios rather than relying on clean lab-quality audio.
