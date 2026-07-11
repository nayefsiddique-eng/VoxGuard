# Clean Audio Spoof Detection Benchmarks

This table compares spoofing detection classifiers trained on **MFCC** features.

| Model | Dev Accuracy | Dev AUC | Dev EER | Eval Accuracy | Eval AUC | Eval EER |
|---|---|---|---|---|---|---|
| Logistic Regression | 86.7% | 0.9384 | 15.62% | 72.3% | 0.9870 | 5.92% |
| Random Forest | 85.8% | 0.9077 | 16.02% | 79.4% | 0.9565 | 11.85% |
| Multi-Layer Perceptron (MLP) | 84.9% | 0.9329 | 13.75% | 77.3% | 0.9400 | 12.65% |

*Best Model: **Logistic Regression** selected for production verification based on lowest Evaluation EER.*
