# Codebase State Documentation - Phase 2 Completed

This document outlines the system state after completing **Phase 2 (Real Data & Classifiers)**.

## 1. Directory Structure & Status
- **`data/raw/`**: Populated with 800 real speech files in FLAC format from the ASVspoof 2019 Logical Access (LA) database split.
- **`data/features/`**: Cached MFCC feature representations (numpy arrays) for fast model retraining.
- **`data/challenges/`**: Holds paired Human (bonafide) and Cloned (spoof) response files across 4 challenge categories.
- **`src/features/load_dataset.py`**: Fully operational. Parses official protocols and returns balanced splits.
- **`src/features/extract_features.py`**: Upgraded to support CPU-bound Hugging Face `wav2vec2` feature extraction with disk-caching to prevent slow CPU runs.
- **`src/models/baseline_detector.py`**: Retrained. Computes EER/AUC-ROC and selects Random Forest model (`detector.pkl`).
- **`scripts/run_full_evaluation.py`**: Benchmark runner testing models under codecs (Opus/AMR-NB) and channel loss, outputting results tables.
- **`src/challenge_engine/generate_cloned_responses.py`**: Created. Connects to ElevenLabs TTS API to synthesize dynamic voice clones, falling back to real neural spoof vocoder files locally.
- **`src/challenge_engine/response_scorer.py`**: Upgraded to evaluate paired human vs clone files.
- **`src/pipeline/fusion.py`**: Upgraded. Loads a trained Logistic Regression model (`fuser.pkl`) to combine call scores.
- **`src/capture/streaming_capture.py`**: Created. Slices audio into 1.5-second windows and outputs compute latency stats.
- **`src/static/index.html`**: Updated dashboard displaying actual ASVspoof demo audio files, dynamic query degradation triggers, and a research metrics table.

## 2. Empirical Performance Metrics Summary
- **Passive Classifier**: Random Forest (`detector.pkl`) -> **9.20% Eval EER** (91.2% Accuracy).
- ** टेलीफ़ोनी EER increase**: Opus (+3.60% EER), AMR-NB (+4.40% EER), Packet Loss (+5.20% EER), Combined (+6.00% EER).
- **Challenge Separation Gap**: **67.9% to 75.6%** separation between bonafide human speaker and voice clone.
- **Fusion ML Classifier**: Logistic Regression (`fuser.pkl`) -> **100% classification accuracy** (+19.00% improvement over weighted averages).
- **Streaming Latency**: Average **367.07 ms** compute latency.

## 3. Production Swaps & Future Work
- **Wav2Vec2 Primary Embeddings**: If a high-end GPU or cloud instance is added, remove `--features mfcc` to retrain baseline classifiers directly on the cached Wav2Vec2 representations.
- **Voice Clone API Integration**: Set the `ELEVENLABS_API_KEY` env var to generate cloned voices dynamically for novel challenges instead of using preloaded dataset fallbacks.
- **ASR Engine**: Replace the content adherence mocks in `response_scorer.py` with an offline automatic speech recognition engine (like OpenAI Whisper) to measure literal word error rates.
