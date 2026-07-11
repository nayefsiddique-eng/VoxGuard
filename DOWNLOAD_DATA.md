# Dataset Download Instructions

To evaluate the system under realistic conditions and train the classifier on genuine human speech and AI-generated voice clones, download the ASVspoof 2019 and WaveFake datasets.

## 1. ASVspoof 2019 (Logical Access / LA)

The ASVspoof 2019 dataset contains genuine speech and various synthesized/cloned voice attacks.

### Download Steps:
1. Go to the official ASVspoof 2019 dataset repository: [ASVspoof 2019 LA on Zenodo](https://zenodo.org/record/4837263).
2. Download the `LA.zip` file (approx. 9.5 GB).
3. Extract the contents.

### Directory Mapping:
Move the relevant folders into the `data/raw/` directory:
- Place the training audio files in: `data/raw/ASVspoof2019_LA_train/flac/`
- Place the validation audio files in: `data/raw/ASVspoof2019_LA_dev/flac/`
- Place the metadata protocols (`ASVspoof2019.LA.cm.train.trn.txt` and `ASVspoof2019.LA.cm.dev.asl.txt`) in: `data/raw/ASVspoof2019_LA_protocols/`

---

## 2. WaveFake Dataset

The WaveFake dataset provides alternative neural vocoder synthetic speech samples (MelGAN, Parallel WaveGAN, Multi-Band MelGAN, WaveGlow) in multiple languages.

### Download Steps:
1. Go to the [WaveFake GitHub Page](https://github.com/asvspoof-challenge/WaveFake) or download directly from [WaveFake on Zenodo](https://zenodo.org/record/5653063).
2. Download the compressed archives (e.g. JSUT and LJSpeech generated audios).
3. Extract and place the WAV files into the `data/cloned/` directory.

---

## 3. Integrating with the Pipeline

Once files are placed, you can update `src/models/baseline_detector.py` to point to the actual protocols instead of `data/placeholder/`. The feature extractor is already set up to read any standard `.wav` or `.flac` file.
