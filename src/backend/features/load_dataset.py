import os
import glob
import soundfile as sf

def parse_protocol_file(protocol_path, audio_dir):
    """
    Parses an ASVspoof 2019 LA protocol file.
    Protocol lines format: speaker_id filename system_id class_name
    Maps filename to its full path in audio_dir and translates labels:
      - bonafide -> 1 (Real)
      - spoof -> 0 (Fake)
    """
    samples = []
    if not os.path.exists(protocol_path):
        print(f"Warning: Protocol file not found: {protocol_path}")
        return samples
        
    with open(protocol_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            speaker_id = parts[0]
            filename = parts[1]
            system_id = parts[2]
            label_str = parts[3]
            
            # Find the file in audio_dir (could be .flac or .wav)
            possible_paths = [
                os.path.join(audio_dir, f"{filename}.flac"),
                os.path.join(audio_dir, f"{filename}.wav")
            ]
            
            actual_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    actual_path = p
                    break
                    
            if actual_path:
                label = 1 if label_str == "bonafide" else 0
                samples.append({
                    "filepath": actual_path,
                    "label": label,
                    "speaker_id": speaker_id,
                    "system_id": system_id,
                    "dataset": "ASVspoof2019_LA"
                })
                
    return samples

def load_wavefake_spoofs(cloned_dir):
    """
    Dynamically scans data/cloned/ for any synthetic/cloned speech files (WaveFake).
    Since they are all synthetic voice clones, they are classified as spoof (label 0).
    """
    samples = []
    if not os.path.exists(cloned_dir):
        return samples
        
    # Search for all wav/flac files
    file_patterns = [
        os.path.join(cloned_dir, "**", "*.wav"),
        os.path.join(cloned_dir, "**", "*.flac")
    ]
    
    found_files = []
    for pattern in file_patterns:
        found_files.extend(glob.glob(pattern, recursive=True))
        
    for fp in found_files:
        samples.append({
            "filepath": os.path.abspath(fp),
            "label": 0, # Spoof
            "speaker_id": "unknown_wavefake",
            "system_id": "WaveFake",
            "dataset": "WaveFake"
        })
        
    if samples:
        print(f"Dynamically loaded {len(samples)} synthetic spoof files from WaveFake directory ({cloned_dir})")
        
    return samples

def get_split_dataset(split="train"):
    """
    Returns a list of samples (dict containing filepath, label, dataset) for the given split.
    Integrates ASVspoof 2019 protocols and dynamically merges WaveFake spoofs for training.
    """
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    cloned_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cloned"))
    
    # Path mappings based on standard ASVspoof structure
    if split == "train":
        protocol_file = os.path.join(raw_dir, "ASVspoof2019_LA_protocols", "ASVspoof2019.LA.cm.train.trn.txt")
        audio_dir = os.path.join(raw_dir, "ASVspoof2019_LA_train", "flac")
    elif split == "dev":
        protocol_file = os.path.join(raw_dir, "ASVspoof2019_LA_protocols", "ASVspoof2019.LA.cm.dev.asl.txt")
        audio_dir = os.path.join(raw_dir, "ASVspoof2019_LA_dev", "flac")
    elif split == "eval":
        protocol_file = os.path.join(raw_dir, "ASVspoof2019_LA_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
        audio_dir = os.path.join(raw_dir, "ASVspoof2019_LA_eval", "flac")
    else:
        raise ValueError(f"Unknown split: {split}")
        
    # Load ASVspoof samples
    samples = parse_protocol_file(protocol_file, audio_dir)
    
    # Confirm file counts match protocol files
    if os.path.exists(protocol_file):
        with open(protocol_file, "r") as f:
            protocol_lines = [l for l in f.readlines() if l.strip()]
        if len(samples) != len(protocol_lines):
            print(f"Warning/Mismatch: Protocol file specifies {len(protocol_lines)} samples, but only {len(samples)} audio files were resolved locally.")
            
    # Include WaveFake samples into train split for spoof diversity
    if split == "train":
        wavefake_samples = load_wavefake_spoofs(cloned_dir)
        samples.extend(wavefake_samples)
        
    return samples

def print_split_statistics(split_name, samples):
    """Calculates and prints statistics for a dataset split."""
    if not samples:
        print(f"\n--- Split: {split_name} (Empty) ---")
        return
        
    total_files = len(samples)
    real_count = sum(1 for s in samples if s["label"] == 1)
    fake_count = total_files - real_count
    
    total_duration_sec = 0.0
    for s in samples:
        try:
            info = sf.info(s["filepath"])
            total_duration_sec += info.duration
        except Exception:
            pass
            
    print(f"\n--- Split: {split_name.upper()} Statistics ---")
    print(f"  Total Audio Files  : {total_files}")
    print(f"  Class Balance      : Real = {real_count} ({real_count/total_files*100:.1f}%), Fake = {fake_count} ({fake_count/total_files*100:.1f}%)")
    print(f"  Total Duration     : {total_duration_sec / 60.0:.2f} minutes ({total_duration_sec:.1f} seconds)")

if __name__ == "__main__":
    for split in ["train", "dev", "eval"]:
        samples = get_split_dataset(split)
        print_split_statistics(split, samples)
