import os
import io
import datasets
from datasets import load_dataset

def main():
    print("==================================================")
    print("PROGRAMMATIC DATASET SUBSET DOWNLOADER")
    print("==================================================")
    
    # Target directories
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    protocols_dir = os.path.join(raw_dir, "ASVspoof2019_LA_protocols")
    
    splits = {
        "train": os.path.join(raw_dir, "ASVspoof2019_LA_train", "flac"),
        "dev": os.path.join(raw_dir, "ASVspoof2019_LA_dev", "flac"),
        "eval": os.path.join(raw_dir, "ASVspoof2019_LA_eval", "flac")
    }
    
    for path in [protocols_dir] + list(splits.values()):
        os.makedirs(path, exist_ok=True)
        
    print("Loading test split from Hugging Face (decode=False)...")
    try:
        ds = load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", split="test", streaming=True)
        ds = ds.cast_column("audio", datasets.Audio(decode=False))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Protocols lists
    protocols = {
        "train": [],
        "dev": [],
        "eval": []
    }
    
    # We will download:
    # - 400 samples for train (200 real, 200 fake)
    # - 150 samples for dev (75 real, 75 fake)
    # - 250 samples for eval (125 real, 125 fake)
    counts = {
        "train": {"real": 0, "fake": 0, "max": 200},
        "dev": {"real": 0, "fake": 0, "max": 75},
        "eval": {"real": 0, "fake": 0, "max": 125}
    }
    
    print("Streaming dataset and saving files locally...")
    
    # We iterate over the stream
    for item in ds:
        label = item["label"] # 1 = bonafide, 0 = spoof
        label_str = "bonafide" if label == 1 else "spoof"
        category = "real" if label == 1 else "fake"
        
        # Decide which split this sample belongs to based on count
        target_split = None
        for s in ["train", "dev", "eval"]:
            if counts[s][category] < counts[s]["max"]:
                target_split = s
                break
                
        if target_split is None:
            # All splits are full
            if all(counts[s]["real"] >= counts[s]["max"] and counts[s]["fake"] >= counts[s]["max"] for s in ["train", "dev", "eval"]):
                break
            continue
            
        # File details
        # HF schema path usually contains something like 'LA_T_1132912.flac'
        original_path = item["path"]
        filename = os.path.basename(original_path)
        if not filename.endswith(".flac"):
            filename = f"{filename}.flac"
            
        # Save the audio bytes
        audio_bytes = item["audio"]["bytes"]
        dest_path = os.path.join(splits[target_split], filename)
        
        with open(dest_path, "wb") as f:
            f.write(audio_bytes)
            
        counts[target_split][category] += 1
        
        # Create standard protocol line: speaker_id filename system_id class_name
        # Notes field might contain speaker/system details, or we can mock/extract them
        speaker_id = "LA_0001"
        system_id = "-" if label == 1 else "A07"
        
        notes = item.get("notes", "")
        if notes and isinstance(notes, str):
            parts = notes.split(",")
            for p in parts:
                if "speaker" in p:
                    speaker_id = p.split(":")[-1].strip()
                if "system" in p:
                    system_id = p.split(":")[-1].strip()
                    
        protocol_line = f"{speaker_id} {os.path.splitext(filename)[0]} {system_id} {label_str}\n"
        protocols[target_split].append(protocol_line)
        
        total_downloaded = sum(sum(counts[s].values()) - counts[s]["max"] for s in counts)
        if total_downloaded % 20 == 0:
            print(f"  Downloaded {total_downloaded} files...")
            
    # Write protocol files
    print("Writing protocol text files...")
    
    # Train protocol: ASVspoof2019.LA.cm.train.trn.txt
    train_proto_path = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    with open(train_proto_path, "w") as f:
        f.writelines(protocols["train"])
    print(f"  Saved train protocol to: {train_proto_path}")
        
    # Dev protocol: ASVspoof2019.LA.cm.dev.asl.txt
    dev_proto_path = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.dev.asl.txt")
    with open(dev_proto_path, "w") as f:
        f.writelines(protocols["dev"])
    print(f"  Saved dev protocol to: {dev_proto_path}")
        
    # Eval protocol: ASVspoof2019.LA.cm.eval.trl.txt
    eval_proto_path = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
    with open(eval_proto_path, "w") as f:
        f.writelines(protocols["eval"])
    print(f"  Saved eval protocol to: {eval_proto_path}")
    
    print("\nDataset subset download complete!")
    print(f"Total files saved in {raw_dir}:")
    print(f"  Train split: {sum(counts['train'].values()) - counts['train']['max']} files")
    print(f"  Dev split: {sum(counts['dev'].values()) - counts['dev']['max']} files")
    print(f"  Eval split: {sum(counts['eval'].values()) - counts['eval']['max']} files")

if __name__ == "__main__":
    main()
