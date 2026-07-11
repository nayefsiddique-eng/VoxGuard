import os
import shutil
import glob

def partition_speaker_disjoint():
    print("==================================================")
    print("ENFORCING STRICT SPEAKER-DISJOINT SPLITS (N=800)")
    print("==================================================")
    
    raw_dir = "c:/Users/Admin pc/Desktop/voice detection/voice-deepfake-verify/data/raw"
    protocol_dir = os.path.join(raw_dir, "ASVspoof2019_LA_protocols")
    
    # Paths for old protocol files
    old_train = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    old_dev = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.dev.asl.txt")
    old_eval = os.path.join(protocol_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
    
    # 1. Read all entries from all protocol files to create a master database
    master_records = {}
    
    def parse_file(path):
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    spk, filename, sys_id, label = parts[0], parts[1], parts[2], parts[3]
                    # Strip any quotation marks from speaker ID
                    spk = spk.replace('"', '')
                    master_records[filename] = {
                        "speaker": spk,
                        "system": sys_id,
                        "label": label
                    }
                    
    parse_file(old_train)
    parse_file(old_dev)
    parse_file(old_eval)
    
    if not master_records:
        print("Error: Master protocol mapping is empty. Downloader splits missing?")
        return
        
    print(f"Loaded master map with {len(master_records)} file records.")
    
    # 2. Gather all downloaded audio files across all directories
    all_audio_files = glob.glob(os.path.join(raw_dir, "ASVspoof2019_LA_*", "flac", "*.flac"))
    print(f"Located {len(all_audio_files)} FLAC audio files on disk.")
    
    # 3. Group files by Speaker ID
    speaker_groups = {}
    for filepath in all_audio_files:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        if base_name in master_records:
            spk = master_records[base_name]["speaker"]
            if spk not in speaker_groups:
                speaker_groups[spk] = []
            speaker_groups[spk].append(filepath)
            
    print(f"Grouped files into {len(speaker_groups)} unique speakers.")
    
    # 4. Partition Speaker IDs strictly:
    # Sort speakers list to be deterministic
    sorted_speakers = sorted(list(speaker_groups.keys()))
    
    # Target sizes: train = 50%, dev = 20%, eval = 30%
    num_spk = len(sorted_speakers)
    train_end = int(num_spk * 0.50)
    dev_end = train_end + int(num_spk * 0.20)
    
    train_speakers = set(sorted_speakers[:train_end])
    dev_speakers = set(sorted_speakers[train_end:dev_end])
    eval_speakers = set(sorted_speakers[dev_end:])
    
    print(f"\nStrict Partition splits:")
    print(f"  Train Speakers : {len(train_speakers)} ({sorted_speakers[0]} to {sorted_speakers[train_end-1]})")
    print(f"  Dev Speakers   : {len(dev_speakers)} ({sorted_speakers[train_end]} to {sorted_speakers[dev_end-1]})")
    print(f"  Eval Speakers  : {len(eval_speakers)} ({sorted_speakers[dev_end]} to {sorted_speakers[-1]})")
    
    # Create target split directories
    splits = {
        "train": os.path.join(raw_dir, "ASVspoof2019_LA_train", "flac"),
        "dev": os.path.join(raw_dir, "ASVspoof2019_LA_dev", "flac"),
        "eval": os.path.join(raw_dir, "ASVspoof2019_LA_eval", "flac")
    }
    for d in splits.values():
        os.makedirs(d, exist_ok=True)
        
    # 5. Move files and generate new protocol contents
    new_protocols = {"train": [], "dev": [], "eval": []}
    move_count = 0
    
    for spk, files in speaker_groups.items():
        if spk in train_speakers:
            target = "train"
        elif spk in dev_speakers:
            target = "dev"
        else:
            target = "eval"
            
        target_dir = splits[target]
        
        for filepath in files:
            filename = os.path.basename(filepath)
            dest_path = os.path.join(target_dir, filename)
            
            # Move file if it's not already in target directory
            if os.path.abspath(filepath) != os.path.abspath(dest_path):
                shutil.move(filepath, dest_path)
                move_count += 1
                
            # Add to protocol
            base_name = os.path.splitext(filename)[0]
            rec = master_records[base_name]
            protocol_line = f"\"{spk}\" {base_name} {rec['system']} {rec['label']}\n"
            new_protocols[target].append(protocol_line)
            
    print(f"Rearranged and moved {move_count} files into disjoint folders.")
    
    # 6. Write out new protocol files
    def write_protocol(lines, path):
        with open(path, "w") as f:
            f.writelines(lines)
            
    write_protocol(new_protocols["train"], old_train)
    write_protocol(new_protocols["dev"], old_dev)
    write_protocol(new_protocols["eval"], old_eval)
    
    print("\nSuccessfully updated protocol text files with speaker-disjoint mappings.")
    
    # 7. Clear feature cache directory to ensure recalculation of newly located samples
    cache_dir = "c:/Users/Admin pc/Desktop/voice detection/voice-deepfake-verify/data/features"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print("Cleared feature caches folder to invalidate old path keys.")
    else:
        print("No features cache folder found to clear.")
        
    print("\nSpeaker-independent split verification complete!")

if __name__ == "__main__":
    partition_speaker_disjoint()
