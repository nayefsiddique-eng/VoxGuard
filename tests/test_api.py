import os
import time
import subprocess
import requests

def is_server_running(url):
    try:
        res = requests.get(f"{url}/challenge/request", timeout=2.0)
        return True
    except requests.exceptions.RequestException:
        return False

def find_real_dataset_files():
    """Parses protocols to locate one real bonafide file and one real spoof file on disk."""
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
    protocol_dir = os.path.join(raw_dir, "ASVspoof2019_LA_protocols")
    
    splits = ["train", "dev", "eval"]
    files_map = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt", 
        "dev": "ASVspoof2019.LA.cm.dev.asl.txt", 
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt"
    }
    
    real_bonafide = None
    real_spoof = None
    
    for split in splits:
        proto_path = os.path.join(protocol_dir, files_map[split])
        if not os.path.exists(proto_path):
            continue
            
        flac_dir = os.path.join(raw_dir, f"ASVspoof2019_LA_{split}", "flac")
        with open(proto_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    filename = parts[1]
                    label = parts[3]
                    filepath = os.path.join(flac_dir, f"{filename}.flac")
                    
                    if os.path.exists(filepath):
                        if label == "bonafide" and real_bonafide is None:
                            real_bonafide = filepath
                        elif label == "spoof" and real_spoof is None:
                            real_spoof = filepath
                            
                if real_bonafide and real_spoof:
                    break
        if real_bonafide and real_spoof:
            break
            
    return real_bonafide, real_spoof

def main():
    print("==================================================")
    print("TESTING API ENDPOINTS (REAL ASVSPOOF AUDIO)")
    print("==================================================")
    
    base_url = "http://127.0.0.1:8080"
    proc = None
    
    # 1. Locate real human vs spoof audio files
    real_wav, fake_wav = find_real_dataset_files()
    if not real_wav or not fake_wav:
        print("Error: Could not locate real ASVspoof audio files. Run download_dataset_subset.py first.")
        return
        
    print(f"Located real Human audio: {os.path.basename(real_wav)}")
    print(f"Located real Spoof audio: {os.path.basename(fake_wav)}")
    
    # 2. Check if server is running, or start it
    if is_server_running(base_url):
        print("FastAPI server is already running. Using the running instance.")
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        python_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "../venv/Scripts/python.exe"))
        
        cmd = [python_exe, "-m", "uvicorn", "src.pipeline.api:app", "--host", "127.0.0.1", "--port", "8080"]
        print(f"Starting FastAPI server on {base_url} (waiting 35s for PyTorch and Whisper initialization)...")
        proc = subprocess.Popen(cmd, env=env)
        time.sleep(35.0)
        
        if proc.poll() is not None:
            print("Error: Server failed to start.")
            return

    try:
        # --- Endpoint 1: Verify Passive (Real Human Voice) ---
        print("\n1. Testing POST /verify-passive with Real Human voice...")
        with open(real_wav, "rb") as f:
            files = {"file": (os.path.basename(real_wav), f, "audio/flac")}
            res = requests.post(f"{base_url}/verify-passive", files=files)
            real_data = res.json()
            print("Response:", real_data)
            
            # Assertions
            assert res.status_code == 200
            assert real_data["passive_authenticity_score"] >= 0.50
            assert real_data["trigger_challenge"] is False
            assert real_data["status"] == "CLEAN"
            
        # --- Endpoint 2: Verify Passive (Fake Spoof Voice) ---
        print("\n2. Testing POST /verify-passive with Real Spoof voice...")
        with open(fake_wav, "rb") as f:
            files = {"file": (os.path.basename(fake_wav), f, "audio/flac")}
            res = requests.post(f"{base_url}/verify-passive", files=files)
            fake_data = res.json()
            print("Response:", fake_data)
            
            # Assertions
            assert res.status_code == 200
            assert fake_data["passive_authenticity_score"] < 0.50
            assert fake_data["trigger_challenge"] is True
            assert fake_data["status"] == "SUSPICIOUS"
            
        # --- Endpoint 3: Get Challenge Phrase ---
        print("\n3. Testing GET /challenge/request...")
        res = requests.get(f"{base_url}/challenge/request")
        challenge_data = res.json()
        print("Response:", challenge_data)
        assert res.status_code == 200
        phrase = challenge_data["challenge_phrase"]
        
        # --- Endpoint 4: Verify Challenge Response ---
        print("\n4. Testing POST /challenge/verify (with response and fusion)...")
        # Submit the spoof file to see if the fuser classifies it as SUSPICIOUS/FAKE
        with open(fake_wav, "rb") as f:
            files = {"file": (os.path.basename(fake_wav), f, "audio/flac")}
            data = {"phrase": phrase, "passive_score": fake_data["passive_authenticity_score"]}
            res = requests.post(f"{base_url}/challenge/verify", files=files, data=data)
            print("Response:", res.json())
            assert res.status_code == 200
            
        print("\nAll endpoints and authentic real-vs-fake classifications verified successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
    finally:
        if proc is not None:
            print("\nStopping FastAPI server...")
            proc.terminate()
            proc.wait()
            print("Server stopped.")

if __name__ == "__main__":
    main()
