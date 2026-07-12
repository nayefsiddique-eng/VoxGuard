import os
import time
import subprocess
import requests

def is_server_running(url):
    try:
        # Simple health check endpoint or just checking connection
        res = requests.get(f"{url}/challenge/request", timeout=2.0)
        return True
    except requests.exceptions.RequestException:
        return False

def main():
    print("==================================================")
    print("TESTING API ENDPOINTS")
    print("==================================================")
    
    base_url = "http://127.0.0.1:8080"
    proc = None
    
    # Check if a server is already running (e.g. from our manual background task)
    if is_server_running(base_url):
        print("FastAPI server is already running. Using the running instance.")
    else:
        # Set up environment with PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # Path to python executable
        python_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "../venv/Scripts/python.exe"))
        
        # Start the API server in a background process
        cmd = [python_exe, "-m", "uvicorn", "src.pipeline.api:app", "--host", "127.0.0.1", "--port", "8080"]
        print(f"Starting FastAPI server on {base_url} (waiting 15s for PyTorch initialization)...")
        proc = subprocess.Popen(cmd, env=env)
        
        # Give the server ample time to start up (PyTorch and Whisper loading is slow on CPU)
        time.sleep(35.0)
        
        if proc.poll() is not None:
            print("Error: Server failed to start.")
            return

    try:
        placeholder_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/placeholder"))
        
        # Test paths
        real_wav = os.path.join(placeholder_dir, "class_a_clean_0.wav")
        fake_wav = os.path.join(placeholder_dir, "class_b_fake_0.wav")
        
        # --- Endpoint 1: Verify Passive (Real Voice) ---
        print("\n1. Testing POST /verify-passive with Real audio...")
        with open(real_wav, "rb") as f:
            files = {"file": ("class_a_clean_0.wav", f, "audio/wav")}
            res = requests.post(f"{base_url}/verify-passive", files=files)
            print("Response:", res.json())
            assert res.status_code == 200
            
        # --- Endpoint 1: Verify Passive (Fake Voice) ---
        print("\n2. Testing POST /verify-passive with Fake audio...")
        with open(fake_wav, "rb") as f:
            files = {"file": ("class_b_fake_0.wav", f, "audio/wav")}
            res = requests.post(f"{base_url}/verify-passive", files=files)
            print("Response:", res.json())
            assert res.status_code == 200
            
        # --- Endpoint 2: Get Challenge Phrase ---
        print("\n3. Testing GET /challenge/request...")
        res = requests.get(f"{base_url}/challenge/request")
        challenge_data = res.json()
        print("Response:", challenge_data)
        assert res.status_code == 200
        phrase = challenge_data["challenge_phrase"]
        
        # --- Endpoint 3: Verify Challenge Response ---
        print("\n4. Testing POST /challenge/verify (with response and fusion)...")
        with open(fake_wav, "rb") as f:
            files = {"file": ("class_b_fake_0.wav", f, "audio/wav")}
            data = {"phrase": phrase, "passive_score": 0.42}
            res = requests.post(f"{base_url}/challenge/verify", files=files, data=data)
            print("Response:", res.json())
            assert res.status_code == 200
            
        print("\nAll endpoints tested successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
    finally:
        if proc is not None:
            # Shutdown uvicorn process if we started it
            print("\nStopping FastAPI server...")
            proc.terminate()
            proc.wait()
            print("Server stopped.")

if __name__ == "__main__":
    main()
