import os
import shutil
import glob
import requests
from gtts import gTTS
import pyttsx3

CHALLENGE_TYPES = {
    "phrase_repetition": {
        "phrase": "Please repeat the numbers: 8, 3, 9, 1, 4.",
        "desc": "Testing sequential digit speech generation."
    },
    "prosody_shift": {
        "phrase": "Please whisper: 'The quick brown fox jumps over the lazy dog'.",
        "desc": "Whisper/Prosody shift challenge."
    },
    "non_verbal_sound": {
        "phrase": "Please laugh loudly for two seconds.",
        "desc": "Non-verbal vocalization challenge."
    },
    "latency_probe": {
        "phrase": "What is 15 plus 27? Answer immediately.",
        "desc": "Cognitive latency / spontaneous response challenge."
    }
}

def synthesize_human_standin(text, output_path):
    """Synthesizes human stand-in audio using gTTS (natural) with offline pyttsx3 fallback."""
    try:
        # Try online gTTS for a natural sound
        tts = gTTS(text=text, lang="en", tld="com")
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"  gTTS failed ({e}). Falling back to pyttsx3 (Voice 1)...")
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            # Select Zira / Voice 1 if available
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            return True
        except Exception as e2:
            print(f"  Offline pyttsx3 synthesis failed: {e2}")
            return False

def synthesize_clone_response(text, output_path):
    """Synthesizes cloned audio using pyttsx3 (robotic SAPI5 Microsoft David)."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if len(voices) > 0:
            engine.setProperty('voice', voices[0].id) # Microsoft David
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"  pyttsx3 synthesis failed: {e}. Falling back to gTTS (UK Accent)...")
        try:
            tts = gTTS(text=text, lang="en", tld="co.uk")
            tts.save(output_path)
            return True
        except Exception as e2:
            print(f"  Clone fallback failed: {e2}")
            return False

def main():
    print("==================================================")
    print("CHALLENGE RESPONSE LOCAL MATCHING-CONTENT SYNTHESIS")
    print("==================================================")
    
    dest_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/challenges"))
    os.makedirs(dest_dir, exist_ok=True)
    
    # Clear existing challenges folder to prevent mixing old files
    for f in os.listdir(dest_dir):
        try:
            os.remove(os.path.join(dest_dir, f))
        except:
            pass

    # Build 10 pairs for each challenge category
    for key, info in CHALLENGE_TYPES.items():
        print(f"\nSynthesizing 10 pairs for challenge type: {key.upper()}")
        
        for idx in range(10):
            phrase_variation = f"{info['phrase']} Code sequence {idx+1}."
            
            # 1. Synthesize Human Stand-In
            human_dest = os.path.join(dest_dir, f"human_{key}_{idx}.wav")
            h_success = synthesize_human_standin(phrase_variation, human_dest)
            
            # 2. Synthesize Robotic Voice Clone
            cloned_dest = os.path.join(dest_dir, f"cloned_{key}_{idx}.wav")
            c_success = synthesize_clone_response(phrase_variation, cloned_dest)
            
            if not h_success or not c_success:
                print(f"  [ERROR] Failed to synthesize pair #{idx+1} for {key}.")
                
        print(f"  Successfully synthesized 10 pairs for {key}.")
        
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
