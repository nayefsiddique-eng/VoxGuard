import os
from src.challenge_engine.response_scorer import score_response

def simulate_replay_attack():
    print("==================================================")
    print("REPLAY ATTACK ROBUSTNESS SIMULATION (OFFLINE WHISPER)")
    print("==================================================")
    
    challenges_dir = "c:/Users/Admin pc/Desktop/voice detection/voice-deepfake-verify/data/challenges"
    replayed_audio = os.path.join(challenges_dir, "human_phrase_repetition_0.wav")
    
    if not os.path.exists(replayed_audio):
        print(f"Error: Replayed source file not found at {replayed_audio}. Run generator first.")
        return
        
    # --- TEST 1: REPLAY ATTACK CASE (MISMATCHED CONTENT) ---
    print("\n--- TEST 1: Replay Attack (Mismatched Challenge) ---")
    issued_challenge_1 = "What is 15 plus 27? Answer immediately."
    print(f"Attacker replays audio to pass challenge: '{issued_challenge_1}'")
    
    res_1 = score_response(replayed_audio, issued_challenge_1)
    
    print("[Evaluation Metrics]")
    print(f"  Actual Transcribed Text  : '{res_1['transcript']}'")
    print(f"  Voice Authenticity Score : {res_1['voice_authenticity_score']*100:.2f}%")
    print(f"  Content Adherence Score  : {res_1['content_adherence_score']*100:.2f}% (Expected: 0.00%)")
    print(f"  Replay Attack Flagged?   : {res_1['replay_attack_detected']}")
    print(f"  Call Blocked/Suspicious? : {res_1['is_suspicious']}")
    
    # --- TEST 2: POSITIVE CONTROL CASE (MATCHING CONTENT) ---
    print("\n--- TEST 2: Positive Control (Matching Challenge) ---")
    issued_challenge_2 = "Please repeat the numbers: 8, 3, 9, 1, 4."
    print(f"Attacker replays matching audio to challenge: '{issued_challenge_2}'")
    
    res_2 = score_response(replayed_audio, issued_challenge_2)
    
    print("[Evaluation Metrics]")
    print(f"  Actual Transcribed Text  : '{res_2['transcript']}'")
    print(f"  Voice Authenticity Score : {res_2['voice_authenticity_score']*100:.2f}%")
    print(f"  Content Adherence Score  : {res_2['content_adherence_score']*100:.2f}% (Expected: High 88-98%)")
    print(f"  Replay Attack Flagged?   : {res_2['replay_attack_detected']}")
    print(f"  Call Blocked/Suspicious? : {res_2['is_suspicious']}")
    
    # Verification assertions
    if res_1["replay_attack_detected"] and not res_2["replay_attack_detected"]:
        print("\nSANITY CHECKS COMPLETE: Robustness checks pass under local Whisper transcription!")
    else:
        print("\nSANITY CHECKS FAILED: Mismatch detection or positive control failed.")

if __name__ == "__main__":
    simulate_replay_attack()
