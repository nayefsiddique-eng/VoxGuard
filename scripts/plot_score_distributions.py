import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.features.load_dataset import get_split_dataset
from src.features.extract_features import extract_features
from src.pipeline.degrade_audio import process_file

def evaluate_scores(samples, codec, packet_loss, jitter, model):
    """Processes audio and returns raw prediction scores categorized by label."""
    bonafide_scores = []
    spoof_scores = []
    
    temp_dir = "data/degraded_temp_dist"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Enforce deterministic random seeding inside this function
    np.random.seed(42)
    
    for s in samples:
        path = process_file(s["filepath"], temp_dir, codec=codec, packet_loss=packet_loss, jitter=jitter)
        feats = extract_features(path, feature_type="mfcc", use_cache=True)
        prob = model.predict_proba([feats])[0, 1]
        
        if s["label"] == 1:
            bonafide_scores.append(prob)
        else:
            spoof_scores.append(prob)
            
        try: os.remove(path)
        except: pass
        
    try: os.rmdir(temp_dir)
    except: pass
    
    return np.array(bonafide_scores), np.array(spoof_scores)

def main():
    print("==================================================")
    print("PLOTTING SCORE DISTRIBUTIONS FOR DEGRADATIONS")
    print("==================================================")
    
    # 1. Load model
    model_path = "src/models/detector.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return
    model = joblib.load(model_path)
    
    # 2. Load eval samples
    eval_samples = get_split_dataset("eval")
    if not eval_samples:
        print("Error: No evaluation samples loaded.")
        return
        
    # 3. Retrieve scores
    print("Evaluating AMR-NB condition...")
    amr_bonafide, amr_spoof = evaluate_scores(eval_samples, "amr", 0.0, 0, model)
    
    print("Evaluating Combined Telephony condition...")
    comb_bonafide, comb_spoof = evaluate_scores(eval_samples, "amr", 0.15, 30, model)
    
    # 4. Generate side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    global_threshold = 0.7943
    
    # AMR-NB plot
    ax1.hist(amr_bonafide, bins=15, range=(0, 1), alpha=0.6, color="#10b981", label="Bonafide (Human)")
    ax1.hist(amr_spoof, bins=15, range=(0, 1), alpha=0.6, color="#ef4444", label="Spoof (Fake)")
    ax1.axvline(global_threshold, color="#3b82f6", linestyle="--", linewidth=2, label=f"Global Threshold ({global_threshold})")
    ax1.set_title("AMR-NB Codec Only", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Authenticity Score", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.legend(loc="upper center", fontsize=9)
    ax1.grid(alpha=0.1)
    
    # Combined Telephony plot
    ax2.hist(comb_bonafide, bins=15, range=(0, 1), alpha=0.6, color="#10b981", label="Bonafide (Human)")
    ax2.hist(comb_spoof, bins=15, range=(0, 1), alpha=0.6, color="#ef4444", label="Spoof (Fake)")
    ax2.axvline(global_threshold, color="#3b82f6", linestyle="--", linewidth=2, label=f"Global Threshold ({global_threshold})")
    ax2.set_title("Combined (AMR + Loss + Jitter)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Authenticity Score", fontsize=10)
    ax2.legend(loc="upper center", fontsize=9)
    ax2.grid(alpha=0.1)
    
    plt.suptitle("Authenticity Score Distributions under Channel Mismatch", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    save_path = os.path.join(docs_dir, "score_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved score distribution plot to: {save_path}")

if __name__ == "__main__":
    main()
