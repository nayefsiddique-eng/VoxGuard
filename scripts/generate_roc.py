import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.backend.features.load_dataset import get_split_dataset
from src.backend.features.extract_features import extract_features
from src.backend.pipeline.degrade_audio import process_file

def get_probabilities(samples, codec, packet_loss, jitter, model):
    X, y_true = [], []
    temp_dir = "data/degraded_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    for s in samples:
        path = process_file(s["filepath"], temp_dir, codec=codec, packet_loss=packet_loss, jitter=jitter)
        feats = extract_features(path, feature_type="mfcc", use_cache=True)
        X.append(feats)
        y_true.append(s["label"])
        try: os.remove(path)
        except: pass
        
    X = np.array(X)
    y_true = np.array(y_true)
    
    probs = model.predict_proba(X)[:, 1]
    
    try: os.rmdir(temp_dir)
    except: pass
    
    return y_true, probs

def main():
    print("Generating ROC curves for evaluation splits...")
    model_path = "src/models/detector.pkl"
    if not os.path.exists(model_path):
        print("Error: Trained model detector.pkl not found.")
        return
    model = joblib.load(model_path)
    
    eval_samples = get_split_dataset("eval")
    if not eval_samples:
        print("Error: No evaluation samples resolved.")
        return
        
    # Get probabilities
    y_true_clean, probs_clean = get_probabilities(eval_samples, "none", 0.0, 0, model)
    y_true_amr, probs_amr = get_probabilities(eval_samples, "amr", 0.0, 0, model)
    y_true_comb, probs_comb = get_probabilities(eval_samples, "amr", 0.15, 30, model)
    
    # Calculate ROC and AUC
    fpr_clean, tpr_clean, _ = roc_curve(y_true_clean, probs_clean)
    auc_clean = auc(fpr_clean, tpr_clean)
    
    fpr_amr, tpr_amr, _ = roc_curve(y_true_amr, probs_amr)
    auc_amr = auc(fpr_amr, tpr_amr)
    
    fpr_comb, tpr_comb, _ = roc_curve(y_true_comb, probs_comb)
    auc_comb = auc(fpr_comb, tpr_comb)
    
    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_clean, tpr_clean, color="#06b6d4", lw=2.5, label=f"Clean Baseline (AUC = {auc_clean:.4f})")
    plt.plot(fpr_amr, tpr_amr, color="#8b5cf6", lw=2.2, label=f"AMR-NB Telephony (AUC = {auc_amr:.4f})")
    plt.plot(fpr_comb, tpr_comb, color="#ec4899", lw=2.2, label=f"Combined Severe (AUC = {auc_comb:.4f})")
    
    plt.plot([0, 1], [0, 1], color="#4b5563", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="semibold", labelpad=8)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="semibold", labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.85)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    save_path = "docs/roc_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC Curves plotted and saved to: {save_path}")

if __name__ == "__main__":
    main()
