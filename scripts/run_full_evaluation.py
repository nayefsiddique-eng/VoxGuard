import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from src.features.load_dataset import get_split_dataset
from src.features.extract_features import extract_features
from src.models.baseline_detector import compute_eer
from src.pipeline.degrade_audio import process_file

def run_eval_on_dataset(samples, feature_type="mfcc", model=None):
    """Utility to run model prediction on a list of samples."""
    X = []
    y = []
    for s in samples:
        feats = extract_features(s["filepath"], feature_type=feature_type, use_cache=True)
        X.append(feats)
        y.append(s["label"])
        
    X = np.array(X)
    y = np.array(y)
    
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    eer = compute_eer(y, probs)
    return acc, auc, eer

def main():
    print("==================================================")
    print("RUNNING DYNAMIC DEGRADATION EVALUATION PIPELINE")
    print("==================================================")
    
    # 1. Load best detector
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/models/detector.pkl"))
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the baseline first.")
        return
        
    model = joblib.load(model_path)
    print(f"Loaded trained detector model.")
    
    # 2. Load clean evaluation dataset
    eval_samples = get_split_dataset("eval")
    if not eval_samples:
        print("Error: No evaluation samples resolved.")
        return
        
    print(f"Loaded {len(eval_samples)} clean evaluation samples.")
    
    # 3. Benchmark Clean Baseline
    print("\nEvaluating Clean baseline...")
    clean_acc, clean_auc, clean_eer = run_eval_on_dataset(eval_samples, feature_type="mfcc", model=model)
    print(f"  [Clean] Acc: {clean_acc*100:.1f}%, AUC: {clean_auc:.4f}, EER: {clean_eer*100:.2f}%")
    
    # 4. Prepare degraded conditions
    degraded_conditions = {
        "AMR-NB Codec": {"codec": "amr", "packet_loss": 0.0, "jitter": 0},
        "Opus Codec": {"codec": "opus", "packet_loss": 0.0, "jitter": 0},
        "Packet Loss Only": {"codec": "none", "packet_loss": 0.05, "jitter": 10},
        "Combined (AMR-NB + Jitter + Loss)": {"codec": "amr", "packet_loss": 0.08, "jitter": 15}
    }
    
    eval_degraded_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/degraded_eval"))
    os.makedirs(eval_degraded_dir, exist_ok=True)
    
    degradation_results = {}
    
    # Run degradation and evaluate
    for name, params in degraded_conditions.items():
        print(f"\nProcessing eval split through: {name}...")
        degraded_samples = []
        
        for idx, s in enumerate(eval_samples):
            # Degrade file
            # If codec is "none", we copy the original file structure and apply loss only
            degraded_path = process_file(
                s["filepath"],
                eval_degraded_dir,
                codec=params["codec"],
                packet_loss=params["packet_loss"],
                jitter=params["jitter"]
            )
            
            degraded_samples.append({
                "filepath": degraded_path,
                "label": s["label"]
            })
            
            if (idx + 1) % 15 == 0:
                print(f"  Processed {idx + 1}/{len(eval_samples)} files...")
                
        # Evaluate degraded samples
        acc, auc, eer = run_eval_on_dataset(degraded_samples, feature_type="mfcc", model=model)
        print(f"  [{name}] Acc: {acc*100:.1f}%, AUC: {auc:.4f}, EER: {eer*100:.2f}%")
        
        # Calculate gaps
        acc_gap = clean_acc - acc
        eer_gap = eer - clean_eer
        
        degradation_results[name] = {
            "acc": acc, "auc": auc, "eer": eer,
            "acc_gap": acc_gap, "eer_gap": eer_gap
        }
        
    # 5. Save results to docs/results_degraded.md
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs"))
    results_path = os.path.join(docs_dir, "results_degraded.md")
    
    with open(results_path, "w") as f:
        f.write("# Realistic Channel Degradation Benchmarks\n\n")
        f.write("This table shows the performance drops when evaluation audios are compressed and degraded by codecs and network channel simulation.\n\n")
        f.write("| Channel Condition | Accuracy | EER | Accuracy Drop | EER Increase |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Clean Baseline** | **{clean_acc*100:.1f}%** | **{clean_eer*100:.2f}%** | - | - |\n")
        for name, res in degradation_results.items():
            f.write(f"| {name} | {res['acc']*100:.1f}% | {res['eer']*100:.2f}% | -{res['acc_gap']*100:.1f}% | +{res['eer_gap']*100:.2f}% |\n")
            
        f.write("\n> [!WARNING]\n")
        f.write("> Codec compression (AMR-NB) and combined channel errors significantly increase Equal Error Rates. This highlights the importance of evaluating deepfake detection algorithms under compressed telephony scenarios rather than relying on clean lab-quality audio.\n")
        
    print(f"\nGenerated degraded results benchmarks file: {results_path}")
    
    # 6. Cleanup degraded_eval dir
    print("Cleaning up temporary evaluation folders...")
    for f in os.listdir(eval_degraded_dir):
        os.remove(os.path.join(eval_degraded_dir, f))
    os.rmdir(eval_degraded_dir)
    print("Done.")

if __name__ == "__main__":
    main()
