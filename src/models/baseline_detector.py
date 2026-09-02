import os
import argparse
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from src.features.load_dataset import get_split_dataset, print_split_statistics
from src.features.extract_features import extract_features

def compute_eer(y_true, y_prob):
    """
    Computes Equal Error Rate (EER) from classification probabilities.
    y_true: binary labels (1 for Real/bonafide, 0 for Spoof)
    y_prob: predicted probability of being Real/bonafide (class 1)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer

def load_features_and_labels(split_name, feature_type="wav2vec2"):
    """Loads dataset split and extracts/caches features for all files."""
    samples = get_split_dataset(split_name)
    if not samples:
        return None, None
        
    print(f"Extracting '{feature_type}' features for '{split_name}' split ({len(samples)} files)...")
    X = []
    y = []
    
    for idx, s in enumerate(samples):
        # We explicitly pass the feature_type
        feats = extract_features(s["filepath"], feature_type=feature_type, use_cache=True)
        X.append(feats)
        y.append(s["label"])
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(samples)} files...")
            
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="Train baseline spoofing detector.")
    parser.add_argument("--features", type=str, default="mfcc", choices=["wav2vec2", "mfcc"],
                        help="Feature type to use (mfcc or wav2vec2).")
    args = parser.parse_args()
    
    feature_type = args.features
    
    print("==================================================")
    print(f"STEP 3: Training & Comparing Models on Real {feature_type.upper()} Embeddings")
    print("==================================================")
    
    # 1. Load splits
    X_train, y_train = load_features_and_labels("train", feature_type=feature_type)
    X_dev, y_dev = load_features_and_labels("dev", feature_type=feature_type)
    X_eval, y_eval = load_features_and_labels("eval", feature_type=feature_type)
    
    if X_train is None or X_dev is None or X_eval is None:
        print("Error: Missing dataset splits. Ensure download completed successfully.")
        return
        
    print(f"\nTrain size: {X_train.shape}, Dev size: {X_dev.shape}, Eval size: {X_eval.shape}")
    
    # 2. Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = {}
    best_dev_eer = 1.0
    best_model_name = None
    best_model = None
    best_scaler = None
    
    # Fit scaler on train, apply to dev and eval
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_eval_scaled = scaler.transform(X_eval)
    
    # 3. Train and benchmark each model
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, y_train)
        
        # Predict on Dev
        dev_probs = clf.predict_proba(X_dev_scaled)[:, 1]
        dev_preds = clf.predict(X_dev_scaled)
        dev_acc = accuracy_score(y_dev, dev_preds)
        dev_auc = roc_auc_score(y_dev, dev_probs)
        dev_eer = compute_eer(y_dev, dev_probs)
        
        # Predict on Eval
        eval_probs = clf.predict_proba(X_eval_scaled)[:, 1]
        eval_preds = clf.predict(X_eval_scaled)
        eval_acc = accuracy_score(y_eval, eval_preds)
        eval_auc = roc_auc_score(y_eval, eval_probs)
        eval_eer = compute_eer(y_eval, eval_probs)
        
        print(f"  [Dev]  Acc: {dev_acc*100:.1f}%, AUC: {dev_auc:.4f}, EER: {dev_eer*100:.2f}%")
        print(f"  [Eval] Acc: {eval_acc*100:.1f}%, AUC: {eval_auc:.4f}, EER: {eval_eer*100:.2f}%")
        
        results[name] = {
            "dev_acc": dev_acc, "dev_auc": dev_auc, "dev_eer": dev_eer,
            "eval_acc": eval_acc, "eval_auc": eval_auc, "eval_eer": eval_eer
        }
        
        # Track best model based on Dev EER
        if dev_eer < best_dev_eer:
            best_dev_eer = dev_eer
            best_model_name = name
            
            # Create a pipeline to bundle the scaler and the model
            from sklearn.pipeline import Pipeline
            best_model = Pipeline([('scaler', scaler), ('clf', clf)])
            
    # Save the best model
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, "detector.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest Model: {best_model_name} (Dev EER: {best_dev_eer*100:.2f}%)")
    print(f"Saved best model to: {model_path}")
    
    # 4. Generate results_clean.md markdown table
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs"))
    os.makedirs(docs_dir, exist_ok=True)
    results_path = os.path.join(docs_dir, "results_clean.md")
    
    with open(results_path, "w") as f:
        f.write("# Clean Audio Spoof Detection Benchmarks\n\n")
        f.write(f"This table compares spoofing detection classifiers trained on **{feature_type.upper()}** features.\n\n")
        f.write("| Model | Dev Accuracy | Dev AUC | Dev EER | Eval Accuracy | Eval AUC | Eval EER |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, metrics in results.items():
            f.write(f"| {name} | {metrics['dev_acc']*100:.1f}% | {metrics['dev_auc']:.4f} | {metrics['dev_eer']*100:.2f}% | {metrics['eval_acc']*100:.1f}% | {metrics['eval_auc']:.4f} | {metrics['eval_eer']*100:.2f}% |\n")
        f.write(f"\n*Best Model: **{best_model_name}** selected for production verification based on lowest Evaluation EER.*\n")
        
    print(f"Generated results clean benchmarks file: {results_path}")

if __name__ == "__main__":
    main()
