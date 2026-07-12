import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.features.load_dataset import get_split_dataset
from src.features.extract_features import extract_features
from src.pipeline.degrade_audio import process_file

def evaluate_predictions(samples, codec, packet_loss, jitter, threshold, model):
    """Generates ground truth and predictions for a specific degradation condition."""
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
    
    # Predict probabilities
    probs = model.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    # Clean temp dir
    try: os.rmdir(temp_dir)
    except: pass
    
    return y_true, y_pred

def plot_cm(y_true, y_pred, title, save_path):
    """Plots a beautiful styled confusion matrix using matplotlib."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap=plt.cm.Purples, alpha=0.8)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Spoof/Fake", "Authentic/Human"], fontsize=11, fontweight="medium")
    ax.set_yticklabels(["Spoof/Fake", "Authentic/Human"], fontsize=11, fontweight="medium")
    
    # Annotate counts and percentages
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", 
                    color="white" if cm[i, j] > total/3 else "black",
                    fontsize=12, fontweight="bold")
            
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(False)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix: {save_path}")

def draw_architecture_diagram(save_path):
    """Draws a clean system architecture diagram flow chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    
    # Helper to draw boxes
    def draw_box(x, y, w, h, text, color="#8b5cf6"):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#6d28d9", 
                             alpha=0.15, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10, 
                fontweight="bold", color="#1e1b4b", wrap=True)
        
    # Helper to draw arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#4b5563", lw=2, shrinkA=5, shrinkB=5))
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            if abs(y1 - y2) < 0.01:
                # Horizontal arrow: place text above
                ax.text(mx, my + 0.02, label, ha="center", va="bottom", fontsize=8, color="#4b5563", fontweight="semibold")
            else:
                # Vertical arrow: place text to the side
                ax.text(mx + 0.02, my, label, ha="left", va="center", fontsize=8, color="#4b5563", fontweight="semibold")

    # Draw boxes
    draw_box(0.05, 0.7, 0.22, 0.15, "1. Audio Capture\n(1.5s Rolling WAV\nvia Stream/Dashboard)", "#3b82f6")
    draw_box(0.35, 0.7, 0.22, 0.15, "2. Passive Monitor\n(120-dim MFCC +\nLogistic Regression)", "#10b981")
    draw_box(0.68, 0.7, 0.22, 0.15, "3. Challenge Trigger\n(Score < Threshold?\nSpontaneous Prompt)", "#f59e0b")
    
    draw_box(0.68, 0.3, 0.22, 0.15, "4. Offline Whisper\n(Local STT Engine\nTranscribes Response)", "#ef4444")
    draw_box(0.35, 0.3, 0.22, 0.15, "5. ML Score Fuser\n(Logistic Regression\nLearns Hyperplane)", "#8b5cf6")
    draw_box(0.05, 0.3, 0.22, 0.15, "6. Verdict Alert\n(Block Injection &\nReplay Attacks)", "#10b981")
    
    # Draw paths / arrows
    draw_arrow(0.27, 0.775, 0.35, 0.775, "Audio Frame")
    draw_arrow(0.57, 0.775, 0.68, 0.775, "Authenticity < Threshold")
    draw_arrow(0.79, 0.7, 0.79, 0.45, "User Speech Response")
    draw_arrow(0.68, 0.375, 0.57, 0.375, "Content Match %")
    draw_arrow(0.35, 0.375, 0.27, 0.375, "Fused Decision")
    
    # Titles
    ax.text(0.5, 0.95, "VoxGuard System Verification Flow", ha="center", va="center", 
            fontsize=14, fontweight="bold", color="#1e1b4b")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved architecture diagram: {save_path}")

def main():
    print("==================================================")
    print("GENERATING VERIFICATION IMAGES & PLOTS")
    print("==================================================")
    
    # Load model
    model_path = "src/models/detector.pkl"
    if not os.path.exists(model_path):
        print("Error: Trained model detector.pkl not found.")
        return
    model = joblib.load(model_path)
    
    # Load eval dataset
    eval_samples = get_split_dataset("eval")
    if not eval_samples:
        print("Error: No evaluation samples resolved.")
        return
        
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    # 1. Generate Clean Confusion Matrix (threshold = 0.7943)
    y_true_clean, y_pred_clean = evaluate_predictions(eval_samples, "none", 0.0, 0, 0.7943, model)
    plot_cm(y_true_clean, y_pred_clean, "Confusion Matrix: Clean Baseline", os.path.join(docs_dir, "confusion_matrix_clean.png"))
    
    # 2. Generate AMR-NB Confusion Matrix (threshold = 0.9791)
    y_true_amr, y_pred_amr = evaluate_predictions(eval_samples, "amr", 0.0, 0, 0.9791, model)
    plot_cm(y_true_amr, y_pred_amr, "Confusion Matrix: AMR-NB Telephony", os.path.join(docs_dir, "confusion_matrix_amr.png"))
    
    # 3. Generate Combined Severe Telephony CM (threshold = 0.9038)
    y_true_comb, y_pred_comb = evaluate_predictions(eval_samples, "amr", 0.15, 30, 0.9038, model)
    plot_cm(y_true_comb, y_pred_comb, "Confusion Matrix: Combined Severe", os.path.join(docs_dir, "confusion_matrix_combined.png"))
    
    # 4. Generate System Architecture Diagram
    draw_architecture_diagram(os.path.join(docs_dir, "architecture_diagram.png"))
    
    # 5. Output Confusion Matrix counts in JSON format for the SOC web console API
    import json
    cm_clean = confusion_matrix(y_true_clean, y_pred_clean)
    cm_amr = confusion_matrix(y_true_amr, y_pred_amr)
    cm_comb = confusion_matrix(y_true_comb, y_pred_comb)
    
    cm_data = {
        "clean": {"tn": int(cm_clean[0, 0]), "fp": int(cm_clean[0, 1]), "fn": int(cm_clean[1, 0]), "tp": int(cm_clean[1, 1])},
        "amr": {"tn": int(cm_amr[0, 0]), "fp": int(cm_amr[0, 1]), "fn": int(cm_amr[1, 0]), "tp": int(cm_amr[1, 1])},
        "combined": {"tn": int(cm_comb[0, 0]), "fp": int(cm_comb[0, 1]), "fn": int(cm_comb[1, 0]), "tp": int(cm_comb[1, 1])}
    }
    
    json_path = os.path.join(docs_dir, "confusion_matrices.json")
    with open(json_path, "w") as f:
        json.dump(cm_data, f, indent=4)
    print(f"Saved confusion matrices JSON to: {json_path}")
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
