import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

def analyze_feature_importance():
    print("==================================================")
    print("FEATURE EXPLAINABILITY & IMPORTANCE ANALYSIS")
    print("==================================================")
    
    model_path = "c:/Users/Admin pc/Desktop/voice detection/voice-deepfake-verify/src/models/detector.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the baseline first.")
        return
        
    clf = joblib.load(model_path)
    model_type = type(clf).__name__
    print(f"Loaded classifier type: {model_type}")
    
    # Expose feature labels
    # 120 features: MFCC Mean (0-19), MFCC Std (20-39), Delta Mean (40-59), Delta Std (60-79), Delta-Delta Mean (80-99), Delta-Delta Std (100-119)
    feature_names = []
    groups = ["MFCC Mean", "MFCC Std", "Delta Mean", "Delta Std", "Delta-Delta Mean", "Delta-Delta Std"]
    
    for group in groups:
        for band in range(1, 21):
            feature_names.append(f"{group} B{band}")
            
    # Calculate feature importances based on model type
    if hasattr(clf, "coef_"):
        # For Logistic Regression, importance is the absolute weight coefficient
        importances = np.abs(clf.coef_[0])
        importance_type = "Coefficient Magnitude (Absolute)"
    elif hasattr(clf, "feature_importances_"):
        # For Random Forest, use tree Gini importances
        importances = clf.feature_importances_
        importance_type = "Mean Decrease in Impurity (Gini)"
    else:
        print("Model does not support coefficient or feature importance attributes.")
        return
        
    # Group importances by category
    group_totals = {}
    for i, name in enumerate(feature_names):
        grp = name.split(" B")[0]
        group_totals[grp] = group_totals.get(grp, 0) + importances[i]
        
    print(f"\nFeature Group Importances ({importance_type}):")
    for grp, val in sorted(group_totals.items(), key=lambda x: x[1], reverse=True):
        print(f"  {grp:<20}: {val:.4f}")
        
    # Find top 15 individual features
    top_indices = np.argsort(importances)[::-1][:15]
    
    print("\nTop 15 Individual Audio Features Contributing to Classification:")
    for idx in top_indices:
        print(f"  Feature #{idx:<3} | {feature_names[idx]:<22}: {importances[idx]:.4f}")
        
    # Plotting top 15 features
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(15)
    ax.barh(y_pos, importances[top_indices][::-1], color='#8b5cf6', alpha=0.9, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_indices][::-1], fontsize=10, fontweight='medium')
    
    ax.set_title("Top 15 Speech Features for AI Voice Spoof Classification", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel(f"Feature Importance Score ({importance_type})", fontsize=11, fontweight='semibold', labelpad=10)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Save Image
    docs_dir = "c:/Users/Admin pc/Desktop/voice detection/voice-deepfake-verify/docs"
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGenerated and saved feature explainability plot to: {plot_path}")

if __name__ == "__main__":
    analyze_feature_importance()
