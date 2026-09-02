import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
FUSER_PATH = os.path.join(MODEL_DIR, "fuser.pkl")

def get_trained_fuser():
    """Loads the trained fuser model if available, otherwise returns None."""
    if os.path.exists(FUSER_PATH):
        try:
            return joblib.load(FUSER_PATH)
        except Exception:
            pass
    return None

def fuse_scores(passive_score, challenge_score=None, challenge_weight=0.5):
    """
    Combines passive score and challenge score.
    If a trained fuser model (fuser.pkl) is found, uses it to predict authentic probability.
    Otherwise, falls back to the weighted-average baseline.
    """
    if challenge_score is None:
        # No challenge was conducted
        return {
            "passive_score": round(passive_score, 4),
            "challenge_score": None,
            "final_authenticity_score": round(passive_score, 4),
            "status": "SUSPICIOUS/FAKE" if passive_score < 0.5 else "AUTHENTIC/REAL",
            "method": "Passive Only"
        }
        
    fuser = get_trained_fuser()
    
    if fuser is not None:
        # Run trained fuser prediction
        features = np.array([[passive_score, challenge_score]])
        # predict_proba returns [prob_fake, prob_real]
        final_score = float(fuser.predict_proba(features)[0][1])
        status_label = fuser.predict(features)[0]
        status = "AUTHENTIC/REAL" if status_label == 1 else "SUSPICIOUS/FAKE"
        method = "Trained ML Fuser"
    else:
        # Fallback to fixed weight
        final_score = ((1.0 - challenge_weight) * passive_score) + (challenge_weight * challenge_score)
        status = "SUSPICIOUS/FAKE" if final_score < 0.5 else "AUTHENTIC/REAL"
        method = "Fixed-Weight Baseline"
        
    return {
        "passive_score": round(passive_score, 4),
        "challenge_score": round(challenge_score, 4),
        "final_authenticity_score": round(final_score, 4),
        "status": status,
        "method": method
    }

def train_fuser(passive_clf=None, response_scorer_fn=None):
    """
    Synthetic fuser training has been removed for scientific integrity.
    The system now relies on the transparent, deterministic Fixed-Weight Baseline.
    """
    print("==================================================")
    print("STEP 6: CONFIGURING SCORE FUSION ALGORITHM")
    print("==================================================")
    
    if os.path.exists(FUSER_PATH):
        try:
            os.remove(FUSER_PATH)
        except Exception:
            pass
            
    print("Enforced transparent Fixed-Weight Baseline for evaluation integrity.")
    print("Fusion model training disabled.")

if __name__ == "__main__":
    train_fuser()
