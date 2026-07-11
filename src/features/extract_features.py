import os
import glob
import hashlib
import numpy as np
import soundfile as sf
import librosa

# Import torch and transformers
try:
    import torch
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Global model cache to avoid reloading model parameters on every function call
W2V_PROCESSOR = None
W2V_MODEL = None

def get_wav2vec2_model(model_name="facebook/wav2vec2-base"):
    """Loads and caches the Wav2Vec2 processor and model on CPU."""
    global W2V_PROCESSOR, W2V_MODEL
    if W2V_PROCESSOR is None or W2V_MODEL is None:
        print(f"Loading Wav2Vec2 model: {model_name} onto CPU...")
        W2V_PROCESSOR = Wav2Vec2Processor.from_pretrained(model_name)
        W2V_MODEL = Wav2Vec2Model.from_pretrained(model_name)
        W2V_MODEL.eval() # Set to evaluation mode (freezes dropout/batchnorm)
    return W2V_PROCESSOR, W2V_MODEL

def get_cache_path(audio_path, feature_type):
    """Generates a cached filepath based on the source file's absolute path hash and type."""
    # Ensure features cache directory exists
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../data/features/{feature_type}"))
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate MD5 hash of absolute filepath
    abs_path = os.path.abspath(audio_path)
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    
    # Check modification time to invalidate cache if file changes
    mtime = int(os.path.getmtime(abs_path))
    
    return os.path.join(cache_dir, f"{path_hash}_{mtime}.npy")

def extract_mfcc(audio_path, n_mfcc=20):
    """Extracts MFCC features + delta + delta-delta averaged over time (120 dimensions)."""
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    features = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1),
        np.mean(delta2_mfcc, axis=1), np.std(delta2_mfcc, axis=1)
    ])
    return features

def extract_wav2vec2(audio_path):
    """
    Extracts frozen Wav2Vec2 base embeddings on CPU.
    Performs average pooling over the temporal sequence to produce a 768-dimensional feature vector.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("huggingface transformers or torch is not installed. Use MFCC features instead.")
        
    y, sr = librosa.load(audio_path, sr=16000)
    processor, model = get_wav2vec2_model()
    
    # Preprocess inputs
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Sequence dimensions: (batch_size, sequence_length, hidden_size=768)
        embeddings = outputs.last_hidden_state
        # Mean pooling over time sequence (dimension 1) to get a static feature vector
        pooled_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()
        
    return pooled_embedding

def extract_features(audio_path, feature_type="wav2vec2", use_cache=True):
    """
    Unified extraction interface with caching and fallbacks.
    - If use_cache is True, reads/writes cached feature numpy arrays.
    - Falls back to MFCC if Wav2Vec2 model loading fails.
    """
    # 1. Check cache first
    if use_cache:
        cache_file = get_cache_path(audio_path, feature_type)
        if os.path.exists(cache_file):
            return np.load(cache_file)
            
    # 2. Extract features
    if feature_type == "wav2vec2" and HAS_TRANSFORMERS:
        try:
            feats = extract_wav2vec2(audio_path)
            if use_cache:
                np.save(cache_file, feats)
            return feats
        except Exception as e:
            print(f"  [Wav2Vec2 extraction failed: {e}]. Falling back to MFCC.")
            
    # Fallback / Default
    feats = extract_mfcc(audio_path)
    if use_cache:
        # Cache the fallback features under 'mfcc' directory
        fallback_cache = get_cache_path(audio_path, "mfcc")
        np.save(fallback_cache, feats)
    return feats

if __name__ == "__main__":
    # Test feature extraction and caching
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    flac_files = glob.glob(os.path.join(raw_dir, "**", "*.flac"), recursive=True)
    
    if flac_files:
        test_file = flac_files[0]
        print(f"Testing extraction on real FLAC: {test_file}")
        
        # Test MFCC extraction
        mfcc_feats = extract_features(test_file, feature_type="mfcc", use_cache=True)
        print(f"  MFCC features shape: {mfcc_feats.shape} (Cached)")
        
        # Test Wav2Vec2 extraction
        w2v_feats = extract_features(test_file, feature_type="wav2vec2", use_cache=True)
        print(f"  Wav2Vec2 features shape: {w2v_feats.shape} (Cached)")
        
        # Test loading from cache (should be immediate)
        import time
        t0 = time.time()
        w2v_feats_cached = extract_features(test_file, feature_type="wav2vec2", use_cache=True)
        t1 = time.time()
        print(f"  Cache read took: {t1 - t0:.6f} seconds")
    else:
        print("No real FLAC files found in data/raw/. Run subset downloader first.")
