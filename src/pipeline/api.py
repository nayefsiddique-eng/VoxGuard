import os
import shutil
import tempfile
import joblib
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.features.extract_features import extract_features
from src.challenge_engine.phrase_challenge import generate_challenge_phrase
from src.challenge_engine.response_scorer import score_response
from src.pipeline.fusion import fuse_scores
from src.pipeline.degrade_audio import process_file, get_ffmpeg_path

app = FastAPI(
    title="Real-Time AI Voice Clone and Deepfake Call Verification API",
    description="Backend API for passive call verification and challenge-response authentication.",
    version="1.0.0"
)

# Global variables to store the loaded model
CLF = None
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/detector.pkl"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))

@app.on_event("startup")
def startup_event():
    global CLF
    if os.path.exists(MODEL_PATH):
        try:
            CLF = joblib.load(MODEL_PATH)
            print(f"Successfully loaded detector model from: {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        print(f"Warning: Detector model not found at {MODEL_PATH}. Endpoints will fail until model is trained.")

def get_classifier():
    if CLF is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the baseline detector first."
        )
    return CLF

@app.post("/verify-passive")
async def verify_passive(
    file: UploadFile = File(...),
    degradation: str = Query("none", enum=["none", "amr", "opus", "packet_loss", "combined"])
):
    """
    Passively monitors live call audio chunks.
    Optionally applies dynamic degradation on-the-fly to simulate call channel effects.
    """
    clf = get_classifier()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        if degradation != "none":
            temp_dir = tempfile.gettempdir()
            
            if degradation == "amr":
                codec, pl, j = "amr", 0.0, 0
            elif degradation == "opus":
                codec, pl, j = "opus", 0.0, 0
            elif degradation == "packet_loss":
                codec, pl, j = "none", 0.05, 10
            elif degradation == "combined":
                codec, pl, j = "amr", 0.08, 15
                
            degraded_file = process_file(
                temp_path,
                temp_dir,
                codec=codec,
                packet_loss=pl,
                jitter=j
            )
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            temp_path = degraded_file
            
        features = extract_features(temp_path, feature_type="mfcc", use_cache=False)
        features = features.reshape(1, -1)
        
        probs = clf.predict_proba(features)[0]
        passive_score = float(probs[1])
        
        trigger_challenge = passive_score < 0.5
        
        return {
            "filename": file.filename,
            "degradation_applied": degradation,
            "passive_authenticity_score": round(passive_score, 4),
            "status": "SUSPICIOUS" if trigger_challenge else "CLEAN",
            "trigger_challenge": trigger_challenge,
            "message": "Call voice is suspicious. Please issue a verification challenge!" if trigger_challenge else "Voice appears authentic."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction or prediction error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/challenge/request")
@app.post("/challenge/request")
def get_challenge():
    """Generates a random challenge phrase for dynamic verification."""
    phrase = generate_challenge_phrase()
    return {
        "challenge_phrase": phrase,
        "instructions": "Prompt the caller to speak/respond to this phrase exactly."
    }

@app.post("/challenge/verify")
async def verify_challenge(
    file: UploadFile = File(...),
    phrase: str = Form(...),
    passive_score: float = Form(...)
):
    """
    Evaluates the response audio to a dynamic challenge.
    Fuses the response fidelity score with the initial passive score using the trained ML fuser.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        res = score_response(temp_path, phrase, model_path=MODEL_PATH)
        
        fusion_result = fuse_scores(
            passive_score=passive_score,
            challenge_score=res["fused_challenge_score"],
            challenge_weight=0.5
        )
        
        return {
            "challenge_phrase": phrase,
            "response_metrics": res,
            "fusion_metrics": fusion_result,
            "final_status": fusion_result["status"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failure: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# WebSocket Real-Time Audio Streaming Endpoint
@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    clf = get_classifier()
    
    # Running byte array to hold streamed audio chunks
    audio_bytes_accumulator = bytearray()
    chunk_counter = 0
    
    print("WebSocket client connected for real-time verification.")
    
    try:
        while True:
            # Receive binary chunk from client mic stream
            data = await websocket.receive_bytes()
            audio_bytes_accumulator.extend(data)
            chunk_counter += 1
            
            # Analyze every ~1.5 seconds of accumulated audio chunks
            # If the user streams at 44.1kHz or standard compressed sizes,
            # we run inference once we've collected enough bytes (e.g. > 48KB of data)
            if len(audio_bytes_accumulator) >= 64000:
                # Write current buffer to a temp container file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
                    temp_input.write(audio_bytes_accumulator)
                    temp_input_path = temp_input.name
                
                temp_wav_path = temp_input_path.replace(".webm", ".wav")
                
                # Convert buffer to a standard WAV (16kHz mono) using ffmpeg
                ffmpeg_cmd = get_ffmpeg_path()
                if ffmpeg_cmd:
                    cmd = [
                        ffmpeg_cmd, "-y", "-i", temp_input_path,
                        "-ar", "16000", "-ac", "1", "-f", "wav", temp_wav_path
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Verify converted WAV file exists
                if os.path.exists(temp_wav_path):
                    try:
                        # Extract features and predict
                        features = extract_features(temp_wav_path, feature_type="mfcc", use_cache=False)
                        features = features.reshape(1, -1)
                        
                        probs = clf.predict_proba(features)[0]
                        passive_score = float(probs[1]) # Prob of being Real
                        
                        # Send back score JSON
                        await websocket.send_json({
                            "chunk_id": chunk_counter,
                            "passive_authenticity_score": round(passive_score, 4),
                            "trigger_challenge": passive_score < 0.5,
                            "status": "SUSPICIOUS" if passive_score < 0.5 else "CLEAN"
                        })
                    except Exception as e:
                        # Keep connection alive even on single chunk decoding failure
                        pass
                    finally:
                        try:
                            os.remove(temp_wav_path)
                        except:
                            pass
                
                try:
                    os.remove(temp_input_path)
                except:
                    pass
                
                # Keep a sliding window: discard older bytes to prevent buffer growth
                # Retain the last ~32KB to allow overlapping feature extraction windows
                audio_bytes_accumulator = audio_bytes_accumulator[-32000:]
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"WebSocket execution error: {e}")

# Serve the data directory statically at /data
if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# Serve the frontend static directory at the root /
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Voice deepfake backend running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.pipeline.api:app", host="127.0.0.1", port=8000, reload=True)
