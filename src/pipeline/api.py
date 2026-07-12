import os
import time
import shutil
import tempfile
import joblib
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.features.extract_features import extract_features
from src.challenge_engine.phrase_challenge import generate_challenge_phrase
from src.challenge_engine.response_scorer import score_response, get_whisper_model
from src.pipeline.fusion import fuse_scores
from src.pipeline.degrade_audio import process_file, get_ffmpeg_path

app = FastAPI(
    title="VoxGuard Verification Pipeline API",
    description="Backend API for continuous passive voice clone detection, challenge dispatching, and score fusion verification.",
    version="1.0.0"
)

# Pydantic Schemas for Swagger / API documentation
class PassiveVerifyResponse(BaseModel):
    filename: str = Field(..., description="Name of the processed file.")
    degradation_applied: str = Field(..., description="Type of degradation/codec applied.")
    passive_authenticity_score: float = Field(..., description="Authenticity score from passive monitor (0.0 to 1.0).")
    status: str = Field(..., description="Call status: CLEAN or SUSPICIOUS.")
    trigger_challenge: bool = Field(..., description="Whether the challenge mechanism is recommended.")
    message: str = Field(..., description="Detailed explanation of the verdict.")

class ChallengeRequestResponse(BaseModel):
    challenge_phrase: str = Field(..., description="Synthesized phrase issued to caller.")
    instructions: str = Field(..., description="Instructions for verification step.")

class ResponseMetrics(BaseModel):
    voice_authenticity_score: float
    content_adherence_score: float
    fused_challenge_score: float
    is_suspicious: bool
    replay_attack_detected: bool
    transcript: str

class FusionMetrics(BaseModel):
    passive_score: float
    challenge_score: float
    final_authenticity_score: float
    status: str
    method: str

class ChallengeVerifyResponse(BaseModel):
    challenge_phrase: str
    response_metrics: ResponseMetrics
    fusion_metrics: FusionMetrics
    final_status: str

class SystemStatusResponse(BaseModel):
    detector_loaded: bool = Field(..., description="Is the passive Logistic Regression model loaded.")
    whisper_loaded: bool = Field(..., description="Is the local Whisper model loaded in memory.")
    uptime_seconds: float = Field(..., description="API server uptime in seconds.")
    last_checked_timestamp: str = Field(..., description="RFC-formatted timestamp of the last check.")

class EventLogEntry(BaseModel):
    timestamp: str = Field(..., description="Logical time (HH:MM:SS) of the event.")
    severity: str = Field(..., description="Event severity: INFO, SUCCESS, WARNING, ERROR.")
    component: str = Field(..., description="Source subsystem: PASSIVE_MONITOR, CHALLENGE_ENGINE, FUSER, SYSTEM.")
    message: str = Field(..., description="Readable event log description.")

class CallHistoryEntry(BaseModel):
    timestamp: str = Field(..., description="Date and time of attempt.")
    filename: str = Field(..., description="Name of audio sample evaluated.")
    passive_score: float = Field(..., description="Authenticity score from passive scan.")
    challenge_score: float | None = Field(None, description="Response adherence score, if challenged.")
    final_score: float | None = Field(None, description="Fused score output.")
    verdict: str = Field(..., description="Call status result.")

class ConfusionMatrixData(BaseModel):
    tp: int
    tn: int
    fp: int
    fn: int

class DegradationRow(BaseModel):
    condition: str
    threshold: float
    accuracy: float
    eer: float
    auc: float
    accuracy_drop: float
    eer_increase: float

class ResultsSummaryResponse(BaseModel):
    confusion_matrices: dict[str, ConfusionMatrixData]
    degradation_benchmarks: list[DegradationRow]

# Middleware for response latency logging
@app.middleware("http")
async def log_latency_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    response.headers["X-Response-Time-Seconds"] = f"{duration:.4f}"
    # Log duration cleanly
    print(f"[Latency Diagnostic] Path: {request.url.path} | Method: {request.method} | Duration: {duration:.4f}s")
    return response

# Global variables to store the loaded model and session logs
CLF = None
START_TIME = time.time()
SESSION_EVENTS = []
SESSION_HISTORY = []

def add_event(message: str, severity: str = "INFO", component: str = "SYSTEM"):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    SESSION_EVENTS.append(
        EventLogEntry(
            timestamp=timestamp,
            severity=severity,
            component=component,
            message=message
        )
    )

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/detector.pkl"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))

# Helper for audio file validation
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
def validate_audio_file(file: UploadFile):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    # Check if size is empty or potentially too large (e.g. 10MB limit)
    # Note: FastAPI file.size is metadata. Let's do a basic size lookup or handle safely.

@app.on_event("startup")
def startup_event():
    global CLF
    add_event("VoxGuard call verification engine starting up...", "INFO", "SYSTEM")
    # 1. Load baseline passive detector
    if os.path.exists(MODEL_PATH):
        try:
            CLF = joblib.load(MODEL_PATH)
            add_event(f"Successfully loaded Logistic Regression passive detector from: {os.path.basename(MODEL_PATH)}", "SUCCESS", "SYSTEM")
            print(f"Successfully loaded detector model from: {MODEL_PATH}")
        except Exception as e:
            add_event(f"Failed to load detector model: {e}", "ERROR", "SYSTEM")
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        add_event(f"Warning: Passive detector not found at {MODEL_PATH}.", "WARNING", "SYSTEM")
        print(f"Warning: Detector model not found at {MODEL_PATH}.")
        
    # 2. Pre-load local Whisper tiny model onto CPU
    try:
        add_event("Loading offline Whisper 'tiny' model onto CPU...", "INFO", "SYSTEM")
        get_whisper_model()
        add_event("Successfully pre-loaded Whisper model. Ready for audio challenges.", "SUCCESS", "SYSTEM")
        print("Successfully pre-loaded Whisper model on startup.")
    except Exception as e:
        add_event(f"Error preloading Whisper model: {e}", "ERROR", "SYSTEM")
        print(f"Error preloading Whisper model: {e}")

def get_classifier():
    if CLF is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the baseline detector first."
        )
    return CLF

@app.post("/verify-passive", response_model=PassiveVerifyResponse, summary="Passive Voice Deepfake Check", description="Passively analyzes 1.5s rolling audio frames using speaker-independent Logistic Regression classification.")
async def verify_passive(
    file: UploadFile = File(..., description="Audio chunk to analyze (WAV/FLAC/MP3/OGG)."),
    degradation: str = Query("none", enum=["none", "amr", "opus", "packet_loss", "combined"], description="Channel condition profile to apply.")
):
    """
    Passively monitors live call audio chunks.
    Optionally applies dynamic degradation on-the-fly to simulate call channel effects.
    """
    validate_audio_file(file)
    clf = get_classifier()
    
    add_event(f"Received passive check request for file: {file.filename} (channel condition: {degradation})", "INFO", "PASSIVE_MONITOR")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        if degradation != "none":
            add_event(f"Applying codec degradation simulation: {degradation}", "INFO", "PASSIVE_MONITOR")
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
            
        add_event("Extracting acoustic Cepstral features (120-dim MFCCs)...", "INFO", "PASSIVE_MONITOR")
        features = extract_features(temp_path, feature_type="mfcc", use_cache=False)
        features = features.reshape(1, -1)
        
        add_event("Running classification through Logistic Regression model...", "INFO", "PASSIVE_MONITOR")
        probs = clf.predict_proba(features)[0]
        passive_score = float(probs[1])
        
        trigger_challenge = passive_score < 0.7943  # Using the calibrated clean baseline threshold
        status = "SUSPICIOUS" if trigger_challenge else "CLEAN"
        
        add_event(f"Passive evaluation complete. Score: {passive_score:.4f} | Verdict: {status}", "WARNING" if trigger_challenge else "SUCCESS", "PASSIVE_MONITOR")
        
        # If call is clean, record it in history immediately as it terminates there
        if not trigger_challenge:
            SESSION_HISTORY.append(
                CallHistoryEntry(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    filename=file.filename,
                    passive_score=round(passive_score, 4),
                    challenge_score=None,
                    final_score=round(passive_score, 4),
                    verdict="CLEAN"
                )
            )
            
        return {
            "filename": file.filename,
            "degradation_applied": degradation,
            "passive_authenticity_score": round(passive_score, 4),
            "status": status,
            "trigger_challenge": trigger_challenge,
            "message": "Call voice is suspicious. Please issue a verification challenge!" if trigger_challenge else "Voice appears authentic."
        }
    except Exception as e:
        add_event(f"Passive monitor prediction failure: {str(e)}", "ERROR", "PASSIVE_MONITOR")
        raise HTTPException(status_code=500, detail=f"Feature extraction or prediction error: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/challenge/request", response_model=ChallengeRequestResponse, summary="Request Dynamic Prompt", description="Generates a spontaneous phrase challenge for context-binding caller verification.")
@app.post("/challenge/request", response_model=ChallengeRequestResponse)
def get_challenge():
    """Generates a random challenge phrase for dynamic verification."""
    phrase = generate_challenge_phrase()
    add_event(f"Generated dynamic challenge prompt: '{phrase}'", "INFO", "CHALLENGE_ENGINE")
    return {
        "challenge_phrase": phrase,
        "instructions": "Prompt the caller to speak/respond to this phrase exactly."
    }

@app.post("/challenge/verify", response_model=ChallengeVerifyResponse, summary="Verify Challenge Response", description="Transcribes response audio offline using Whisper and fuses metrics using trained ML model.")
async def verify_challenge(
    file: UploadFile = File(..., description="Challenge-response speech audio file."),
    phrase: str = Form(..., description="The issued prompt challenge text."),
    passive_score: float = Form(..., description="Initial passive authentication score (0.0 to 1.0).")
):
    """
    Evaluates the response audio to a dynamic challenge.
    Fuses the response fidelity score with the initial passive score using the trained ML fuser.
    """
    validate_audio_file(file)
    add_event(f"Received user challenge response audio: {file.filename}", "INFO", "CHALLENGE_ENGINE")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        add_event("Evaluating acoustic features of challenge response...", "INFO", "CHALLENGE_ENGINE")
        res = score_response(temp_path, phrase, model_path=MODEL_PATH)
        
        add_event(f"Offline Whisper transcription complete: '{res['transcript']}'", "INFO", "CHALLENGE_ENGINE")
        add_event(f"Voice footprint score: {res['voice_authenticity_score']:.4f} | Content alignment score: {res['content_adherence_score']:.4f}", "INFO", "CHALLENGE_ENGINE")
        
        if res['replay_attack_detected']:
            add_event("[ALERT] Replay attack detected! Content does not align with the issued challenge.", "ERROR", "CHALLENGE_ENGINE")
            
        add_event("Fusing passive classifier and challenge response scores...", "INFO", "FUSER")
        fusion_result = fuse_scores(
            passive_score=passive_score,
            challenge_score=res["fused_challenge_score"],
            challenge_weight=0.5
        )
        
        fscore = fusion_result["final_authenticity_score"]
        fstatus = fusion_result["status"]
        
        add_event(f"Fusion complete. Combined Score: {fscore:.4f} | Final Verdict: {fstatus}", "SUCCESS" if fscore >= 0.5 else "ERROR", "FUSER")
        
        SESSION_HISTORY.append(
            CallHistoryEntry(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                filename=file.filename,
                passive_score=round(passive_score, 4),
                challenge_score=round(res["fused_challenge_score"], 4),
                final_score=round(fscore, 4),
                verdict=fstatus
            )
        )
        
        return {
            "challenge_phrase": phrase,
            "response_metrics": res,
            "fusion_metrics": fusion_result,
            "final_status": fstatus
        }
    except Exception as e:
        add_event(f"Challenge verification failure: {str(e)}", "ERROR", "CHALLENGE_ENGINE")
        raise HTTPException(status_code=500, detail=f"Verification failure: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/events", response_model=list[EventLogEntry], summary="Session Events Log", description="Exposes server-side logs for events tracking.")
def get_events():
    return SESSION_EVENTS

@app.get("/history", response_model=list[CallHistoryEntry], summary="Call History Log", description="Exposes all past verification attempts in this session.")
def get_history():
    return SESSION_HISTORY

@app.get("/system-status", response_model=SystemStatusResponse, summary="System Status Indicator", description="Returns models loading states, uptime, and last-check timestamp.")
def get_system_status():
    from src.challenge_engine.response_scorer import WHISPER_MODEL
    return {
        "detector_loaded": CLF is not None,
        "whisper_loaded": WHISPER_MODEL is not None,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "last_checked_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

@app.get("/results/summary", response_model=ResultsSummaryResponse, summary="Results Summary Data", description="Returns structured evaluation matrices and channel degradation benchmarks.")
def get_results_summary():
    CONFUSION_MATRICES = {
        "clean": {"tp": 16, "tn": 98, "fp": 27, "fn": 0},
        "amr": {"tp": 15, "tn": 87, "fp": 38, "fn": 1},
        "combined": {"tp": 13, "tn": 103, "fp": 22, "fn": 3}
    }
    DEGRADATION_BENCHMARKS = [
        {"condition": "Clean Baseline", "threshold": 0.7943, "accuracy": 0.809, "eer": 0.0592, "auc": 0.9870, "accuracy_drop": 0.0, "eer_increase": 0.0},
        {"condition": "AMR-NB Codec", "threshold": 0.9791, "accuracy": 0.723, "eer": 0.1265, "auc": 0.9645, "accuracy_drop": 0.085, "eer_increase": 0.0673},
        {"condition": "GSM Codec (Cellular)", "threshold": 0.1863, "accuracy": 0.766, "eer": 0.1818, "auc": 0.9420, "accuracy_drop": 0.043, "eer_increase": 0.1225},
        {"condition": "Low Loss (5% Loss, 10ms Jitter)", "threshold": 0.0557, "accuracy": 0.596, "eer": 0.0673, "auc": 0.9835, "accuracy_drop": 0.213, "eer_increase": 0.0080},
        {"condition": "High Loss (15% Loss, 30ms Jitter)", "threshold": 0.1241, "accuracy": 0.787, "eer": 0.1145, "auc": 0.9705, "accuracy_drop": 0.021, "eer_increase": 0.0552}
    ]
    return {
        "confusion_matrices": CONFUSION_MATRICES,
        "degradation_benchmarks": DEGRADATION_BENCHMARKS
    }

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

# Serve the docs directory statically at /docs
docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs"))
if os.path.exists(docs_dir):
    app.mount("/docs", StaticFiles(directory=docs_dir), name="docs")

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
