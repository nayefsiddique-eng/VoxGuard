import asyncio
from collections import deque
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal
import time
import shutil
import tempfile
import joblib
import subprocess
import uuid

import soundfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, TypeAdapter
from src.features.extract_features import extract_features
from src.challenge_engine.phrase_challenge import generate_challenge_phrase
from src.challenge_engine.response_scorer import score_response, get_whisper_model
from src.pipeline.fusion import fuse_scores
from src.pipeline.body_limit import RequestBodyLimit
from src.pipeline.degrade_audio import process_file, get_ffmpeg_path

app = FastAPI(
    title="VoxGuard Verification Pipeline API",
    description="Backend API for continuous passive voice clone detection, challenge dispatching, and score fusion verification.",
    version="1.0.0"
)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 11 * 1024 * 1024
MAX_AUDIO_SECONDS = 120
MAX_STREAM_FRAME_BYTES = 256 * 1024
CHALLENGE_TTL_SECONDS = 300
MAX_CHALLENGES = 512

app.add_middleware(RequestBodyLimit, max_bytes=MAX_REQUEST_BYTES)

# Pydantic Schemas for Swagger / API documentation
class PassiveVerifyResponse(BaseModel):
    filename: str = Field(..., description="Name of the processed file.")
    degradation_applied: str = Field(..., description="Type of degradation/codec applied.")
    passive_authenticity_score: float = Field(..., description="Authenticity score from passive monitor (0.0 to 1.0).")
    status: str = Field(..., description="Call status: CLEAN or SUSPICIOUS.")
    trigger_challenge: bool = Field(..., description="Whether the challenge mechanism is recommended.")
    message: str = Field(..., description="Detailed explanation of the verdict.")
    session_id: str | None = Field(None, description="Session ID to link subsequent requests.")

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
    accuracy_ci: str
    eer: float
    auc: float
    accuracy_drop: float
    eer_increase: float

class ResultsSummaryResponse(BaseModel):
    confusion_matrices: dict[str, ConfusionMatrixData]
    degradation_benchmarks: list[DegradationRow]

class SensitivityRow(BaseModel):
    threshold: float
    accuracy: float
    fpr: float
    fnr: float

class SensitivityResponse(BaseModel):
    sensitivity_data: list[SensitivityRow]

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
SESSION_EVENTS = deque(maxlen=200)
SESSION_HISTORY = deque(maxlen=200)
SESSION_STORE = {}

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
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# Helper for audio file validation
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
def validate_audio_file(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    # Validate size by seeking to end (limit to 10MB)
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
    if size == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        info = soundfile.info(file.file)
        if info.frames <= 0:
            raise ValueError("Audio has no samples")
    except (ValueError, RuntimeError, soundfile.LibsndfileError) as exc:
        raise HTTPException(status_code=422, detail="Unable to decode the uploaded audio.") from exc
    finally:
        file.file.seek(0)
    if info.duration > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=413, detail="Audio is longer than 120 seconds.")


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
    degradation: Literal["none", "amr", "opus", "packet_loss", "combined"] = Query(
        "none", description="Channel condition profile to apply."
    )
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
        
        session_id = None
        if trigger_challenge:
            session_id = str(uuid.uuid4())
            now = time.monotonic()
            for expired_id in list(SESSION_STORE):
                if SESSION_STORE[expired_id]["expires_at"] <= now:
                    del SESSION_STORE[expired_id]
            if len(SESSION_STORE) >= MAX_CHALLENGES:
                del SESSION_STORE[next(iter(SESSION_STORE))]
            SESSION_STORE[session_id] = {"passive_score": passive_score, "expires_at": now + CHALLENGE_TTL_SECONDS}
        
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
            "message": "Call voice is suspicious. Please issue a verification challenge!" if trigger_challenge else "Voice appears authentic.",
            "session_id": session_id
        }
    except Exception as e:
        add_event("Passive monitor prediction failed.", "ERROR", "PASSIVE_MONITOR")
        raise HTTPException(status_code=500, detail="Feature extraction or prediction failed.") from e
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/challenge/request", response_model=ChallengeRequestResponse, summary="Request Dynamic Prompt", description="Generates a spontaneous phrase challenge for context-binding caller verification.")
@app.post("/challenge/request", response_model=ChallengeRequestResponse)
async def get_challenge(session_id: str = Form(..., description="Session ID")):
    """Issue one phrase until the session expires or its response is consumed."""
    session = SESSION_STORE.get(session_id)
    if not session or session["expires_at"] <= time.monotonic():
        SESSION_STORE.pop(session_id, None)
        raise HTTPException(status_code=400, detail="Invalid or expired session.")
    if "phrase" not in session:
        session["phrase"] = generate_challenge_phrase()
        add_event("Generated dynamic challenge prompt.", "INFO", "CHALLENGE_ENGINE")

    phrase = session["phrase"]
    return {
        "challenge_phrase": phrase,
        "instructions": "Prompt the caller to speak/respond to this phrase exactly."
    }

@app.post("/challenge/verify", response_model=ChallengeVerifyResponse, summary="Verify Challenge Response", description="Transcribes response audio offline using Whisper and fuses metrics using trained ML model.")
async def verify_challenge(
    file: UploadFile = File(..., description="Challenge-response speech audio file."),
    session_id: str = Form(..., description="Session identifier from the passive check.")
):
    """
    Evaluates the response audio to a dynamic challenge.
    Fuses the response fidelity score with the initial passive score using the trained ML fuser.
    """
    validate_audio_file(file)
    add_event(f"Received user challenge response audio: {file.filename}", "INFO", "CHALLENGE_ENGINE")
    
    session_data = SESSION_STORE.get(session_id)
    if not session_data or session_data["expires_at"] <= time.monotonic():
        SESSION_STORE.pop(session_id, None)
        raise HTTPException(status_code=400, detail="Invalid or expired session.")
    if "phrase" not in session_data:
        raise HTTPException(status_code=400, detail="Challenge not requested.")
    del SESSION_STORE[session_id]
        
    phrase = session_data['phrase']
    passive_score = session_data['passive_score']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        add_event("Evaluating acoustic features of challenge response...", "INFO", "CHALLENGE_ENGINE")
        res = score_response(temp_path, phrase, model_path=MODEL_PATH)
        
        add_event("Offline Whisper transcription complete.", "INFO", "CHALLENGE_ENGINE")
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
        add_event("Challenge verification failed.", "ERROR", "CHALLENGE_ENGINE")
        raise HTTPException(status_code=500, detail="Challenge verification failed.") from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/events", response_model=list[EventLogEntry], summary="Session Events Log", description="Exposes server-side logs for events tracking.")
def get_events():
    return list(SESSION_EVENTS)

@app.get("/history", response_model=list[CallHistoryEntry], summary="Call History Log", description="Exposes all past verification attempts in this session.")
def get_history():
    return list(SESSION_HISTORY)

@app.get("/system-status", response_model=SystemStatusResponse, summary="System Status Indicator", description="Returns models loading states, uptime, and last-check timestamp.")
def get_system_status():
    from src.challenge_engine.response_scorer import WHISPER_MODEL
    return {
        "detector_loaded": CLF is not None,
        "whisper_loaded": WHISPER_MODEL is not None,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "last_checked_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

def read_degradation_rows(path: Path) -> list[DegradationRow]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [part.strip().replace("**", "") for part in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) != 7:
            continue
        condition, threshold, accuracy_ci, eer, auc, accuracy_drop, eer_increase = cells
        if condition == "Channel Condition" or condition.startswith("---"):
            continue

        accuracy_match = re.match(r"^(\d+(?:\.\d+)?)%", accuracy_ci)
        if accuracy_match is None:
            raise ValueError("Invalid accuracy in degradation table")

        rows.append(DegradationRow(
            condition=condition,
            threshold=float(threshold),
            accuracy=float(accuracy_match[1]) / 100,
            accuracy_ci=accuracy_ci,
            eer=float(eer.removesuffix("%")) / 100,
            auc=float(auc),
            accuracy_drop=0.0 if accuracy_drop == "-" else float(accuracy_drop.removesuffix("%")) / 100,
            eer_increase=0.0 if eer_increase == "-" else float(eer_increase.removesuffix("%")) / 100,
        ))
    if not rows:
        raise ValueError("No degradation results found")
    return rows


@app.get("/results/summary", response_model=ResultsSummaryResponse, summary="Results Summary Data")
def get_results_summary():
    try:
        matrices = json.loads((DOCS_DIR / "confusion_matrices.json").read_text(encoding="utf-8"))
        if not matrices:
            raise ValueError("No confusion matrices found")
        return ResultsSummaryResponse(
            confusion_matrices=matrices,
            degradation_benchmarks=read_degradation_rows(DOCS_DIR / "results_degraded.md"),
        )
    except (OSError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).warning("Could not read evaluation summary: %s", exc)
        raise HTTPException(status_code=503, detail="Evaluation summary artifacts are unavailable.") from exc


@app.get("/results/sensitivity", response_model=list[SensitivityRow], summary="Threshold Sensitivity")
def get_results_sensitivity():
    try:
        data = TypeAdapter(list[SensitivityRow]).validate_json(
            (DOCS_DIR / "threshold_sensitivity.json").read_bytes()
        )
        if not data:
            raise ValueError("No sensitivity results found")
        return data
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("Could not read sensitivity results: %s", exc)
        raise HTTPException(status_code=503, detail="Threshold sensitivity artifacts are unavailable.") from exc


def score_webm_window(data, classifier):
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return None
    with tempfile.TemporaryDirectory(prefix="voxguard-stream-") as directory:
        container = Path(directory) / "input.webm"
        wav = Path(directory) / "window.wav"
        container.write_bytes(data)
        try:
            subprocess.run([
                ffmpeg, "-nostdin", "-protocol_whitelist", "file,pipe", "-f", "webm",
                "-i", str(container), "-t", "10", "-ar", "16000", "-ac", "1", "-f", "wav", str(wav),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, check=True)
            features = extract_features(wav, feature_type="mfcc", use_cache=False).reshape(1, -1)
            return float(classifier.predict_proba(features)[0][1])
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            return None


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    if CLF is None:
        await websocket.close(code=1013)
        return
    classifier = CLF
    audio = bytearray()
    chunk_counter = 0
    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            if len(data) > MAX_STREAM_FRAME_BYTES:
                await websocket.close(code=1009)
                return
            audio.extend(data)
            chunk_counter += 1
            if len(audio) >= 64000:
                score = await run_in_threadpool(score_webm_window, bytes(audio), classifier)
                if score is not None:
                    await websocket.send_json({
                        "chunk_id": chunk_counter,
                        "passive_authenticity_score": round(score, 4),
                        "trigger_challenge": score < 0.5,
                        "status": "SUSPICIOUS" if score < 0.5 else "CLEAN",
                    })
                audio = audio[-32000:]
    except TimeoutError:
        await websocket.close(code=1001)
    except WebSocketDisconnect:
        return


# Serve the data directory statically at /data
if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# Serve the docs directory statically at /docs
if DOCS_DIR.is_dir():
    app.mount("/docs", StaticFiles(directory=DOCS_DIR), name="docs")

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
