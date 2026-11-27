import os
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.pipeline import api

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("filename", "content", "status"), [
    ("demo.txt", b"sample text", 400),
    ("empty.wav", b"", 400),
    ("invalid.wav", b"not audio", 422),
    ("oversize.wav", bytes(api.MAX_REQUEST_BYTES + 1), 413),
])
def test_upload_validation(filename, content, status):
    response = TestClient(api.app).post("/verify-passive", files={"file": (filename, content)})
    assert response.status_code == status
    assert "/tmp/" not in response.text


def test_stream_rejects_oversized_frames(monkeypatch):
    monkeypatch.setattr(api, "CLF", object())
    with TestClient(api.app).websocket_connect("/stream") as connection:
        connection.send_bytes(bytes(api.MAX_STREAM_FRAME_BYTES + 1))
        with pytest.raises(WebSocketDisconnect) as closed:
            connection.receive_json()
        assert closed.value.code == 1009


def test_expired_challenge_cannot_be_issued(monkeypatch):
    monkeypatch.setattr(api, "SESSION_STORE", {"expired": {"expires_at": time.monotonic() - 1}})
    response = TestClient(api.app).post("/challenge/request", data={"session_id": "expired"})
    assert response.status_code == 400
    assert not api.SESSION_STORE


@pytest.mark.skipif(os.environ.get("VOXGUARD_RUN_MODEL_TEST") != "1", reason="Requires local Whisper weights")
def test_shipped_model_and_one_time_whisper_challenge():
    with TestClient(api.app) as client:
        status = client.get("/system-status").json()
        assert status["detector_loaded"] and status["whisper_loaded"]
        sample = (ROOT / "data/demo/class_b_fake_0.wav").read_bytes()
        files = {"file": ("demo.wav", sample, "audio/wav")}
        passive = client.post("/verify-passive", files=files)
        assert passive.status_code == 200, passive.text
        values = passive.json()
        assert values["trigger_challenge"]
        payload = {"session_id": values["session_id"]}
        phrase = client.post("/challenge/request", data=payload)
        assert phrase.status_code == 200
        assert client.post("/challenge/request", data=payload).json() == phrase.json()
        result = client.post("/challenge/verify", data=payload, files=files)
        assert result.status_code == 200, result.text
        assert client.post("/challenge/verify", data=payload, files=files).status_code == 400


@pytest.mark.skipif(os.environ.get("VOXGUARD_RUN_MODEL_TEST") != "1", reason="Requires FFmpeg and the model")
def test_two_real_webm_windows_can_issue_challenges(tmp_path):
    import subprocess

    window = tmp_path / "sample.webm"
    subprocess.run([
        api.get_ffmpeg_path(), "-nostdin", "-y", "-i", str(ROOT / "data/demo/class_b_fake_0.wav"),
        "-t", "1.6", "-c:a", "libopus", "-f", "webm", str(window),
    ], check=True, capture_output=True, timeout=15)
    samples = window.read_bytes()
    with TestClient(api.app) as client:
        with client.websocket_connect("/stream?segments=true") as stream:
            for index in (1, 2):
                stream.send_bytes(samples)
                message = stream.receive_json()
                assert message["chunk_id"] == index
                assert 0 <= message["passive_authenticity_score"] <= 1
                if message["trigger_challenge"]:
                    assert client.post("/challenge/request", data={"session_id": message["session_id"]}).is_success
