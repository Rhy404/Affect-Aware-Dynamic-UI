"""FastAPI backend for the Affect-Aware Dynamic UI (Member 4).

Run from the repository root:

    uvicorn backend.app:app --reload --port 8000
    # or
    python -m backend.app

Endpoints
---------
GET  /health        server status, model vs mock mode, loaded checkpoints
GET  /state         the current cognitive-load state
POST /predict-load  predict from precomputed feature vectors (JSON)
POST /predict-live  predict from raw webcam frames (JPEG) + mic audio (WAV)
POST /simulate      manually force a state (demo backup; also works from curl/phone)
POST /reset         clear prediction smoothing and return to Normal Load
WS   /ws            pushes every state change to connected UIs
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import features
from .inference import ALL_STATES, NORMAL, InferenceEngine, state_triggers_focus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("affect-backend")

app = FastAPI(title="Affect-Aware Dynamic UI - Inference Backend", version="1.0.0")

# Let the React dev server (any localhost port) call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InferenceEngine.from_env()

StateName = Literal["Normal Load", "High Mental Demand", "High Temporal Demand", "High Effort"]


# --------------------------------------------------------------------------- schemas

class PredictRequest(BaseModel):
    video_features: list[float] | None = Field(None, description="Video feature vector (e.g. 256 values)")
    audio_features: list[float] | None = Field(None, description="Audio feature vector (e.g. 128 values)")


class SimulateRequest(BaseModel):
    state: StateName


class StateResponse(BaseModel):
    cognitive_state: str
    trigger_focus_mode: bool
    source: Literal["model", "mock", "manual", "init"]
    probabilities: dict[str, float] = {}
    smoothed: dict[str, float] = {}
    timestamp: float
    face_detected: bool | None = None


# --------------------------------------------------------------------------- live state + websockets

class StateHub:
    """Holds the latest state and pushes changes to every connected UI."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.current: dict = {
            "cognitive_state": NORMAL,
            "trigger_focus_mode": False,
            "source": "init",
            "probabilities": {},
            "smoothed": {},
            "timestamp": time.time(),
        }

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)
        await ws.send_json({"type": "state", **self.current})

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def publish(self, state: dict):
        self.current = state
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json({"type": "state", **state})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = StateHub()


# --------------------------------------------------------------------------- routes

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": engine.mode,  # "model" once checkpoints are in backend/, else "mock"
        "dimensions": engine.predictor.dimensions,
        "input_dims": engine.predictor.input_dims(),
        "checkpoint_errors": engine.load_errors,
        "placeholder_features": features.USING_PLACEHOLDER,
        "states": ALL_STATES,
        "connected_clients": len(hub.clients),
    }


@app.get("/state", response_model=StateResponse)
async def get_state():
    return hub.current


@app.post("/predict-load", response_model=StateResponse)
async def predict_load(req: PredictRequest | None = None):
    """Predict from precomputed feature vectors.

    With no body (or in mock mode) this behaves like the plan's mock endpoint,
    so the frontend can be tested before any model exists.
    """
    import numpy as np

    req = req or PredictRequest()
    video = np.asarray(req.video_features, dtype=np.float32) if req.video_features else None
    audio = np.asarray(req.audio_features, dtype=np.float32) if req.audio_features else None
    try:
        result = await run_in_threadpool(engine.run, video, audio)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await hub.publish(result)
    return result


@app.post("/predict-live", response_model=StateResponse)
async def predict_live(
    frames: list[UploadFile] = File(default=[], description="~2 s of webcam frames as JPEG images"),
    audio: UploadFile | None = File(default=None, description="~2 s of microphone audio, 16-bit PCM WAV"),
):
    """Predict from raw webcam frames + microphone audio sent by the browser."""
    frame_bytes = [await f.read() for f in frames]
    audio_bytes = await audio.read() if audio else None
    if not frame_bytes and not audio_bytes:
        raise HTTPException(status_code=422, detail="Send at least one frame or an audio clip")

    def work():
        images = [img for img in (features.decode_image(b) for b in frame_bytes) if img is not None]
        video_feat, face = (features.video_features(images) if images else (None, None))
        audio_feat = None
        if audio_bytes:
            wave, sr = features.decode_wav(audio_bytes)
            audio_feat = features.audio_features(wave, sr)
        result = engine.run(video_feat, audio_feat)
        result["face_detected"] = face
        return result

    try:
        result = await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    await hub.publish(result)
    return result


@app.post("/simulate", response_model=StateResponse)
async def simulate(req: SimulateRequest):
    """Manual override: force a state (e.g. `curl -X POST localhost:8000/simulate
    -H 'Content-Type: application/json' -d '{"state":"High Effort"}'`)."""
    result = {
        "cognitive_state": req.state,
        "trigger_focus_mode": state_triggers_focus(req.state),
        "source": "manual",
        "probabilities": {},
        "smoothed": {},
        "timestamp": time.time(),
    }
    await hub.publish(result)
    return result


@app.post("/reset", response_model=StateResponse)
async def reset():
    engine.smoother.reset()
    result = {**hub.current, "cognitive_state": NORMAL, "trigger_focus_mode": False,
              "source": "manual", "smoothed": {}, "timestamp": time.time()}
    await hub.publish(result)
    return result


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()  # clients may send "ping" to keep the socket alive
            if msg == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        hub.disconnect(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000)
