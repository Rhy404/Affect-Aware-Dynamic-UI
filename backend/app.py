"""FastAPI backend for the Affect-Aware Dynamic UI (Member 4).

Run from the repository root:

    uvicorn backend.app:app --reload --port 8000

Endpoints
---------
GET  /health          status: model vs mock, Member 1 preprocessing found, expected shapes
GET  /state           the current cognitive-load state
POST /predict-load    plan-compatible endpoint (no body); mock prediction
POST /predict-live    raw webcam frames (JPEG) + mic audio (WAV) -> Member 1 tensors -> model
POST /predict-window  one preprocessed window (.npz with video + audio), e.g. from ClipReader
POST /simulate        manually force a state (demo backup; also works from curl/phone)
POST /reset           clear prediction smoothing and return to Normal Load
WS   /ws              pushes every state change to connected UIs
"""

from __future__ import annotations

import io
import logging
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import features
from .inference import ALL_STATES, NORMAL, TARGET_ORDER, InferenceEngine, state_triggers_focus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("affect-backend")

app = FastAPI(title="Affect-Aware Dynamic UI - Inference Backend", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = InferenceEngine.from_env()

StateName = Literal["Normal Load", "High Mental Demand", "High Temporal Demand", "High Effort"]


class SimulateRequest(BaseModel):
    state: StateName


class StateResponse(BaseModel):
    cognitive_state: str
    trigger_focus_mode: bool
    source: Literal["model", "mock", "manual", "init"]
    probabilities: dict[str, float] = {}
    smoothed: dict[str, float] = {}
    timestamp: float
    face_fraction: float | None = None   # share of the 16 frames with a detected face
    skipped: bool = False                 # True if the window was not used (e.g. no face)
    note: str | None = None
    ground_truth: dict[str, int] | None = None  # only for /predict-window with labels


# --------------------------------------------------------------------------- live state + websockets

class StateHub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.current: dict = {"cognitive_state": NORMAL, "trigger_focus_mode": False, "source": "init",
                              "probabilities": {}, "smoothed": {}, "timestamp": time.time()}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)
        await ws.send_json({"type": "state", **self.current})

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def publish(self, state: dict):
        self.current = state
        for ws in list(self.clients):
            try:
                await ws.send_json({"type": "state", **state})
            except Exception:
                self.disconnect(ws)


hub = StateHub()


async def _run_and_publish(video, audio, extra: dict | None = None) -> dict:
    try:
        result = await run_in_threadpool(engine.run, video, audio)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result.update(extra or {})
    await hub.publish(result)
    return result


# --------------------------------------------------------------------------- routes

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": engine.mode,  # "model" once a model file is in backend/, else "mock"
        "model": engine.predictor.describe(),
        "model_error": engine.load_error,
        "targets": list(TARGET_ORDER),
        "preprocessing": features.PRE.info(),
        "live_ready": engine.mode == "mock" or features.PRE.found,
        "states": ALL_STATES,
        "connected_clients": len(hub.clients),
    }


@app.get("/state", response_model=StateResponse)
async def get_state():
    return hub.current


@app.post("/predict-load", response_model=StateResponse)
async def predict_load():
    """The plan's original endpoint: returns a prediction without any input.
    Only meaningful in mock mode; with a real model use /predict-live or /predict-window."""
    if engine.mode == "model":
        raise HTTPException(422, "A model is loaded: send data to /predict-live or /predict-window")
    return await _run_and_publish(None, None)


@app.post("/predict-live", response_model=StateResponse)
async def predict_live(
    frames: list[UploadFile] = File(default=[], description="~2 s of webcam frames as JPEG images"),
    audio: UploadFile | None = File(default=None, description="~2 s of microphone audio, 16-bit PCM WAV"),
):
    """Webcam frames + mic audio from the browser -> Member 1's tensors -> prediction."""
    frame_bytes = [await f.read() for f in frames]
    audio_bytes = await audio.read() if audio else None
    if not frame_bytes and not audio_bytes:
        raise HTTPException(422, "Send at least one frame or an audio clip")

    if engine.mode == "mock":  # the mock ignores the data, so no preprocessing needed
        return await _run_and_publish(None, None)

    if not features.PRE.found:
        raise HTTPException(503, f"Member 1's preprocessing is not available: {features.PRE.error}")
    if not frame_bytes or not audio_bytes:
        raise HTTPException(422, "The model needs both webcam frames and microphone audio")

    def preprocess():
        images = [im for im in (features.decode_image(b) for b in frame_bytes) if im is not None]
        if not images:
            raise ValueError("None of the frames could be decoded as images")
        video, face_fraction = features.video_window(images)
        wave_, sr = features.decode_wav(audio_bytes)
        return video, features.audio_window(wave_, sr), face_fraction

    try:
        video, audio_t, face_fraction = await run_in_threadpool(preprocess)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    if face_fraction < features.MIN_VALID:
        # Training never used windows with < 75 % face frames, so don't predict on them.
        return {**hub.current, "face_fraction": round(face_fraction, 3), "skipped": True,
                "note": "No face in enough frames - window skipped, state unchanged", "timestamp": time.time()}
    return await _run_and_publish(video, audio_t, {"face_fraction": round(face_fraction, 3)})


@app.post("/predict-window", response_model=StateResponse)
async def predict_window(window: UploadFile = File(..., description=".npz with 'video' and 'audio' arrays")):
    """Predict on ONE window already preprocessed by Member 1's ClipReader:
    video (3,16,112,112) and audio (1,80,200), normalised (a batch dim is optional).
    An optional 'labels' array (Member 1's target order) is echoed back as ground_truth.
    Used by backend/replay_val_clips.py to drive the UI with real AVCAffe clips."""
    try:
        with np.load(io.BytesIO(await window.read())) as z:
            video = z["video"].astype(np.float32)
            audio = z["audio"].astype(np.float32)
            labels = z["labels"].tolist() if "labels" in z.files else None
    except Exception as exc:
        raise HTTPException(422, f"Expected an .npz with 'video' and 'audio' arrays ({exc})")
    video = video[None] if video.ndim == 4 else video
    audio = audio[None] if audio.ndim == 3 else audio
    if video.shape[1:] != (3, 16, 112, 112) or audio.shape[1:] != (1, 80, 200):
        raise HTTPException(422, f"Expected video (3,16,112,112) and audio (1,80,200); "
                                 f"got {video.shape[1:]} and {audio.shape[1:]}")
    extra = {}
    if labels is not None:
        extra["ground_truth"] = {t: int(v) for t, v in zip(TARGET_ORDER, labels)}
    return await _run_and_publish(video[:1], audio[:1], extra)


@app.post("/simulate", response_model=StateResponse)
async def simulate(req: SimulateRequest):
    """Manual override, e.g.  curl -X POST localhost:8000/simulate
    -H 'Content-Type: application/json' -d '{"state":"High Effort"}'"""
    result = {"cognitive_state": req.state, "trigger_focus_mode": state_triggers_focus(req.state),
              "source": "manual", "probabilities": {}, "smoothed": {}, "timestamp": time.time()}
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
            if await ws.receive_text() == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        hub.disconnect(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000)
