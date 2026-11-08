"""Backend tests. Run from the repo root:  python -m pytest backend/tests -q"""

import io
import wave

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.fusion_model import MultimodalFusionModel
from backend.inference import NORMAL, InferenceEngine, StateSmoother, build_predictor


def _jpeg(seed=0):
    img = np.random.default_rng(seed).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _wav(seconds=2.0, sr=48_000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture
def mock_client(monkeypatch):
    predictor, errors = build_predictor(force_mock=True)
    monkeypatch.setattr(app_module, "engine", InferenceEngine(predictor, errors))
    monkeypatch.setattr(app_module, "hub", app_module.StateHub())
    return TestClient(app_module.app)


@pytest.fixture
def model_client(monkeypatch, tmp_path):
    """Save an untrained model as a fake checkpoint, exactly as Member 2's train.py would."""
    torch.manual_seed(0)
    torch.save(MultimodalFusionModel().state_dict(), tmp_path / "multimodal_model_checkpoint.pth")
    predictor, errors = build_predictor(checkpoint_dir=tmp_path)
    monkeypatch.setattr(app_module, "engine", InferenceEngine(predictor, errors))
    monkeypatch.setattr(app_module, "hub", app_module.StateHub())
    return TestClient(app_module.app)


def test_health_mock(mock_client):
    body = mock_client.get("/health").json()
    assert body["status"] == "ok" and body["mode"] == "mock"
    assert "High Effort" in body["states"]


def test_predict_load_without_body_matches_plan_contract(mock_client):
    body = mock_client.post("/predict-load").json()
    assert {"cognitive_state", "trigger_focus_mode"} <= body.keys()
    assert body["source"] == "mock"


def test_simulate_and_state(mock_client):
    r = mock_client.post("/simulate", json={"state": "High Temporal Demand"})
    assert r.status_code == 200 and r.json()["trigger_focus_mode"] is True
    assert mock_client.get("/state").json()["cognitive_state"] == "High Temporal Demand"
    assert mock_client.post("/simulate", json={"state": "Bogus"}).status_code == 422
    assert mock_client.post("/reset").json()["cognitive_state"] == NORMAL


def test_websocket_receives_updates(mock_client):
    with mock_client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["cognitive_state"] == NORMAL  # initial snapshot
        mock_client.post("/simulate", json={"state": "High Effort"})
        msg = ws.receive_json()
        assert msg["cognitive_state"] == "High Effort" and msg["source"] == "manual"


def test_checkpoint_loads_and_predicts(model_client):
    health = model_client.get("/health").json()
    assert health["mode"] == "model"
    assert health["input_dims"]["mental_demand"] == {"video": 256, "audio": 128}

    body = model_client.post(
        "/predict-load", json={"video_features": [0.1] * 256, "audio_features": [0.2] * 128}
    ).json()
    assert body["source"] == "model"
    assert 0.0 <= body["probabilities"]["mental_demand"] <= 1.0


def test_wrong_feature_size_is_a_clear_422(model_client):
    r = model_client.post("/predict-load", json={"video_features": [0.0] * 10, "audio_features": [0.0] * 128})
    assert r.status_code == 422 and "expects video=256" in r.json()["detail"]


def test_predict_live_end_to_end_with_model(model_client):
    files = [("frames", (f"f{i}.jpg", _jpeg(i), "image/jpeg")) for i in range(4)]
    files.append(("audio", ("clip.wav", _wav(), "audio/wav")))
    r = model_client.post("/predict-live", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "model" and body["face_detected"] is False  # noise image, no face


def test_predict_live_requires_input(mock_client):
    assert mock_client.post("/predict-live").status_code == 422


def test_smoother_hysteresis_prevents_flicker():
    s = StateSmoother(alpha=1.0)  # no averaging, test thresholds only
    assert s.update({"mental_demand": 0.55}) == NORMAL  # below enter (0.6)
    assert s.update({"mental_demand": 0.7}) == "High Mental Demand"
    assert s.update({"mental_demand": 0.5}) == "High Mental Demand"  # still above exit (0.4)
    assert s.update({"mental_demand": 0.3}) == NORMAL
