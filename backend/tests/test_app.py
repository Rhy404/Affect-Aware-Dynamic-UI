"""Backend tests. Run from the repo root:  python -m pytest backend/tests -q

Tests marked `needs_member1` use Member 1's preprocessing code (read-only). They
are skipped unless it can be found, e.g.:
    AFFECT_PREPROCESSING_DIR=../ml-CSD361/src/preprocessing python -m pytest backend/tests -q
"""

import io
import wave

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend import app as app_module
from backend import features
from backend.inference import NORMAL, TARGET_ORDER, InferenceEngine, StateSmoother, build_predictor, decode_output

needs_member1 = pytest.mark.skipif(not features.PRE.found, reason=f"Member 1 preprocessing: {features.PRE.error}")


# --------------------------------------------------------------------------- helpers

class TinyAV(torch.nn.Module):
    """Stand-in for Member 2's model with the agreed input/output contract."""

    def __init__(self):
        super().__init__()
        self.v = torch.nn.Conv3d(3, 4, 3)
        self.a = torch.nn.Conv2d(1, 4, 3)
        self.head = torch.nn.Linear(8, 3)

    def forward(self, video, audio):
        v = self.v(video).mean(dim=(2, 3, 4))
        a = self.a(audio).mean(dim=(2, 3))
        return self.head(torch.cat([v, a], dim=1))  # (B, 3) logits in TARGET_ORDER


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


def _npz(video_shape=(3, 16, 112, 112), audio_shape=(1, 80, 200), labels=(1, 0, 1)):
    buf = io.BytesIO()
    np.savez(buf, video=np.zeros(video_shape, np.float32), audio=np.zeros(audio_shape, np.float32),
             labels=np.array(labels, np.int64))
    return {"window": ("w.npz", buf.getvalue(), "application/octet-stream")}


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setattr(app_module, "engine", InferenceEngine(*build_predictor(force_mock=True)))
    monkeypatch.setattr(app_module, "hub", app_module.StateHub())
    return TestClient(app_module.app)


@pytest.fixture
def model_client(monkeypatch, tmp_path):
    """Export the stand-in model as TorchScript, exactly as the README asks Member 2 to."""
    torch.manual_seed(0)
    model = TinyAV().eval()
    torch.jit.trace(model, (torch.zeros(1, 3, 16, 112, 112), torch.zeros(1, 1, 80, 200))).save(
        str(tmp_path / "multimodal_model.pt"))
    predictor, err = build_predictor(model_dir=tmp_path)
    assert err is None
    monkeypatch.setattr(app_module, "engine", InferenceEngine(predictor))
    monkeypatch.setattr(app_module, "hub", app_module.StateHub())
    return TestClient(app_module.app)


# --------------------------------------------------------------------------- mock mode

def test_health_mock(mock_client):
    body = mock_client.get("/health").json()
    assert body["mode"] == "mock" and body["targets"] == list(TARGET_ORDER)
    assert body["live_ready"] is True


def test_predict_load_matches_plan_contract(mock_client):
    body = mock_client.post("/predict-load").json()
    assert {"cognitive_state", "trigger_focus_mode"} <= body.keys() and body["source"] == "mock"


def test_predict_live_in_mock_mode_needs_no_preprocessing(mock_client):
    r = mock_client.post("/predict-live", files=[("frames", ("f.jpg", _jpeg(), "image/jpeg"))])
    assert r.status_code == 200 and r.json()["source"] == "mock"
    assert mock_client.post("/predict-live").status_code == 422


def test_simulate_state_reset(mock_client):
    assert mock_client.post("/simulate", json={"state": "High Temporal Demand"}).json()["trigger_focus_mode"]
    assert mock_client.get("/state").json()["cognitive_state"] == "High Temporal Demand"
    assert mock_client.post("/simulate", json={"state": "Bogus"}).status_code == 422
    assert mock_client.post("/reset").json()["cognitive_state"] == NORMAL


def test_websocket_receives_updates(mock_client):
    with mock_client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["cognitive_state"] == NORMAL
        mock_client.post("/simulate", json={"state": "High Effort"})
        assert ws.receive_json()["cognitive_state"] == "High Effort"


# --------------------------------------------------------------------------- model contract

@pytest.mark.parametrize("out, expected", [
    (torch.tensor([[10.0, -10.0, 10.0]]), {"mental_demand": 1, "effort": 0, "temporal_demand": 1}),
    (torch.tensor([[[0.0, 9.0], [9.0, 0.0], [0.0, 9.0]]]), {"mental_demand": 1, "effort": 0, "temporal_demand": 1}),
    (torch.tensor([[9.0, 0.0]]), {"mental_demand": 0}),
    (torch.tensor([[9.0]]), {"mental_demand": 1}),
    ((torch.tensor([[0.0, 9.0]]), torch.tensor([[9.0, 0.0]]), torch.tensor([[9.0]])),
     {"mental_demand": 1, "effort": 0, "temporal_demand": 1}),
])
def test_decode_output_shapes(out, expected):
    probs = decode_output(out)
    assert {k: round(v) for k, v in probs.items()} == expected


def test_predict_window_with_torchscript_model(model_client):
    health = model_client.get("/health").json()
    assert health["mode"] == "model" and health["model"]["model_file"] == "multimodal_model.pt"
    body = model_client.post("/predict-window", files=_npz()).json()
    assert body["source"] == "model" and set(body["probabilities"]) == set(TARGET_ORDER)
    assert body["ground_truth"] == {"mental_demand": 1, "effort": 0, "temporal_demand": 1}


def test_torch_export_model_file_is_supported(tmp_path):
    model = TinyAV().eval()
    ep = torch.export.export(model, (torch.zeros(1, 3, 16, 112, 112), torch.zeros(1, 1, 80, 200)))
    torch.export.save(ep, str(tmp_path / "multimodal_model.pt2"))
    predictor, err = build_predictor(model_dir=tmp_path)
    assert err is None and predictor.source == "model"
    probs = predictor.predict(np.zeros((1, 3, 16, 112, 112), np.float32), np.zeros((1, 1, 80, 200), np.float32))
    assert set(probs) == set(TARGET_ORDER)


def test_predict_window_rejects_wrong_shapes(model_client):
    r = model_client.post("/predict-window", files=_npz(audio_shape=(1, 128, 200)))
    assert r.status_code == 422 and "(1,80,200)" in r.json()["detail"]


def test_bad_model_file_falls_back_to_mock_with_error(tmp_path):
    (tmp_path / "multimodal_model.pt").write_bytes(b"not a model")
    predictor, err = build_predictor(model_dir=tmp_path)
    assert predictor.source == "mock" and err


def test_smoother_hysteresis_prevents_flicker():
    s = StateSmoother(alpha=1.0)
    assert s.update({"mental_demand": 0.55}) == NORMAL
    assert s.update({"mental_demand": 0.7}) == "High Mental Demand"
    assert s.update({"mental_demand": 0.5}) == "High Mental Demand"
    assert s.update({"mental_demand": 0.3}) == NORMAL


# --------------------------------------------------------------------------- Member 1 compatibility

def _member1_clip_reader_pack():
    """Member 1's ClipReader normalisation, called read-only on a hand-built window."""
    from backend.replay_val_clips import load_clip_reader

    ClipReader = load_clip_reader(features.PRE.directory)
    reader = ClipReader.__new__(ClipReader)  # skip __init__ (it needs the dataset on disk)
    reader.mode, reader.seed, reader.epoch, reader.normalize, reader.targets = "random", 0, 0, True, ()
    reader.a_mean, reader.a_std = features.PRE.a_mean, features.PRE.a_std
    reader.v_mean = features.PRE.v_mean.reshape(1, 1, 1, 1, 3)
    reader.v_std = features.PRE.v_std.reshape(1, 1, 1, 1, 3)
    return reader


@needs_member1
def test_live_tensors_identical_to_member1_clipreader(monkeypatch):
    rng = np.random.default_rng(0)
    crops_rgb = rng.integers(0, 255, (16, 112, 112, 3), dtype=np.uint8)
    wave_ = (0.2 * rng.standard_normal(32000)).astype(np.float32)

    # Member 1's path: crops + log-mel window -> ClipReader._pack
    reader = _member1_clip_reader_pack()
    d = dict(video=crops_rgb[None], audio=features.PRE.audio.log_mel(wave_)[None, :, :200],
             n=1, window_idx=np.array([0]), has_audio=True, n_audio=1)
    theirs = reader._pack({"clip_id": "synthetic"}, d, 0)

    # Our live path, with the face detector replaced by the same crops
    crops = iter(crops_rgb)
    monkeypatch.setattr(features, "face_crop_rgb", lambda frame, size: next(crops))
    frames = [np.zeros((240, 320, 3), np.uint8) for _ in range(16)]
    video, face_fraction = features.video_window(frames)
    audio = features.audio_window(wave_, 16000)

    assert video.shape == (1, 3, 16, 112, 112) and audio.shape == (1, 1, 80, 200)
    assert face_fraction == 1.0
    np.testing.assert_allclose(video[0], theirs["video"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(audio[0], theirs["audio"], rtol=1e-5, atol=1e-5)


@needs_member1
def test_predict_live_end_to_end_with_model(model_client, monkeypatch):
    monkeypatch.setattr(features, "face_crop_rgb", lambda frame, size: np.full((size, size, 3), 128, np.uint8))
    files = [("frames", (f"f{i}.jpg", _jpeg(i), "image/jpeg")) for i in range(16)]
    files.append(("audio", ("clip.wav", _wav(), "audio/wav")))
    body = model_client.post("/predict-live", files=files).json()
    assert body["source"] == "model" and body["face_fraction"] == 1.0 and not body["skipped"]


@needs_member1
def test_predict_live_skips_window_without_face(model_client):
    files = [("frames", (f"f{i}.jpg", _jpeg(i), "image/jpeg")) for i in range(16)]
    files.append(("audio", ("clip.wav", _wav(), "audio/wav")))
    body = model_client.post("/predict-live", files=files).json()  # random noise: no face
    assert body["skipped"] is True and body["face_fraction"] == 0.0


def test_predict_live_with_model_but_no_member1_code_is_clear(model_client, monkeypatch):
    monkeypatch.setattr(features.PRE, "found", False)
    files = [("frames", ("f.jpg", _jpeg(), "image/jpeg")), ("audio", ("a.wav", _wav(), "audio/wav"))]
    r = model_client.post("/predict-live", files=files)
    assert r.status_code == 503 and "Member 1" in r.json()["detail"]
