"""Loads the trained model, runs predictions, and smooths them into a UI state.

MODEL CONTRACT WITH MEMBER 2
----------------------------
The model takes Member 1's tensors and is called as  model(video, audio):

    video  (B, 3, 16, 112, 112)  float32   (ClipReader output, batched)
    audio  (B, 1, 80, 200)       float32

Preferred: export it as TorchScript, so the backend needs no copy of the class:

    model.eval()
    traced = torch.jit.trace(model, (torch.zeros(1, 3, 16, 112, 112), torch.zeros(1, 1, 80, 200)))
    traced.save("backend/multimodal_model.pt")

Also accepted (newer PyTorch, where TorchScript is deprecated):

    ep = torch.export.export(model, (torch.zeros(1, 3, 16, 112, 112), torch.zeros(1, 1, 80, 200)))
    torch.export.save(ep, "backend/multimodal_model.pt2")

Alternative: a plain state_dict saved as backend/multimodal_model_checkpoint.pth
plus the environment variable AFFECT_MODEL_CLASS="package.module:ClassName"
(the class must be constructible with no arguments).

Accepted outputs, in Member 1's label order (dataset.DEFAULT_TARGETS):
    (B, 3)      one logit per target (multi-label)  -> sigmoid
    (B, 3, 2)   two-class logits per target          -> softmax, class 1 = High
    (B, 2)      one target only (Mental Demand)      -> softmax, class 1 = High
    (B, 1)      one target only, single logit        -> sigmoid
    a tuple/list of 3 heads, each (B, 2) or (B, 1)

With no model file the server runs in MOCK mode so the UI can be demoed.
"""

from __future__ import annotations

import importlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
NORMAL = "Normal Load"

# Same order as Member 1's dataset.DEFAULT_TARGETS ("mental_demand_bin", "effort_bin", "temporal_demand_bin")
TARGET_ORDER = ("mental_demand", "effort", "temporal_demand")

STATE_FOR = {
    "mental_demand": "High Mental Demand",
    "effort": "High Effort",
    "temporal_demand": "High Temporal Demand",
}
ALL_STATES = [NORMAL, "High Temporal Demand", "High Mental Demand", "High Effort"]


def state_triggers_focus(state: str) -> bool:
    return state != NORMAL


# --------------------------------------------------------------------------- output decoding

def decode_output(out) -> dict[str, float]:
    """Turn the model's raw output into P(High) per target, for the first item in the batch."""
    import torch

    if isinstance(out, (tuple, list)):
        if len(out) != len(TARGET_ORDER):
            raise ValueError(f"Model returned {len(out)} heads; expected {len(TARGET_ORDER)} or a single tensor")
        return {t: decode_output(head)["mental_demand"] for t, head in zip(TARGET_ORDER, out)}

    out = out.detach().float()
    shape = tuple(out.shape)
    if out.dim() == 3 and shape[1:] == (3, 2):
        p = torch.softmax(out[0], dim=-1)[:, 1]
        return {t: float(p[i]) for i, t in enumerate(TARGET_ORDER)}
    if out.dim() == 2 and shape[1] == 3:
        p = torch.sigmoid(out[0])
        return {t: float(p[i]) for i, t in enumerate(TARGET_ORDER)}
    if out.dim() == 2 and shape[1] == 2:
        return {"mental_demand": float(torch.softmax(out[0], dim=-1)[1])}
    if out.dim() == 2 and shape[1] == 1 or out.dim() == 1:
        return {"mental_demand": float(torch.sigmoid(out.reshape(-1)[0]))}
    raise ValueError(f"Unsupported model output shape {shape}; see the contract in backend/inference.py")


# --------------------------------------------------------------------------- predictors

class ModelPredictor:
    source = "model"

    def __init__(self, model, path: Path):
        self.model = model
        self.path = path

    @property
    def dimensions(self) -> list[str]:
        return list(TARGET_ORDER)

    def describe(self) -> dict:
        return {"model_file": self.path.name}

    def predict(self, video: np.ndarray, audio: np.ndarray) -> dict[str, float]:
        import torch

        with torch.no_grad():
            out = self.model(torch.from_numpy(video), torch.from_numpy(audio))
        return decode_output(out)


class MockPredictor:
    """Stand-in until the trained model exists: a slow random walk per target,
    mostly low with occasional spikes, so the UI visibly reacts in Auto mode."""

    source = "mock"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.levels = {t: 0.2 for t in TARGET_ORDER}

    @property
    def dimensions(self) -> list[str]:
        return list(TARGET_ORDER)

    def describe(self) -> dict:
        return {"model_file": None}

    def predict(self, video=None, audio=None) -> dict[str, float]:
        for t, level in self.levels.items():
            drift = self.rng.gauss(0, 0.12) + (0.2 - level) * 0.15
            if self.rng.random() < 0.06:
                drift += 0.5
            self.levels[t] = min(1.0, max(0.0, level + drift))
        return dict(self.levels)


def _load_model(model_dir: Path):
    """Returns (model, path) or (None, None). Raises with a readable message on a bad file."""
    import torch

    exported = model_dir / "multimodal_model.pt2"  # torch.export format (newer PyTorch)
    if exported.exists():
        model = torch.export.load(str(exported)).module()
        return model, exported

    scripted = model_dir / "multimodal_model.pt"
    if scripted.exists():
        model = torch.jit.load(str(scripted), map_location="cpu")
        model.eval()
        return model, scripted

    state_file = model_dir / "multimodal_model_checkpoint.pth"
    if state_file.exists():
        target = os.getenv("AFFECT_MODEL_CLASS")
        if not target:
            raise RuntimeError(
                "Found multimodal_model_checkpoint.pth (a state_dict) but no model class. Either save the model "
                "as TorchScript (backend/multimodal_model.pt) or set AFFECT_MODEL_CLASS=package.module:ClassName"
            )
        module_name, class_name = target.split(":")
        model = getattr(importlib.import_module(module_name), class_name)()
        state = torch.load(state_file, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        model.eval()
        return model, state_file
    return None, None


def build_predictor(model_dir: Path | None = None, force_mock: bool = False):
    """Returns (predictor, load_error_or_None)."""
    model_dir = Path(model_dir or os.getenv("AFFECT_MODEL_DIR", BACKEND_DIR))
    if force_mock or os.getenv("AFFECT_FORCE_MOCK") == "1":
        return MockPredictor(), None
    try:
        model, path = _load_model(model_dir)
    except Exception as exc:
        log.error("Could not load the model: %s", exc)
        return MockPredictor(), f"{type(exc).__name__}: {exc}"
    if model is None:
        log.warning("No model in %s - running in MOCK mode", model_dir)
        return MockPredictor(), None
    log.info("Loaded model from %s", path)
    return ModelPredictor(model, path), None


# --------------------------------------------------------------------------- smoothing

@dataclass
class StateSmoother:
    """Stops the UI flickering on noisy predictions: an exponential moving average
    per target, plus hysteresis (enter High above `enter`, leave below `exit`)."""

    alpha: float = 0.5
    enter: float = 0.6
    exit: float = 0.4
    smoothed: dict[str, float] = field(default_factory=dict)
    high: dict[str, bool] = field(default_factory=dict)

    def update(self, probs: dict[str, float]) -> str:
        for t, p in probs.items():
            s = self.alpha * p + (1 - self.alpha) * self.smoothed.get(t, p)
            self.smoothed[t] = s
            self.high[t] = s >= self.exit if self.high.get(t, False) else s > self.enter
        active = [t for t, is_high in self.high.items() if is_high]
        if not active:
            return NORMAL
        return STATE_FOR[max(active, key=lambda t: self.smoothed[t])]

    def reset(self):
        self.smoothed.clear()
        self.high.clear()


# --------------------------------------------------------------------------- engine

class InferenceEngine:
    def __init__(self, predictor, load_error: str | None = None, smoother: StateSmoother | None = None):
        self.predictor = predictor
        self.load_error = load_error
        self.smoother = smoother or StateSmoother()

    @classmethod
    def from_env(cls) -> "InferenceEngine":
        return cls(*build_predictor())

    @property
    def mode(self) -> str:
        return self.predictor.source

    def run(self, video: np.ndarray | None, audio: np.ndarray | None) -> dict:
        if self.mode == "model":
            if video is None or audio is None:
                raise ValueError("The trained model needs both a video and an audio window")
            probs = self.predictor.predict(video, audio)
        else:
            probs = self.predictor.predict()
        state = self.smoother.update(probs)
        return {
            "cognitive_state": state,
            "trigger_focus_mode": state_triggers_focus(state),
            "source": self.mode,
            "probabilities": {t: round(p, 4) for t, p in probs.items()},
            "smoothed": {t: round(p, 4) for t, p in self.smoother.smoothed.items()},
            "timestamp": time.time(),
        }
