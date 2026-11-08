"""Loads the trained checkpoint(s), runs predictions, and smooths them into a UI state.

Checkpoints (put them in backend/, or point AFFECT_CHECKPOINT_DIR elsewhere):

    multimodal_model_checkpoint.pth   -> Mental Demand   (the file named in the plan)
    temporal_demand_checkpoint.pth    -> Temporal Demand (optional)
    effort_checkpoint.pth             -> Effort          (optional)

Each is a binary High/Low MultimodalFusionModel (class 1 = High, i.e. NASA-TLX
score > 10). Any subset may be present. With none present the server runs in
MOCK mode so the UI can be developed and demoed before training finishes.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

NORMAL = "Normal Load"

# dimension key -> (UI state name, checkpoint filename)
DIMENSIONS: dict[str, tuple[str, str]] = {
    "mental_demand": ("High Mental Demand", "multimodal_model_checkpoint.pth"),
    "temporal_demand": ("High Temporal Demand", "temporal_demand_checkpoint.pth"),
    "effort": ("High Effort", "effort_checkpoint.pth"),
}
ALL_STATES = [NORMAL] + [name for name, _ in DIMENSIONS.values()]

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent


def state_triggers_focus(state: str) -> bool:
    return state != NORMAL


# --------------------------------------------------------------------------- predictors

class ModelPredictor:
    """Runs one or more trained MultimodalFusionModel checkpoints on CPU."""

    source = "model"

    def __init__(self, models: dict):
        self.models = models  # dim -> (torch model, video_dim, audio_dim)

    @property
    def dimensions(self) -> list[str]:
        return list(self.models)

    def input_dims(self) -> dict[str, dict[str, int]]:
        return {d: {"video": v, "audio": a} for d, (_, v, a) in self.models.items()}

    def predict(self, video_feat: np.ndarray, audio_feat: np.ndarray) -> dict[str, float]:
        import torch

        probs = {}
        with torch.no_grad():
            for dim, (model, vdim, adim) in self.models.items():
                if video_feat.size != vdim or audio_feat.size != adim:
                    raise ValueError(
                        f"{dim} model expects video={vdim}, audio={adim} features; "
                        f"got video={video_feat.size}, audio={audio_feat.size}"
                    )
                v = torch.from_numpy(video_feat.reshape(1, -1).astype(np.float32))
                a = torch.from_numpy(audio_feat.reshape(1, -1).astype(np.float32))
                p = torch.softmax(model(v, a), dim=1)[0]
                probs[dim] = float(p[1]) if p.numel() > 1 else float(torch.sigmoid(p[0]))
        return probs


class MockPredictor:
    """Stand-in until real checkpoints exist: a slow random walk per dimension,
    mostly low with occasional spikes, so the UI visibly reacts in Auto mode."""

    source = "mock"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.levels = {d: 0.2 for d in DIMENSIONS}

    @property
    def dimensions(self) -> list[str]:
        return list(DIMENSIONS)

    def input_dims(self) -> dict[str, dict[str, int]]:
        return {}

    def predict(self, video_feat=None, audio_feat=None) -> dict[str, float]:
        for d, level in self.levels.items():
            drift = self.rng.gauss(0, 0.12) + (0.2 - level) * 0.15  # pull back towards low load
            if self.rng.random() < 0.06:  # occasional spike
                drift += 0.5
            self.levels[d] = min(1.0, max(0.0, level + drift))
        return dict(self.levels)


def _load_checkpoint(path: Path):
    """Build a MultimodalFusionModel whose layer sizes match the saved weights."""
    import torch

    from .fusion_model import MultimodalFusionModel

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:  # common checkpoint wrapper
        state = state["model_state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}  # DataParallel prefix
    vdim = state["video_branch.0.weight"].shape[1]
    adim = state["audio_branch.0.weight"].shape[1]
    ncls = state["fusion_mlp.3.weight"].shape[0]
    model = MultimodalFusionModel(video_dim=vdim, audio_dim=adim, num_classes=ncls)
    model.load_state_dict(state)
    model.eval()
    return model, vdim, adim


def build_predictor(checkpoint_dir: Path | None = None, force_mock: bool = False):
    checkpoint_dir = Path(checkpoint_dir or os.getenv("AFFECT_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
    if force_mock or os.getenv("AFFECT_FORCE_MOCK") == "1":
        return MockPredictor(), {}

    loaded, errors = {}, {}
    for dim, (_, filename) in DIMENSIONS.items():
        path = checkpoint_dir / filename
        if not path.exists():
            continue
        try:
            loaded[dim] = _load_checkpoint(path)
            log.info("Loaded %s checkpoint from %s", dim, path)
        except Exception as exc:  # wrong architecture, corrupt file, ...
            errors[dim] = f"{type(exc).__name__}: {exc}"
            log.error("Could not load %s: %s", path, exc)

    if loaded:
        return ModelPredictor(loaded), errors
    log.warning("No checkpoints found in %s - running in MOCK mode", checkpoint_dir)
    return MockPredictor(), errors


# --------------------------------------------------------------------------- smoothing

@dataclass
class StateSmoother:
    """Stops the UI flickering between modes on every noisy prediction.

    Each dimension's P(High) is smoothed with an exponential moving average and
    passed through hysteresis: it must rise above `enter` to count as High, and
    fall below `exit` to count as Low again.
    """

    alpha: float = 0.5
    enter: float = 0.6
    exit: float = 0.4
    smoothed: dict[str, float] = field(default_factory=dict)
    high: dict[str, bool] = field(default_factory=dict)

    def update(self, probs: dict[str, float]) -> str:
        for dim, p in probs.items():
            prev = self.smoothed.get(dim, p)
            s = self.alpha * p + (1 - self.alpha) * prev
            self.smoothed[dim] = s
            if self.high.get(dim, False):
                self.high[dim] = s >= self.exit
            else:
                self.high[dim] = s > self.enter
        active = [d for d, is_high in self.high.items() if is_high]
        if not active:
            return NORMAL
        strongest = max(active, key=lambda d: self.smoothed[d])
        return DIMENSIONS[strongest][0]

    def reset(self):
        self.smoothed.clear()
        self.high.clear()


# --------------------------------------------------------------------------- engine

class InferenceEngine:
    def __init__(self, predictor, load_errors: dict | None = None, smoother: StateSmoother | None = None):
        self.predictor = predictor
        self.load_errors = load_errors or {}
        self.smoother = smoother or StateSmoother()

    @classmethod
    def from_env(cls) -> "InferenceEngine":
        predictor, errors = build_predictor()
        return cls(predictor, errors)

    @property
    def mode(self) -> str:
        return self.predictor.source

    def run(self, video_feat: np.ndarray | None, audio_feat: np.ndarray | None) -> dict:
        if self.mode == "model":
            if video_feat is None or audio_feat is None:
                raise ValueError("The trained model needs both video and audio features")
            probs = self.predictor.predict(video_feat, audio_feat)
        else:
            probs = self.predictor.predict()
        state = self.smoother.update(probs)
        return {
            "cognitive_state": state,
            "trigger_focus_mode": state_triggers_focus(state),
            "source": self.mode,
            "probabilities": {d: round(p, 4) for d, p in probs.items()},
            "smoothed": {d: round(p, 4) for d, p in self.smoother.smoothed.items()},
            "timestamp": time.time(),
        }
