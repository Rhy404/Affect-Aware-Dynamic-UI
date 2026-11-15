"""Live webcam + microphone -> model input tensors, matching Member 1's training data.

Member 1's preprocessing (preprocessing/ in this repo) turns each
AVCAffe clip into 2-second windows, normalised in ClipReader (dataset.py):

    video  (3, 16, 112, 112) float32  RGB face crops, 16 frames at 8 fps,
                                      x/255, then (x - mean) / std per channel
    audio  (1, 80, 200)      float32  80-bin log-mel, 16 kHz, 10 ms hop,
                                      then (x - mean) / std

This module produces the SAME tensors from a live 2-second webcam/mic window.
It does not copy or change Member 1's code: it loads his audio_features.py and
video_features.py (for log_mel and the constants) and configs/norm_stats.json
read-only from wherever they live. Search order:

    1. $AFFECT_PREPROCESSING_DIR             (folder containing audio_features.py)
    2. <repo>/preprocessing/  and  <repo>/src/preprocessing/
    3. ../ml-CSD361/src/preprocessing/        (his repo cloned next to this one)

and for the stats: $AFFECT_NORM_STATS, then configs/norm_stats.json next to
his preprocessing folder, then <repo>/configs/norm_stats.json.

If his code cannot be found, the backend still runs in mock mode, but a real
model cannot be fed live data; /health reports this.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_VALID = 0.75  # same default as Member 1's build_clip(min_valid=0.75): windows with
# fewer than 75 % real face frames are never used in training, so we skip them live too.


# --------------------------------------------------------------------------- locate Member 1's code

def _candidate_dirs() -> list[Path]:
    dirs = []
    if os.getenv("AFFECT_PREPROCESSING_DIR"):
        dirs.append(Path(os.environ["AFFECT_PREPROCESSING_DIR"]))
    dirs += [
        REPO_ROOT / "preprocessing",
        REPO_ROOT / "src" / "preprocessing",
        REPO_ROOT.parent / "ml-CSD361" / "src" / "preprocessing",
        REPO_ROOT.parent / "ml-CSD361-main" / "src" / "preprocessing",
    ]
    return dirs


def _load_module(name: str, path: Path):
    """Import a file by path without writing __pycache__ into Member 1's folder."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(f"member1_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def _find_stats(pre_dir: Path) -> Path | None:
    options = []
    if os.getenv("AFFECT_NORM_STATS"):
        options.append(Path(os.environ["AFFECT_NORM_STATS"]))
    options += [pre_dir.parent / "configs" / "norm_stats.json", pre_dir.parent.parent / "configs" / "norm_stats.json", REPO_ROOT / "configs" / "norm_stats.json"]
    return next((p for p in options if p.is_file()), None)


class Member1Preprocessing:
    """Holds Member 1's functions, constants and normalisation stats."""

    def __init__(self):
        self.found = False
        self.directory: Path | None = None
        self.stats_path: Path | None = None
        self.error: str | None = None
        for d in _candidate_dirs():
            if (d / "audio_features.py").is_file() and (d / "video_features.py").is_file():
                self.directory = d
                break
        if self.directory is None:
            self.error = "audio_features.py / video_features.py not found (set AFFECT_PREPROCESSING_DIR)"
            return
        self.stats_path = _find_stats(self.directory)
        if self.stats_path is None:
            self.error = "configs/norm_stats.json not found (set AFFECT_NORM_STATS)"
            return
        try:
            self.audio = _load_module("audio_features", self.directory / "audio_features.py")
            self.video = _load_module("video_features", self.directory / "video_features.py")
            stats = json.loads(self.stats_path.read_text())
        except Exception as exc:  # e.g. PyAV not installed
            self.error = f"could not load Member 1's preprocessing: {type(exc).__name__}: {exc}"
            return
        self.a_mean = float(stats["audio_logmel_mean"])
        self.a_std = float(stats["audio_logmel_std"])
        self.v_mean = np.asarray(stats["video_rgb_mean_0to1"], np.float32).reshape(1, 1, 1, 3)
        self.v_std = np.asarray(stats["video_rgb_std_0to1"], np.float32).reshape(1, 1, 1, 3)
        self.sr = int(self.audio.SR)
        self.win_frames = int(self.video.WIN_FRAMES)
        self.size = int(self.video.SIZE)
        self.mel_frames = int(self.audio.WIN)
        self.n_mels = int(self.audio.N_MELS)
        self.found = True
        log.info("Using Member 1's preprocessing from %s (stats: %s)", self.directory, self.stats_path)

    def info(self) -> dict:
        out = {"found": self.found, "directory": str(self.directory) if self.directory else None,
               "norm_stats": str(self.stats_path) if self.stats_path else None, "error": self.error}
        if self.found:
            out["video_shape"] = [3, self.win_frames, self.size, self.size]
            out["audio_shape"] = [1, self.n_mels, self.mel_frames]
            out["window_seconds"] = self.mel_frames * self.audio.HOP / self.sr
        return out


PRE = Member1Preprocessing()


# --------------------------------------------------------------------------- decoding uploads

def decode_image(data: bytes) -> np.ndarray | None:
    """JPEG/PNG bytes -> BGR uint8 image (None if undecodable)."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV bytes -> (mono float32 waveform in [-1, 1], sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {8 * width}-bit")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


# --------------------------------------------------------------------------- live face crops

def _load_face_detector():
    """OpenCV's bundled Haar cascade (in opencv-python 4.x). None -> no face crops."""
    try:
        detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        if not detector.empty():
            return detector
    except Exception:
        pass
    log.warning("OpenCV face detector not available (install opencv-python<5)")
    return None


_face_detector = _load_face_detector()


def face_crop_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray | None:
    """Largest detected face, square-padded by 20 %, resized like Member 1's load_crops
    (INTER_AREA to size x size, BGR -> RGB). None if no face is found."""
    if _face_detector is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if not len(faces):
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy, side = x + w / 2, y + h / 2, int(max(w, h) * 1.2)
    H, W = gray.shape
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    crop = frame_bgr[y0 : min(H, y0 + side), x0 : min(W, x0 + side)]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- window -> tensors

def video_window(frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """~2 s of webcam frames -> (normalised video (1,3,16,112,112), fraction of frames with a face).

    Mirrors Member 1: 16 frames evenly spaced over the window; a frame with no face is
    left black (as video_windows_by_time does for missing crops); then ClipReader's
    x/255 and per-channel (x - mean) / std, transposed to (C, T, H, W)."""
    n, size = PRE.win_frames, PRE.size
    picks = np.linspace(0, len(frames_bgr) - 1, n).round().astype(int)  # pad/resample to 16 frames
    clip = np.zeros((n, size, size, 3), np.uint8)
    valid = np.zeros(n, bool)
    cache: dict[int, np.ndarray | None] = {}
    for j, i in enumerate(picks):
        if i not in cache:
            cache[i] = face_crop_rgb(frames_bgr[i], size)
        if cache[i] is not None:
            clip[j], valid[j] = cache[i], True
    video = (clip.astype(np.float32) / 255.0 - PRE.v_mean) / PRE.v_std
    video = np.ascontiguousarray(np.transpose(video, (3, 0, 1, 2)))[None]  # (1,3,16,112,112)
    return video, float(valid.mean())


def _resample(wave_: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return wave_
    n_out = int(round(len(wave_) * target / sr))
    return np.interp(np.linspace(0, len(wave_) - 1, n_out), np.arange(len(wave_)), wave_).astype(np.float32)


def audio_window(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """~2 s of mic audio -> normalised log-mel (1,1,80,200), using Member 1's log_mel."""
    wave_ = _resample(waveform.astype(np.float32), sample_rate, PRE.sr)
    need = PRE.mel_frames * PRE.audio.HOP  # 200 frames * 160 samples = 2 s at 16 kHz
    wave_ = wave_[-need:] if len(wave_) >= need else np.pad(wave_, (need - len(wave_), 0))
    mel = PRE.audio.log_mel(wave_)[:, : PRE.mel_frames]  # (80, 200), exactly his windows()
    mel = (mel - PRE.a_mean) / PRE.a_std
    return mel[None, None].astype(np.float32)  # (1,1,80,200)
