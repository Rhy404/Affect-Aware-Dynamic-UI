"""Turns raw webcam frames + microphone audio into model input features.

INTEGRATION CONTRACT WITH MEMBER 1 (preprocessing/extract_features.py)
---------------------------------------------------------------------
The live demo only works if the backend preprocesses webcam/mic data EXACTLY the
way the training data was preprocessed. So if preprocessing/extract_features.py
defines these two functions, the backend uses them automatically:

    def video_features_from_frames(frames: list[np.ndarray]) -> np.ndarray
        # frames: list of BGR uint8 images (as read by cv2), ~2 s of webcam video
        # returns: 1-D float array, same length as the training video features

    def audio_features_from_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray
        # waveform: 1-D float32 mono audio in [-1, 1], ~2 s long
        # returns: 1-D float array, same length as the training audio features

Until those exist, the PLACEHOLDER functions below are used. They produce
vectors of the right size (256 video / 128 audio) so the whole pipeline runs
end to end, but they are NOT the training preprocessing - predictions from a
real checkpoint will only be meaningful once Member 1's functions are in place.
"""

from __future__ import annotations

import io
import logging
import os
import wave

import cv2
import numpy as np

log = logging.getLogger(__name__)

VIDEO_DIM = 256
AUDIO_DIM = 128
TARGET_SAMPLE_RATE = 16_000  # plan: resample audio to 16 kHz
FACE_SIZE = 112  # plan: resize frames to 112 x 112

def _load_face_detector():
    """OpenCV's bundled Haar cascade (shipped with opencv-python 4.x; some 5.x
    builds no longer include it). Returns None if unavailable -> centre crop."""
    try:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(path)
        if not detector.empty():
            return detector
    except Exception:
        pass
    log.warning("OpenCV face detector not available - using centre crops instead of face crops")
    return None


_face_detector = _load_face_detector()


# --------------------------------------------------------------------------- decoding

def decode_image(data: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG bytes into a BGR uint8 image (None if undecodable)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit PCM WAV bytes into (mono float32 waveform in [-1, 1], sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {8 * width}-bit")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


# --------------------------------------------------------------------------- face crop

def crop_face(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (112x112 grayscale face crop, face_found). Falls back to a centre crop."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = () if _face_detector is None else _face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces):
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        crop = gray[y : y + h, x : x + w]
        found = True
    else:
        h, w = gray.shape
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = gray[y0 : y0 + side, x0 : x0 + side]
        found = False
    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA), found


# --------------------------------------------------------------------------- placeholders

def _placeholder_video_features(frames: list[np.ndarray]) -> np.ndarray:
    """PLACEHOLDER: face crops -> temporal mean -> 16x16 thumbnail -> 256-d vector."""
    crops = [crop_face(f)[0].astype(np.float32) / 255.0 for f in frames]
    mean_face = np.mean(crops, axis=0)
    thumb = cv2.resize(mean_face, (16, 16), interpolation=cv2.INTER_AREA).flatten()
    return ((thumb - thumb.mean()) / (thumb.std() + 1e-6)).astype(np.float32)


def _placeholder_audio_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """PLACEHOLDER: resample to 16 kHz -> 128-bin log-mel spectrogram -> mean over time."""
    import librosa  # imported lazily: slow to import, only needed for live audio

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    if waveform.size < 2048:
        waveform = np.pad(waveform, (0, 2048 - waveform.size))
    mel = librosa.feature.melspectrogram(y=waveform, sr=TARGET_SAMPLE_RATE, n_mels=AUDIO_DIM)
    log_mel = librosa.power_to_db(mel, ref=np.max).mean(axis=1)
    return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)).astype(np.float32)


# Use Member 1's preprocessing when it is available.
try:
    from preprocessing.extract_features import (  # type: ignore
        audio_features_from_waveform as _audio_fn,
        video_features_from_frames as _video_fn,
    )

    USING_PLACEHOLDER = False
    log.info("Using Member 1's preprocessing from preprocessing/extract_features.py")
except Exception:  # ImportError, or the file is still empty
    _video_fn = _placeholder_video_features
    _audio_fn = _placeholder_audio_features
    USING_PLACEHOLDER = True
    log.warning("preprocessing.extract_features not ready - using placeholder features")


def video_features(frames: list[np.ndarray]) -> tuple[np.ndarray, bool]:
    """Return (video feature vector, whether a face was found in any frame)."""
    face_found = any(crop_face(f)[1] for f in frames[:: max(1, len(frames) // 3)])
    return np.asarray(_video_fn(frames), dtype=np.float32).ravel(), face_found


def audio_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.asarray(_audio_fn(waveform, sample_rate), dtype=np.float32).ravel()
