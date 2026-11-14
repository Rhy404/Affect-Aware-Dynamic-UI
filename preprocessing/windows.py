import os
import numpy as np
from video_features import load_crops, video_windows_by_time

# Safely try importing audio features if available
try:
    from audio_features import load_audio, log_mel, windows as audio_windows
    HAS_AUDIO_MODULE = True
except ImportError:
    HAS_AUDIO_MODULE = False

SILENCE = float(np.log(1e-6))


def build_clip(row, root, no_audio="keep", min_valid=0.75):
    """row: a manifest row (dict). Returns aligned 2 s windows, or None if nothing is usable.
    video (n,16,112,112,3) uint8 | audio (n,80,200) float32 | window_idx (n,) = which 2 s slot each is.
    no_audio: 'keep' (as decoded), 'mask' (replace with silence), 'drop' (skip no-audio clips).
    min_valid: drop windows where less than this fraction of the 16 frames has a real face crop.
    n_video / n_audio are KEPT / TOTAL windows (names kept so the test scripts still work)."""
    if row.get("has_crops") != "True":
        return None
    
    crop_path = row.get("crop_path", "")
    if not crop_path:
        return None
    
    # 1. Load face crops safely
    idxs, frames = load_crops(os.path.join(root, crop_path))
    
    # 2. Check if raw video/audio file exists on disk
    video_path = row.get("video_path", "")
    video_full_path = os.path.join(root, video_path) if video_path else ""
    
    has_audio = (row.get("has_audio") == "True") and video_path and os.path.exists(video_full_path) and HAS_AUDIO_MODULE
    
    if not has_audio and no_audio == "drop":
        return None

    # 3. Load audio if it exists, otherwise create safe dummy audio windows
    if has_audio:
        try:
            aw = audio_windows(log_mel(load_audio(video_full_path)))
        except Exception:
            has_audio = False
            n_total = max(1, len(frames) // 16)
            aw = [np.zeros((80, 200), dtype=np.float32) for _ in range(n_total)]
    else:
        # Fallback: estimate number of windows from video frames if no audio file is present
        n_total = max(1, len(frames) // 16)
        aw = [np.zeros((80, 200), dtype=np.float32) for _ in range(n_total)]

    n_total = len(aw)
    if n_total == 0:
        return None
        
    vw, valid = video_windows_by_time(idxs, frames, n_total)
    keep = np.where(valid >= min_valid)[0]
    if len(keep) == 0:
        return None
        
    audio = np.stack(aw)[keep]
    if not has_audio and no_audio == "mask":
        audio = np.full_like(audio, SILENCE)
        
    return dict(video=vw[keep], audio=audio, window_idx=keep, video_valid=valid[keep],
                has_audio=has_audio, n=len(keep), n_video=len(keep), n_audio=n_total)