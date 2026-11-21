import io, re, zipfile
import numpy as np
from PIL import Image

SRC_FPS = 16
OUT_FPS = 8
SIZE = 112
WIN_FRAMES = 16
IDX_RE = re.compile(r"img_(\d+)\.jpg$")


def load_crops(zip_path):
    """Read face crops from nested ZIP files. Returns sorted indices and RGB frames."""
    with zipfile.ZipFile(zip_path) as outer:
        inner = outer.read(outer.namelist()[0])

    items = []
    with zipfile.ZipFile(io.BytesIO(inner)) as z:
        for n in z.namelist():
            m = IDX_RE.search(n)
            if m:
                items.append((int(m.group(1)), n))

        items.sort()
        idxs, frames = [], []

        for i, n in items:
            try:
                with Image.open(io.BytesIO(z.read(n))) as im:
                    im = im.convert("RGB")
                    im = im.resize((SIZE, SIZE), Image.Resampling.BOX)
                    frames.append(np.asarray(im, dtype=np.uint8))
                    idxs.append(i)
            except Exception as e:
                print(f"Warning: skipping unreadable crop {n}: {e}")

    arr = np.stack(frames) if frames else np.zeros((0, SIZE, SIZE, 3), np.uint8)
    return idxs, arr


def video_windows(frames, src_fps=SRC_FPS, out_fps=OUT_FPS, win=WIN_FRAMES):
    sub = frames[::max(1, round(src_fps / out_fps))]
    n = len(sub) // win
    return sub[:n * win].reshape(n, win, SIZE, SIZE, 3)


TOL = 4


def video_windows_by_time(idxs, frames, n_windows, src_fps=SRC_FPS,
                          out_fps=OUT_FPS, win=WIN_FRAMES, tol=TOL):
    step = src_fps / out_fps
    video = np.zeros((n_windows, win, SIZE, SIZE, 3), np.uint8)
    valid = np.zeros((n_windows, win), bool)

    if len(idxs) == 0:
        return video, valid.mean(axis=1)

    idx = np.asarray(idxs)

    for k in range(n_windows):
        for j in range(win):
            target = (k * win + j) * step
            pos = np.searchsorted(idx, target)
            cands = [p for p in (pos - 1, pos) if 0 <= p < len(idx)]
            best = min(cands, key=lambda p: abs(idx[p] - target))

            if abs(idx[best] - target) <= tol:
                video[k, j] = frames[best]
                valid[k, j] = True

    return video, valid.mean(axis=1)
