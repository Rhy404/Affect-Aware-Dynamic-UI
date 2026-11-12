"""Replay real AVCAffe validation windows through the running backend (demo "Option C").

Live webcam crops never look exactly like the dataset's face crops, so the most
honest way to show the TRAINED model driving the UI is to stream held-out
validation windows (participants the model never saw) into it. The UI, in Auto
mode, adapts to each prediction, and this script prints prediction vs truth.

It reads clips with Member 1's ClipReader (src/preprocessing/dataset.py)
read-only, exactly as the training code does, and POSTs each window to
/predict-window.

Usage (backend already running, from the repo root):

    python -m backend.replay_val_clips --data C:\\dev\\data\\avcaffe \\
        --preprocessing ..\\ml-CSD361\\src\\preprocessing --n 30 --interval 2.5

Clips are played in alternating blocks of Low / High Mental Demand (--block),
so the audience can watch the UI switch modes as the load changes.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from .features import _load_module


def load_clip_reader(pre_dir: Path):
    """Import Member 1's dataset.py by path (no bytecode written to his folder)."""
    sys.path.insert(0, str(pre_dir))  # his dataset.py does `from windows import build_clip`
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return _load_module("dataset", pre_dir / "dataset.py").ClipReader
    finally:
        sys.dont_write_bytecode = previous


def order_clips(rows, n, block, seed):
    rng = random.Random(seed)
    low = [i for i, r in enumerate(rows) if r.get("mental_demand_bin") == "0"]
    high = [i for i, r in enumerate(rows) if r.get("mental_demand_bin") == "1"]
    rng.shuffle(low)
    rng.shuffle(high)
    order, take_high = [], False
    while len(order) < n and (low or high):
        src = high if take_high and high else low if low else high
        order += [src.pop() for _ in range(min(block, len(src), n - len(order)))]
        take_high = not take_high
    return order


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="AVCAffe root (folder with manifest.csv)")
    ap.add_argument("--preprocessing", required=True, help="Member 1's src/preprocessing folder")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=30, help="number of windows to play")
    ap.add_argument("--block", type=int, default=5, help="windows per Low/High block")
    ap.add_argument("--interval", type=float, default=2.5, help="seconds between windows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ClipReader = load_clip_reader(Path(args.preprocessing).resolve())
    reader = ClipReader(args.data, "val", mode="random", seed=args.seed)
    order = order_clips(reader.rows, args.n, args.block, args.seed)
    print(f"Replaying {len(order)} validation windows to {args.url}/predict-window\n")
    print(f"{'clip':28} {'true MD/EF/TD':>13}  {'P(High) MD / EF / TD':>22}  UI state")

    with httpx.Client(timeout=30) as client:
        for i in order:
            item = reader[i]
            buf = io.BytesIO()
            np.savez(buf, video=item["video"], audio=item["audio"], labels=item["labels"])
            r = client.post(f"{args.url}/predict-window",
                            files={"window": ("window.npz", buf.getvalue(), "application/octet-stream")})
            r.raise_for_status()
            body = r.json()
            p = body["probabilities"]
            truth = "/".join(str(v) for v in item["labels"].tolist())
            probs = " / ".join(f"{p[t]:.2f}" if t in p else "  - " for t in ("mental_demand", "effort", "temporal_demand"))
            print(f"{item['clip_id']:28} {truth:>13}  {probs:>22}  {body['cognitive_state']}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
