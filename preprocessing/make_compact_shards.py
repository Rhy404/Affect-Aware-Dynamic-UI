import argparse
import io
import os
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image


def load_ground_truths(zf: zipfile.ZipFile, targets: list[str]) -> dict[str, dict[str, float]]:
    gt_map: dict[str, dict[str, float]] = {}
    for t in targets:
        path = f"avcaffe/ground_truths/{t}.txt"
        mapping: dict[str, float] = {}
        if path in zf.namelist():
            raw = zf.read(path).decode("utf-8", errors="replace")
            for line in raw.splitlines():
                line = line.strip()
                if not line or "," not in line:
                    continue
                key, val = line.split(",", 1)
                try:
                    mapping[key.strip()] = float(val.strip())
                except ValueError:
                    continue
        gt_map[t] = mapping
    return gt_map


def clip_to_task_key(clip_id: str) -> str:
    if "_clip_" in clip_id:
        return clip_id.split("_clip_", 1)[0]
    return clip_id


def subject_from_clip_id(clip_id: str) -> str:
    return clip_id.split("_", 1)[0]


def read_clip_frames(zf: zipfile.ZipFile, crop_path: str, frames_per_clip: int, img_size: int) -> torch.Tensor | None:
    inner_name = "avcaffe/" + crop_path.lstrip("/\\").replace("\\", "/")
    if inner_name not in zf.namelist():
        return None

    inner_bytes = io.BytesIO(zf.read(inner_name))
    with zipfile.ZipFile(inner_bytes) as clip_zf:
        img_names = sorted(
            [n for n in clip_zf.namelist() if not n.endswith("/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        if not img_names:
            return None

        if len(img_names) >= frames_per_clip:
            idxs = np.linspace(0, len(img_names) - 1, frames_per_clip, dtype=int)
        else:
            idxs = np.resize(np.arange(len(img_names)), frames_per_clip)

        frames = []
        for idx in idxs:
            with clip_zf.open(img_names[int(idx)]) as f:
                im = Image.open(f).convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
                arr = np.asarray(im, dtype=np.uint8)  # (H, W, C)
                frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # (C, H, W)

        # Return shape: (C, T, H, W) uint8 to keep disk footprint small
        return torch.stack(frames, dim=1)


def flush_shard(out_dir: str, shard_idx: int, videos: list, labels: list, clip_ids: list):
    if not videos:
        return
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"shard_{shard_idx:04d}.pt")
    torch.save(
        {
            "video": torch.stack(videos, dim=0),  # (N, C, T, H, W) uint8
            "labels": torch.tensor(labels, dtype=torch.int64),  # (N, num_targets)
            "clip_id": clip_ids,
        },
        out_path,
    )
    print(f"Wrote {out_path} ({len(videos)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default=r"D:\dev\data\avcaffe.zip")
    parser.add_argument("--out-dir", default=r"D:\dev\data\shards_compact")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=10.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all valid clips; >0 for quick test")
    args = parser.parse_args()

    targets = ["mental_demand", "effort", "temporal_demand"]

    with zipfile.ZipFile(args.zip_path, "r") as zf:
        manifest_bytes = zf.read("avcaffe/manifest.csv")
        df = pd.read_csv(io.BytesIO(manifest_bytes))
        gt_map = load_ground_truths(zf, targets)

        df = df[df["has_crops"] == True].dropna(subset=["crop_path"]).reset_index(drop=True)
        if args.max_clips > 0:
            df = df.iloc[: args.max_clips].copy()

        subjects = sorted(df["clip_id"].map(subject_from_clip_id).unique())
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(subjects))
        n_val = max(1, int(len(subjects) * args.val_ratio))
        val_subjects = {subjects[i] for i in perm[:n_val]}

        buffers = {
            "train": {"v": [], "y": [], "ids": [], "shard_idx": 0},
            "val": {"v": [], "y": [], "ids": [], "shard_idx": 0},
        }

        processed = 0
        for row in df.itertuples(index=False):
            clip_id = str(row.clip_id)
            subj = subject_from_clip_id(clip_id)
            split = "val" if subj in val_subjects else "train"

            task_key = clip_to_task_key(clip_id)
            row_labels = []
            for t in targets:
                score = gt_map[t].get(task_key, None)
                if score is None or np.isnan(score):
                    row_labels.append(-1)
                else:
                    row_labels.append(1 if score >= args.threshold else 0)

            vid = read_clip_frames(zf, str(row.crop_path), args.frames_per_clip, args.img_size)
            if vid is None:
                continue

            buf = buffers[split]
            buf["v"].append(vid)
            buf["y"].append(row_labels)
            buf["ids"].append(clip_id)
            processed += 1

            if len(buf["v"]) >= args.shard_size:
                flush_shard(os.path.join(args.out_dir, split), buf["shard_idx"], buf["v"], buf["y"], buf["ids"])
                buf["shard_idx"] += 1
                buf["v"].clear()
                buf["y"].clear()
                buf["ids"].clear()

            if processed % 500 == 0:
                print(f"Processed {processed} clips...")

        for split, buf in buffers.items():
            if buf["v"]:
                flush_shard(os.path.join(args.out_dir, split), buf["shard_idx"], buf["v"], buf["y"], buf["ids"])

    print(f"Done. Total clips written: {processed}")


if __name__ == "__main__":
    main()
