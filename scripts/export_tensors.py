import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("src/preprocessing"))

from torch_dataset import AVCAffeDataset
from windows import build_clip

ROOT = r"C:\dev\data\avcaffe"
OUTPUT = "processed_data"
SHARD_SIZE = 64
MIN_VALID = 0.75

def export_split(split):
    print(f"\n========== EXPORTING {split.upper()} ==========")

    ds = AVCAffeDataset(
        root=ROOT,
        split=split,
        mode="all",
        no_audio="keep"
    )

    rows = ds.reader.rows
    out_dir = os.path.join(OUTPUT, split)
    os.makedirs(out_dir, exist_ok=True)

    videos, audios, labels = [], [], []
    audio_flags, clip_ids, window_indices = [], [], []

    shard_number = 0
    total_windows = 0
    skipped = 0
    errors = 0

    def save_shard():
        nonlocal shard_number, total_windows

        if not videos:
            return

        path = os.path.join(
            out_dir, f"shard_{shard_number:05d}.pt"
        )

        torch.save({
            "video": torch.stack(videos),
            "audio": torch.stack(audios),
            "labels": torch.stack(labels),
            "has_audio": torch.tensor(audio_flags, dtype=torch.bool),
            "clip_id": clip_ids.copy(),
            "window_idx": torch.tensor(window_indices, dtype=torch.long),
        }, path)

        count = len(videos)
        total_windows += count

        print(f"Saved {path} ({count} windows)")

        videos.clear()
        audios.clear()
        labels.clear()
        audio_flags.clear()
        clip_ids.clear()
        window_indices.clear()

        shard_number += 1

    for i, row in enumerate(rows):
        try:
            data = build_clip(
                row, ROOT,
                no_audio="keep",
                min_valid=MIN_VALID
            )

            if data is None:
                skipped += 1
                continue

            packed = ds.reader._pack(row, data, i)

            for j in range(packed["video"].shape[0]):
                videos.append(torch.from_numpy(
                    packed["video"][j].copy()
                ))
                audios.append(torch.from_numpy(
                    packed["audio"][j].copy()
                ))
                labels.append(torch.from_numpy(
                    packed["labels"].copy()
                ))

                audio_flags.append(packed["has_audio"])
                clip_ids.append(packed["clip_id"])
                window_indices.append(
                    int(packed["window_idx"][j])
                )

                if len(videos) >= SHARD_SIZE:
                    save_shard()

        except Exception as e:
            errors += 1
            print(f"ERROR on {row['clip_id']}: {e}")

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(rows)} clips | "
                f"Skipped: {skipped} | Errors: {errors}"
            )

    save_shard()

    summary = {
        "split": split,
        "clips_in_manifest": len(rows),
        "windows_exported": total_windows,
        "shards": shard_number,
        "skipped_clips": skipped,
        "errors": errors,
        "labels": list(ds.reader.targets),
        "video_shape": [3, 16, 112, 112],
        "audio_shape": [1, 80, 200],
        "shard_size": SHARD_SIZE
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSplit summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)

    export_split("train")
    export_split("val")

    print("\n========== EXPORT COMPLETE ==========")
