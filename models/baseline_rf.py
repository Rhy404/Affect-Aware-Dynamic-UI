import argparse
import glob
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def pool_video(video_tensor: torch.Tensor, pool_size=(2, 4, 4)) -> np.ndarray:
    if video_tensor.ndim == 5 and video_tensor.shape[2] in (1, 3):
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
    pooled = nn.functional.adaptive_avg_pool3d(video_tensor.float(), pool_size)
    return pooled.flatten(1).numpy()


def collect_capped_split(split_dir: str, max_samples: int, pool_size=(2, 4, 4)):
    shard_paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {split_dir}")

    xs, ys = [], []
    collected = 0
    for path in shard_paths:
        need = max_samples - collected
        if need <= 0:
            break
        shard = torch.load(path, map_location="cpu")
        x_np = pool_video(shard["video"], pool_size=pool_size)[:need]
        y_np = shard["labels"].long().numpy()[:need]
        del shard

        xs.append(x_np)
        ys.append(y_np)
        collected += len(y_np)

    return np.vstack(xs), np.vstack(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--out-dir", default="results_clean")
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pool_size = (2, 4, 4)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    X_tr, y_tr = collect_capped_split(train_dir, args.max_train_samples, pool_size=pool_size)
    X_va, y_va = collect_capped_split(val_dir, max_samples=10**9, pool_size=pool_size)

    models = []
    preds_val = np.zeros_like(y_va)
    for col in range(y_tr.shape[1]):
        tr_mask = y_tr[:, col] >= 0
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr[tr_mask], y_tr[tr_mask, col])
        models.append(clf)

        preds_val[:, col] = clf.predict(X_va)
        va_mask = y_va[:, col] >= 0
        if va_mask.any():
            acc = accuracy_score(y_va[va_mask, col], preds_val[va_mask, col])
            f1 = f1_score(y_va[va_mask, col], preds_val[va_mask, col], zero_division=0)
            print(f"[RF | Target {col}] Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

    joblib.dump(models, os.path.join(args.out_dir, "model2_rf.joblib"))
    np.save(os.path.join(args.out_dir, "preds_model2.npy"), preds_val)
    print("Saved model2_rf.joblib and preds_model2.npy to:", args.out_dir)


if __name__ == "__main__":
    main()
