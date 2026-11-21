import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn


class UnimodalVideoMLP(nn.Module):
    def __init__(self, in_features: int, num_targets: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pool_video(video_tensor: torch.Tensor, pool_size=(2, 4, 4)) -> torch.Tensor:
    if video_tensor.ndim == 5 and video_tensor.shape[2] in (1, 3):
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
    pooled = nn.functional.adaptive_avg_pool3d(video_tensor.float(), pool_size)
    return pooled.flatten(1)


def iter_batches(split_dir: str, batch_size: int, pool_size=(2, 4, 4), shuffle: bool = True):
    shard_paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {split_dir}")

    if shuffle:
        rng = np.random.default_rng(42)
        shard_paths = [shard_paths[i] for i in rng.permutation(len(shard_paths))]

    for path in shard_paths:
        shard = torch.load(path, map_location="cpu")
        x = pool_video(shard["video"], pool_size=pool_size)
        y = shard["labels"].long()
        del shard

        n = len(x)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            yield x[idx], y[idx]


def masked_bce_and_acc(logits: torch.Tensor, targets: torch.Tensor):
    mask = targets >= 0
    if not mask.any():
        return logits.sum() * 0.0, 0, 0

    valid_logits = logits[mask]
    valid_targets = targets[mask].float()
    loss = nn.functional.binary_cross_entropy_with_logits(valid_logits, valid_targets)

    preds = (torch.sigmoid(valid_logits) >= 0.5).long()
    correct = (preds == targets[mask]).sum().item()
    count = mask.sum().item()
    return loss, correct, count


@torch.no_grad()
def predict_val(model: nn.Module, val_dir: str, batch_size: int, pool_size, device) -> np.ndarray:
    model.eval()
    preds_all = []
    for bx, _ in iter_batches(val_dir, batch_size=batch_size, pool_size=pool_size, shuffle=False):
        bx = bx.to(device, non_blocking=True)
        probs = torch.sigmoid(model(bx))
        preds_all.append((probs >= 0.5).long().cpu().numpy())
    return np.vstack(preds_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="processed_data")
    parser.add_argument("--out-dir", default="results_clean")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_size = (2, 4, 4)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    sample_x, sample_y = next(iter_batches(train_dir, batch_size=2, pool_size=pool_size, shuffle=False))
    model = UnimodalVideoMLP(in_features=sample_x.shape[1], num_targets=sample_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0

        for bx, by in iter_batches(train_dir, batch_size=args.batch_size, pool_size=pool_size, shuffle=True):
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss, correct, count = masked_bce_and_acc(logits, by)
            loss.backward()
            optimizer.step()

            if count > 0:
                tr_loss_sum += loss.item() * count
                tr_correct += correct
                tr_total += count

        model.eval()
        va_loss_sum, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in iter_batches(val_dir, batch_size=args.batch_size, pool_size=pool_size, shuffle=False):
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                logits = model(bx)
                loss, correct, count = masked_bce_and_acc(logits, by)
                if count > 0:
                    va_loss_sum += loss.item() * count
                    va_correct += correct
                    va_total += count

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {tr_loss_sum / max(tr_total, 1):.4f}, Train Acc: {tr_correct / max(tr_total, 1):.4f} | "
            f"Val Loss: {va_loss_sum / max(va_total, 1):.4f}, Val Acc: {va_correct / max(va_total, 1):.4f}"
        )

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model1_video.pt"))
    preds = predict_val(model, val_dir, args.batch_size, pool_size, device)
    np.save(os.path.join(args.out_dir, "preds_model1.npy"), preds)
    print("Saved model1_video.pt and preds_model1.npy to:", args.out_dir)


if __name__ == "__main__":
    main()
