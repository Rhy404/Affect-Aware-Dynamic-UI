"""Model 1 (unimodal video baseline) and Model 2 (Random Forest) on processed_data/."""
import argparse
import glob
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


POOL = (4, 8, 8)


def load_split(root, split):
    """Pooled video features (N, 768), labels (N, 3) and (clip_id, window_idx) per row."""
    paths = sorted(glob.glob(os.path.join(root, split, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError("no shard_*.pt in %s" %
                                os.path.join(root, split))
    xs, ys, ids = [], [], []
    for p in paths:
        s = torch.load(p, map_location="cpu")
        xs.append(nn.functional.adaptive_avg_pool3d(
            s["video"].float(), POOL).flatten(1))
        ys.append(s["labels"].long())
        ids += list(zip(s["clip_id"], s["window_idx"].tolist()))
    return torch.cat(xs), torch.cat(ys), ids


class UnimodalBaselineModel(nn.Module):
    """Video-only MLP, one logit per target. Audio is ignored on purpose."""

    def __init__(self, n_in, n_out=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_out))

    def forward(self, video_feat, audio_feat=None):
        return self.classifier(video_feat.flatten(1))


def masked_bce(logits, y):
    """BCE over the targets that have a label (-1 means missing)."""
    m = y >= 0
    if not m.any():
        return logits.sum() * 0
    return nn.functional.binary_cross_entropy_with_logits(logits[m], y[m].float())


def train_unimodal(x, y, epochs=10, bs=32, lr=3e-4):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnimodalBaselineModel(x.shape[1], y.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        perm, tot = torch.randperm(len(x)), 0.0
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            loss = masked_bce(model(x[idx].to(dev)), y[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print("[model1] epoch %d/%d loss %.4f" %
              (ep + 1, epochs, tot / len(x)))
    return model


@torch.no_grad()
def predict_unimodal(model, x, bs=256):
    model.eval()
    dev = next(model.parameters()).device
    out = [(model(x[i:i + bs].to(dev)) > 0).long().cpu()
           for i in range(0, len(x), bs)]
    return torch.cat(out).numpy()


def train_rf(x, y, max_n=None):
    """One Random Forest per target, rows with a missing label dropped."""
    x, y = x.numpy(), y.numpy()
    if max_n and len(x) > max_n:
        idx = np.random.RandomState(42).choice(len(x), max_n, replace=False)
        x, y = x[idx], y[idx]
    rfs = []
    for t in range(y.shape[1]):
        m = y[:, t] >= 0
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)
        rfs.append(rf.fit(x[m], y[m, t]))
    return rfs


def predict_rf(rfs, x):
    x = x.numpy()
    return np.stack([rf.predict(x) for rf in rfs], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="processed_data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-rf", type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    xtr, ytr, _ = load_split(args.root, "train")
    xva, yva, vids = load_split(args.root, "val")
    with open(os.path.join(args.out, "val_windows.txt"), "w") as f:
        f.writelines("%s,%d\n" % v for v in vids)
    np.save(os.path.join(args.out, "y_train.npy"), ytr.numpy())
    np.save(os.path.join(args.out, "y_val.npy"), yva.numpy())

    m1 = train_unimodal(xtr, ytr, epochs=args.epochs)
    torch.save(m1.state_dict(), os.path.join(args.out, "model1_video.pt"))
    np.save(os.path.join(args.out, "preds_model1.npy"),
            predict_unimodal(m1, xva))

    rfs = train_rf(xtr, ytr, args.max_rf)
    joblib.dump(rfs, os.path.join(args.out, "model2_rf.joblib"))
    np.save(os.path.join(args.out, "preds_model2.npy"), predict_rf(rfs, xva))
    print("saved predictions to", args.out)
