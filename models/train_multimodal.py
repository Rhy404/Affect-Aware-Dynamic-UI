"""Train the AffectAwareFusionModel on cached tensor shards (shard_*.pt).

Each shard is a dict with:  "video" (N,C,T,H,W) or (N,T,C,H,W), "labels" (N,3) with 1=High,
0=Low, -1=missing, and optionally "audio" (N,1,F,T) or (N,F,T). The format is detected
automatically; if there is no audio the model is built video-only.

Run from the repo root:
  python models/train_multimodal.py --dummy                       # test everything on random data
  python models/train_multimodal.py --train-dir D:\\...\\train --val-dir D:\\...\\val --sanity
  python models/train_multimodal.py --train-dir ... --val-dir ... --epochs 15 --export backend/multimodal_model.pt
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fusion_model import AffectAwareFusionModel  # noqa: E402

TARGETS = ["Mental Demand", "Effort", "Temporal Demand"]


# ----------------------------------------------------------------------------- data
def _fix_video(v):
    v = torch.as_tensor(v)
    if v.ndim != 5:
        raise ValueError(f"video should be 5-D, got shape {tuple(v.shape)}")
    if v.shape[1] not in (1, 3) and v.shape[2] in (1, 3):   # (N,T,C,H,W) -> (N,C,T,H,W)
        v = v.permute(0, 2, 1, 3, 4)
    if v.shape[2] < 8:
        raise ValueError(f"need at least 8 frames per clip, got {v.shape[2]}")
    return v.contiguous()


def _fix_audio(a):
    a = torch.as_tensor(a)
    if a.ndim == 3:
        a = a.unsqueeze(1)
    if a.ndim != 4:
        raise ValueError(
            f"audio should be 3-D or 4-D, got shape {tuple(a.shape)}")
    return a.contiguous()


def load_split(folder, max_clips=0):
    paths = sorted(glob.glob(os.path.join(folder, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt files in {folder}")
    vids, auds, labs, n = [], [], [], 0
    for p in paths:
        s = torch.load(p, map_location="cpu")
        if "video" not in s or "labels" not in s:
            raise KeyError(
                f"{p} has keys {list(s.keys())}; need at least 'video' and 'labels'")
        vids.append(_fix_video(s["video"]))
        labs.append(torch.as_tensor(s["labels"]).long())
        if "audio" in s:
            auds.append(_fix_audio(s["audio"]))
        n += len(labs[-1])
        if max_clips and n >= max_clips:
            break
    video, labels = torch.cat(vids), torch.cat(labs)
    audio = torch.cat(auds) if len(auds) == len(vids) else None
    if max_clips:
        video, labels = video[:max_clips], labels[:max_clips]
        audio = audio[:max_clips] if audio is not None else None
    return video, audio, labels


def make_dummy(n, use_audio, seed):
    g = torch.Generator().manual_seed(seed)
    video = torch.randint(0, 256, (n, 3, 8, 64, 64),
                          dtype=torch.uint8, generator=g)
    audio = torch.randn(n, 1, 80, 200, generator=g) if use_audio else None
    labels = (torch.rand(n, 3, generator=g) <
              torch.tensor([0.5, 0.5, 0.36])).long()
    return video, audio, labels


class ClipDataset(Dataset):
    def __init__(self, video, audio, labels, a_mean=0.0, a_std=1.0):
        self.video, self.audio, self.labels = video, audio, labels
        self.a_mean, self.a_std = a_mean, a_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        v = self.video[i]
        v = v.float() / 255.0 if v.dtype == torch.uint8 else v.float()
        v = (v - 0.5) / 0.5
        a = torch.zeros(1) if self.audio is None else (
            self.audio[i].float() - self.a_mean) / self.a_std
        return v, a, self.labels[i]


# ----------------------------------------------------------------------------- loss / metrics
def compute_pos_weight(labels):
    w = []
    for j, name in enumerate(TARGETS):
        col = labels[:, j]
        high, low = int((col == 1).sum()), int((col == 0).sum())
        w.append(low / high if high > 0 else 1.0)
        print(
            f"  {name:16s} High={high:6d}  Low={low:6d}  pos_weight={w[-1]:.2f}")
    return w


def masked_loss(bce, logits, labels):
    mask = (labels >= 0).float()
    loss = bce(logits, labels.float().clamp(min=0))
    return (loss * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labs = [], []
    for v, a, y in loader:
        logits = model(v.to(device), a.to(device))
        probs.append(torch.sigmoid(logits).cpu())
        labs.append(y)
    return torch.cat(probs).numpy(), torch.cat(labs).numpy()


def best_thresholds(probs, y):
    ths = []
    grid = np.linspace(0.05, 0.95, 91)
    for j in range(3):
        m = y[:, j] >= 0
        if m.sum() == 0:
            ths.append(0.5)
            continue
        f1s = [f1_score(y[m, j], probs[m, j] >= t, zero_division=0)
               for t in grid]
        ths.append(float(grid[int(np.argmax(f1s))]))
    return ths


def score(probs, y, ths):
    out = []
    for j in range(3):
        m = y[:, j] >= 0
        t, p = y[m, j], (probs[m, j] >= ths[j]).astype(int)
        out.append(dict(acc=accuracy_score(t, p), prec=precision_score(t, p, zero_division=0),
                        rec=recall_score(t, p, zero_division=0), f1=f1_score(t, p, zero_division=0)))
    return out


def majority_baseline(train_y, val_y):
    out = []
    for j in range(3):
        mt, mv = train_y[:, j] >= 0, val_y[:, j] >= 0
        maj = int((train_y[mt, j] == 1).sum() >= (train_y[mt, j] == 0).sum())
        p = np.full(mv.sum(), maj)
        out.append(dict(acc=accuracy_score(val_y[mv, j], p), f1=f1_score(
            val_y[mv, j], p, zero_division=0)))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir")
    ap.add_argument("--val-dir")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-clips", type=int, default=0,
                    help="limit clips per split (0 = all)")
    ap.add_argument("--sanity", action="store_true",
                    help="64 clips, 50 epochs: loss must drop well below 0.69")
    ap.add_argument("--dummy", action="store_true",
                    help="use random data to test the pipeline")
    ap.add_argument("--dummy-audio", action="store_true",
                    help="with --dummy: include a random audio input")
    ap.add_argument("--export", default="",
                    help="path for the TorchScript model, e.g. backend/multimodal_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    print("device:", device)

    if args.dummy:
        tr_v, tr_a, tr_y = make_dummy(256, args.dummy_audio, 0)
        va_v, va_a, va_y = make_dummy(64, args.dummy_audio, 1)
        args.epochs = min(args.epochs, 3)
    else:
        if not args.train_dir or not args.val_dir:
            ap.error("--train-dir and --val-dir are required (or use --dummy)")
        print("loading train shards ...")
        tr_v, tr_a, tr_y = load_split(args.train_dir, args.max_clips)
        print("loading val shards ...")
        va_v, va_a, va_y = load_split(args.val_dir, args.max_clips)

    use_audio = tr_a is not None and va_a is not None
    print(f"train clips: {len(tr_y)}  val clips: {len(va_y)}")
    print(f"video shape per clip: {tuple(tr_v.shape[1:])} ({tr_v.dtype});  audio: "
          f"{tuple(tr_a.shape[1:]) if use_audio else 'none -> video-only model'}")

    if args.sanity:
        tr_v, tr_y = tr_v[:64], tr_y[:64]
        tr_a = tr_a[:64] if use_audio else None
        args.epochs = 50
        print("SANITY MODE: 64 clips, 50 epochs")

    a_mean, a_std = 0.0, 1.0
    if use_audio:
        sample = tr_a[:2000].float()
        a_mean, a_std = float(sample.mean()), float(sample.std() + 1e-6)

    print("class weights (from TRAIN labels):")
    pos_w = compute_pos_weight(tr_y)

    train_loader = DataLoader(ClipDataset(tr_v, tr_a if use_audio else None, tr_y, a_mean, a_std),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ClipDataset(va_v, va_a if use_audio else None, va_y, a_mean, a_std),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AffectAwareFusionModel(use_audio=use_audio).to(device)
    print("model parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        pos_w, dtype=torch.float32, device=device), reduction="none")

    os.makedirs(args.out_dir, exist_ok=True)
    best_f1, best_state, best_ths, first_loss, last_loss = - \
        1.0, None, [0.5] * 3, None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, count = time.time(), 0.0, 0
        for v, a, y in train_loader:
            v, a, y = v.to(device), a.to(device), y.to(device)
            loss = masked_loss(bce, model(v, a), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
        last_loss = total / count
        first_loss = first_loss if first_loss is not None else last_loss

        if args.sanity:
            if epoch % 10 == 0 or epoch == 1:
                print(f"epoch {epoch:3d}  train loss {last_loss:.4f}")
            continue

        probs, y_val = predict(model, val_loader, device)
        ths = best_thresholds(probs, y_val)
        res = score(probs, y_val, ths)
        macro = float(np.mean([r["f1"] for r in res]))
        flag = ""
        if macro > best_f1:
            best_f1, best_ths, flag = macro, ths, "  <- best"
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {last_loss:.4f}  val macro-F1 {macro:.3f}  "
              f"({time.time() - t0:.0f}s){flag}")

    if args.sanity:
        ok = last_loss < 0.6 * first_loss
        print(f"\nloss went {first_loss:.3f} -> {last_loss:.3f}.  "
              + ("PASS: the model can learn. Now run the full training."
                 if ok else "FAIL: the loss did not drop clearly. Something is broken; do not train on everything yet."))
        return

    model.load_state_dict(best_state)
    probs, y_val = predict(model, val_loader, device)
    res = score(probs, y_val, best_ths)
    base = majority_baseline(tr_y.numpy(), y_val)
    print("\n=========== BEST MODEL vs MAJORITY-CLASS BASELINE (validation) ===========")
    for j, name in enumerate(TARGETS):
        r, b = res[j], base[j]
        print(f"{name:16s} thr={best_ths[j]:.2f}  acc {r['acc']:.3f}  prec {r['prec']:.3f}  rec {r['rec']:.3f}  "
              f"F1 {r['f1']:.3f}   | majority: acc {b['acc']:.3f}  F1 {b['f1']:.3f}")
    print(
        f"macro F1: model {np.mean([r['f1'] for r in res]):.3f}   majority {np.mean([b['f1'] for b in base]):.3f}")
    print("(thresholds were tuned on the same validation set, so these numbers are slightly optimistic)")

    torch.save({"state_dict": best_state, "use_audio": use_audio, "thresholds": best_ths, "pos_weight": pos_w},
               os.path.join(args.out_dir, "best_fusion.pt"))
    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(
            dict(zip(["mental_demand", "effort", "temporal_demand"], best_ths)), f, indent=2)
    print("saved", os.path.join(args.out_dir,
          "best_fusion.pt"), "and thresholds.json")

    if args.export:
        model.eval().cpu()
        vshape = (1,) + tuple(tr_v.shape[1:])
        ashape = (1,) + (tuple(tr_a.shape[1:]) if use_audio else (1, 80, 200))
        torch.jit.trace(model, (torch.zeros(vshape),
                        torch.zeros(ashape))).save(args.export)
        print("exported TorchScript model to", args.export)


if __name__ == "__main__":
    main()
