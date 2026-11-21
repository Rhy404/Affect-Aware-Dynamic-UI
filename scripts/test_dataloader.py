import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("src/preprocessing"))

from torch_dataset import AVCAffeDataset

ROOT = r"C:\dev\data\avcaffe"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading datasets...")

train_ds = AVCAffeDataset(
    root=ROOT,
    split="train",
    mode="random",
    no_audio="mask"
)

val_ds = AVCAffeDataset(
    root=ROOT,
    split="val",
    mode="random",
    no_audio="mask"
)

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\n========== DATASET INFO ==========")
print("Device:", DEVICE)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Training clips:", len(train_ds))
print("Validation clips:", len(val_ds))

print("\nLoading one training batch...")
batch = next(iter(train_loader))

print("\n========== BATCH SHAPES ==========")
print("Video:", tuple(batch["video"].shape))
print("Audio:", tuple(batch["audio"].shape))
print("Labels:", tuple(batch["labels"].shape))
print("Has audio:", tuple(batch["has_audio"].shape))

video = batch["video"].to(DEVICE, non_blocking=True)
audio = batch["audio"].to(DEVICE, non_blocking=True)
labels = batch["labels"].to(DEVICE, non_blocking=True)

print("\n========== VALIDATION ==========")
print("Video device:", video.device)
print("Audio device:", audio.device)
print("Labels device:", labels.device)
print("Video finite:", torch.isfinite(video).all().item())
print("Audio finite:", torch.isfinite(audio).all().item())
print("Labels:", labels.tolist())
print("Audio availability:", batch["has_audio"].tolist())

print("\nDataLoader test PASSED!")
