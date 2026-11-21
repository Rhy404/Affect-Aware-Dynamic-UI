import os
import json
import random
import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath("src/preprocessing"))
from torch_dataset import AVCAffeDataset

ROOT = r"C:\dev\data\avcaffe"
MODEL_PATH = "checkpoints/friend_multimodal_model.pt"
OUTPUT_PATH = "checkpoints/finetuned_multimodal_model.pt"
REPORT_PATH = "reports/finetuning_results.json"

TRAIN_LIMIT = 4000
VAL_LIMIT = 1000
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

TARGETS = ["Mental Demand", "Effort", "Temporal Demand"]

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("\nLoading original model...")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad]
if not trainable_params:
    raise RuntimeError("The loaded model has no trainable parameters.")

optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

print("Loading training dataset...")
train_full = AVCAffeDataset(ROOT, "train", mode="random", no_audio="mask")
val_full = AVCAffeDataset(ROOT, "val", mode="random", no_audio="mask")

train_count = min(TRAIN_LIMIT, len(train_full))
val_count = min(VAL_LIMIT, len(val_full))

train_indices = random.Random(42).sample(range(len(train_full)), train_count)
val_indices = random.Random(42).sample(range(len(val_full)), val_count)

train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print(f"Training clips: {len(train_dataset)}")
print(f"Validation clips: {len(val_dataset)}")
print(f"Maximum epochs: {EPOCHS}")

def calculate_metrics(all_labels, all_predictions):
    results = {}
    f1_values = []

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)

    for j, name in enumerate(TARGETS):
        mask = y_true[:, j] >= 0
        true = y_true[mask, j].astype(int)
        pred = y_pred[mask, j].astype(int)

        if len(true) == 0:
            results[name] = {"valid_samples": 0}
            continue

        tp = int(((true == 1) & (pred == 1)).sum())
        fp = int(((true == 0) & (pred == 1)).sum())
        fn = int(((true == 1) & (pred == 0)).sum())

        accuracy = float((true == pred).mean())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results[name] = {
            "valid_samples": len(true),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
        f1_values.append(f1)

    results["macro_f1"] = (
        round(float(np.mean(f1_values)), 4) if f1_values else None
    )
    return results

best_val_loss = float("inf")
best_state = None
history = []

for epoch in range(EPOCHS):
    train_full.set_epoch(epoch)
    model.train()

    train_loss_sum = 0.0
    train_batches = 0

    print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========")

    for batch_idx, batch in enumerate(train_loader):
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True).float()

        valid_mask = labels >= 0
        safe_labels = labels.clamp(min=0)

        optimizer.zero_grad(set_to_none=True)

        logits = model(video, audio)
        raw_loss = F.binary_cross_entropy_with_logits(
            logits, safe_labels, reduction="none"
        )
        loss = (raw_loss * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_batches += 1

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Training batch {batch_idx + 1}/{len(train_loader)} "
                f"| loss: {loss.item():.4f}"
            )

    train_loss = train_loss_sum / max(train_batches, 1)

    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            video = batch["video"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).float()

            valid_mask = labels >= 0
            safe_labels = labels.clamp(min=0)

            logits = model(video, audio)

            raw_loss = F.binary_cross_entropy_with_logits(
                logits, safe_labels, reduction="none"
            )
            loss = (
                (raw_loss * valid_mask.float()).sum()
                / valid_mask.sum().clamp(min=1)
            )

            val_loss_sum += loss.item()
            val_batches += 1

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(
                (logits >= 0).int().cpu().numpy()
            )

    val_loss = val_loss_sum / max(val_batches, 1)
    metrics = calculate_metrics(all_labels, all_predictions)

    print(f"\nTrain loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print("Validation macro F1:", metrics["macro_f1"])

    for name in TARGETS:
        print(name, ":", metrics.get(name))

    history.append({
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 6),
        "validation_loss": round(val_loss, 6),
        "validation_metrics": metrics
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        print("New best model.")

if best_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

model.load_state_dict(best_state)
model.eval()
torch.jit.save(model, OUTPUT_PATH)

report = {
    "training_clips": len(train_dataset),
    "validation_clips": len(val_dataset),
    "epochs_completed": len(history),
    "best_validation_loss": round(best_val_loss, 6),
    "best_checkpoint": OUTPUT_PATH,
    "history": history
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\n========== TRAINING COMPLETE ==========")
print("Best validation loss:", round(best_val_loss, 6))
print("Saved model:", OUTPUT_PATH)
print("Saved report:", REPORT_PATH)
