import argparse
import glob
import json
import os
import numpy as np
import torch


def load_val_labels(val_dir: str) -> np.ndarray:
    shard_paths = sorted(glob.glob(os.path.join(val_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {val_dir}")
    ys = [torch.load(p, map_location="cpu")["labels"] for p in shard_paths]
    return torch.cat(ys, dim=0).numpy()


def binary_f1_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    f1s = []
    for t in range(y_true.shape[1]):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else 0.0)
    return f1s


def evaluate_split(name: str, val_dir: str, results_dir: str) -> dict:
    y_true = load_val_labels(val_dir)
    m1 = np.load(os.path.join(results_dir, "preds_model1.npy"))
    m2 = np.load(os.path.join(results_dir, "preds_model2.npy"))

    m1_acc_t = (m1 == y_true).mean(axis=0).tolist()
    m1_f1_t = binary_f1_per_target(y_true, m1)

    m2_acc_t = (m2 == y_true).mean(axis=0).tolist()
    m2_f1_t = binary_f1_per_target(y_true, m2)

    agree_t = (m1 == m2).mean(axis=0).tolist()

    return {
        "split": name,
        "num_val_samples": int(y_true.shape[0]),
        "val_pos_rate_per_target": y_true.mean(axis=0).tolist(),
        "model1_video_mlp": {
            "val_acc_per_target": m1_acc_t,
            "val_f1_per_target": m1_f1_t,
            "mean_val_acc": float(np.mean(m1_acc_t)),
            "mean_val_f1": float(np.mean(m1_f1_t)),
            "pred_pos_rate_per_target": m1.mean(axis=0).tolist(),
        },
        "model2_rf": {
            "val_acc_per_target": m2_acc_t,
            "val_f1_per_target": m2_f1_t,
            "mean_val_acc": float(np.mean(m2_acc_t)),
            "mean_val_f1": float(np.mean(m2_f1_t)),
            "pred_pos_rate_per_target": m2.mean(axis=0).tolist(),
        },
        "agreement_per_target": agree_t,
        "mean_agreement": float(np.mean(agree_t)),
    }


def to_markdown(records: list[dict]) -> str:
    lines = ["# Baseline Model Results Summary\n"]
    for r in records:
        m1 = r["model1_video_mlp"]
        m2 = r["model2_rf"]
        lines.append(f"## {r['split']} (Val Samples: {r['num_val_samples']})\n")
        lines.append(
            f"- Ground-truth val positive rate per target: "
            f"{[round(x, 4) for x in r['val_pos_rate_per_target']]}\n"
        )
        lines.append(
            f"- Model 1 (Video MLP) Mean Val Acc: `{m1['mean_val_acc']:.4f}` | "
            f"Mean Val F1: `{m1['mean_val_f1']:.4f}`\n"
        )
        lines.append(
            f"- Model 2 (Random Forest) Mean Val Acc: `{m2['mean_val_acc']:.4f}` | "
            f"Mean Val F1: `{m2['mean_val_f1']:.4f}`\n"
        )
        lines.append(
            f"- Mean Prediction Agreement (M1 vs M2): `{r['mean_agreement']:.4f}`\n"
        )
        lines.append("| Target | M1 Val Acc | M1 Val F1 | M2 Val Acc | M2 Val F1 | M1-M2 Agreement |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for idx in range(len(m1["val_acc_per_target"])):
            lines.append(
                f"| Target {idx} | {m1['val_acc_per_target'][idx]:.4f} | "
                f"{m1['val_f1_per_target'][idx]:.4f} | {m2['val_acc_per_target'][idx]:.4f} | "
                f"{m2['val_f1_per_target'][idx]:.4f} | {r['agreement_per_target'][idx]:.4f} |"
            )
        lines.append("\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    splits = [
        ("shards_test", r"D:\dev\data\shards_test\val", "results_test"),
        ("shards_scaled", r"D:\dev\data\shards_scaled\val", "results_scaled"),
    ]

    records = []
    for name, val_dir, res_dir in splits:
        if os.path.isdir(val_dir) and os.path.isfile(os.path.join(res_dir, "preds_model1.npy")) and os.path.isfile(os.path.join(res_dir, "preds_model2.npy")):
            records.append(evaluate_split(name, val_dir, res_dir))

    json_path = os.path.join(args.out_dir, "baseline_metrics.json")
    md_path = os.path.join(args.out_dir, "baseline_summary.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(records))

    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
