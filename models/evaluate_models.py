"""Compare Models 1-3 on the same validation windows. Reads preds_*.npy from --out."""
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

TARGETS = ["mental_demand", "effort", "temporal_demand"]
MODELS = [("Model 1 (Unimodal Baseline)", "preds_model1.npy"),
          ("Model 2 (Random Forest Baseline)", "preds_model2.npy"),
          ("Model 3 (Multimodal Late-Fusion)", "preds_model3.npy")]


def target_scores(y, p):
    """Precision, recall, F1 of the High class and macro-F1, skipping missing labels."""
    m = y >= 0
    pr, rc, f1, _ = precision_recall_fscore_support(
        y[m], p[m], labels=[0, 1], zero_division=0)
    return {"precision": float(pr[1]), "recall": float(rc[1]),
            "f1_high": float(f1[1]), "macro_f1": float(f1.mean())}


def model_scores(y, p):
    per = {t: target_scores(y[:, i], p[:, i])
           for i, t in enumerate(TARGETS[:y.shape[1]])}
    avg = {k: float(np.mean([s[k] for s in per.values()]))
           for k in next(iter(per.values()))}
    return {"per_target": per, "average": avg}


def majority_preds(y_train, n):
    """Always predict the most common training class, per target."""
    cols = []
    for i in range(y_train.shape[1]):
        col = y_train[y_train[:, i] >= 0, i]
        cols.append(np.full(n, np.bincount(col).argmax()))
    return np.stack(cols, axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    y_val = np.load(os.path.join(args.out, "y_val.npy"))
    y_train = np.load(os.path.join(args.out, "y_train.npy"))
    preds = {"Majority class": majority_preds(y_train, len(y_val))}
    for name, f in MODELS:
        path = os.path.join(args.out, f)
        if os.path.exists(path):
            p = np.load(path)
            assert p.shape == y_val.shape, "%s: shape %s vs labels %s" % (
                f, p.shape, y_val.shape)
            preds[name] = p
        else:
            print("skipping %s (no %s yet)" % (name, f))

    results = {n: model_scores(y_val, p) for n, p in preds.items()}
    with open(os.path.join(args.out, "model_comparison_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("%-34s %9s %7s %8s %9s" %
          ("Model", "Precision", "Recall", "F1(High)", "Macro-F1"))
    for n, r in results.items():
        a = r["average"]
        print("%-34s %9.3f %7.3f %8.3f %9.3f" % (
            n, a["precision"], a["recall"], a["f1_high"], a["macro_f1"]))

    names = list(results)
    w = 0.8 / len(TARGETS[:y_val.shape[1]])
    plt.figure(figsize=(9, 4))
    for i, t in enumerate(TARGETS[:y_val.shape[1]]):
        plt.bar(np.arange(len(names)) + i * w,
                [results[n]["per_target"][t]["f1_high"] for n in names], w, label=t)
    plt.xticks(np.arange(len(names)) + w, [n.split(" (")[0] for n in names])
    plt.ylabel("F1 (High)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "f1_comparison.png"), dpi=150)
    plt.close()

    nt = y_val.shape[1]
    fig, ax = plt.subplots(len(names), nt, figsize=(
        3 * nt, 3 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        for c in range(nt):
            m = y_val[:, c] >= 0
            ax[r][c].imshow(confusion_matrix(y_val[m, c], preds[n][m, c], labels=[0, 1]),
                            cmap="Blues")
            ax[r][c].set_title("%s\n%s" %
                               (n.split(" (")[0], TARGETS[c]), fontsize=8)
            ax[r][c].set_xticks([0, 1], ["Low", "High"])
            ax[r][c].set_yticks([0, 1], ["Low", "High"])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_matrices.png"), dpi=150)
    plt.close()
