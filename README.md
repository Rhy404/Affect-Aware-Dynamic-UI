# Affect-Aware Dynamic UI

A Teams-style video-meeting interface that estimates a participant's **cognitive load** from face video (and audio, when available) and adapts the UI to it: **Normal**, **High Temporal Demand**, or **Focus Mode** (for example, collapsing the chat panel).

The load estimator is a multimodal model trained on the [AVCAffe](https://github.com/) dataset. It predicts three binary labels per clip:

| Output | Meaning |
|---|---|
| Mental Demand | High / Low |
| Effort | High / Low |
| Temporal Demand | High / Low |

A clip is labelled **High** when its NASA-TLX style score is at or above the threshold used by the shard generator (`10.5`). Missing labels are stored as `-1` and ignored during training.

## Team

| Member | Role |
|---|---|
| Siddharth | Dataset and preprocessing |
| Rhythima | Multimodal fusion model |
| Bharat | Baseline models and model comparison |
| Sanya | Teams-style UI, FastAPI backend, integration, repo |

## How it works

```
webcam / mic ──► preprocessing ──► fusion model ──► 3 High/Low scores ──► UI mode
                                   (video + audio)       (per-label thresholds)
```

1. **Preprocessing** turns each clip into a face-crop video tensor and, when available, a log-mel audio tensor.
2. **The fusion model** (`models/fusion_model.py`) runs a 3D-conv branch over the video and a 2D-conv branch over the audio, concatenates the two feature vectors, and outputs 3 logits.
3. **Per-label thresholds** (tuned on validation data, saved in `thresholds.json`) turn probabilities into High/Low.
4. **The backend** serves predictions to the frontend, which switches UI mode. A manual toggle (Normal / High Temporal Demand / Focus Mode) works without the model and serves as the backup demo.

### Model

`AffectAwareFusionModel` has four conv blocks per branch (Conv, BatchNorm, ReLU, pooling) and a small fusion head.

| Variant | Inputs | Parameters |
|---|---|---|
| With audio | video `(B,3,T,H,W)` + audio `(B,1,F,T)` | 423,235 |
| Video only | video `(B,3,T,H,W)` | 309,219 |

It works for any frame size and for `T >= 8` frames, and it builds the video-only variant automatically when the shards contain no audio.

The original placeholder had 2,627 parameters, and its loss stayed at about 0.69 (random guessing). It only learned each label's most common answer.

## Repository layout

```
backend/            FastAPI server and the exported model (multimodal_model.pt)
frontend/           React + Tailwind Teams-style UI
models/
  fusion_model.py       the fusion model
  train_multimodal.py   training, evaluation, threshold tuning, export
  baseline_rf.py        Random Forest baseline
  baseline_video_mlp.py video MLP baseline
  export_baseline_results.py
preprocessing/      compact shard generator and AVCAffe preprocessing
src/preprocessing/  dataset readers, audio/video features, audits
scripts/            tensor export, fine-tuning and data download scripts
notebooks/          baseline training notebook
results_fusion/     saved checkpoint and tuned thresholds
reports/            result files
docs/               PREPROCESSING.md and other notes
```

> `preprocessing/` and `src/preprocessing/` contain overlapping code. `dev` keeps the `src/preprocessing/` versions for the three files that existed in both (`manifest.py`, `video_features.py`, `windows.py`).

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Use a GPU build of PyTorch for training. The scripts pick CUDA automatically when it is available.

## Data

Training reads cached tensor shards: `shard_*.pt` files in a `train/` folder and a `val/` folder. Each shard is a dict with:

| Key | Shape | Notes |
|---|---|---|
| `video` | `(N,C,T,H,W)` (or `(N,T,C,H,W)`) | uint8 or float |
| `labels` | `(N,3)` | 1 = High, 0 = Low, -1 = missing |
| `audio` | `(N,1,F,T)` or `(N,F,T)` | optional |

The compact shards from `preprocessing/make_compact_shards.py` are video only: 8 frames at 64×64. The shards are not stored in git. Ask Siddharth where they are.

## Training

Run everything from the repo root.

**1. Test the pipeline on random data** (no shards needed). The numbers are meaningless, because the data is random:
```powershell
python models/train_multimodal.py --dummy
python models/train_multimodal.py --dummy --dummy-audio
```

**2. Sanity check on real data.** It trains on 64 clips for 50 epochs. The loss must fall clearly below its starting value (the script prints PASS or FAIL):
```powershell
python models/train_multimodal.py --train-dir "PATH\TO\train" --val-dir "PATH\TO\val" --sanity
```

**3. Full training:**
```powershell
python models/train_multimodal.py --train-dir "PATH\TO\train" --val-dir "PATH\TO\val" --epochs 15
```

**4. Export for the backend**, only after the model clearly beats the majority-class baseline:
```powershell
python models/train_multimodal.py --train-dir "PATH\TO\train" --val-dir "PATH\TO\val" --epochs 15 --export backend/multimodal_model.pt
```

What training does:
- Class weights (`pos_weight = Low / High`) are computed from the **training** labels, so the rarer class is not ignored.
- Missing labels (`-1`) are masked out of the loss.
- The High/Low threshold is tuned **per label** on the validation set, and the best epoch is chosen by validation macro-F1.
- The script prints a table comparing the model against a majority-class baseline, and saves `checkpoints/best_fusion.pt` and `thresholds.json`.

If memory runs out, add `--max-clips 3000`.

## Results

| Model | Macro F1 (val) | Notes |
|---|---|---|
| Majority class | TODO | baseline to beat |
| Random Forest | TODO | `models/baseline_rf.py` |
| Video MLP | TODO | `models/baseline_video_mlp.py` |
| Fusion model | TODO | `models/train_multimodal.py` |

Even the dataset authors' 40-70M-parameter models reached only about 60-65% on these tasks, so the goal is to clearly beat the majority-class baseline, not to reach high accuracy. Thresholds are tuned on the validation set, so the reported numbers are slightly optimistic.

## Git workflow

```
feature/*  →  dev  →  main
```

- Work on a `feature/*` branch and open a pull request into `dev`.
- `main` holds only stable, demo-ready code.
- Do not commit `.pt` checkpoints or shards unless the team has agreed to. `backend/multimodal_model.pt` and `results_fusion/` are the agreed exceptions.

## Status and known issues

- The model has not yet been trained and evaluated on the full real dataset. Do not treat `backend/multimodal_model.pt` as a trained model until it beats the baseline.
- If the compact shards have no audio, the video-only variant is used.
- Input shapes must match between the backend and the model. Check them whenever the model changes.
- If trained-from-scratch results stay near the majority baseline, the fallback plan is frozen pretrained features (`r3d_18` or similar) plus classical classifiers.