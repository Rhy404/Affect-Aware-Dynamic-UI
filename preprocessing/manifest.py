import os
import csv
from pathlib import Path

BASE_DIR = Path(r"D:\dev\data\avcaffe")
FACE_CROPS_DIR = BASE_DIR / "face_crops"
INFO_DIR = BASE_DIR / "info"
OUTPUT_MANIFEST = BASE_DIR / "manifest.csv"

def generate_manifest():
    print("🔄 Generating manifest from face-crop zip archives and source videos...")
    
    # Load train and val splits
    train_set = {line.strip() for line in open(INFO_DIR / "train.txt", encoding="utf-8") if line.strip()}
    val_set = {line.strip() for line in open(INFO_DIR / "val.txt", encoding="utf-8") if line.strip()}
    
    # Map clip stem -> relative path string for face crop zips
    zip_files = {p.stem: str(p.relative_to(BASE_DIR)) for p in FACE_CROPS_DIR.rglob("*.zip")}
    
    # Map clip/video stem -> relative path string for source videos (checking common video extensions)
    video_files = {}
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for p in BASE_DIR.rglob(ext):
            # Avoid matching inside face_crops folders if they contain videos
            if "face_crops" not in p.parts:
                video_files[p.stem] = str(p.relative_to(BASE_DIR))
                
    print(f"📊 Found {len(zip_files)} zip files and {len(video_files)} video files on disk.")
    
    # Load ground truth files if available
    gt_data = {}
    gt_dir = BASE_DIR / "ground_truths"
    if gt_dir.exists():
        for gt_file in gt_dir.glob("*.txt"):
            metric = gt_file.stem
            gt_dict = {}
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().rsplit(',', 1)
                    if len(parts) == 2:
                        gt_dict[parts[0].strip()] = parts[1].strip()
            gt_data[metric] = gt_dict

    metrics = list(gt_data.keys())
    
    # All unique clip IDs from splits and zips
    all_clip_ids = sorted(list(train_set.union(val_set).union(zip_files.keys())))
    
    rows = []
    for clip_id in all_clip_ids:
        if clip_id in train_set:
            split = "train"
        elif clip_id in val_set:
            split = "val"
        else:
            split = "train"
            
        has_crops = clip_id in zip_files
        has_audio = clip_id in video_files
        crop_path = zip_files.get(clip_id, "")
        video_path = video_files.get(clip_id, "")
        
        row = {
            "clip_id": clip_id,
            "split": split,
            "has_crops": str(has_crops),
            "has_audio": str(has_audio),
            "crop_path": crop_path,
            "video_path": video_path
        }
        for m in metrics:
            row[m] = gt_data[m].get(clip_id, "")
            
        rows.append(row)
        
    fieldnames = ["clip_id", "split", "has_crops", "has_audio", "crop_path", "video_path"] + metrics
    with open(OUTPUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"✅ Successfully wrote manifest to {OUTPUT_MANIFEST} with {len(rows)} entries.")

if __name__ == "__main__":
    generate_manifest()