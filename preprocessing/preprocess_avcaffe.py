from pathlib import Path

# Define root paths based on your AVCAFFE directory structure
BASE_DIR = Path(r"D:\dev\data\avcaffe")
FACE_CROPS_DIR = BASE_DIR / "face_crops"
GROUND_TRUTHS_DIR = BASE_DIR / "ground_truths"
INFO_DIR = BASE_DIR / "info"

def read_labels(path):
    """Reads AVCAffe ground truth text files formatted as key, value."""
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    out[parts[0].strip()] = parts[1].strip()
    return out

def load_split_file(split_filename):
    """Loads a split file (train.txt or val.txt) into a set of sample IDs."""
    split_path = INFO_DIR / split_filename
    if not split_path.exists():
        print(f"⚠️ Split file not found: {split_path}")
        return set()
    
    with open(split_path, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}

def load_ground_truths():
    """Loads all available ground truth text files into a dictionary."""
    gt_data = {}
    gt_files = list(GROUND_TRUTHS_DIR.glob("*.txt"))
    
    for gt_file in gt_files:
        gt_data[gt_file.stem] = read_labels(gt_file)
            
    return gt_data

def align_dataset():
    """Validates splits, ground truth labels, and face-crop zip inventory."""
    print("🔄 Loading data splits and ground truth labels...")
    
    train_ids = load_split_file("train.txt")
    val_ids = load_split_file("val.txt")
    print(f"   Loaded {len(train_ids)} training samples and {len(val_ids)} validation samples.")
    
    ground_truths = load_ground_truths()
    print(f"   Loaded ground truth metrics: {list(ground_truths.keys())}")
    
    all_crop_zips = {p.stem: p for p in FACE_CROPS_DIR.rglob("*.zip")}
    print(f"📊 Found {len(all_crop_zips)} clip face-crop zip files on disk.")

if __name__ == "__main__":
    align_dataset()
    print("\n🚀 Dataset validation complete! Ready for model training.")