import shutil
import random
from pathlib import Path
from tqdm import tqdm

ROOT = Path(r"D:\Sonar-Drishti\ml")
SRC = ROOT / "data" / "raw" / "subpipe_tiled"
DST = ROOT / "data" / "splits"
SMART_CAP = 1000

print("Merging Submarine Pipelines (Class 1) into final splits...")
random.seed(42)

for split in ["train", "val", "test"]:
    img_files = list((SRC / split / "images").glob("*.jpg"))
    
    # Filter to only keep tiles that have actual bounding boxes
    non_empty = [f for f in img_files if (SRC / split / "labels" / f"{f.stem}.txt").read_text().strip()]
    
    # Enforce cap on train split
    if split == "train" and len(non_empty) > SMART_CAP:
        random.shuffle(non_empty)
        non_empty = non_empty[:SMART_CAP]
        
    for img_p in tqdm(non_empty, desc=f"Merging {split}"):
        lbl_p = SRC / split / "labels" / f"{img_p.stem}.txt"
        shutil.copy2(img_p, DST / split / "images" / f"pipe_{img_p.name}")
        shutil.copy2(lbl_p, DST / split / "labels" / f"pipe_{lbl_p.name}")

print("Merge complete! Class 1 (submarine_pipeline) is fully integrated.")
