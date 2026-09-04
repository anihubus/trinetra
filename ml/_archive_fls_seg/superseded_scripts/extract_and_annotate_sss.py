import json
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ---- Paths & Config ----
ROOT = Path(r"D:\Sonar-Drishti\ml")
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"

CLASSES = {"crab_pot": 0, "shipwreck": 2, "mine_cylinder": 4}
SMART_CAP = 1000

# Clear old splits
if SPLITS.exists():
    shutil.rmtree(SPLITS)
for split in ["train", "val", "test"]:
    (SPLITS / split / "images").mkdir(parents=True, exist_ok=True)
    (SPLITS / split / "labels").mkdir(parents=True, exist_ok=True)

def save_sample(img_path, lines, split, prefix):
    dst_img = SPLITS / split / "images" / f"{prefix}_{img_path.name}"
    dst_lbl = SPLITS / split / "labels" / f"{prefix}_{img_path.stem}.txt"
    shutil.copy2(img_path, dst_img)
    dst_lbl.write_text("\n".join(lines) + "\n")

# ---- 1. Crab Pots (Class 0) ----
print("Extracting Crab Pots...")
cp_src = RAW / "sss_crab_pot"
for s_src, s_dst in [("train", "train"), ("validation", "val"), ("test", "test")]:
    img_dir, ann_dir = cp_src / s_src / "images", cp_src / s_src / "annotations_raw"
    if not img_dir.exists(): continue
    
    files = list(img_dir.glob("*.jpg"))
    if s_dst == "train":
        random.seed(42)
        random.shuffle(files)
        files = files[:SMART_CAP]
        
    for img_p in tqdm(files, desc=f"CrabPot {s_dst}"):
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]
        
        ann_p = ann_dir / f"{img_p.stem}.json"
        lines = []
        if ann_p.exists():
            ann = json.loads(ann_p.read_text())
            for bbox, cat in zip(ann["objects"]["bbox"], ann["objects"]["category"]):
                if cat == "Crab-Pot":
                    bx, by, bw, bh = bbox
                    xc, yc = (bx + bw/2)/w, (by + bh/2)/h
                    lines.append(f"{CLASSES['crab_pot']} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")
        if lines: save_sample(img_p, lines, s_dst, "cp")

# ---- 2. Shipwrecks (Class 2) ----
print("\nExtracting Shipwrecks...")
sw_src = RAW / "ai4shipwrecks"
for s_src in ["train", "test"]:
    img_dir, lbl_dir = sw_src / s_src / "images", sw_src / s_src / "labels"
    if not img_dir.exists(): continue
    
    files = list(img_dir.glob("*.png"))
    splits = [("train", files[:-15]), ("val", files[-15:])] if s_src == "train" else [("test", files)]
    
    for s_dst, split_files in splits:
        for img_p in tqdm(split_files, desc=f"Shipwreck {s_dst}"):
            img = cv2.imread(str(img_p))
            mask_p = lbl_dir / f"{img_p.stem}.png"
            if img is None or not mask_p.exists(): continue
            
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            lines = []
            
            # Convert mask to bounding box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(c)
                xc, yc = (x + bw/2)/w, (y + bh/2)/h
                lines.append(f"{CLASSES['shipwreck']} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")
            if lines: save_sample(img_p, lines, s_dst, "wreck")

# ---- 3. Mine Cylinders (Class 4) ----
print("\nExtracting Mine Cylinders...")
mine_src = RAW / "kaggle_sonar_mine"
for s in ["train", "val", "test"]:
    img_dir = mine_src / s / s / "images" if (mine_src / s / s).exists() else mine_src / s / "images"
    lbl_dir = mine_src / s / s / "labels" if (mine_src / s / s).exists() else mine_src / s / "labels"
    if not img_dir.exists(): continue
    
    for img_p in tqdm(list(img_dir.glob("*.jpg")), desc=f"Mine {s}"):
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        lines = []
        if lbl_p.exists():
            for line in lbl_p.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "0":  # Keep only MILCO (0)
                    lines.append(f"{CLASSES['mine_cylinder']} {' '.join(parts[1:5])}")
        if lines: save_sample(img_p, lines, s, "mine")

print("\nExtraction Complete! Classes 0, 2, and 4 are safely in ml/data/splits/.")
