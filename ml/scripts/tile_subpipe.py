import cv2
import random
from pathlib import Path
from tqdm import tqdm

# ---- Configurations ----
HF_SRC = Path(r"D:\Sonar-Drishti\ml\data\raw\SubPipeMini2\DATA\SSS_HF_images")
LF_SRC = Path(r"D:\Sonar-Drishti\ml\data\raw\SubPipeMini2\DATA\SSS_LF_images")
OUT_DIR = Path(r"D:\Sonar-Drishti\ml\data\raw\subpipe_tiled")

TILE_W, TILE_H = 640, 500  # SubPipe strips are typically ~500px high
STRIDE = 500               # Non-overlapping
TARGET_CLASS = 1           # DRISHTI taxonomy: 1 = submarine_pipeline
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# ---- 1. Merge HF & LF Sources ----
all_files = []
for src in [HF_SRC, LF_SRC]:
    img_dir = src / "Image"
    lbl_dir = src / "YOLO_Annotation"
    if img_dir.exists():
        for img_path in img_dir.glob("*.pbm"):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            all_files.append((img_path, lbl_path))

print(f"Total HF/LF source images found: {len(all_files)}")

# ---- 2. File-Level Split (Prevents Leakage) ----
random.seed(42)
random.shuffle(all_files)

n = len(all_files)
n_train = int(n * SPLIT_RATIOS["train"])
n_val = int(n * SPLIT_RATIOS["val"])
splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
file_to_split = dict(zip([f[0] for f in all_files], splits))

for split in ["train", "val", "test"]:
    (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# ---- 3. Tile & Adjust Boxes ----
tile_count = 0
valid_box_count = 0

for img_path, lbl_path in tqdm(all_files, desc="Tiling SubPipe"):
    split = file_to_split[img_path]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    
    h, w = img.shape
    
    # Load original boxes (normalized xc yc w h)
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                xc, yc, bw, bh = map(float, parts[1:5])
                # Convert to absolute pixel coordinates
                boxes.append((xc * w, yc * h, bw * w, bh * h))

    # Slide horizontally across the strip
    for x_start in range(0, w - TILE_W + 1, STRIDE):
        tile = img[0:TILE_H, x_start:x_start + TILE_W]
        tile_boxes = []
        
        for xc_abs, yc_abs, bw_abs, bh_abs in boxes:
            x1, y1 = xc_abs - bw_abs/2, yc_abs - bh_abs/2
            x2, y2 = xc_abs + bw_abs/2, yc_abs + bh_abs/2
            
            # Check overlap with current tile
            ix1, iy1 = max(x1, x_start), max(y1, 0)
            ix2, iy2 = min(x2, x_start + TILE_W), min(y2, TILE_H)
            
            if ix2 <= ix1 or iy2 <= iy1:
                continue # No overlap
                
            box_area = (x2 - x1) * (y2 - y1)
            overlap_area = (ix2 - ix1) * (iy2 - iy1)
            
            # Keep if >50% of the box is inside this tile
            if overlap_area / box_area >= 0.5:
                # Convert back to normalized YOLO coordinates RELATIVE to the tile
                nx1, nx2 = ix1 - x_start, ix2 - x_start
                nxc = ((nx1 + nx2) / 2) / TILE_W
                nyc = ((iy1 + iy2) / 2) / TILE_H
                nbw = (nx2 - nx1) / TILE_W
                nbh = (iy2 - iy1) / TILE_H
                
                tile_boxes.append(f"{TARGET_CLASS} {nxc:.6f} {nyc:.6f} {nbw:.6f} {nbh:.6f}")
                valid_box_count += 1
        
        # Save tile image and label (even if label is empty, for background training)
        tile_name = f"{img_path.stem}_x{x_start}"
        cv2.imwrite(str(OUT_DIR / split / "images" / f"{tile_name}.jpg"), tile)
        (OUT_DIR / split / "labels" / f"{tile_name}.txt").write_text("\n".join(tile_boxes))
        tile_count += 1

print(f"\nDone. {tile_count} total tiles written.")
print(f"Successfully calculated {valid_box_count} normalized bounding boxes inside tiles.")