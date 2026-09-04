"""
DRISHTI SSS - Tile AI4Shipwrecks transect strips into YOLO detection tiles.

AI4Shipwrecks ships full sonar transect strips (~5600 x 1728 px) with a binary
PNG segmentation mask per strip. Feeding a whole strip to a 640-input detector
squashes it ~9x and turns every wreck into one giant box. This script instead:

  1. Slides a 640x640 window over each strip (non-overlapping).
  2. Runs connectedComponents on the mask crop -> one tight bbox per wreck blob.
  3. Writes YOLO detection labels (class 2 = shipwreck), normalized to the tile.
  4. Splits BY SOURCE STRIP (never by tile) so no transect leaks across splits.

Output: ml/data/raw/ai4shipwrecks_tiled/{train,val,test}/{images,labels}
Run once; build_dataset.py consumes the result.
"""

import random
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RAW = SCRIPT_DIR.parent / "data" / "raw"
SRC = RAW / "ai4shipwrecks"
OUT = RAW / "ai4shipwrecks_tiled"

TILE = 640
STRIDE = 320                 # 50% overlap -> boundary wrecks land whole in >=1 tile
SHIPWRECK_CLASS = 2
MIN_BLOB_AREA = 120          # px in the tile; smaller = mask speckle, drop it
MIN_BOX_FRAC = 0.010         # drop boxes narrower/shorter than this fraction of the tile
MASK_CLOSE_PX = 9            # morphological close: merge a broken hull / debris into one blob
VAL_FRAC_OF_TRAIN_STRIPS = 0.15
SEED = 42


def strip_pairs(split: str):
    img_dir, lbl_dir = SRC / split / "images", SRC / split / "labels"
    pairs = []
    for img_p in sorted(img_dir.glob("*.png")):
        mask_p = lbl_dir / f"{img_p.stem}.png"
        if mask_p.exists():
            pairs.append((img_p, mask_p))
    return pairs


def tile_strip(img_p: Path, mask_p: Path, dst_img: Path, dst_lbl: Path) -> tuple[int, int]:
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return 0, 0
    if mask.shape != img.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    binm = (mask > 0).astype(np.uint8)
    if MASK_CLOSE_PX > 1:                      # merge hull fragments / debris fields
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_CLOSE_PX, MASK_CLOSE_PX))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    h, w = img.shape
    tiles_written = boxes_written = 0
    xs = list(range(0, max(1, w - TILE + 1), STRIDE))
    ys = list(range(0, max(1, h - TILE + 1), STRIDE))
    if w > TILE and xs[-1] != w - TILE:
        xs.append(w - TILE)
    if h > TILE and ys[-1] != h - TILE:
        ys.append(h - TILE)

    for y0 in ys:
        for x0 in xs:
            tile_img = img[y0:y0 + TILE, x0:x0 + TILE]
            tile_msk = binm[y0:y0 + TILE, x0:x0 + TILE]
            th, tw = tile_img.shape
            if th < TILE or tw < TILE:                       # pad edge tiles
                tile_img = cv2.copyMakeBorder(tile_img, 0, TILE - th, 0, TILE - tw, cv2.BORDER_CONSTANT, value=0)
                tile_msk = cv2.copyMakeBorder(tile_msk, 0, TILE - th, 0, TILE - tw, cv2.BORDER_CONSTANT, value=0)

            n, _, stats, _ = cv2.connectedComponentsWithStats(tile_msk, connectivity=8)
            lines = []
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < MIN_BLOB_AREA:
                    continue
                if bw / TILE < MIN_BOX_FRAC or bh / TILE < MIN_BOX_FRAC:
                    continue
                xc, yc = (x + bw / 2) / TILE, (y + bh / 2) / TILE
                lines.append(f"{SHIPWRECK_CLASS} {xc:.6f} {yc:.6f} {bw / TILE:.6f} {bh / TILE:.6f}")

            if not lines:                                    # keep only tiles that contain a wreck
                continue
            name = f"{img_p.stem}_y{y0}_x{x0}"
            cv2.imwrite(str(dst_img / f"{name}.jpg"), tile_img)
            (dst_lbl / f"{name}.txt").write_text("\n".join(lines) + "\n")
            tiles_written += 1
            boxes_written += len(lines)
    return tiles_written, boxes_written


def main():
    for s in ("train", "val", "test"):
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

    train_strips = strip_pairs("train")
    test_strips = strip_pairs("test")
    random.seed(SEED)
    random.shuffle(train_strips)
    n_val = int(len(train_strips) * VAL_FRAC_OF_TRAIN_STRIPS)
    assignment = [("val", p) for p in train_strips[:n_val]] \
        + [("train", p) for p in train_strips[n_val:]] \
        + [("test", p) for p in test_strips]

    logger.info(f"strips: {len(train_strips) - n_val} train / {n_val} val / {len(test_strips)} test")
    totals = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
    for split, (img_p, mask_p) in assignment:
        t, b = tile_strip(img_p, mask_p, OUT / split / "images", OUT / split / "labels")
        totals[split][0] += t
        totals[split][1] += b
    for s, (t, b) in totals.items():
        logger.info(f"  {s}: {t} wreck tiles, {b} boxes -> {OUT / s}")


if __name__ == "__main__":
    main()
