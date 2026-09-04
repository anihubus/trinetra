"""
DRISHTI SSS - Master dataset assembler (pure side-scan sonar, box detection).

Replaces the earlier extract_and_annotate_sss.py + merge_pipes.py. Consolidates
every real SSS source into ml/data/splits/{train,val,test} with a single 5-class
taxonomy, per-class caps so no source dominates, real hard-negative background
tiles, and leakage-safe splits (each source's own train/val/test boundary is
preserved - tiles/crops never cross it).

Taxonomy (matches ml/configs/drishti.yaml):
  0 crab_pot           <- sss_crab_pot (HF, COCO json)
  1 submarine_pipeline <- subpipe_tiled (SubPipeMini2 HF+LF, pre-tiled)
  2 shipwreck          <- ai4shipwrecks_tiled + roboflow_sss (Ship+Plane) + KLSG crops
  3 ghost_net          <- (none here) run build_synthetic_data.py afterwards
  4 mine_cylinder      <- kaggle_sonar_mine (source class 0 / MILCO only)
  background           <- empty subpipe_tiled tiles (no label lines)

Usage:  python ml/scripts/build_dataset.py
        python ml/scripts/build_dataset.py --dry-run      # audit only, no copy
"""

import argparse
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ML = SCRIPT_DIR.parent
RAW = ML / "data" / "raw"
SPLITS = ML / "data" / "splits"
SEED = 42

# per (class, split) instance/image caps  -  None = take everything
# caps are on IMAGES per (class, split); instance counts run a little higher.
# v3 (preprocessed restart): crab_pot kept in training as a hard-negative-ish
# class (model learns "small blob != shipwreck") but FILTERED downstream ->
# the product is nc=4. shipwreck cap raised: the overlap re-tile now yields
# ~890 real wreck tiles instead of 152. mine_cylinder now includes NonMILCO.
CAPS = {
    "crab_pot":           {"train": 900, "val": 150, "test": 150},
    "submarine_pipeline": {"train": 1000, "val": 150, "test": 175},
    "shipwreck":          {"train": 900, "val": 250, "test": 300},
    "mine_cylinder":      {"train": None, "val": None, "test": None},
}
BACKGROUND = {"train": 500, "val": 70, "test": 70}
# KLSG weak full-frame boxes hurt more than they helped (shipwreck FPs + loose
# boxes in the first run). Set > 0 to re-enable; 0 = use only real tight boxes.
KLSG_TRAIN_CAP = 0

CLASS_ID = {"crab_pot": 0, "submarine_pipeline": 1, "shipwreck": 2, "ghost_net": 3, "mine_cylinder": 4}
_rng = random.Random(SEED)


def reset_splits():
    if SPLITS.exists():
        shutil.rmtree(SPLITS)
    for s in ("train", "val", "test"):
        (SPLITS / s / "images").mkdir(parents=True, exist_ok=True)
        (SPLITS / s / "labels").mkdir(parents=True, exist_ok=True)


def write(split: str, prefix: str, img_src: Path, label_lines: list[str], dry: bool):
    """Copy one image + write its label file (empty list -> background negative)."""
    if dry:
        return
    dst_img = SPLITS / split / "images" / f"{prefix}_{img_src.name}"
    dst_lbl = SPLITS / split / "labels" / f"{prefix}_{img_src.stem}.txt"
    shutil.copy2(img_src, dst_img)
    dst_lbl.write_text(("\n".join(label_lines) + "\n") if label_lines else "")


def capped(items, cap):
    if cap is None or len(items) <= cap:
        return items
    items = list(items)
    _rng.shuffle(items)
    return items[:cap]


# --------------------------------------------------------------------------- #
def ingest_crab_pot(dry):
    src = RAW / "sss_crab_pot"
    stats = Counter()
    for s_src, s_dst in (("train", "train"), ("validation", "val"), ("test", "test")):
        img_dir, ann_dir = src / s_src / "images", src / s_src / "annotations_raw"
        if not img_dir.exists():
            continue
        rows = []
        for img_p in sorted(img_dir.glob("*.jpg")):
            ann_p = ann_dir / f"{img_p.stem}.json"
            if not ann_p.exists():
                continue
            ann = json.loads(ann_p.read_text())
            w, h = ann.get("width", 640), ann.get("height", 640)
            lines = []
            for bbox, cat in zip(ann["objects"]["bbox"], ann["objects"]["category"]):
                if cat != "Crab-Pot":
                    continue
                bx, by, bw, bh = bbox
                lines.append(f"0 {(bx + bw / 2) / w:.6f} {(by + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            if lines:
                rows.append((img_p, lines))
        for img_p, lines in capped(rows, CAPS["crab_pot"][s_dst]):
            write(s_dst, "cp", img_p, lines, dry)
            stats[s_dst] += len(lines)
    return stats


def ingest_pipeline(dry):
    src = RAW / "subpipe_tiled"
    stats = Counter()
    for s in ("train", "val", "test"):
        img_dir, lbl_dir = src / s / "images", src / s / "labels"
        rows = []
        for img_p in sorted(img_dir.glob("*.jpg")):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            txt = lbl_p.read_text().strip() if lbl_p.exists() else ""
            if not txt:
                continue
            lines = [f"1 {' '.join(ln.split()[1:5])}" for ln in txt.splitlines() if len(ln.split()) >= 5]
            if lines:
                rows.append((img_p, lines))
        for img_p, lines in capped(rows, CAPS["submarine_pipeline"][s]):
            write(s, "pipe", img_p, lines, dry)
            stats[s] += len(lines)
    return stats


def _remap_yolo_txt(lbl_p: Path, new_id: int, keep_src=None):
    lines = []
    for ln in lbl_p.read_text().strip().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        if keep_src is not None and p[0] not in keep_src:
            continue
        lines.append(f"{new_id} {' '.join(p[1:5])}")
    return lines


def ingest_shipwreck(dry):
    stats = Counter()
    per_split_rows = {"train": [], "val": [], "test": []}

    # a) AI4Shipwrecks tiled (already class 2, real tight boxes)
    a = RAW / "ai4shipwrecks_tiled"
    for s in ("train", "val", "test"):
        for img_p in sorted((a / s / "images").glob("*.jpg")):
            lbl_p = a / s / "labels" / f"{img_p.stem}.txt"
            lines = _remap_yolo_txt(lbl_p, 2) if lbl_p.exists() else []
            if lines:
                per_split_rows[s].append(("wreckA", img_p, lines))

    # b) Roboflow SSS: class 0 Plane + class 1 Ship  -> shipwreck
    r = RAW / "roboflow_sss"
    for s_src, s_dst in (("train", "train"), ("valid", "val"), ("test", "test")):
        for img_p in sorted((r / s_src / "images").glob("*")):
            if img_p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_p = r / s_src / "labels" / f"{img_p.stem}.txt"
            lines = _remap_yolo_txt(lbl_p, 2, keep_src={"0", "1"}) if lbl_p.exists() else []
            if lines:
                per_split_rows[s_dst].append(("wreckR", img_p, lines))

    # c) KLSG classification crops (no boxes) -> weak centred box, TRAIN ONLY
    klsg = RAW / "SeabedObjects-Ship-and-Airplane-dataset-master"
    klsg_rows = []
    for sub in ("ship-real-1", "ship-real-2", "ship-real-3", "plane-real"):
        d = klsg / sub
        if not d.exists():
            continue
        for img_p in sorted(d.glob("*")):
            if img_p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                klsg_rows.append(("wreckK", img_p, ["2 0.500000 0.500000 0.860000 0.860000"]))
    per_split_rows["train"] += capped(klsg_rows, KLSG_TRAIN_CAP)

    for s in ("train", "val", "test"):
        for prefix, img_p, lines in capped(per_split_rows[s], CAPS["shipwreck"][s]):
            write(s, prefix, img_p, lines, dry)
            stats[s] += len(lines)
    return stats


def ingest_mine(dry):
    src = RAW / "kaggle_sonar_mine"
    stats = Counter()
    for s in ("train", "val", "test"):
        img_dir, lbl_dir = src / s / "images", src / s / "labels"
        if not img_dir.exists():
            continue
        rows = []
        for img_p in sorted(img_dir.glob("*.jpg")):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            # PS asks for "cylinders", not "mines": keep MILCO (0) AND NonMILCO (1)
            # - both are cylindrical man-made seabed contacts.
            lines = _remap_yolo_txt(lbl_p, 4, keep_src={"0", "1"})
            if lines:
                rows.append((img_p, lines))
        for img_p, lines in capped(rows, CAPS["mine_cylinder"][s]):
            write(s, "mine", img_p, lines, dry)
            stats[s] += len(lines)
    return stats


def ingest_background(dry):
    """Empty SubPipe tiles = real seabed / rock-ripple clutter with no target."""
    src = RAW / "subpipe_tiled"
    stats = Counter()
    for s in ("train", "val", "test"):
        img_dir, lbl_dir = src / s / "images", src / s / "labels"
        empties = [p for p in sorted(img_dir.glob("*.jpg"))
                   if not (lbl_dir / f"{p.stem}.txt").exists()
                   or not (lbl_dir / f"{p.stem}.txt").read_text().strip()]
        for img_p in capped(empties, BACKGROUND[s]):
            write(s, "bg", img_p, [], dry)
            stats[s] += 1
    return stats


# --------------------------------------------------------------------------- #
def audit():
    logger.info("=" * 68)
    logger.info("  FINAL SPLIT AUDIT  (instances per class, per split)")
    logger.info("=" * 68)
    names = {v: k for k, v in CLASS_ID.items()}
    grand = {}
    for s in ("train", "val", "test"):
        c = Counter()
        n_img = n_bg = 0
        for lbl_p in (SPLITS / s / "labels").glob("*.txt"):
            n_img += 1
            txt = lbl_p.read_text().strip()
            if not txt:
                n_bg += 1
                continue
            for ln in txt.splitlines():
                c[int(ln.split()[0])] += 1
        grand[s] = (c, n_img, n_bg)
        logger.info(f"[{s}]  images={n_img}  background={n_bg}")
        for cid in sorted(names):
            logger.info(f"    {cid} {names[cid]:20s} {c.get(cid, 0):6d}")
    tot = sum(v[1] for v in grand.values())
    if tot:
        r = {s: round(100 * grand[s][1] / tot) for s in grand}
        logger.info(f"  split ratio  train/val/test = {r['train']}/{r['val']}/{r['test']}  (target ~80/10/10)")
    if grand["train"][0].get(3, 0) == 0:
        logger.warning("  ghost_net (3) still empty -> run: python ml/scripts/build_synthetic_data.py "
                       "--classes 3 --split train --count 900  (then val/test 100 each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        reset_splits()
    logger.info("Assembling DRISHTI SSS detection dataset...")
    for name, fn in (("crab_pot", ingest_crab_pot), ("submarine_pipeline", ingest_pipeline),
                     ("shipwreck", ingest_shipwreck), ("mine_cylinder", ingest_mine),
                     ("background", ingest_background)):
        st = fn(args.dry_run)
        logger.info(f"  {name:20s} {dict(st)}")
    if not args.dry_run:
        audit()
        logger.info("Next: generate ghost_net (synthetic), then train_yolo_seg.py --model yolov8s.pt")


if __name__ == "__main__":
    main()
