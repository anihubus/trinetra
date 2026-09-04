"""
Module 0 - preprocess the assembled dataset IN PLACE.

Runs the same tile-level filter (Lee despeckle + CLAHE, native resolution) over
every image in ml/data/splits/{train,val,test}/images. Labels are untouched.

The identical filter runs at serve time (ml/inference/preprocess.despeckle_clahe),
so the model never sees a distribution it wasn't trained on.

    python ml/scripts/preprocess_splits.py            # all splits, in place
    python ml/scripts/preprocess_splits.py --dry-run  # count only
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML / "scripts"))
from preprocess_sonar import despeckle_clahe

SPLITS = ML / "data" / "splits"
EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", type=Path, default=SPLITS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = done = failed = 0
    for split in ("train", "val", "test"):
        img_dir = args.splits_dir / split / "images"
        if not img_dir.exists():
            continue
        imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in EXT]
        total += len(imgs)
        print(f"[{split}] {len(imgs)} images"
              + (" (dry run)" if args.dry_run else " - preprocessing..."))
        if args.dry_run:
            continue
        t0 = time.time()
        for i, p in enumerate(imgs, 1):
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                failed += 1
                continue
            out = despeckle_clahe(im)                       # -> single-channel uint8
            cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
            done += 1
            if i % 1000 == 0 or i == len(imgs):
                print(f"  {i}/{len(imgs)}  ({(time.time()-t0):.0f}s)")

    print(f"\ndone: {done} preprocessed, {failed} unreadable, {total} total")
    if not args.dry_run:
        print("serve path matches via ml/inference/preprocess.despeckle_clahe")


if __name__ == "__main__":
    main()
