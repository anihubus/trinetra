"""
Does the geometric shadow check actually separate true objects from false alarms?

Runs the detector on splits/test, matches to ground truth (so TP/FP are known),
then applies ShadowVerifier.compute_penalty with plausible assigned geometry
(our test tiles carry no real altitude/range) and reports:

  - mean shadow penalty for TP vs FP
  - how many FPs it demotes below the keep threshold vs how many TPs it costs

A useful check demotes clearly more FPs than TPs. If TP and FP penalties look
alike, the check needs real ping geometry (Module 3) or better shadow extraction.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ML = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ML.parent))
from ml.inference.shadow_verification import ShadowVerifier

SPLITS = ML / "data" / "splits"
NAMES = {0: "crab_pot", 1: "submarine_pipeline", 2: "shipwreck", 3: "ghost_net", 4: "mine_cylinder"}
IOU_MATCH = 0.5
KEEP_THR = 0.35              # raw-confidence keep threshold we test demotion against


def load_gt(p):
    out = []
    if not p.exists():
        return out
    for ln in p.read_text().strip().splitlines():
        q = ln.split()
        if len(q) < 5:
            continue
        c = int(q[0]); xc, yc, bw, bh = map(float, q[1:5])
        out.append((c, xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ML / "models" / "checkpoints" / "best_detector.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--altitude", type=float, default=5.0)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    verifier = ShadowVerifier()

    img_dir, lbl_dir = SPLITS / args.split / "images", SPLITS / args.split / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    logger.info(f"{len(imgs)} {args.split} images  (assigned geometry: alt {args.altitude} m, range {args.max_range} m)")

    tp_pen, fp_pen = [], []
    demoted_fp = demoted_tp = kept_fp = kept_tp = 0

    for ip in imgs:
        gts = load_gt(lbl_dir / f"{ip.stem}.txt")
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        H, W = gray.shape[:2]
        meta = {"altitude": args.altitude, "max_range": args.max_range,
                "image_height": H, "image_width": W}
        res = model.predict(str(ip), conf=KEEP_THR, iou=0.6, imgsz=args.imgsz, verbose=False)
        if not res or res[0].boxes is None:
            continue
        b = res[0].boxes
        used = set()
        for i in np.argsort(-b.conf.cpu().numpy()):
            conf = float(b.conf[i]); pc = int(b.cls[i])
            x1, y1, x2, y2 = b.xyxy[i].cpu().numpy()
            pn = (x1 / W, y1 / H, x2 / W, y2 / H)
            best, bj = 0.0, -1
            for j, (gc, *gb) in enumerate(gts):
                if gc == pc and j not in used:
                    v = iou(pn, gb)
                    if v > best:
                        best, bj = v, j
            is_tp = best >= IOU_MATCH
            if is_tp:
                used.add(bj)

            det = {"bbox": [float(x1), float(y1), float(x2), float(y2)],
                   "confidence_raw": conf, "class_label": NAMES.get(pc, "?")}
            pen = verifier.compute_penalty(det, meta, gray)
            (tp_pen if is_tp else fp_pen).append(pen)
            after = conf - pen
            if is_tp:
                demoted_tp += after < KEEP_THR
                kept_tp += after >= KEEP_THR
            else:
                demoted_fp += after < KEEP_THR
                kept_fp += after >= KEEP_THR

    def stat(x):
        x = np.array(x) if x else np.array([0.0])
        return f"mean {x.mean():.3f}  >0: {(x > 0).mean():.2f}  n={len(x)}"

    print("\n" + "=" * 64)
    print("  SHADOW-CHECK PENALTY  (geometry assigned, not from real ping headers)")
    print("=" * 64)
    print(f"  true positives : {stat(tp_pen)}")
    print(f"  false positives: {stat(fp_pen)}")
    print("\n  effect at keep-threshold", KEEP_THR)
    print(f"    false positives demoted below threshold : {demoted_fp:4d}  (kept {kept_fp})")
    print(f"    true  positives demoted below threshold : {demoted_tp:4d}  (kept {kept_tp})")
    net = demoted_fp - demoted_tp
    print(f"\n  net: removes {demoted_fp} FP for {demoted_tp} lost TP  "
          f"({'useful' if net > 0 and demoted_tp <= demoted_fp * 0.3 else 'marginal / needs real geometry'})")
    print("=" * 64)


if __name__ == "__main__":
    main()
