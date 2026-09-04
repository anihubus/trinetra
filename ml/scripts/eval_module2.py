"""
Module 2 measurement harness.

Runs the detector on the held-out test split, matches predictions to ground
truth, then reports what the confidence pipeline can and cannot do:

  1. Calibration honesty  - raw ECE vs calibrated ECE, reliability table.
  2. Score separation      - per class, do true positives score higher than
                             false positives? (i.e. can a threshold help?)
  3. Operating points      - precision / recall at a few calibrated cut-offs.
  4. Shadow-check feasibility - of TP vs FP detections, how many have a
                             detectable dark trailing region? (metadata-free
                             version of shadow_verification.py)

Usage:
  python ml/scripts/eval_module2.py --model ml/models/checkpoints/best_detector.pt
"""

import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ML = Path(__file__).resolve().parent.parent
SPLITS = ML / "data" / "splits"
NAMES = {0: "crab_pot", 1: "submarine_pipeline", 2: "shipwreck", 3: "ghost_net", 4: "mine_cylinder"}
IOU_MATCH = 0.5


def load_gt(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    for ln in label_path.read_text().strip().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        xc, yc, bw, bh = map(float, p[1:5])
        boxes.append((c, xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def has_trailing_shadow(gray, bbox_px, search=2.0, min_px=10):
    """Metadata-free: is there a dark region just below the detection?"""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    oh = max(y2 - y1, 1)
    ys, ye = y2, min(y2 + int(oh * search), h)
    xs, xe = max(x1 - 5, 0), min(x2 + 5, w)
    if ye <= ys or xe <= xs:
        return False, 0.0
    roi = gray[ys:ye, xs:xe]
    if roi.size == 0:
        return False, 0.0
    thr = max(roi.mean() * 0.45, 10)
    dark = (roi < thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    frac = dark.sum() / dark.size
    return dark.sum() >= min_px and frac > 0.08, float(frac)


def ece(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct)
    edges = np.linspace(0, 1, n_bins + 1)
    e, rows = 0.0, []
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= 1.0)
        if m.sum() == 0:
            rows.append((edges[i], edges[i + 1], 0, np.nan, np.nan))
            continue
        c_mean, a_mean = conf[m].mean(), correct[m].mean()
        e += (m.sum() / len(conf)) * abs(a_mean - c_mean)
        rows.append((edges[i], edges[i + 1], int(m.sum()), c_mean, a_mean))
    return e, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ML / "models" / "checkpoints" / "best_detector.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--calibrator", type=Path, default=ML / "models" / "exported" / "calibrator.pkl")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    calib = pickle.load(open(args.calibrator, "rb")) if args.calibrator.exists() else None

    img_dir, lbl_dir = SPLITS / args.split / "images", SPLITS / args.split / "labels"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
    logger.info(f"{len(imgs)} {args.split} images")

    raw, correct, cls_of = [], [], []
    sh_tp, sh_fp = [], []          # trailing-shadow present? split by TP/FP
    per_cls = defaultdict(lambda: {"tp": 0, "fp": 0, "tp_conf": [], "fp_conf": [], "n_gt": 0})

    for ip in imgs:
        gts = load_gt(lbl_dir / f"{ip.stem}.txt")
        g_img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        H, W = g_img.shape[:2]
        for c, *_ in gts:
            per_cls[c]["n_gt"] += 1
        res = model.predict(str(ip), conf=0.01, iou=0.6, imgsz=args.imgsz, verbose=False)
        if not res or res[0].boxes is None:
            continue
        b = res[0].boxes
        used = set()
        order = np.argsort(-b.conf.cpu().numpy())
        for i in order:
            conf = float(b.conf[i]); pc = int(b.cls[i])
            x1, y1, x2, y2 = b.xyxy[i].cpu().numpy()
            pn = (x1 / W, y1 / H, x2 / W, y2 / H)
            best, bj = 0.0, -1
            for j, (gc, *gb) in enumerate(gts):
                if gc != pc or j in used:
                    continue
                v = iou(pn, gb)
                if v > best:
                    best, bj = v, j
            ok = int(best >= IOU_MATCH)
            if ok:
                used.add(bj)
            raw.append(conf); correct.append(ok); cls_of.append(pc)
            per_cls[pc]["tp" if ok else "fp"] += 1
            per_cls[pc]["tp_conf" if ok else "fp_conf"].append(conf)
            if conf >= 0.25:  # shadow feasibility only on real candidate detections
                present, _ = has_trailing_shadow(g_img, (x1, y1, x2, y2))
                (sh_tp if ok else sh_fp).append(int(present))

    raw = np.array(raw); correct = np.array(correct); cls_arr0 = np.array(cls_of)

    def _apply_cal(scores, clss):
        if calib is None:
            return scores.copy()
        if isinstance(calib, dict) and calib.get("kind") == "per_class":
            out = scores.copy()
            for c in np.unique(clss):
                m = calib["models"].get(int(c)) or calib.get("fallback")
                if m is None:
                    continue
                idx = clss == c
                out[idx] = m.predict_proba(scores[idx].reshape(-1, 1))[:, 1]
            return out
        return calib.predict_proba(scores.reshape(-1, 1))[:, 1]

    cal = _apply_cal(raw, cls_arr0)

    print("\n" + "=" * 66)
    print("  1. CALIBRATION HONESTY  (all classes, all detections conf>=0.01)")
    print("=" * 66)
    e_raw, rows_raw = ece(raw, correct)
    e_cal, rows_cal = ece(cal, correct)
    print(f"  ECE raw scores        : {e_raw:.4f}")
    print(f"  ECE calibrated scores : {e_cal:.4f}   ({'better' if e_cal < e_raw else 'worse'})")
    print("\n  reliability (calibrated): bin   n   mean_pred   actual_acc")
    for lo, hi, n, cp, ac in rows_cal:
        if n == 0:
            continue
        print(f"    {lo:.1f}-{hi:.1f}  {n:5d}   {cp:.3f}      {ac:.3f}")

    print("\n" + "=" * 66)
    print("  2. SCORE SEPARATION  (can a threshold tell TP from FP?)")
    print("=" * 66)
    print(f"  {'class':20s} {'nGT':>5} {'TP':>5} {'FP':>5}  {'meanConf TP':>11} {'meanConf FP':>11}  gap")
    for c in sorted(per_cls):
        d = per_cls[c]
        mt = np.mean(d["tp_conf"]) if d["tp_conf"] else 0
        mf = np.mean(d["fp_conf"]) if d["fp_conf"] else 0
        print(f"  {NAMES[c]:20s} {d['n_gt']:5d} {d['tp']:5d} {d['fp']:5d}  {mt:11.3f} {mf:11.3f}  {mt-mf:+.3f}")

    print("\n" + "=" * 66)
    print("  3. OPERATING POINTS  (calibrated-score cut-offs, all classes)")
    print("=" * 66)
    total_gt = sum(d["n_gt"] for d in per_cls.values())
    for t in (0.30, 0.50, 0.70, 0.85):
        keep = cal >= t
        tp = int(correct[keep].sum()); fp = int(keep.sum() - tp)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(total_gt, 1)
        print(f"  cal>={t:.2f}   precision={prec:.3f}   recall={rec:.3f}   ({tp} TP / {fp} FP)")

    print("\n" + "=" * 66)
    print("  3b. PER-CLASS operating points  (RAW score cut-offs)")
    print("=" * 66)
    cls_arr = np.array(cls_of)
    for c in sorted(per_cls):
        ng = per_cls[c]["n_gt"]
        line = f"  {NAMES[c]:20s} nGT={ng:4d} | "
        for t in (0.25, 0.35, 0.45):
            m = (cls_arr == c) & (raw >= t)
            tp = int(correct[m].sum()); fp = int(m.sum() - tp)
            p = tp / max(tp + fp, 1); r = tp / max(ng, 1)
            line += f"raw>={t}: P={p:.2f} R={r:.2f}  "
        print(line)

    print("\n" + "=" * 66)
    print("  4. SHADOW-CHECK FEASIBILITY  (detections conf>=0.25)")
    print("=" * 66)
    if sh_tp and sh_fp:
        print(f"  true positives  with a trailing dark region : {np.mean(sh_tp):.2f}  (n={len(sh_tp)})")
        print(f"  false positives with a trailing dark region : {np.mean(sh_fp):.2f}  (n={len(sh_fp)})")
        print("  -> a metadata-free 'no shadow => demote' rule is useful only if")
        print("     FP fraction is clearly lower than TP fraction.")
    else:
        print("  not enough detections >=0.25 to assess")
    print("=" * 66)


if __name__ == "__main__":
    main()
