"""
Accuracy check for the exported ONNX models via the torch-free EdgeDetector.

Runs FP32 and INT8 ONNX over splits/test, matches to ground truth (IoU>=0.5),
and reports per-class precision / recall at a fixed confidence - so we can see
whether INT8 quantization damages the faint classes (the known sonar risk).

    python edge/eval_edge_onnx.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.edge_infer import EdgeDetector, CLASS_NAMES

SPLITS = ROOT / "ml" / "data" / "splits"
EXP = ROOT / "ml" / "models" / "exported"
CONF = 0.25
IOU_MATCH = 0.5


def load_gt(p):
    out = []
    if p.exists():
        for ln in p.read_text().strip().splitlines():
            q = ln.split()
            if len(q) == 5:
                c = int(q[0]); xc, yc, w, h = map(float, q[1:])
                out.append((c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    i = (ix2 - ix1) * (iy2 - iy1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


def evaluate(onnx_path: Path, images: list[Path]):
    det = EdgeDetector(onnx_path)
    per = defaultdict(lambda: {"tp": 0, "fp": 0, "gt": 0})
    t0 = time.perf_counter()
    for ip in images:
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        gts = load_gt(SPLITS / "test" / "labels" / f"{ip.stem}.txt")
        for c, *_ in gts:
            per[c]["gt"] += 1
        used = set()
        dets = sorted(det.detect(bgr, conf=CONF), key=lambda d: -d["confidence_raw"])
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            pn = (x1 / W, y1 / H, x2 / W, y2 / H)
            cid = CLASS_NAMES.index(d["class_label"]) if d["class_label"] in CLASS_NAMES else -1
            best, bj = 0.0, -1
            for j, (gc, *gb) in enumerate(gts):
                if gc == cid and j not in used:
                    v = iou(pn, gb)
                    if v > best:
                        best, bj = v, j
            if best >= IOU_MATCH:
                per[cid]["tp"] += 1; used.add(bj)
            else:
                per[cid]["fp"] += 1
    dt = time.perf_counter() - t0
    return per, dt / max(len(images), 1) * 1000


def main():
    images = sorted(p for p in (SPLITS / "test" / "images").iterdir()
                    if p.suffix.lower() in (".jpg", ".png"))
    models = [("FP32", EXP / "best_detector.onnx"), ("INT8", EXP / "best_detector_int8.onnx")]
    rows = {}
    for tag, mp in models:
        if not mp.exists():
            continue
        print(f"running {tag} ({mp.stat().st_size/1e6:.1f} MB) over {len(images)} tiles ...")
        per, ms = evaluate(mp, images)
        rows[tag] = (per, ms)

    print("\n" + "=" * 70)
    print(f"  ONNX accuracy @ conf {CONF}  (torch-free EdgeDetector, CPU)")
    print("=" * 70)
    hdr = f"  {'class':20s}"
    for tag in rows:
        hdr += f" | {tag}  P     R "
    print(hdr)
    for cid in range(len(CLASS_NAMES)):
        line = f"  {CLASS_NAMES[cid]:20s}"
        for tag, (per, _) in rows.items():
            d = per[cid]
            p = d["tp"] / max(d["tp"] + d["fp"], 1)
            r = d["tp"] / max(d["gt"], 1)
            line += f" | {p:.2f}  {r:.2f} "
        print(line)
    print("-" * 70)
    for tag, (_, ms) in rows.items():
        print(f"  {tag}: {ms:.1f} ms/tile")
    print("=" * 70)


if __name__ == "__main__":
    main()
