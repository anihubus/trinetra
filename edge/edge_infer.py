"""
Torch-free edge inference: onnxruntime + numpy + opencv only.

The AUV / Jetson path. Loads the exported ONNX detector, decodes YOLOv8's raw
output in numpy, runs numpy NMS, then reuses the same Module 2 confidence filter
and Module 3 geotagger as the server path - none of which need PyTorch.

    python edge/edge_infer.py --image tile.png --onnx ml/models/exported/best_detector.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from preprocess_sonar import despeckle_clahe                      # cv2 + numpy only
from ml.inference.confidence_filter import ConfidenceFilter        # numpy only

CLASS_NAMES = ["crab_pot", "submarine_pipeline", "shipwreck", "ghost_net", "mine_cylinder"]
DEFAULT_ONNX = ROOT / "ml" / "models" / "exported" / "best_detector.onnx"
DEFAULT_CAL = ROOT / "ml" / "models" / "exported" / "calibrator.pkl"


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """Plain numpy NMS. boxes = [N,4] xyxy."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class EdgeDetector:
    """ONNX YOLOv8 detector - no torch, no ultralytics."""

    def __init__(self, onnx_path: str | Path, imgsz: int = 640, providers=None):
        import onnxruntime as ort
        self.imgsz = imgsz
        self.sess = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.inp = self.sess.get_inputs()[0].name
        self.size_mb = Path(onnx_path).stat().st_size / 1e6

    def _preprocess(self, bgr: np.ndarray):
        h0, w0 = bgr.shape[:2]
        img = cv2.resize(bgr, (self.imgsz, self.imgsz))
        x = img[:, :, ::-1].astype(np.float32) / 255.0     # BGR->RGB, 0-1
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        return x, (w0, h0)

    def detect(self, bgr: np.ndarray, conf: float = 0.10, iou: float = 0.5) -> list[dict]:
        x, (w0, h0) = self._preprocess(bgr)
        out = self.sess.run(None, {self.inp: x})[0]          # [1, 4+nc, 8400]
        pred = np.squeeze(out, 0).T                          # [8400, 4+nc]
        boxes_xywh, cls_scores = pred[:, :4], pred[:, 4:]
        cls_id = cls_scores.argmax(1)
        cls_conf = cls_scores.max(1)
        m = cls_conf >= conf
        boxes_xywh, cls_id, cls_conf = boxes_xywh[m], cls_id[m], cls_conf[m]
        if len(boxes_xywh) == 0:
            return []
        # xywh(center, 640-space) -> xyxy in original pixels
        cx, cy, bw, bh = boxes_xywh.T
        sx, sy = w0 / self.imgsz, h0 / self.imgsz
        xyxy = np.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy,
                         (cx + bw / 2) * sx, (cy + bh / 2) * sy], 1)
        dets = []
        for c in np.unique(cls_id):
            idx = np.where(cls_id == c)[0]
            for k in _nms(xyxy[idx], cls_conf[idx], iou):
                j = idx[k]
                dets.append({
                    "class_label": CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}",
                    "confidence_raw": round(float(cls_conf[j]), 4),
                    "bbox": [round(float(v), 2) for v in xyxy[j]],
                    "mask_polygon": [],
                })
        return dets


def run(image_path, onnx_path=DEFAULT_ONNX, calibrator=DEFAULT_CAL, preprocess=True):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    if preprocess:
        bgr = cv2.cvtColor(despeckle_clahe(bgr), cv2.COLOR_GRAY2BGR)

    det = EdgeDetector(onnx_path)
    t0 = time.perf_counter()
    raw = det.detect(bgr, conf=0.10)
    infer_ms = (time.perf_counter() - t0) * 1000

    cal = str(calibrator) if Path(calibrator).exists() else None
    cf = ConfidenceFilter(calibrator_path=cal, use_shadow_check=False)
    scored = cf.filter(raw)
    return {
        "runtime": "onnxruntime-cpu (torch-free)",
        "onnx_size_mb": round(det.size_mb, 2),
        "inference_ms": round(infer_ms, 1),
        "raw": len(raw),
        "kept": len(scored),
        "detections": [
            {"class_label": d["class_label"], "confidence_score": d["confidence_score"],
             "bbox": d["bbox"], "review_status": ("auto_confirmed" if d["confidence_score"] >= 80
                                                  else "pending_review")}
            for d in scored
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--calibrator", type=Path, default=DEFAULT_CAL)
    ap.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    args = ap.parse_args()
    print(json.dumps(run(args.image, args.onnx, args.calibrator, args.preprocess), indent=2))


if __name__ == "__main__":
    main()
