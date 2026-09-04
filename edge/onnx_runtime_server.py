"""
Lightweight FastAPI microservice for on-vehicle / edge inference.

Same JSON contract as the Django API's detection response, so the frontend needs
zero changes whether inference ran server-side or here. Torch-free: the whole
request path is onnxruntime + numpy + opencv (see edge/edge_infer.py).

  Tier 0 (laptop / AUV compute): CPUExecutionProvider   -> this file, as is
  Tier 1 (Jetson):               onnxruntime-gpu + TensorRTExecutionProvider

Run:
  pip install -r edge/requirements.txt
  uvicorn edge.onnx_runtime_server:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml" / "scripts"))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from edge.edge_infer import EdgeDetector, DEFAULT_ONNX, DEFAULT_CAL, CLASS_NAMES
from preprocess_sonar import despeckle_clahe
from ml.inference.confidence_filter import ConfidenceFilter

ONNX_PATH = Path(os.environ.get("DRISHTI_ONNX", DEFAULT_ONNX))
CAL_PATH = Path(os.environ.get("DRISHTI_CALIBRATOR", DEFAULT_CAL))
PREPROCESS = os.environ.get("DRISHTI_PREPROCESS", "1") != "0"

app = FastAPI(title="DRISHTI edge inference", version="1.0")

_detector: EdgeDetector | None = None
_filter: ConfidenceFilter | None = None


@app.on_event("startup")
def _load():
    global _detector, _filter
    if not ONNX_PATH.exists():
        raise RuntimeError(f"ONNX model not found: {ONNX_PATH} — run ml/scripts/export_onnx.py")
    _detector = EdgeDetector(ONNX_PATH)
    _filter = ConfidenceFilter(
        calibrator_path=str(CAL_PATH) if CAL_PATH.exists() else None,
        use_shadow_check=False,          # tiles: no per-ping geometry
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "runtime": "onnxruntime-cpu (torch-free)",
        "model": ONNX_PATH.name,
        "model_size_mb": round(_detector.size_mb, 2) if _detector else None,
        "preprocess": PREPROCESS,
        "classes": CLASS_NAMES,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """One sonar tile in -> filtered, scored detections out (API-contract shape)."""
    if _detector is None or _filter is None:
        raise HTTPException(503, "model not loaded")
    raw = np.frombuffer(await file.read(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    if PREPROCESS:
        bgr = cv2.cvtColor(despeckle_clahe(bgr), cv2.COLOR_GRAY2BGR)

    import time
    t0 = time.perf_counter()
    dets_raw = _detector.detect(bgr, conf=0.10)
    scored = _filter.filter(dets_raw)
    infer_ms = round((time.perf_counter() - t0) * 1000, 1)

    return JSONResponse({
        "source_file": file.filename,
        "runtime": "onnxruntime-cpu",
        "inference_ms": infer_ms,
        "detections": [
            {
                "class_label": d["class_label"],
                "confidence_score": d["confidence_score"],
                "bounding_geometry": {"bbox": d["bbox"], "mask_polygon": []},
                "review_status": "auto_confirmed" if d["confidence_score"] >= 80 else "pending_review",
            }
            for d in scored
        ],
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("edge.onnx_runtime_server:app", host="0.0.0.0", port=8100, reload=False)
