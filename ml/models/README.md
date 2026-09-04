# ml/models/ — weights not in git

Only `exported/calibrator.pkl` (2 KB, per-class Platt calibrators) is committed, so
`ml/inference/pipeline.py` runs after a plain clone.

Everything else is on the Hugging Face model repo — [rehan9599/drishti-detector](https://huggingface.co/rehan9599/drishti-detector):

| File | What |
|---|---|
| `best_detector.pt` | shipped — preprocessed Run 3 YOLOv8s |
| `best_detector_prep.pt` / `best_detector_raw.pt` | Run 3 / Run 2 backups |
| `exported/best_detector.onnx` | FP32 ONNX — the edge deployment model (44.8 MB) |
| `exported/best_detector_fp16.onnx` | FP16 — Jetson GPU target |
| `exported/best_detector_int8.onnx` | INT8 — excluded (accuracy collapse), kept for the record |

Fetch:
```bash
hf download rehan9599/drishti-detector --local-dir ml/models/exported
```
Or re-derive: train (`ml/scripts/train_yolo_seg.py`) then `ml/scripts/export_onnx.py`.
