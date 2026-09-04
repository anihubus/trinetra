---
license: mit
tags:
  - object-detection
  - sonar
  - side-scan-sonar
  - marine-debris
  - underwater
  - yolov8
  - onnx
  - edge-ai
pipeline_tag: object-detection
---

<!-- NOTE: `library_name: ultralytics` is deliberately NOT set. It makes the Hub emit an
     auto-generated snippet (`YOLOvv8.from_pretrained(...)`) that does not work for this repo —
     wrong class name, no from_pretrained support for this file layout, a COCO cat photo as the
     input, and it would skip preprocessing + calibration. Use the quickstart below instead. -->

# DRISHTI — marine-debris detector for side-scan sonar

## ▶ Start here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rehan9599/Sonar-Drishti/blob/main/notebooks/01_quickstart_inference.ipynb)

Runs in ~2 minutes, CPU only, no PyTorch and no account needed:

```bash
pip install onnxruntime opencv-python-headless numpy scikit-learn
git clone https://github.com/Rehan9599/Sonar-Drishti.git && cd Sonar-Drishti
```
```python
from edge.edge_infer import run
run("demo/tiles/synth_ghost_net_00002.jpg", preprocess=False)
# -> {'class_label': 'ghost_net', 'confidence_score': 95.3, 'review_status': 'auto_confirmed', ...}
```

The weights are committed to that repo, so the clone is the whole setup — sample tiles included.

> **Do not just call `model.predict()` on a raw image.** Three things break silently:
> the model needs **Lee + CLAHE** preprocessing (it was trained that way), the **Module 2**
> per-class gate + calibration is what produces the honest 0–100 % score, and **coordinates**
> require an XTF/navigation log. `run()` and `run_pipeline()` handle all three.


YOLOv8s fine-tuned to detect man-made seabed hazards in **side-scan sonar (SSS)** imagery.
Built for Smart India Hackathon 2026, Problem Statement 26057 (Ministry of Earth Sciences / NIOT).

Ships as **PyTorch** and as **ONNX** — the ONNX path runs the full pipeline with
`onnxruntime + numpy + opencv` and **no PyTorch import at all** (~50 MB runtime vs ~1.5 GB),
which is what makes it deployable on an AUV.

- Code: https://github.com/Rehan9599/Sonar-Drishti
- Full technical record: `docs/PROJECT_RECORD.html` in that repo

## Classes

**5 trained, 4 shipped.** `crab_pot` is trained as a hard-negative-ish class and filtered
downstream — it never reached a usable AP (see Limitations).

| id | class | shipped |
|---|---|---|
| 0 | `crab_pot` | ✗ filtered |
| 1 | `submarine_pipeline` | ✓ |
| 2 | `shipwreck` | ✓ |
| 3 | `ghost_net` | ✓ (synthetic training data) |
| 4 | `mine_cylinder` | ✓ |

## Files

| File | Size | Notes |
|---|---|---|
| `best_detector.pt` | 22.5 MB | **shipped model** — Run 3, trained on Lee+CLAHE preprocessed tiles |
| `best_detector.onnx` | 44.8 MB | FP32 ONNX — the edge deployment model, same weights |
| `best_detector_fp16.onnx` | 21.4 MB | FP16 — Jetson GPU target |
| `best_detector_int8.onnx` | 11.0 MB | INT8 — **excluded**, kept for the record (see Limitations) |
| `best_detector_prep.pt` / `best_detector_raw.pt` | 22 MB ea. | Run 3 / Run 2 backups |
| `calibrator.pkl` | 2 KB | per-class Platt-scaling calibrators (Module 2) |
| `results.csv`, `args.yaml`, `*.png` | ~1 MB | training curves, confusion matrix, PR curves, exact hyper-parameters |

## Results — held-out test set (850 tiles)

| Metric | Value |
|---|---|
| mAP@50 | 0.580 |
| mAP@50-95 | 0.434 |
| precision | 0.734 |
| recall | 0.629 |
| false-positive rate | 0.266 |

Per class (AP@50):

| Class | AP@50 | Note |
|---|---|---|
| `ghost_net` | 0.995 | **synthetic data — not a field number** |
| `submarine_pipeline` | 0.984 | production-grade |
| `mine_cylinder` | 0.424 | real-data ceiling |
| `shipwreck` | 0.302 | see below |
| `crab_pot` | 0.193 | filtered from the product |

**On the shipwreck number.** It is measured on a deliberately hard test set (a 50 %-overlap
re-tile tripled the shipwreck instances with partial and near-duplicate tiles). The 2026
literature on the *same* dataset (AI4Shipwrecks) reports a vanilla YOLOv8 detection baseline of
**mAP50 0.716** and a best-in-class **0.755** (DFSE-YOLO), while human inter-annotator agreement
on SSS wrecks is only **50–60 %** (SW-Net). Our number is that ceiling minus a harder split and
multi-class dilution — not a training failure.

## Confidence calibration

Raw detector scores are not probabilities. `calibrator.pkl` holds **per-class** Platt-scaling
models; a single global calibrator made things *worse*.

| | Expected Calibration Error (test) |
|---|---|
| raw scores | 0.052 |
| per-class calibrated | **0.037** |

## Usage

### Torch-free (recommended for deployment)

```python
import cv2, numpy as np, onnxruntime as ort

sess = ort.InferenceSession("best_detector.onnx", providers=["CPUExecutionProvider"])
img = cv2.resize(cv2.imread("tile.png"), (640, 640))
x = np.ascontiguousarray((img[:, :, ::-1] / 255.0).astype(np.float32).transpose(2, 0, 1)[None])
out = sess.run(None, {sess.get_inputs()[0].name: x})[0]   # [1, 4+nc, 8400]
```
Full decode + NMS + calibration: `edge/edge_infer.py` in the GitHub repo.

### Ultralytics

```python
from ultralytics import YOLO
model = YOLO("best_detector.pt")
results = model.predict("tile.png", conf=0.10)
```

> **Preprocess your input.** This model was trained on **Lee speckle filter + CLAHE** tiles.
> Apply `despeckle_clahe()` (`ml/scripts/preprocess_sonar.py`) before inference, or accuracy
> degrades silently. Use `conf=0.10` — the per-class gate in Module 2 does the real cut.

## Training

| | |
|---|---|
| base | `yolov8s.pt` (COCO-pretrained) |
| image size | 640 × 640 |
| batch | 16 |
| epochs | 120 requested, early-stopped at 96 (patience 25) |
| hardware | RTX 4050 laptop, ~2.1 h |
| preprocessing | Lee speckle filter + CLAHE, applied identically at train and serve |
| augmentation | SSS-tuned: colour off (grayscale), **vertical flip off** (preserves highlight→shadow polarity), mosaic 0.8, mixup 0.1, erasing 0.4, rotate ±10°, shear 2° |
| loss | CIoU + BCE + DFL |

Exact hyper-parameters in `args.yaml`; the epoch-by-epoch log in `results.csv`.

## Training data

~4,775 training tiles assembled from six real SSS surveys plus procedural synthetic data.
`ghost_net` is **100 % synthetic** — no public real ghost-net-in-SSS dataset exists (a
Microsoft AI for Good / WWF effort had 412 real segments total and called it a feasibility study).

The dataset is not redistributed here; source provenance and the rebuild scripts are in the
GitHub repo (`docs/PROJECT_RECORD.html` §15, `ml/scripts/`).

## Limitations

- **Does not transfer across surveys without fine-tuning.** On an unseen instrument/seabed
  (AURORA) it produced near-zero detections. This is documented literature behaviour — GhostNetZero
  reports IoU 0.740 → 0.547 across regions — and the remedy is the same: fine-tune on a small
  labelled sample of the target sonar.
- **`ghost_net` metrics are synthetic-on-synthetic.** Treat as proof-of-capability, not field performance.
- **`crab_pot` is not usable** (AP@50 0.193 after two runs and a data doubling). It is a separability
  problem, not a data-volume one. Filtered from the product.
- **INT8 is broken for this task.** Static PTQ collapsed accuracy to **0.00 precision / 0.00 recall
  on every class**, and was *slower* than FP32 on CPU (164 ms vs 90 ms) without VNNI/AVX-512.
  The file is published only so the result is reproducible. Use FP32 or FP16.
- **Positional accuracy is a search area, not a survey fix.** The tow-fish position recovers to
  < 1 m, but detection position also depends on across-track scale; two navigation paths can
  disagree by ~100 m on the same target.

## Benchmarks (laptop CPU, onnxruntime 1.29, no CUDA EP)

| Model | Size | Latency | Accuracy |
|---|---|---|---|
| FP32 ONNX | 44.8 MB | ~90 ms/tile (11 FPS) | matches PyTorch |
| FP16 ONNX | 21.4 MB | n/a on CPU | Jetson GPU target |
| INT8 ONNX | 11.0 MB | 164 ms/tile | collapsed |

## Citation

```bibtex
@software{drishti2026,
  title  = {DRISHTI: AI-Powered Marine Debris Detection from Side-Scan Sonar},
  author = {Fazal, Rehan and others},
  year   = {2026},
  note   = {Smart India Hackathon 2026, Problem Statement 26057},
  url    = {https://github.com/Rehan9599/Sonar-Drishti}
}
```
