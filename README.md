# DRISHTI

**AI-powered underwater marine-debris & anomaly detection from side-scan sonar imagery.**
Smart India Hackathon 2026 · Problem Statement **26057** · Ministry of Earth Sciences / NIOT.

DRISHTI ingests a raw side-scan sonar (SSS) log, detects man-made seabed hazards
(shipwrecks, pipelines/cables, cylinders, entangled "ghost" nets), scores each detection
0–100 %, geotags it to latitude/longitude, and emits a JSON/CSV/GeoJSON report — running
**on the edge**, torch-free, without a cloud.

> Full technical record — problem → 3 training runs → edge/ONNX → per-module theory & physics
> with diagrams → literature benchmark → directions: **`docs/PROJECT_RECORD.html`**

**Try it in 2 minutes, no setup:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rehan9599/Sonar-Drishti/blob/main/notebooks/01_quickstart_inference.ipynb)
— the model and sample tiles are committed, so one `git clone` runs the detector. CPU only.

---

## 1. Repository layout

| Path | Contents |
|---|---|
| `ml/` | dataset assembly, training, evaluation, ONNX export, inference pipeline |
| `ml/scripts/` | tiling · `build_dataset.py` · `train_yolo_seg.py` · `calibrate_confidence.py` · `evaluate.py` · `export_onnx.py` · `run_aurora_survey.py` |
| `ml/inference/` | `detector.py` · `confidence_filter.py` · `shadow_verification.py` · `pipeline.py` (chains M0→M3) |
| `backend/` | Django REST + Channels (WebSocket) + Celery · `geotagging/` (XTF + nav parsing, projection) · `reporting/` (schema, JSON/CSV/GeoJSON) |
| `edge/` | torch-free inference: `edge_infer.py` · `onnx_runtime_server.py` (FastAPI) · `benchmark.py` · `Dockerfile.edge` |
| `frontend/` | React + Leaflet dashboard (Module 4) |
| `infra/` | docker-compose, nginx, postgres init |
| `docs/` | `PROJECT_RECORD.html` (the master doc) · `api_contract.md` (**frozen** — build against this) · per-run reports · `pre_final_training_changes.md` |
| `notebooks/` | Colab-ready: explore dataset · train detector · inference demo |

Data (34 GB) and model weights are **not in git** — see §4.

---

## 2. The model  ← main focus

### 2.1 What it is
- **YOLOv8s** object detector, fine-tuned from COCO (Ultralytics 8.4.x).
- **Box detection**, not segmentation — every real SSS dataset we could obtain is box-labelled;
  segmentation is a documented roadmap (AI4Shipwrecks, SW-Net benchmark).
- **5 classes trained, 4 shipped:** `submarine_pipeline`, `shipwreck`, `mine_cylinder`,
  `ghost_net` ship; `crab_pot` is trained as a hard-negative-ish class and **filtered downstream**.
- **SSS-specific choices:** grayscale (colour aug off); **vertical flip off** (preserves the
  highlight→shadow polarity that identifies an object); brightness/scale/erase aug mapped to
  real sonar nuisances.
- **Preprocessing (Module 0):** Lee speckle filter + CLAHE, applied identically at train and
  serve time (`ml/scripts/preprocess_sonar.py :: despeckle_clahe`).

### 2.2 Training
```bash
python ml/scripts/preprocess_splits.py                       # Module 0, in place
python ml/scripts/train_yolo_seg.py --model yolov8s.pt \
       --epochs 120 --batch 16 --patience 25 --workers 2     # resume: resume_training.py
python ml/scripts/calibrate_confidence.py --model ml/models/checkpoints/best_detector.pt --split val
python ml/scripts/evaluate.py --model ml/models/checkpoints/best_detector.pt
```
RTX 4050 · ~2 h/run · early-stopped. Config rationale in `docs/PROJECT_RECORD.html` §10.1.

### 2.3 Results — three runs, held-out test

| Metric | Run 1 | Run 2 (raw, backup) | **Run 3 (preprocessed, shipped)** |
|---|---|---|---|
| mAP@50 | 0.596 | 0.606 | **0.580** |
| precision | 0.687 | 0.734 | **0.734** |
| false-positive rate | 0.313 | 0.266 | **0.266** |
| `submarine_pipeline` AP50 | 0.982 | 0.994 | **0.984** |
| `mine_cylinder` AP50 | 0.431 | 0.398 | **0.424** |
| `shipwreck` AP50 | 0.453 | 0.450 | **0.302** \* |
| `ghost_net` AP50 (synthetic) | 0.995 | 0.995 | **0.995** |

\* Run 3's test set is deliberately harder (overlap re-tile 3×'d shipwreck instances). Run 2's
model scores **0.278** on the *same* set — the two are tied. Single-class YOLO on a clean
AI4Shipwrecks split tops out at ~0.72–0.76 in the 2026 literature; human annotators agree only
50–60 %. See `PROJECT_RECORD.html` §08 and §12.

**Ablation verdict:** Lee+CLAHE preprocessing gave *no* accuracy gain on a matched protocol
(identical precision & FP-rate). Shipped anyway — the data-quality changes bundled with it were
kept, and CLAHE'd input makes acoustic shadows more detectable for Module 2.

### 2.4 Module 2 — confidence & noise filtering
NMS → per-class raw-score gate → **per-class Platt calibration** (ECE 0.052 → 0.037 on test;
a single global calibrator made it *worse*) → opt-in shadow-geometry check (`L = h·G/(H−h)`,
transects only). `ml/models/exported/calibrator.pkl`.

### 2.5 Module 3 — geotagging
Header-only XTF parser + navigation CSV → pixel → slant range → ground range `G = √(R²−H²)` →
geodesic offset ⟂ heading → (lat, lon) + size in metres. Validated on the AURORA survey:
position agrees with recorded navigation to **< 1 m**. `backend/geotagging/`, `backend/reporting/`.

### 2.6 Edge / ONNX
| Model | Size | CPU latency | Accuracy | Role |
|---|---|---|---|---|
| **FP32 ONNX** | 44.8 MB | ~90 ms/tile | matches PyTorch | **shipped** (torch-free: onnxruntime + numpy + opencv) |
| FP16 ONNX | 22.4 MB | — | — | Jetson GPU target |
| INT8 ONNX | 11.5 MB | 164 ms (slower) | **collapsed (0.0)** | excluded |

```bash
python ml/scripts/export_onnx.py --model ml/models/checkpoints/best_detector.pt --benchmark
python edge/edge_infer.py --image tile.png --no-preprocess     # torch-free path
uvicorn edge.onnx_runtime_server:app --port 8100               # edge service (same JSON contract)
```

---

## 3. End-to-end pipeline

```
raw SSS log ─► tile ─► M0 preprocess ─► M1 detect (YOLOv8s) ─► M2 confidence ─► M3 geotag ─► JSON/CSV/GeoJSON
                                                        ▲                 │
                                                        └── altitude H + slant range R ┘
```
- One call: `python -m ml.inference.pipeline --image tile.png --xtf <file>.xtf --nav navigation.csv --out report`
- Full transect: `python ml/scripts/run_aurora_survey.py --preprocess` (~8 s/transect)
- **Module 4 (dashboard):** React + Leaflet + review queue — scaffold on mock data, wiring to the
  live API in progress.

Modules 0–3: **functional.** Module 4: **in progress.** Cross-survey transfer: **needs fine-tuning**
(documented literature behaviour — see `PROJECT_RECORD.html` §12).

---

## 4. Data & weights (not in git)

**The three files the pipeline needs are committed** (72 MB total) — `git clone` and it runs, no
fetch step, no account, no Git LFS:

| In the repo | Size | Used by |
|---|---|---|
| `ml/models/checkpoints/best_detector.pt` | 21.5 MB | `pipeline.py` default (needs torch) |
| `ml/models/exported/best_detector.onnx` | 42.7 MB | same model, torch-free onnxruntime path |
| `ml/models/exported/calibrator.pkl` | 2 KB | Module 2 per-class calibration |

Everything else is training-only and lives outside git:

| Asset | Where | How to get it |
|---|---|---|
| `.pt` backups, FP16/INT8 ONNX, per-epoch checkpoints (3.4 GB) | [`rehan9599/drishti-detector`](https://huggingface.co/rehan9599/drishti-detector) | `hf download rehan9599/drishti-detector` |
| Training splits (2.1 GB, preprocessed) | [`rehan9599/drishti-sss`](https://huggingface.co/datasets/rehan9599/drishti-sss) *(private)* | `hf download rehan9599/drishti-sss --repo-type dataset --local-dir ml/data/splits` |
| Raw source datasets (32 GB) | not hosted — reconstructible | public sources in `PROJECT_RECORD.html` §15 + `ml/scripts/tile_*.py` + `build_dataset.py` |

Module 4 and the edge service need **none** of the above — only the three committed files.

---

## 5. Reproduce / run

```bash
# ML env (heavy — training / export only)
python -m venv .venv && .venv/Scripts/activate      # or: source .venv/bin/activate
pip install -r ml/requirements.txt

# backend + infra
cp .env.example .env
docker compose -f infra/docker-compose.yml up --build

# frontend
cd frontend && npm install && npm run dev
```

Colab: open a notebook in `notebooks/` — it installs deps and pulls data + weights from Hugging Face.

**Read `docs/api_contract.md` before writing code against another module — it is frozen.**

---

## 6. Docs

- **`docs/INTEGRATION.md` — start here if you are wiring Module 4.** One function call, the three
  Django stubs to fill, a worked Celery task.
- `docs/PROJECT_RECORD.html` — the master technical record (problem → runs → edge → per-module
  theory & physics → literature benchmark → directions)
- `docs/api_contract.md` — frozen interface between tracks
- `docs/pre_final_training_changes.md` — the change list for an optional final training run
- `docs/run1_detector_report.html`, `run2_detector_report.html`, `module3_geotag_report.html` — deep dives
