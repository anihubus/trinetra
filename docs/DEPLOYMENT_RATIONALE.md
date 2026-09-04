# Deployment architecture — recommendation

**Question on the table:** host the ML model on its own server, host frontend + backend on
another, and have them talk over HTTP.

**Recommendation:** don't. For this project the model is a **Python library**, not a service.
Run it inside the Celery worker in the same compose stack as the backend. One repo, one
`docker compose up`, no ML server, no inter-service network hop.

This doc says why, and gives the concrete layout to build against.

---

## 1. The split architecture solves a problem we don't have

Splitting the model onto its own server is the right call when **any** of these is true:

| Reason to split out an ML service | True for DRISHTI? |
|---|---|
| Model needs a GPU the app server shouldn't pay for | No — torch-free CPU inference, ~90 ms/tile |
| Model and app ship on different release cadences | No — model is frozen (`best_detector.onnx`, Run 3) |
| Model must scale independently of web traffic | No — it already scales, as Celery worker replicas |
| Multiple products consume the same model | No — one consumer, this app |
| A different team owns the model, in a different language | No — same repo, same Python |

None hold. What the split **costs** instead:

- **A service that doesn't exist yet.** Today the integration is one function call —
  `from ml.inference.pipeline import run_pipeline`. Splitting means building and maintaining an
  HTTP wrapper around it (a FastAPI app, its own Dockerfile, request/response schemas, health
  checks, error mapping) that adds zero capability.
- **A second thing that can be down.** The backend gains a hard dependency on a network call
  that can time out, 500, or refuse connection mid-survey. Every tile becomes a round trip.
- **A second deploy, second set of env vars, second log stream, and CORS/auth between services.**
- **Harder local dev.** Every contributor now runs two stacks to see a detection render.
- **A weaker story for judges.** "We put the model behind its own microservice" invites the
  question *why?* — and the honest answer is *we didn't need to*.

The latency fact that matters: the model runs at ~90 ms/tile. A cross-container HTTP hop adds
~5–30 ms of overhead per tile plus the cost of serializing the image payload. On a full
transect (hundreds of tiles) that is real, avoidable waste.

---

## 2. Recommended: one stack, model as a library

```
  SPLIT  (avoid)                              UNIFIED  (recommended)
  ─────────────                               ──────────────────────
  ┌──────────┐   HTTP    ┌──────────┐         ┌──────────┐
  │ frontend │──────────▶│ backend  │         │ frontend │
  └──────────┘           │ (Django) │         └────┬─────┘
                         └────┬─────┘              │ HTTP + WS
                              │                    ▼
                     HTTP hop │  ◀── added    ┌──────────┐   in-process
                   failure    │      failure  │ backend  │   import
                   mode       ▼      mode     │ (Django) │◀───────────────┐
                         ┌──────────┐         └────┬─────┘                │
                         │ ML server│              │ Celery task         │
                         │ FastAPI  │              ▼                     │
                         │ + model  │         ┌──────────┐         ┌──────────┐
                         └──────────┘         │  worker  │────────▶│  model   │
                       (build + run +         │ (Celery) │ import  │  ml/ pkg │
                        monitor this)         └──────────┘         └──────────┘
```

The worker container **is** the ML runtime. `tasks.py` calls `run_pipeline()` directly. There
is no ML endpoint because there is no ML service.

### Compose services

| Service | Image | Job |
|---|---|---|
| `frontend` | node (dev) or a static build served by `web` | React dashboard (Vite) |
| `web` | `python:3.12-slim` + `backend/requirements.txt` | Django REST + Channels (ASGI via daphne). Uploads, detections, WebSocket push. **No model import.** |
| `worker` | `web` image + `worker-requirements.txt` (onnxruntime, numpy, opencv-headless) + `COPY ml/` | Celery worker. Imports `ml.inference.pipeline`, runs inference per tile, writes `Detection` rows, sends WS events. |
| `redis` | `redis:7-alpine` | Celery broker + Channels layer |
| `postgres` | `postgres:16-alpine` | `DetectionJob`, `Detection`, `AuditLogEntry` |

Scale inference with `docker compose up --scale worker=4`. Each worker is **~250 MB RAM**
(onnxruntime + numpy + opencv + Django), not ~2 GB.

### Turborepo layout

The Python side does not need to become JS. Keep it as a sibling the worker image builds from.

```
drishti/
├─ turbo.json
├─ package.json                 # workspaces: apps/*
├─ apps/
│  ├─ dashboard/                # the React frontend (was frontend/)
│  │  └─ package.json
│  └─ api/                      # Django project (was backend/)
│     ├─ Dockerfile             #   Turbo runs its dev/build via a script that shells python
│     ├─ requirements.txt
│     └─ ...
├─ services/
│  └─ worker/
│     ├─ Dockerfile             # FROM api image + worker-requirements.txt + COPY ml/
│     └─ worker-requirements.txt
├─ ml/                          # unchanged — inference package + weights (committed)
│  ├─ inference/                # run_pipeline() lives here
│  └─ models/exported/best_detector.onnx   (+ calibrator.pkl)
├─ compose.yaml
└─ docs/
```

Turbo orchestrates `dev` / `build` / `lint` / `test` across `apps/*`; the Python services are
driven by compose. The root `dev` task can itself run `docker compose up` so one command
brings the whole thing up.

---

## 3. Why the worker runs `best_detector.onnx`, not `best_detector.pt`

Clear a misconception first: **the ONNX file is not a shrunk or lossy model.**

| | `best_detector.pt` | `best_detector.onnx` (FP32) |
|---|---|---|
| Weights | Run 3, epoch 91 | **the same weights** |
| Architecture | YOLOv8s | the same graph, op-fused |
| Accuracy | — | **identical** — verified detection-for-detection on all 850 test tiles |
| File size on disk | 21.5 MB | 42.7 MB |
| Precision | FP32 | FP32 |

The `.pt` is actually the *smaller* file. "Real-sized model" is not a meaningful distinction —
both are the full-precision model. (The genuinely reduced one is `best_detector_int8.onnx`,
which we **excluded** because static quantization collapsed its accuracy to zero on faint
sonar targets.)

The choice between `.pt` and `.onnx` is entirely about **what must be installed to run it**:

| | `.pt` path | `.onnx` path |
|---|---|---|
| Python deps | `torch` + `ultralytics` | `onnxruntime` + `numpy` + `opencv-python-headless` |
| Installed size (site-packages) | **~4.4 GB** (`torch` alone) | **~210 MB** (`onnxruntime`) |
| RAM to `import` | ~1.5 GB before any inference | ~50 MB |
| CUDA / driver libs | pulled in even for CPU-only | none |
| Docker image | ~5–6 GB | ~600–800 MB |
| Cold start | slow (large pull, heavy import) | fast |
| Reproducible on the team's Win / Mac / Linux machines | fragile (CUDA, platform wheels) | yes |

For a compose stack with 4 worker replicas on one commodity box, that is the difference
between the stack fitting in **~2 GB** and needing **~12 GB+**. Accuracy is identical, so there
is no trade-off — the ONNX path is strictly better for server deployment.

`best_detector.pt` stays in the repo, but for **retraining and experimentation**, not serving.
`SonarDetector` already dispatches on file extension (`ml/inference/detector.py`) — the worker
points `DRISHTI_MODEL` at the `.onnx` and nothing else changes.

---

## 4. What the dev team builds

**No new endpoints between model and backend.** That interface is a function call. The
endpoints are frontend↔backend only — see `docs/API_ENDPOINTS.md` (upload, job status,
detections, export, and the `job_<id>` WebSocket group).

1. **`apps/api` (was `backend/`)** — fill the three stubs in `detections/`: `models.py`,
   `tasks.py` (imports `run_pipeline`), `views.py` + `serializers.py` + `consumers.py`. Worked
   example in `docs/INTEGRATION.md`.
2. **`services/worker/Dockerfile`** — `FROM` the api image, add `worker-requirements.txt`,
   `COPY ml/`. `CMD celery -A drishti_api worker`.
3. **`compose.yaml`** — the five services above.
4. **`apps/dashboard` (was `frontend/`)** — already scaffolded (Vite + React + react-leaflet +
   axios + recharts). Point `axios` at the `web` service; subscribe the detection socket to
   `job_<id>`.

### `worker-requirements.txt`

```
onnxruntime>=1.18
numpy>=1.26
opencv-python-headless>=4.10
scikit-learn>=1.4        # calibrator.pkl (per-class Platt models) unpickles against this
```

That is the entire ML runtime. No `torch`, no `ultralytics`.

---

## TL;DR

- **Don't split the model onto its own server.** It is a frozen, GPU-free Python library with
  one consumer — a separate service adds a deploy, a failure mode, and a network hop per tile,
  and buys nothing.
- **Run it in the Celery worker** in the same compose stack. Scale = more worker replicas.
- **The worker loads `best_detector.onnx`** — not because it is a smaller model (it is the same
  model, same accuracy) but because the ONNX runtime is ~210 MB against `torch`'s ~4.4 GB. That
  keeps the worker image ~800 MB and each replica ~250 MB RAM.
- **`best_detector.pt` stays for retraining only.**
