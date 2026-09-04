# DRISHTI — API surface

**Status:** the backend is scaffolded but every route file is empty. This document is the contract
both teams build against. Frontend can mock all of it today; backend fills the stubs to match.

Nothing here is negotiable without telling both teams — the record shape is already produced by
working code (`backend/reporting/schema.py`) and consumed by the report exporters.

---

## 0. The seam

The frontend never talks to the model. Everything crosses **one function**:

```python
from ml.inference.pipeline import run_pipeline

report = run_pipeline(image_path, source_file,
                      model_path="ml/models/exported/best_detector.onnx",
                      xtf=xtf_path, nav=nav_csv)
report["detections"]   # ← already the exact wire shape below
```

| Layer | Owns | Location |
|---|---|---|
| Model | preprocess → detect → calibrate → geotag | `ml/inference/pipeline.py` |
| Backend | jobs, persistence, auth, push, exports | `backend/detections/` |
| Frontend | upload, map, review queue | `frontend/src/` |

Swapping or retraining the model changes nothing above this line.

---

## 1. The canonical record

Produced by `DetectionRecord.to_dict()`. Every REST response and WebSocket event carries this object.

```json
{
  "detection_id": "a138c61c-59b4-4eb2-815b-205b9d15b74f",
  "job_id": "176771df-7360-4b31-a4ef-c92e3cddf393",
  "ping_id": "DATA0000106.H-PU#1401",
  "timestamp": "2015-08-12T09:08:28.650000+00:00",
  "latitude": 50.3937068,
  "longitude": -7.7132752,
  "class_label": "shipwreck",
  "confidence_score": 64.0,
  "bounding_geometry": {
    "bbox": [475.9, 11.4, 640.0, 153.1],
    "mask_polygon": [],
    "width_m": 55.2,
    "height_m": 47.63
  },
  "across_track_m": 80.29,
  "side": "starboard",
  "review_status": "pending_review",
  "source_file": "DATA0000106.H-PU"
}
```

**Gotchas the frontend will hit:**
- `bbox` is nested inside `bounding_geometry`, but `across_track_m` and `side` are **top-level**.
- `bbox` is `[x_min, y_min, x_max, y_max]` in **pixels of the source tile**, not normalised.
- `confidence_score` is **0–100**, already calibrated. Do not re-scale it.
- `ping_id` is `"{source_file}#{ping_number}"` — the audit pointer back into the sonar log.
- `class_label` ∈ `submarine_pipeline` · `shipwreck` · `mine_cylinder` · `ghost_net`.
  `crab_pot` is trained but filtered out; it will never appear.

### review_status

| Score | Value | Meaning |
|---|---|---|
| ≥ 80 | `auto_confirmed` | render on the map, no action needed |
| 30–79 | `pending_review` | **this is the review-queue feed** |
| < 30 | — | dropped before it reaches the API |
| — | `analyst_confirmed` / `analyst_rejected` | set by `PATCH`, never by the model |

Thresholds live in `backend/reporting/schema.py` (`AUTO_CONFIRM_THRESHOLD = 80.0`,
`REVIEW_FLOOR = 30.0`).

---

## 2. REST endpoints

Base path `/api/`. Wire in `backend/detections/urls.py` (empty), implement in
`backend/detections/views.py` (docstring only), serialize in
`backend/detections/serializers.py` (empty).

### `POST /api/upload/`
Accept a sonar image or log, create a job, enqueue the Celery task, return immediately.

**Request** — `multipart/form-data`

| Field | Required | Notes |
|---|---|---|
| `file` | ✅ | sonar image or waterfall TIF |
| `xtf` | — | `.xtf` log. **Without `xtf` or `nav` there are no coordinates** |
| `nav` | — | `navigation.csv` |
| `source_file` | — | defaults to the uploaded filename |

**202 Accepted**
```json
{ "job_id": "176771df-…", "status": "queued", "websocket": "/ws/jobs/176771df-…/" }
```
Client immediately opens the WebSocket and waits — it does not poll.

---

### `GET /api/jobs/<job_id>/`
Job status. Polling fallback for clients without a WebSocket.

```json
{
  "job_id": "176771df-…",
  "status": "running",
  "source_file": "DATA0000106.H-PU",
  "tiles_total": 42,
  "tiles_done": 17,
  "detection_count": 3,
  "created_at": "2026-09-02T10:14:00Z",
  "error": null
}
```
`status` ∈ `queued` · `running` · `done` · `failed`.

---

### `GET /api/jobs/`
List jobs, newest first — the dashboard history view.
Query: `?status=done&limit=20&offset=0`. Returns `{ "count": n, "results": [ …job objects… ] }`.

---

### `GET /api/detections/<job_id>/`
All detections for a job. What the map and results table render.

Query: `?review_status=pending_review` · `?class_label=shipwreck` · `?min_confidence=50`

```json
{
  "job_id": "176771df-…",
  "source_file": "DATA0000106.H-PU",
  "generated_at": "2026-09-02T10:16:12Z",
  "detection_count": 3,
  "class_counts": { "shipwreck": 1, "submarine_pipeline": 2 },
  "detections": [ /* records from §1 */ ]
}
```

This envelope is exactly `build_report()` in `backend/reporting/json_export.py` — reuse it, don't
rebuild it.

---

### `PATCH /api/detections/<detection_id>/`
The review queue writes here. The **only** endpoint that mutates a detection.

```json
{ "review_status": "analyst_confirmed", "note": "confirmed against 2019 survey" }
```
Accepts `analyst_confirmed` or `analyst_rejected` only. Returns the updated record.
Append to `AuditLogEntry` — these decisions become fine-tuning labels later, so keep who and when.

---

### `GET /api/export/<job_id>/?format=json|csv|geojson`
Report download. Delegate to the existing exporters, do not hand-roll.

| format | Content-Type | Producer |
|---|---|---|
| `json` | `application/json` | `json_export.build_report()` |
| `csv` | `text/csv` | `csv_export.write_csv()` — 18 flat columns |
| `geojson` | `application/geo+json` | `json_export.to_geojson()` — **Leaflet-ready** |

Set `Content-Disposition: attachment; filename="drishti_<job_id>.<ext>"`.

CSV flattens the bbox: `detection_id, job_id, ping_id, timestamp, latitude, longitude, class_label,
confidence_score, review_status, side, across_track_m, width_m, height_m, bbox_x_min, bbox_y_min,
bbox_x_max, bbox_y_max, source_file`.

GeoJSON `Feature.properties` carries `detection_id, class_label, confidence_score, review_status,
ping_id, timestamp, width_m, height_m` — enough to style pins and fill a popup without a second call.

---

### `GET /api/health/`
Dashboard status strip and deployment smoke test.

```json
{
  "status": "ok",
  "model": "best_detector.onnx",
  "runtime": "onnxruntime-cpu",
  "classes": ["submarine_pipeline", "shipwreck", "mine_cylinder", "ghost_net"],
  "calibrator_loaded": true
}
```

---

## 3. WebSocket

`ws://<host>/ws/jobs/<job_id>/` — Django Channels. Route in `backend/detections/routing.py`
(empty), implement in `backend/detections/consumers.py` (docstring only). Group name: `job_<job_id>`.

Frontend: `frontend/src/api/websocket.js` → `frontend/src/hooks/useDetectionSocket.js`.

**`detection.partial`** — one per finished tile, this is what makes the demo feel live:
```json
{ "type": "detection.partial", "tile_index": 3, "tiles_total": 42,
  "detections": [ /* records from §1 */ ] }
```

**`detection.complete`**
```json
{ "type": "detection.complete", "job_id": "176771df-…", "detection_count": 3,
  "report_url": "/api/export/176771df-…/?format=json" }
```

**`detection.failed`**
```json
{ "type": "detection.failed", "job_id": "176771df-…", "error": "could not read image" }
```

Accumulate `detection.partial` client-side; don't re-fetch on every tile. Fall back to polling
`GET /api/jobs/<id>/` if the socket drops.

---

## 4. Where each piece goes

| File | State | To do |
|---|---|---|
| `backend/detections/models.py` | docstring only | `DetectionJob`, `Detection`, `AuditLogEntry` — field names identical to §1 so the serializer is `fields = "__all__"` |
| `backend/detections/tasks.py` | docstring only | `run_inference_job` — tile → `run_pipeline` → `bulk_create` → `group_send`. Worked example in `docs/INTEGRATION.md` |
| `backend/detections/views.py` | docstring only | the six routes above |
| `backend/detections/urls.py` | **empty** | urlpatterns |
| `backend/detections/serializers.py` | **empty** | DRF serializers |
| `backend/detections/routing.py` | **empty** | `websocket_urlpatterns` |
| `backend/detections/consumers.py` | docstring only | group join/leave + forward |
| `backend/reporting/*` | ✅ **done** | reuse as-is |
| `backend/geotagging/*` | ✅ **done** | reuse as-is |
| `ml/inference/pipeline.py` | ✅ **done** | call it, don't modify it |

---

## 5. Turborepo layout — one thing to decide first

Turborepo orchestrates **JS/TS workspaces**. A Django backend can live in the monorepo, but `turbo`
won't build or cache it the way it does a Next app. Two workable shapes:

**A — keep Django, wrap it for turbo** *(recommended; nothing is rewritten)*
```
apps/
  web/          package.json   → Vite/Next frontend
  api/          package.json   → thin wrapper whose scripts shell out to Python
                manage.py, drishti_api/, detections/, reporting/, geotagging/
packages/
  api-types/    shared TS types generated from §1
ml/  edge/  docs/            (outside the turbo pipeline)
```
`apps/api/package.json` just needs `"dev": "python manage.py runserver"` and
`"worker": "celery -A drishti_api worker -l info"` so `turbo dev` starts everything.

**B — Next.js API routes, Python as a sidecar**
Frontend and REST in one Next app; a small FastAPI service (`edge/onnx_runtime_server.py` is already
one) does inference over HTTP. Fewer languages in the pipeline, but you rewrite the job/queue layer
and lose the Channels WebSocket.

**The contract in §1–3 is identical either way** — start the frontend against mocks now, the choice
doesn't block it.

One shared type package is worth the effort: generate TS interfaces from §1 once, import in both
`apps/web` and any Node code, so a field rename breaks the build instead of the demo.

---

## 6. Rules

1. **Don't change §1 or `backend/reporting/schema.py`.** Frontend, exporters and the ML side all
   depend on it. Change it in one place and tell both teams.
2. **Don't call the model from the frontend.** Everything goes through `/api/`.
3. **Don't set `preprocess=False`** in `run_pipeline` — the shipped model needs Lee+CLAHE.
4. **Don't touch `detector_conf=0.10`.** Module 2's per-class gate is the real threshold.
5. **`review_status` is computed, not stored** — except when an analyst sets `analyst_*` via PATCH.
6. Positional accuracy is a **search area, not a survey fix**. Two nav paths can disagree by ~100 m.
   Render an uncertainty radius, not a pinpoint.

## 7. See also

- `docs/INTEGRATION.md` — how to call the model, worked Celery task
- `docs/api_contract.md` — the original frozen contract
- `docs/MODEL_DOSSIER.html` — what the model does and how well
