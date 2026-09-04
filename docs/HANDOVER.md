# DRISHTI — work handover, backend & frontend

**Audited against the actual repo, not the status report.** Two claims in the status PDF are
optimistic, and one contract divergence will break the frontend if it isn't fixed before they
build. Read §1 and §2 before assigning anything.

Reference docs (all already in `docs/`):
`DEPLOYMENT_RATIONALE.md` · `TURBOREPO_SETUP.md` · `INTEGRATION.md` · `API_ENDPOINTS.md`

---

## 1. Status audit — what is actually done

### ✅ Genuinely done and working

| Item | Evidence |
|---|---|
| pnpm + Turborepo monorepo | `package.json`, `turbo.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml` at root |
| `apps/api` Django project | `settings.py` 224 lines — Postgres, Redis, Channels, Celery, CORS, DRF all configured |
| `sys.path` wired to the ML package | `settings.py` inserts `REPO_ROOT` and `REPO_ROOT/ml/scripts` — `import ml.inference.pipeline` will work |
| Model paths configured | `DRISHTI_MODEL` and `DRISHTI_CALIBRATOR` read from env with correct defaults |
| `MEDIA_ROOT` configured | uploads have somewhere to land |
| ASGI + Daphne | `asgi.py` with `ProtocolTypeRouter`, HTTP + WebSocket |
| Celery app | `celery.py` with Django autodiscovery; `architecture_test` task proven working |
| WebSocket consumer | `consumers.py` — event names `detection.partial` / `.complete` / `.failed` **match the contract exactly** |
| WS route | `routing.py` → `/ws/jobs/<job_id>/` |
| Postgres + initial migration | `detections/migrations/0001_initial.py` applied |
| Health endpoint | `GET /api/health/` |
| `geotagging/` + `reporting/` moved in | moved to `ml/geotagging/` + `ml/reporting/` (they are Module 3 + the report exporters — ML-pipeline concerns, imported by `run_pipeline`) |
| `ml/` left untouched | correct — it stays a shared package |

**Credit where due:** `settings.py`, `asgi.py`, `celery.py`, `consumers.py` and `routing.py` follow
`TURBOREPO_SETUP.md` closely and are correct. The plumbing works.

### ⚠️ Report says done — actually not

| Report claim | Reality |
|---|---|
| "Dockerfiles … created" | **`apps/api/Dockerfile` is 0 bytes — empty.** No web image exists. |
| "compose services currently defined: postgres, redis, **web**, **worker**" | `compose.yaml` defines **only `postgres` and `redis`**. |
| "`services/worker/` Celery worker Docker definition" | **`services/` directory does not exist.** |
| "Separate Celery worker architecture prepared" | The Celery *app* is configured, but there is no worker container — it runs from the local venv only. |

Not a criticism of the work done — a correction so nobody plans against Docker that isn't there.
Full file contents for all three are in **`TURBOREPO_SETUP.md` Phases 3, 4 and 6**.

### 🧹 Cleanup

`backend/` and `frontend/` still exist, containing only stale `__pycache__/*.pyc`. The real code
moved to `apps/`. Delete them:

```bash
git rm -r --cached backend frontend 2>/dev/null; rm -rf backend frontend
```

---

## 2. 🔴 BLOCKING — the Detection model diverges from the frozen contract

`run_pipeline()` returns records in the exact shape `API_ENDPOINTS.md` specifies. The current model
uses different field names, so `Detection(job=job, **record)` **will raise `TypeError`**, and the
frontend cannot build against the contract.

| Contract field | Current model | Problem |
|---|---|---|
| `class_label` | `class_name` | rename |
| `confidence_score` (0–100) | `confidence` (0–1) | rename **and** rescale |
| `bounding_geometry` (JSON) | `x1 y1 x2 y2` floats | wrong shape — loses `mask_polygon`, `width_m`, `height_m` |
| `source_file` | `source_tile` | rename |
| `detection_id` (UUID PK) | implicit auto-int PK | add |
| `ping_id` | — | **missing** — the audit pointer back to the exact sonar ping |
| `timestamp` | — | **missing** |
| `review_status` | — | **missing — the entire review queue depends on this** |
| `side`, `across_track_m` | — | missing |

**Fix it now.** The database has one migration and zero real rows — this is the cheapest it will
ever be. Replace `apps/api/detections/models.py`:

```python
import uuid

from django.db import models


class DetectionJob(models.Model):
    STATUS = [("queued", "queued"), ("running", "running"),
              ("completed", "completed"), ("failed", "failed")]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_file = models.CharField(max_length=255)          # original log/image name
    input_path = models.CharField(max_length=1024)          # where it landed on disk
    xtf_path = models.CharField(max_length=1024, blank=True, default="")
    nav_path = models.CharField(max_length=1024, blank=True, default="")
    status = models.CharField(max_length=16, choices=STATUS, default="queued")
    progress = models.FloatField(default=0.0)
    celery_task_id = models.CharField(max_length=128, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)


class Detection(models.Model):
    detection_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(DetectionJob, related_name="detections", on_delete=models.CASCADE)

    ping_id = models.CharField(max_length=255, blank=True, default="")
    timestamp = models.DateTimeField(null=True, blank=True)

    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    class_label = models.CharField(max_length=64)
    confidence_score = models.FloatField()                  # 0–100, calibrated
    bounding_geometry = models.JSONField(default=dict)      # {bbox, mask_polygon, width_m, height_m}

    across_track_m = models.FloatField(null=True, blank=True)
    side = models.CharField(max_length=12, blank=True, default="")

    review_status = models.CharField(max_length=24, default="pending_review")
    source_file = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["job", "review_status"])]


class AuditLogEntry(models.Model):
    job = models.ForeignKey(DetectionJob, related_name="audit", on_delete=models.CASCADE)
    detection = models.ForeignKey(Detection, related_name="audit", null=True, blank=True,
                                 on_delete=models.CASCADE)
    action = models.CharField(max_length=255)
    actor = models.CharField(max_length=128, blank=True, default="")
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
```

Then reset the migration (safe — no data yet):

```bash
cd apps/api
rm detections/migrations/0001_initial.py
python manage.py makemigrations detections
python manage.py migrate
```

---

## 3. BACKEND — work package

Do these **in order**. B1–B3 unblock the frontend; don't reorder them.

### B1 · Fix the Detection model  🔴 blocking
Exactly §2 above. Nothing else starts until this lands.

### B2 · Serializers match the contract 1:1
`apps/api/detections/serializers.py` — field names must equal the JSON keys so no translation layer
is ever needed.

```python
from rest_framework import serializers
from .models import Detection, DetectionJob


class DetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detection
        exclude = ("job", "created_at")


class DetectionJobSerializer(serializers.ModelSerializer):
    detection_count = serializers.IntegerField(source="detections.count", read_only=True)

    class Meta:
        model = DetectionJob
        fields = ("id", "source_file", "status", "progress", "error_message",
                  "created_at", "started_at", "completed_at", "detection_count")
```

### B3 · The five routes  🔴 blocking the frontend
Currently only `POST/GET /api/jobs/` and `GET /api/jobs/<id>/` exist. Per `API_ENDPOINTS.md`:

| Method | Route | Status | Notes |
|---|---|---|---|
| POST | `/api/upload/` | **build** | multipart `file` (+ optional `xtf`, `nav`), save to `MEDIA_ROOT`, create job, `.delay()`, return `202 {job_id, status}` |
| GET | `/api/jobs/<job_id>/` | ✅ exists | keep |
| GET | `/api/jobs/` | ✅ exists | keep |
| GET | `/api/detections/<job_id>/` | **build** | list serialized detections for a job |
| PATCH | `/api/detections/<detection_id>/` | **build** | `{review_status: analyst_confirmed\|analyst_rejected, actor}` → also write an `AuditLogEntry` |
| GET | `/api/export/<job_id>/?format=json\|csv\|geojson` | **build** | reuse `reporting/json_export.py` + `csv_export.py` — do **not** re-implement |
| GET | `/api/health/` | ✅ exists | keep |

Keep `POST /api/jobs/` with `input_path` as a dev convenience if you like, but `/api/upload/` is
what the frontend calls.

### B4 · Wire the ML seam in `tasks.py`
The TODO comment at the seam is correct and well-placed — now fill it. Full worked example in
**`INTEGRATION.md` §3b**. The shape:

```python
from pathlib import Path
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.conf import settings
from ml.inference.pipeline import run_pipeline

# inside run_detection_job, replacing the seam comment:
layer = get_channel_layer()
group = f"job_{job_id}"

report = run_pipeline(
    job.input_path,
    job.source_file,
    model_path=settings.DRISHTI_MODEL,
    calibrator_path=settings.DRISHTI_CALIBRATOR,
    xtf=Path(job.xtf_path) if job.xtf_path else None,
    nav=Path(job.nav_path) if job.nav_path else None,
)
recs = report["detections"]

Detection.objects.bulk_create([
    Detection(job=job, **{k: v for k, v in r.items() if k != "job_id"}) for r in recs
])
async_to_sync(layer.group_send)(group, {
    "type": "detection.partial", "tile_index": 0, "detections": recs,
})
```

**Three things not to get wrong** (all in `INTEGRATION.md`):
- Leave `preprocess=True` (the default). The shipped model was trained on Lee+CLAHE input; turning it off silently degrades every prediction.
- Don't touch `detector_conf` — Module 2's per-class gate is the real cut.
- The parseable XTF lives in `xtf/xtf-navigation/*.H-PU.xtf`, **not** `xtf/xtf-data/*.H.xtf`.

Verify the ML path standalone *before* wiring it:

```bash
python -m ml.inference.pipeline \
  --image demo/tiles/synth_ghost_net_00002.jpg --source-file demo \
  --model ml/models/exported/best_detector.onnx --no-preprocess --out /tmp/r
# expect: ghost_net ~95%, auto_confirmed
```

### B5 · Tile large uploads
`run_pipeline()` handles **one image**. A full sonar waterfall must be tiled first — copy the
640/stride-512 loop and the box→full-image mapping from `ml/scripts/run_aurora_survey.py`. Send one
`detection.partial` per tile with the real `tile_index` so the progress bar means something.

### B6 · Finish Docker
Three files, full contents in `TURBOREPO_SETUP.md`:
- `apps/api/Dockerfile` — Phase 3 (currently empty)
- `services/worker/Dockerfile` — Phase 4 (`FROM drishti-api`, swap `CMD` to `celery`)
- `compose.yaml` — Phase 6, add the `web` and `worker` services plus the shared `media` volume

Target: `docker compose up --build` brings up all four services and
`docker compose exec web python manage.py migrate` succeeds.

### B7 · Point the worker at the ONNX model
Set `DRISHTI_MODEL` to **`ml/models/exported/best_detector.onnx`**, not the `.pt`. Same weights,
same accuracy — but the ONNX path needs only `onnxruntime` (~210 MB) instead of torch (~4.4 GB),
which keeps each worker replica near 250 MB RAM. Reasoning in `DEPLOYMENT_RATIONALE.md` §3.
Confirm at runtime: `import sys; "torch" in sys.modules` → `False`.

---

## 4. FRONTEND — work package

`apps/dashboard` is a Vite + React skeleton. Build against `API_ENDPOINTS.md` — **not** against
whatever the backend currently returns, which is mid-change. Mock the five routes and you can work
in parallel from today.

### F1 · API client + socket hook
`src/api/client.js`, `src/api/websocket.js`, `src/hooks/useDetectionSocket.js` — full contents in
`TURBOREPO_SETUP.md` Phase 5. Vite proxies `/api` → `:8000` and `/ws` → `ws://:8000`.

The WebSocket is **already live on the backend** and matches the contract:
`detection.partial` (carries `tile_index`, `detections[]`) · `detection.complete` · `detection.failed`.

### F2 · Five screens

| Screen | Does | Calls |
|---|---|---|
| **Upload** | drag a sonar image/log + optional XTF & nav CSV → create job → go to live feed | `POST /api/upload/` |
| **Live feed** | subscribe to `job_<id>`, append detections as tiles complete, show progress + running count | `WS /ws/jobs/<id>/` |
| **Map** | Leaflet; one marker per detection, coloured by `review_status`, popup with class + score | `GET /api/detections/<job_id>/` |
| **Review queue** | every detection with `review_status == "pending_review"`, confirm/reject | `PATCH /api/detections/<id>/` |
| **Export** | download JSON / CSV / GeoJSON | `GET /api/export/<job_id>/?format=` |

### F3 · 🔴 Draw uncertainty circles, not bare pins
A correctness requirement, not styling. A detection is a **search area, not a survey fix** — our two
navigation paths placed the same target ~122 m apart. Render each detection as an `L.circle` sized to
its positional uncertainty with the marker inside it. A bare pin implies metre accuracy we do not
have, and someone could act on it.

### F4 · Colour by review band
Read the band from the data; don't hardcode thresholds in the UI.
- `auto_confirmed` (≥ 80 %) → green
- `pending_review` (30–80 %) → amber — and these are the **only** rows in the review queue
- `analyst_confirmed` → solid green · `analyst_rejected` → grey / struck through

### F5 · Render four classes, not five
`submarine_pipeline`, `shipwreck`, `mine_cylinder`, `ghost_net`. `crab_pot` is trained as a hard
negative and filtered out of the product — if one ever appears in a response, treat it as a bug.

---

## 5. Definition of done — the shared smoke test

Both of you are finished when this passes end to end:

```bash
docker compose up --build -d
docker compose exec web python manage.py migrate

curl -F "file=@demo/tiles/synth_ghost_net_00002.jpg" http://localhost:8000/api/upload/
# → 202 {"job_id": "...", "status": "queued"}

curl http://localhost:8000/api/detections/<job_id>/
# → [{ "class_label": "ghost_net", "confidence_score": 95.3,
#      "review_status": "auto_confirmed", "bounding_geometry": {...}, ... }]
```

- [ ] Detection model field names match `API_ENDPOINTS.md` exactly
- [ ] All five routes respond
- [ ] `tasks.py` calls `run_pipeline` and persists real detections
- [ ] WebSocket delivers `detection.partial` to a connected browser before the job completes
- [ ] `docker compose up` brings up postgres, redis, web, worker
- [ ] Worker carries no torch — `docker compose exec worker python -c "import sys; print('torch' in sys.modules)"` → `False`
- [ ] Dashboard renders the detection on a Leaflet map with an uncertainty circle
- [ ] Review queue lists `pending_review` items and PATCH updates them
- [ ] Export returns all three formats

---

## 6. Rules neither of you may change alone

1. **`API_ENDPOINTS.md` is frozen.** Any change needs both sides told — the record shape is produced
   by working ML code and consumed by the report exporters.
2. **Do not modify anything under `ml/`.** It is a shared package with a validated model behind it.
   If you need something from it, it goes through `run_pipeline()`.
3. **Never pass `preprocess=False`** on a raw sonar image.
4. **Quote metrics from `docs/METRICS_PROTOCOL.md`** — mAP@50 is **0.641**, not 0.580. The lower
   figure came from a wrong evaluation protocol.
