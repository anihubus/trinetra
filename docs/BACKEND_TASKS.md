# DRISHTI — Backend tasks

**You own:** `apps/api/`, `services/worker/`, `compose.yaml`.
**You must not touch:** anything under `ml/`, and `docs/API_ENDPOINTS.md` (frozen contract).

Work top to bottom. Steps 1–5 unblock the frontend — do those first, tell them when they land.

---

## 0 · Setup (once)

```bash
git clone https://github.com/Rehan9599/Sonar-Drishti.git
cd Sonar-Drishti
git checkout -b feat/backend-api

# node side (root workspace)
corepack enable
pnpm install

# python side
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash
# or: .venv\Scripts\activate           # PowerShell
pip install -r apps/api/requirements.txt
```

### 0.1 · Create `.env` — this is why `migrate` currently fails

There is no `.env` in the repo, only `.env.example`. Without it Django falls back to
`config()` defaults and the Postgres password won't match the existing volume.

```bash
cp .env.example .env
```

### 0.2 · Reset the database volume

Postgres bakes `POSTGRES_PASSWORD` into its data directory on **first init only**. The existing
`pgdata` volume has a stale password. There is no real data in it yet, so destroy it:

```bash
docker compose down -v
docker compose up -d              # postgres + redis re-init from .env
```

If `migrate` still fails after this, something non-Docker owns port 5432:

```bash
netstat -ano | findstr :5432
```
If the PID isn't docker, stop the local PostgreSQL service (`services.msc` →
`postgresql-x64-16` → Stop), or set `POSTGRES_PORT=5433` in `.env` and map `5433:5432` in
`compose.yaml`.

### 0.3 · Migrate and verify

```bash
cd apps/api
python manage.py migrate
python manage.py runserver 8000          # or: python -m daphne -b 0.0.0.0 -p 8000 drishti_api.asgi:application
curl http://localhost:8000/api/health/   # → {"status":"ok","service":"drishti-api"}
```

In a second terminal:
```bash
cd apps/api && celery -A drishti_api worker -l info
```

### 0.4 · Delete the stray lockfile

Someone ran `npm` inside `apps/api`. This repo is pnpm-only.

```bash
rm apps/api/package-lock.json
```

---

## What is already done — do not redo

> **Repo-structure fix already on `main`:** `geotagging/` and `reporting/` were left at
> `apps/api/` by the reorg but four files still imported them as `backend.*` (which no longer
> exists), so `run_pipeline()` was broken. They are now `ml/geotagging/` + `ml/reporting/`
> (Module 3 + the report exporters — ML-pipeline concerns), all imports rewritten to `ml.*`,
> and `requirements.txt` files fixed. `apps/api/detections/` code never imports these directly
> — it reaches them only through `run_pipeline()`. Nothing for you to do here; just don't be
> surprised the paths differ from the status PDF.

| Item | State |
|---|---|
| Turborepo root (`package.json`, `turbo.json`, `pnpm-workspace.yaml`) | ✅ |
| `settings.py` — Postgres, Redis, Channels, Celery, CORS, DRF | ✅ 224 lines |
| `sys.path` → `REPO_ROOT` + `ml/scripts` (so `import ml.inference.pipeline` works) | ✅ |
| `DRISHTI_MODEL`, `DRISHTI_CALIBRATOR`, `MEDIA_ROOT` | ✅ configured |
| `asgi.py`, `celery.py`, `wsgi.py`, `drishti_api/urls.py` | ✅ |
| `consumers.py` + `routing.py` — WS event names match the contract | ✅ |
| **`models.py`** — all 13 contract fields present | ✅ **already fixed** |
| `migrations/0001_initial.py` regenerated | ✅ |
| `ml/geotagging/` + `ml/reporting/` | ✅ Module 3 + report exporters — imported by `run_pipeline`, verified working |
| `GET /api/health/` | ✅ |

---

## 1 · Fix `serializers.py`

`DetectionJobSerializer` still lists `input_path` and nests every detection (heavy). Replace the
whole file:

```python
from rest_framework import serializers

from .models import AuditLogEntry, Detection, DetectionJob


class DetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detection
        exclude = ("job", "created_at")


class AuditLogEntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditLogEntry
        fields = "__all__"


class DetectionJobSerializer(serializers.ModelSerializer):
    detection_count = serializers.IntegerField(source="detections.count", read_only=True)

    class Meta:
        model = DetectionJob
        fields = ("id", "source_file", "status", "progress", "error_message",
                  "created_at", "started_at", "completed_at", "detection_count")
```

`exclude` rather than `fields = "__all__"` keeps the `job` FK out of the payload so the JSON
matches the contract record exactly.

---

## 2 · `views.py` — replace entirely

Current `views.py` takes `input_path` as a JSON string. The frontend uploads a **file**. Replace
the whole file:

```python
import csv
import io
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import AuditLogEntry, Detection, DetectionJob
from .serializers import DetectionJobSerializer, DetectionSerializer
from .tasks import run_detection_job


def _save_upload(f, subdir="uploads"):
    dest_dir = Path(settings.MEDIA_ROOT) / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f.name
    with open(dest, "wb") as out:
        for chunk in f.chunks():
            out.write(chunk)
    return str(dest)


@api_view(["POST"])
def upload(request):
    """multipart: file (required), xtf (optional), nav (optional)."""
    f = request.FILES.get("file")
    if not f:
        return Response({"detail": "file is required"}, status=400)

    job = DetectionJob.objects.create(
        source_file=f.name,
        input_path=_save_upload(f),
        xtf_path=_save_upload(request.FILES["xtf"], "nav") if request.FILES.get("xtf") else "",
        nav_path=_save_upload(request.FILES["nav"], "nav") if request.FILES.get("nav") else "",
    )
    async_result = run_detection_job.delay(str(job.id))
    job.celery_task_id = async_result.id
    job.save(update_fields=["celery_task_id"])

    return Response({"job_id": str(job.id), "status": job.status},
                    status=status.HTTP_202_ACCEPTED)


@api_view(["GET"])
def job_list(request):
    qs = DetectionJob.objects.order_by("-created_at")
    return Response(DetectionJobSerializer(qs, many=True).data)


@api_view(["GET"])
def job_detail(request, job_id):
    try:
        job = DetectionJob.objects.get(id=job_id)
    except DetectionJob.DoesNotExist:
        return Response({"detail": "job not found"}, status=404)
    return Response(DetectionJobSerializer(job).data)


@api_view(["GET"])
def detections(request, job_id):
    qs = Detection.objects.filter(job_id=job_id).order_by("created_at")
    return Response(DetectionSerializer(qs, many=True).data)


@api_view(["PATCH"])
def review(request, detection_id):
    try:
        d = Detection.objects.get(detection_id=detection_id)
    except Detection.DoesNotExist:
        return Response({"detail": "detection not found"}, status=404)

    new = request.data.get("review_status")
    if new not in ("analyst_confirmed", "analyst_rejected"):
        return Response(
            {"detail": "review_status must be analyst_confirmed or analyst_rejected"},
            status=400,
        )

    d.review_status = new
    d.save(update_fields=["review_status"])
    AuditLogEntry.objects.create(
        job=d.job, detection=d, action=new,
        actor=request.data.get("actor", ""),
        details={"previous": "pending_review"},
    )
    return Response(DetectionSerializer(d).data)


@api_view(["GET"])
def export(request, job_id):
    fmt = request.query_params.get("format", "json")
    qs = Detection.objects.filter(job_id=job_id).order_by("created_at")
    recs = DetectionSerializer(qs, many=True).data

    if fmt == "geojson":
        return Response({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [r["longitude"], r["latitude"]]},
                    "properties": {k: v for k, v in r.items()
                                   if k not in ("latitude", "longitude")},
                }
                for r in recs if r.get("latitude") is not None
            ],
        })

    if fmt == "csv":
        buf = io.StringIO()
        if recs:
            w = csv.DictWriter(buf, fieldnames=list(recs[0].keys()), extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow({k: (v if not isinstance(v, dict) else str(v))
                            for k, v in r.items()})
        resp = HttpResponse(buf.getvalue(), content_type="text/csv")
        resp["Content-Disposition"] = f'attachment; filename="{job_id}.csv"'
        return resp

    return Response({"job_id": str(job_id), "detection_count": len(recs),
                     "detections": recs})
```

---

## 3 · `urls.py` — replace entirely

```python
from django.urls import path

from . import views

urlpatterns = [
    path("upload/", views.upload),
    path("jobs/", views.job_list),
    path("jobs/<uuid:job_id>/", views.job_detail),
    path("detections/<uuid:job_id>/", views.detections),
    path("detections/<uuid:detection_id>/review/", views.review),
    path("export/<uuid:job_id>/", views.export),
]
```

> Route order matters: `detections/<job_id>/` and `detections/<detection_id>/review/` differ only
> by the trailing segment, so Django resolves them correctly — but keep `review/` second.

**Tell the frontend dev the moment this lands.** These five routes are what they build against.

---

## 4 · `tasks.py` — wire the ML seam

The `# ML INTEGRATION SEAM` comment is in the right place. Verify the ML path standalone **first**:

```bash
cd /d/Sonar-Drishti
python -m ml.inference.pipeline \
  --image demo/tiles/synth_ghost_net_00002.jpg --source-file demo \
  --model ml/models/exported/best_detector.onnx --no-preprocess --out /tmp/r
# expect: ghost_net ~95%, auto_confirmed
```

Then replace the seam block inside `run_detection_job`:

```python
# ---- at the top of tasks.py ----
from pathlib import Path

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.conf import settings

from ml.inference.pipeline import run_pipeline

# ---- replacing the seam comment, inside the try: ----
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
    Detection(job=job, **{k: v for k, v in r.items() if k != "job_id"})
    for r in recs
])

async_to_sync(layer.group_send)(group, {
    "type": "detection.partial",
    "tile_index": 0,
    "detections": recs,
})

job.progress = 1.0
job.save(update_fields=["progress"])
```

And after the status flips to `completed`, push the final event:

```python
async_to_sync(layer.group_send)(group, {
    "type": "detection.complete",
    "job_id": str(job.id),
    "total": job.detections.count(),
})
```

In the `except` block, add:

```python
async_to_sync(get_channel_layer().group_send)(f"job_{job_id}", {
    "type": "detection.failed",
    "job_id": str(job_id),
    "error": str(exc),
})
```

### Three things not to get wrong

1. **Leave `preprocess` at its default (`True`).** The shipped model was trained on Lee+CLAHE
   input. Passing `False` on a raw sonar image silently degrades every prediction.
2. **Do not change `detector_conf`.** Module 2's per-class gate is the real cut; the detector
   threshold is deliberately loose at 0.10.
3. **The parseable XTF is `xtf/xtf-navigation/*.H-PU.xtf`**, *not* `xtf/xtf-data/*.H.xtf`.
   The latter raises `KeyError: no XTF pings parsed`.

---

## 5 · Tile large uploads

`run_pipeline()` handles **one image**. A full sonar waterfall must be tiled first. Copy the
tiling loop and the box→full-image coordinate mapping from `ml/scripts/run_aurora_survey.py`
(640 px window, stride 512), then loop:

```python
for i, tile_path in enumerate(tiles):
    report = run_pipeline(tile_path, job.source_file, ...)
    recs = report["detections"]
    Detection.objects.bulk_create([...])
    async_to_sync(layer.group_send)(group, {
        "type": "detection.partial", "tile_index": i, "detections": recs,
    })
    job.progress = (i + 1) / len(tiles)
    job.save(update_fields=["progress"])
```

One `detection.partial` per tile with the real `tile_index` is what makes the frontend progress
bar meaningful.

---

## 6 · `apps/api/Dockerfile` — currently 0 bytes

```dockerfile
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

COPY apps/api/requirements.txt apps/api/requirements.txt
RUN pip install --no-cache-dir -r apps/api/requirements.txt

COPY ml/ ml/
COPY apps/api/ apps/api/

WORKDIR /repo/apps/api
ENV DJANGO_SETTINGS_MODULE=drishti_api.settings PYTHONPATH=/repo

EXPOSE 8000
CMD ["python", "-m", "daphne", "-b", "0.0.0.0", "-p", "8000", "drishti_api.asgi:application"]
```

Build context is the **repo root** (it copies `ml/`), so `compose.yaml` must use
`context: .` with `dockerfile: apps/api/Dockerfile`.

---

## 7 · `services/worker/` — does not exist yet

```bash
mkdir -p services/worker
```

`services/worker/Dockerfile`:

```dockerfile
# Reuse the api image; only the command differs.
ARG API_IMAGE=drishti-api:latest
FROM ${API_IMAGE}

WORKDIR /repo/apps/api
CMD ["celery", "-A", "drishti_api", "worker", "-l", "info", "--concurrency", "2"]
```

---

## 8 · `compose.yaml` — add `web` and `worker`

Keep the existing `postgres` and `redis` blocks (they're correct, with healthchecks). Append:

```yaml
  web:
    build:
      context: .
      dockerfile: apps/api/Dockerfile
    image: drishti-api:latest
    env_file: .env
    environment:
      POSTGRES_HOST: postgres
      REDIS_URL: redis://redis:6379/0
    volumes:
      - media:/repo/apps/api/media
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
    ports:
      - "8000:8000"

  worker:
    build:
      context: .
      dockerfile: services/worker/Dockerfile
      args:
        API_IMAGE: drishti-api:latest
    env_file: .env
    environment:
      POSTGRES_HOST: postgres
      REDIS_URL: redis://redis:6379/0
      DRISHTI_MODEL: /repo/ml/models/exported/best_detector.onnx
      DRISHTI_CALIBRATOR: /repo/ml/models/exported/calibrator.pkl
    volumes:
      - media:/repo/apps/api/media
    depends_on:
      - web
      - redis
```

And extend the `volumes:` block at the bottom:

```yaml
volumes:
  pgdata:
  media:
```

**Two things that matter here:**
- `POSTGRES_HOST: postgres` overrides the `localhost` in `.env` — inside the network, containers reach the *service name*.
- `DRISHTI_MODEL` points at the **`.onnx`**, not the `.pt`. Same weights, same accuracy, but the ONNX path needs only `onnxruntime` (~210 MB) instead of torch (~4.4 GB), keeping each worker replica near 250 MB RAM. Reasoning in `docs/DEPLOYMENT_RATIONALE.md` §3.

---

## 9 · Test everything

```bash
docker compose up --build -d
docker compose exec web python manage.py migrate

# upload a committed demo tile
curl -F "file=@demo/tiles/synth_ghost_net_00002.jpg" http://localhost:8000/api/upload/
# → 202 {"job_id":"...","status":"queued"}

JOB=<paste job_id>
curl http://localhost:8000/api/jobs/$JOB/
curl http://localhost:8000/api/detections/$JOB/
# → [{"class_label":"ghost_net","confidence_score":95.3,
#     "review_status":"auto_confirmed","bounding_geometry":{...},...}]

curl "http://localhost:8000/api/export/$JOB/?format=geojson"

# confirm the worker is torch-free
docker compose exec worker python -c "import sys; print('torch' in sys.modules)"
# → False
```

### Checklist

- [ ] `.env` created; `migrate` succeeds
- [ ] `serializers.py` — no `input_path`, detections use `exclude`
- [ ] All six routes respond (`upload`, `jobs`, `jobs/<id>`, `detections/<job>`, `review`, `export`)
- [ ] `tasks.py` calls `run_pipeline` and `bulk_create`s real detections
- [ ] WebSocket delivers `detection.partial` before the job completes
- [ ] `docker compose up` brings up **four** services
- [ ] `torch in sys.modules` → `False` in the worker
- [ ] Export returns json, csv and geojson

---

## 10 · Open the PR

```bash
cd /d/Sonar-Drishti
git status                      # confirm no .env, no *.pyc, no media/ staged
git add -A
git commit -m "feat(api): upload, detections, review and export endpoints + ML pipeline wiring

- serializers aligned to the frozen API contract (docs/API_ENDPOINTS.md)
- POST /api/upload/ accepts multipart file + optional xtf/nav
- GET /api/detections/<job_id>/, PATCH review, GET /api/export/ (json|csv|geojson)
- tasks.py calls ml.inference.run_pipeline and persists DetectionRecords
- per-tile detection.partial WebSocket events with real tile_index
- apps/api/Dockerfile + services/worker/Dockerfile
- compose.yaml: added web and worker services with shared media volume
- worker loads best_detector.onnx (torch-free, ~250 MB/replica)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"

git push -u origin feat/backend-api
gh pr create --title "Backend: API endpoints + ML pipeline wiring" --body "$(cat <<'EOF'
Implements the backend half of docs/HANDOVER.md.

## Done
- serializers match `docs/API_ENDPOINTS.md` exactly
- all six REST routes
- Celery task calls `ml.inference.run_pipeline()` — no ML microservice, no network hop
- per-tile WebSocket progress events
- Docker: api image, worker image (reuses api), compose web + worker

## Verified
- `curl -F file=@demo/tiles/synth_ghost_net_00002.jpg .../api/upload/` → 202
- `GET /api/detections/<job>/` returns ghost_net 95.3% auto_confirmed
- worker has no torch: `'torch' in sys.modules` → False
- all four compose services healthy

## Not in scope
- authentication (Phase 2)
- the public `/api/public/debris/` endpoint (Mode C)

EOF
)"
```

> **Before pushing:** `.gitignore` must cover `.env` and `apps/api/media/`. Run `git status`
> after `git add -A` and read the list. Never commit `.env` — it holds the DB password.

---

## Rules

1. **`docs/API_ENDPOINTS.md` is frozen.** Changing a field name breaks the frontend and the report exporters. Tell both sides first.
2. **Do not modify anything under `ml/`.** It is a shared package behind a validated model. Everything goes through `run_pipeline()`.
3. **Never pass `preprocess=False`** on a raw sonar image.
4. **Quote metrics from `docs/METRICS_PROTOCOL.md`** — mAP@50 is **0.641**, not 0.580.
