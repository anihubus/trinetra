# DRISHTI — Turborepo setup, step by step

A complete action script to restructure this repo into a Turborepo monorepo with the frontend,
backend, Celery worker, and Docker stack wired end to end. Every file that cannot be derived is
given in full.

**Current state (verified):** `frontend/` and `backend/` are scaffolds — the directory trees
exist but almost every source file is empty (`settings.py`, `celery.py`, `asgi.py`, `manage.py`,
`main.jsx`, `vite.config.js` … all 0 lines). `ml/` is complete and working. `infra/docker-compose.yml`
exists but predates the ONNX serving decision.

**What this produces:**

```
drishti/
├─ package.json                  # pnpm workspace root
├─ pnpm-workspace.yaml
├─ turbo.json
├─ compose.yaml                  # 5 services, one `docker compose up`
├─ .env  /  .env.example
├─ apps/
│  ├─ dashboard/                 # React + Vite (was frontend/)
│  │  ├─ package.json
│  │  ├─ Dockerfile
│  │  ├─ vite.config.js
│  │  └─ src/...
│  └─ api/                       # Django REST + Channels (was backend/)
│     ├─ package.json            # thin — Turbo runs python through these scripts
│     ├─ Dockerfile
│     ├─ requirements.txt
│     ├─ manage.py
│     ├─ drishti_api/{settings,asgi,celery,urls}.py
│     └─ detections/{models,tasks,views,serializers,consumers,urls,routing}.py
├─ services/
│  └─ worker/
│     ├─ Dockerfile              # api image + onnxruntime, NO torch
│     └─ requirements.txt
├─ ml/                           # unchanged — inference package + committed weights
└─ docs/
```

The model is **not** a service. The worker imports `ml.inference.pipeline` directly. See
`docs/DEPLOYMENT_RATIONALE.md`.

---

## Phase 0 — prerequisites

```bash
node --version      # >= 20
corepack enable     # ships with Node; provides pnpm
pnpm --version      # >= 9  (corepack will fetch it)
docker --version    # >= 24, with compose v2
```

Work on a branch:

```bash
cd /d/Sonar-Drishti
git checkout -b chore/turborepo
```

---

## Phase 1 — move the scaffold into the monorepo layout

Use `git mv` so history follows.

```bash
mkdir -p apps services/worker

git mv frontend apps/dashboard
git mv backend  apps/api

# infra/ is superseded by the root compose.yaml + per-app Dockerfiles
git rm -r infra
git rm -f apps/api/Dockerfile apps/dashboard/vite.config.js   # empty; recreated below
```

`ml/`, `edge/`, `demo/`, `notebooks/`, `scripts/`, `docs/` stay at the repo root untouched.
The worker and api Docker builds reach up into `ml/` from the build context.

---

## Phase 2 — root workspace + Turbo config

### `package.json` (repo root — new)

```json
{
  "name": "drishti",
  "private": true,
  "packageManager": "pnpm@9.12.0",
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "lint": "turbo run lint",
    "test": "turbo run test",
    "stack": "docker compose up --build",
    "stack:down": "docker compose down -v"
  },
  "devDependencies": {
    "turbo": "^2.1.0"
  }
}
```

### `pnpm-workspace.yaml` (repo root — new)

```yaml
packages:
  - "apps/*"
```

`services/worker` and `ml/` are Python; they are **not** pnpm workspaces. Turbo only
orchestrates the two apps; Docker orchestrates the Python side.

### `turbo.json` (repo root — new)

```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "dev": {
      "cache": false,
      "persistent": true
    },
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", "staticfiles/**"]
    },
    "lint": {},
    "test": {
      "dependsOn": ["^build"],
      "outputs": []
    }
  }
}
```

### `.gitignore` additions (repo root)

```gitignore
# turborepo
.turbo/
node_modules/
apps/*/node_modules/
apps/*/dist/
apps/api/staticfiles/
```

### Install

```bash
pnpm install
```

---

## Phase 3 — `apps/api` (Django REST + Channels)

### `apps/api/package.json` (new — the Turbo bridge)

Turbo drives Python through npm scripts. No JS dependencies.

```json
{
  "name": "@drishti/api",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "python -m daphne -b 0.0.0.0 -p 8000 drishti_api.asgi:application",
    "build": "python manage.py collectstatic --noinput",
    "lint": "python -m ruff check .",
    "test": "python -m pytest -q"
  }
}
```

### `apps/api/requirements.txt`

Keep the existing list; add three lines the serving path needs. **No `torch`, no `ultralytics`.**

```
# ---- framework ----
Django>=5.0,<5.3
djangorestframework>=3.15
django-cors-headers>=4.4
drf-spectacular>=0.27

# ---- realtime ----
channels>=4.1
channels-redis>=4.2
daphne>=4.1

# ---- jobs ----
celery>=5.4
redis>=5.0

# ---- db ----
psycopg[binary]>=3.2

# ---- geo ----
pyproj>=3.6
shapely>=2.0

# ---- model serving (torch-free) ----
onnxruntime>=1.18
numpy>=1.26
opencv-python-headless>=4.10
scikit-learn>=1.4          # calibrator.pkl unpickles to sklearn LogisticRegression

# ---- config / misc ----
python-decouple>=3.8
Pillow>=10.3

# ---- test ----
pytest>=8.2
pytest-django>=4.8
ruff>=0.6
```

> The `web` service technically doesn't need onnxruntime — only the `worker` calls the model.
> They share one image for simplicity; if image size becomes a concern later, split
> `requirements.txt` into `requirements-web.txt` + `requirements-ml.txt`.

### `apps/api/manage.py`

```python
#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "drishti_api.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
```

### `apps/api/drishti_api/settings.py`

```python
from pathlib import Path
from decouple import Csv, config

BASE_DIR = Path(__file__).resolve().parent.parent
# repo root, so the worker can `import ml.inference.pipeline`
REPO_ROOT = BASE_DIR.parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ml" / "scripts"))   # despeckle_clahe import

SECRET_KEY = config("SECRET_KEY", default="dev-insecure-change-me")
DEBUG = config("DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="localhost,127.0.0.1", cast=Csv())

INSTALLED_APPS = [
    "daphne",
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "drf_spectacular",
    "channels",
    "detections",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.security.SecurityMiddleware",
]

ROOT_URLCONF = "drishti_api.urls"
WSGI_APPLICATION = "drishti_api.wsgi.application"
ASGI_APPLICATION = "drishti_api.asgi.application"
TEMPLATES = [{"BACKEND": "django.template.backends.django.DjangoTemplates",
             "APP_DIRS": True, "DIRS": [], "OPTIONS": {}}]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": config("POSTGRES_DB", default="drishti"),
        "USER": config("POSTGRES_USER", default="drishti"),
        "PASSWORD": config("POSTGRES_PASSWORD", default="drishti"),
        "HOST": config("POSTGRES_HOST", default="localhost"),
        "PORT": config("POSTGRES_PORT", default="5432"),
    }
}

REDIS_URL = config("REDIS_URL", default="redis://localhost:6379/0")

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [REDIS_URL]},
    }
}

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_TASK_TRACK_STARTED = True

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
}
SPECTACULAR_SETTINGS = {"TITLE": "DRISHTI API", "VERSION": "1.0"}

CORS_ALLOWED_ORIGINS = config(
    "CORS_ALLOWED_ORIGINS", default="http://localhost:5173", cast=Csv()
)

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = config("MEDIA_ROOT", default=str(BASE_DIR / "media"))
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---- model serving ----
DRISHTI_MODEL = config(
    "DRISHTI_MODEL",
    default=str(REPO_ROOT / "ml" / "models" / "exported" / "best_detector.onnx"),
)
DRISHTI_CALIBRATOR = config(
    "DRISHTI_CALIBRATOR",
    default=str(REPO_ROOT / "ml" / "models" / "exported" / "calibrator.pkl"),
)
```

### `apps/api/drishti_api/asgi.py`

```python
import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "drishti_api.settings")
django_asgi_app = get_asgi_application()

from detections.routing import websocket_urlpatterns  # noqa: E402  (after app registry)

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": URLRouter(websocket_urlpatterns),
    }
)
```

### `apps/api/drishti_api/wsgi.py`

```python
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "drishti_api.settings")
application = get_wsgi_application()
```

### `apps/api/drishti_api/celery.py`

```python
import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "drishti_api.settings")

app = Celery("drishti")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
```

### `apps/api/drishti_api/__init__.py`

```python
from .celery import app as celery_app

__all__ = ("celery_app",)
```

### `apps/api/drishti_api/urls.py`

```python
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path("api/", include("detections.urls")),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema")),
]
```

### `apps/api/detections/models.py`

Mirrors `backend/reporting/schema.py` exactly so the serializer is `fields = "__all__"`.

```python
import uuid

from django.db import models


class DetectionJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_file = models.CharField(max_length=255)
    status = models.CharField(max_length=16, default="queued")  # queued|processing|done|failed
    celery_task_id = models.CharField(max_length=128, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Detection(models.Model):
    detection_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(DetectionJob, related_name="detections", on_delete=models.CASCADE)
    ping_id = models.CharField(max_length=255)
    timestamp = models.DateTimeField(null=True)
    latitude = models.FloatField(null=True)
    longitude = models.FloatField(null=True)
    class_label = models.CharField(max_length=64)
    confidence_score = models.FloatField()
    bounding_geometry = models.JSONField(default=dict)
    review_status = models.CharField(max_length=24)  # auto_confirmed|pending_review|analyst_*
    source_file = models.CharField(max_length=255)
    side = models.CharField(max_length=12, blank=True)
    across_track_m = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class AuditLogEntry(models.Model):
    detection = models.ForeignKey(Detection, related_name="audit", on_delete=models.CASCADE)
    action = models.CharField(max_length=32)   # analyst_confirmed | analyst_rejected
    actor = models.CharField(max_length=128, blank=True)
    at = models.DateTimeField(auto_now_add=True)
```

### `apps/api/detections/serializers.py`

```python
from rest_framework import serializers

from .models import Detection, DetectionJob


class DetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detection
        exclude = ("job", "created_at")


class DetectionJobSerializer(serializers.ModelSerializer):
    detections = DetectionSerializer(many=True, read_only=True)

    class Meta:
        model = DetectionJob
        fields = ("id", "source_file", "status", "error", "created_at", "detections")
```

### `apps/api/detections/tasks.py` — the model wiring

```python
from pathlib import Path

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.conf import settings

from ml.inference.pipeline import run_pipeline

from .models import Detection, DetectionJob


@shared_task(bind=True)
def run_inference_job(self, job_id: str, image_paths: list[str],
                      source_file: str, xtf: str | None = None, nav: str | None = None):
    job = DetectionJob.objects.get(id=job_id)
    job.status = "processing"
    job.celery_task_id = self.request.id
    job.save(update_fields=["status", "celery_task_id"])
    layer = get_channel_layer()
    group = f"job_{job_id}"

    try:
        for i, path in enumerate(image_paths):
            report = run_pipeline(
                path, source_file,
                model_path=settings.DRISHTI_MODEL,
                calibrator_path=settings.DRISHTI_CALIBRATOR,
                xtf=Path(xtf) if xtf else None,
                nav=Path(nav) if nav else None,
            )
            recs = report["detections"]
            Detection.objects.bulk_create([
                Detection(job=job, **{k: v for k, v in r.items() if k != "job_id"})
                for r in recs
            ])
            async_to_sync(layer.group_send)(group, {
                "type": "detection.partial", "tile_index": i, "detections": recs,
            })

        job.status = "done"
        job.save(update_fields=["status"])
        async_to_sync(layer.group_send)(group, {
            "type": "detection.complete", "job_id": str(job_id),
            "total": job.detections.count(),
        })
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        job.save(update_fields=["status", "error"])
        async_to_sync(layer.group_send)(group, {"type": "detection.failed", "error": str(exc)})
        raise
```

> `run_pipeline` handles **one image**. For a full uploaded waterfall, tile it first in the
> view (or a pre-task) — reuse the 640/stride-512 loop and box→full-image mapping from
> `ml/scripts/run_aurora_survey.py`, then pass the tile paths as `image_paths`.

### `apps/api/detections/views.py`

```python
import tempfile
from pathlib import Path

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ml.inference.pipeline import run_pipeline  # only for the sync fallback path

from .models import Detection, DetectionJob
from .serializers import DetectionJobSerializer, DetectionSerializer
from .tasks import run_inference_job


@api_view(["POST"])
def upload(request):
    f = request.FILES.get("file")
    if not f:
        return Response({"detail": "no file"}, status=400)
    media = Path(settings.MEDIA_ROOT)
    media.mkdir(parents=True, exist_ok=True)
    dest = media / f.name
    with open(dest, "wb") as out:
        for chunk in f.chunks():
            out.write(chunk)

    job = DetectionJob.objects.create(source_file=f.name)
    # xtf / nav optionally uploaded alongside; omitted here for brevity
    run_inference_job.delay(str(job.id), [str(dest)], f.name)
    return Response({"job_id": str(job.id), "status": job.status}, status=202)


@api_view(["GET"])
def job_detail(request, job_id):
    try:
        job = DetectionJob.objects.get(id=job_id)
    except DetectionJob.DoesNotExist:
        return Response(status=404)
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
        return Response(status=404)
    new = request.data.get("review_status")
    if new not in ("analyst_confirmed", "analyst_rejected"):
        return Response({"detail": "review_status must be analyst_confirmed|analyst_rejected"}, status=400)
    d.review_status = new
    d.save(update_fields=["review_status"])
    d.audit.create(action=new, actor=request.data.get("actor", ""))
    return Response(DetectionSerializer(d).data)


@api_view(["GET"])
def export(request, job_id):
    fmt = request.query_params.get("format", "json")
    recs = list(Detection.objects.filter(job_id=job_id).values())
    if fmt == "csv":
        import csv, io
        from ml.inference.pipeline import CSV_COLUMNS  # or backend.reporting.schema
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(recs[0].keys()) if recs else [])
        w.writeheader()
        w.writerows(recs)
        resp = Response(buf.getvalue(), content_type="text/csv")
        resp["Content-Disposition"] = f'attachment; filename="{job_id}.csv"'
        return resp
    return Response({"job_id": job_id, "detections": recs})
```

### `apps/api/detections/urls.py`

```python
from django.urls import path

from . import views

urlpatterns = [
    path("upload/", views.upload),
    path("jobs/<uuid:job_id>/", views.job_detail),
    path("detections/<uuid:job_id>/", views.detections),
    path("detections/<uuid:detection_id>/review/", views.review),
    path("export/<uuid:job_id>/", views.export),
]
```

### `apps/api/detections/consumers.py`

```python
from channels.generic.websocket import AsyncJsonWebsocketConsumer


class JobConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.job_id = self.scope["url_route"]["kwargs"]["job_id"]
        self.group = f"job_{self.job_id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        await self.accept()

    async def disconnect(self, code):
        await self.channel_layer.group_discard(self.group, self.channel_name)

    # channel_layer.group_send type="detection.partial" -> this handler
    async def detection_partial(self, event):
        await self.send_json({"type": "detection.partial", **event})

    async def detection_complete(self, event):
        await self.send_json({"type": "detection.complete", **event})

    async def detection_failed(self, event):
        await self.send_json({"type": "detection.failed", **event})
```

### `apps/api/detections/routing.py`

```python
from django.urls import re_path

from .consumers import JobConsumer

websocket_urlpatterns = [
    re_path(r"^ws/jobs/(?P<job_id>[0-9a-f-]+)/$", JobConsumer.as_asgi()),
]
```

### `apps/api/detections/apps.py`

```python
from django.apps import AppConfig


class DetectionsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "detections"
```

### `apps/api/detections/__init__.py`

```python
default_app_config = "detections.apps.DetectionsConfig"
```

### `apps/api/pytest.ini`

```ini
[pytest]
DJANGO_SETTINGS_MODULE = drishti_api.settings
python_files = test_*.py
```

### `apps/api/Dockerfile`

```dockerfile
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# deps first for layer caching
COPY apps/api/requirements.txt apps/api/requirements.txt
RUN pip install --no-cache-dir -r apps/api/requirements.txt

# the model package + committed weights, and the Django project
COPY ml/ ml/
COPY apps/api/ apps/api/

WORKDIR /repo/apps/api
ENV DJANGO_SETTINGS_MODULE=drishti_api.settings PYTHONPATH=/repo

EXPOSE 8000
CMD ["python", "-m", "daphne", "-b", "0.0.0.0", "-p", "8000", "drishti_api.asgi:application"]
```

---

## Phase 4 — `services/worker`

The worker shares the api image and just changes the command. Nothing extra to install —
`onnxruntime`, `numpy`, `opencv-python-headless`, `scikit-learn` are already in
`apps/api/requirements.txt`.

### `services/worker/Dockerfile`

```dockerfile
# Reuse the api image built by compose (target: drishti-api), only swap the command.
ARG API_IMAGE=drishti-api:latest
FROM ${API_IMAGE}

WORKDIR /repo/apps/api
CMD ["celery", "-A", "drishti_api", "worker", "-l", "info", "--concurrency", "2"]
```

If you prefer a standalone build (no `API_IMAGE` arg), copy the `apps/api/Dockerfile` body and
change the last `CMD` line only.

---

## Phase 5 — `apps/dashboard` (React + Vite)

The component tree already exists (`MapView`, `ReviewQueue`, `UploadPanel`, `ImageOverlay`,
`useDetectionSocket`). Fill it against `docs/API_ENDPOINTS.md`. These are the files needed for
the app to build and talk to the API.

### `apps/dashboard/package.json`

```json
{
  "name": "@drishti/dashboard",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host --port 5173",
    "build": "vite build",
    "preview": "vite preview --port 4173",
    "lint": "eslint src",
    "test": "vitest run"
  },
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.24.0",
    "react-leaflet": "^4.2.1",
    "leaflet": "^1.9.4",
    "axios": "^1.7.0",
    "recharts": "^2.12.0"
  },
  "devDependencies": {
    "vite": "^5.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "vitest": "^1.6.0",
    "eslint": "^8.57.0"
  }
}
```

### `apps/dashboard/vite.config.js`

```javascript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  test: { environment: "jsdom", globals: true },
});
```

### `apps/dashboard/src/main.jsx`

```jsx
import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import "./styles/globals.css";
import "leaflet/dist/leaflet.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

### `apps/dashboard/src/api/client.js`

```javascript
import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";
export const api = axios.create({ baseURL: BASE });

export const uploadLog = (file) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post("/upload/", fd).then((r) => r.data); // -> { job_id, status }
};
export const getJob = (id) => api.get(`/jobs/${id}/`).then((r) => r.data);
export const getDetections = (id) => api.get(`/detections/${id}/`).then((r) => r.data);
export const reviewDetection = (id, review_status, actor) =>
  api.patch(`/detections/${id}/review/`, { review_status, actor }).then((r) => r.data);
export const exportUrl = (id, format = "json") => `${BASE}/export/${id}/?format=${format}`;
```

### `apps/dashboard/src/api/websocket.js`

```javascript
const WS_BASE =
  import.meta.env.VITE_WS_BASE ??
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;

export function openJobSocket(jobId, { onPartial, onComplete, onFailed }) {
  const ws = new WebSocket(`${WS_BASE}/ws/jobs/${jobId}/`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "detection.partial") onPartial?.(msg);
    else if (msg.type === "detection.complete") onComplete?.(msg);
    else if (msg.type === "detection.failed") onFailed?.(msg);
  };
  return ws; // caller closes on unmount
}
```

### `apps/dashboard/src/hooks/useDetectionSocket.js`

```javascript
import { useEffect, useRef, useState } from "react";
import { openJobSocket } from "../api/websocket";

export function useDetectionSocket(jobId) {
  const [detections, setDetections] = useState([]);
  const [done, setDone] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;
    setDetections([]);
    setDone(false);
    wsRef.current = openJobSocket(jobId, {
      onPartial: (m) => setDetections((prev) => [...prev, ...m.detections]),
      onComplete: () => setDone(true),
      onFailed: () => setDone(true),
    });
    return () => wsRef.current?.close();
  }, [jobId]);

  return { detections, done };
}
```

### `apps/dashboard/Dockerfile`

```dockerfile
FROM node:20-slim AS build
WORKDIR /app
COPY apps/dashboard/package.json apps/dashboard/
COPY package.json pnpm-workspace.yaml ./
RUN corepack enable && pnpm install --filter @drishti/dashboard
COPY apps/dashboard/ apps/dashboard/
RUN pnpm --filter @drishti/dashboard build

FROM nginx:1.27-alpine
COPY --from=build /app/apps/dashboard/dist /usr/share/nginx/html
COPY apps/dashboard/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

### `apps/dashboard/nginx.conf`

```nginx
server {
  listen 80;
  root /usr/share/nginx/html;
  index index.html;

  location /api/ { proxy_pass http://web:8000; proxy_set_header Host $host; }
  location /ws/  {
    proxy_pass http://web:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
  }
  location / { try_files $uri $uri/ /index.html; }
}
```

---

## Phase 6 — `compose.yaml` (repo root)

```yaml
name: drishti

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-drishti}
      POSTGRES_USER: ${POSTGRES_USER:-drishti}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-drishti}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-drishti}"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

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
    volumes:
      - media:/repo/apps/api/media
    depends_on:
      - web
      - redis

  frontend:
    build:
      context: .
      dockerfile: apps/dashboard/Dockerfile
    depends_on:
      - web
    ports:
      - "5173:80"

volumes:
  pgdata:
  media:
```

`worker` depends on `web` only so the `drishti-api:latest` image exists before its `FROM`.
Scale inference: `docker compose up --scale worker=4`.

---

## Phase 7 — environment

### `.env.example` (repo root — replace)

```dotenv
# ---- Django ----
DEBUG=True
SECRET_KEY=dev-insecure-change-me
ALLOWED_HOSTS=localhost,127.0.0.1,web
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:4173

# ---- Postgres ----
POSTGRES_DB=drishti
POSTGRES_USER=drishti
POSTGRES_PASSWORD=drishti
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# ---- Redis (broker + channels) ----
REDIS_URL=redis://localhost:6379/0

# ---- Model (defaults resolve inside settings.py; override only to swap) ----
# DRISHTI_MODEL=/repo/ml/models/exported/best_detector.onnx
# DRISHTI_CALIBRATOR=/repo/ml/models/exported/calibrator.pkl

# ---- Frontend (Vite) ----
VITE_API_BASE=/api
VITE_WS_BASE=ws://localhost:8000
```

```bash
cp .env.example .env
```

---

## Phase 8 — bring it up

### Local dev (no Docker, fastest inner loop)

```bash
# terminal 1 — infra only
docker compose up postgres redis

# terminal 2 — api + worker + frontend via Turbo
pnpm install
python -m venv .venv && . .venv/Scripts/activate       # or your existing .venv
pip install -r apps/api/requirements.txt
(cd apps/api && python manage.py migrate)
pnpm dev                # turbo runs: dashboard `vite` + api `daphne`

# terminal 3 — the Celery worker (Turbo doesn't manage it)
cd apps/api && celery -A drishti_api worker -l info
```

### Full stack in Docker

```bash
docker compose up --build            # or: pnpm stack
docker compose exec web python manage.py migrate
```

Open **http://localhost:5173**. API docs at **http://localhost:8000/api/docs/**.

### Smoke test

```bash
# upload one of the committed demo tiles
curl -F "file=@demo/tiles/synth_ghost_net_00002.jpg" http://localhost:8000/api/upload/
# -> {"job_id":"...","status":"queued"}

curl http://localhost:8000/api/detections/<job_id>/
# -> [ { class_label:"ghost_net", confidence_score:95.3, review_status:"auto_confirmed", ... } ]
```

WebSocket: connect a client to `ws://localhost:8000/ws/jobs/<job_id>/` before uploading to see
`detection.partial` events stream in.

---

## Phase 9 — the Turbo pipeline

| Command | What runs |
|---|---|
| `pnpm dev` | `turbo run dev` → `@drishti/dashboard` (`vite`, port 5173) + `@drishti/api` (`daphne`, port 8000), both persistent, no cache |
| `pnpm build` | `@drishti/dashboard` → `vite build` (`dist/`), `@drishti/api` → `collectstatic` (`staticfiles/`) |
| `pnpm lint` | `eslint` on the dashboard, `ruff` on the api |
| `pnpm test` | `vitest` on the dashboard, `pytest` on the api |
| `pnpm stack` | `docker compose up --build` — the real 5-service stack |

The **Celery worker is not a Turbo task** — it's a compose service (or a manual terminal in
local dev). Turbo orchestrates the two things that build and hot-reload; Docker orchestrates
persistence and background work.

---

## Phase 10 — commit

```bash
git add -A
git commit -m "chore: restructure into Turborepo (apps/api, apps/dashboard, services/worker) + compose stack

- git mv frontend->apps/dashboard, backend->apps/api
- root pnpm workspace + turbo.json (dev/build/lint/test)
- Django project filled: settings/asgi/celery/urls + detections models/tasks/views/consumers
- Celery task calls ml.inference.run_pipeline directly (no ML service)
- worker image = api image + celery command; loads best_detector.onnx (torch-free)
- compose.yaml: postgres, redis, web, worker, frontend with healthchecks
- drop infra/ (superseded)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

Open a PR from `chore/turborepo`.

---

## Checklist for the reviewer

- [ ] `pnpm install` clean at root
- [ ] `docker compose up --build` — all 5 services healthy
- [ ] `docker compose exec web python manage.py migrate` succeeds
- [ ] `curl -F file=@demo/tiles/synth_ghost_net_00002.jpg localhost:8000/api/upload/` returns a `job_id`
- [ ] `GET /api/detections/<job_id>/` returns the ghost_net detection at ~95 %
- [ ] WebSocket `ws/jobs/<job_id>/` streams `detection.partial`
- [ ] `worker` container has **no** `torch` — `docker compose exec worker python -c "import sys; print('torch' in sys.modules)"` after an inference → `False`
- [ ] `pnpm build` produces `apps/dashboard/dist/` and `apps/api/staticfiles/`
