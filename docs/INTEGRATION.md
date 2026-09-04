# Wiring the model into the backend — dev-team guide

**TL;DR:** the ML side is one function. `run_pipeline()` takes an image + nav files and returns
a dict whose `detections` list already matches `docs/api_contract.md` exactly. You do not need to
know anything about YOLO, ONNX, or sonar geometry. Your job is the three Django stubs.

---

## 1. Setup (once, ~5 min)

```bash
git clone https://github.com/Rehan9599/Sonar-Drishti.git && cd Sonar-Drishti
```

The trained model ships **in the repo** — no download, no Hugging Face account, no Git LFS:

```
ml/models/checkpoints/best_detector.pt     21.5 MB   ← default, PyTorch
ml/models/exported/best_detector.onnx      42.7 MB   ← same model, no-torch runtime
ml/models/exported/calibrator.pkl           2 KB     ← Module 2 calibration
```

**Pick your runtime** — both produce identical predictions (verified on the 850-tile test set):

| | install | model file | use when |
|---|---|---|---|
| **Light (recommended)** | `pip install onnxruntime numpy opencv-python-headless` (~50 MB) | `best_detector.onnx` | you just need detections to render |
| Full | `pip install -r ml/requirements.txt` (torch + ultralytics, ~2 GB) | `best_detector.pt` | you also want to retrain/export |

Plus the backend's own deps: `pip install -r backend/requirements.txt`.

### Verify before touching Django

```bash
python -m ml.inference.pipeline \
  --image ml/data/splits/test/images/wreckA_Artificial_Reef_06_y1280_x0.jpg \
  --source-file DATA0000106.H-PU \
  --model ml/models/exported/best_detector.onnx \
  --xtf ml/data/raw/AURORA-SSS/side-scan-sonar/xtf/xtf-navigation/DATA0000106.H-PU.xtf \
  --nav ml/data/raw/AURORA-SSS/side-scan-sonar/navigation.csv \
  --no-preprocess --out /tmp/report
```

Expected (this is a real run, 2026-09-02):

```
1 raw -> 1 geotagged  (alt 3.0 m, slant 107.7 m)
  shipwreck   64.0%  (50.393707, -7.713276)  starboard 80.36m  DATA0000106.H-PU#1401  [pending_review]
-> /tmp/report.{json,geojson,csv}
```

If you see that, the ML side is done and every remaining problem is a Django problem.

> **Two gotchas that will cost you an hour if you miss them.**
> - The parseable XTF lives in **`xtf/xtf-navigation/*.H-PU.xtf`**, *not* `xtf/xtf-data/*.H.xtf`.
>   The latter raises `KeyError: no XTF pings parsed`.
> - `--no-preprocess` is correct **only** for tiles already in `ml/data/splits/` (they were
>   preprocessed at build time). For a raw sonar log, drop the flag — `preprocess=True` is the default
>   and the shipped model requires it.

`.pt` and `.onnx` were verified to give the same answer on this tile:

| runtime | class | score | lat, lon | bbox |
|---|---|---|---|---|
| `.pt` | shipwreck | 64.0 % | 50.393707, −7.713275 | [476, 11, 640, 153] |
| `.onnx` | shipwreck | 64.0 % | 50.393707, −7.713276 | [476, 11, 640, 153] |

…and the ONNX path is genuinely torch-free — after loading the detector + Module 2 + Module 3,
`torch` and `ultralytics` are both absent from `sys.modules`.

---

## 2. The one call

```python
from ml.inference.pipeline import run_pipeline

report = run_pipeline(
    image_path,                 # str | Path — one sonar tile / waterfall image
    source_file,                # str — original log name, used for the ping_id audit pointer
    model_path="ml/models/exported/best_detector.onnx",   # or .../best_detector.pt
    calibrator_path="ml/models/exported/calibrator.pkl",
    xtf=None,                   # Path to the .xtf log  ─┐ supply at least one of these
    nav=None,                   # Path to navigation.csv ─┤ for real lat/lon
    ping_index=None,            # Path to the ping index ─┘
    detector_conf=0.10,         # leave it — Module 2's per-class gate does the real cut
    preprocess=True,            # leave it True — the shipped model is Lee+CLAHE-trained
)
```

### What comes back

```python
{
  "job_id": "uuid",
  "source_file": "DATA0000106",
  "image": "tile_0003.png",
  "image_size": [1024, 640],
  "geometry": {"altitude_m": 42.1, "max_slant_range_m": 75.0},
  "detector_raw": 15,          # before Module 2
  "kept": 3,                   # after Module 2 (≥30 % score)
  "detections": [ ... ]        # ← THIS is the api_contract.md shape, ready to serialize
}
```

Each item in `detections`:

```json
{
  "detection_id": "uuid",
  "job_id": "uuid",
  "ping_id": "DATA0000106#4821",
  "timestamp": "2026-09-02T10:14:33Z",
  "latitude": 48.712345,
  "longitude": -9.874321,
  "class_label": "shipwreck",
  "confidence_score": 91.3,
  "bounding_geometry": {"bbox": [475.9, 11.4, 640.4, 153.0], "mask_polygon": []},
  "review_status": "auto_confirmed",
  "source_file": "DATA0000106",
  "side": "starboard",
  "across_track_m": 31.4
}
```

`review_status` is already decided for you: **≥ 80 %** → `auto_confirmed`, **30–80 %** →
`pending_review` (this is the review-queue feed), **< 30 %** → dropped before it reaches you.

### Coordinates for the map

`latitude` / `longitude` are on every record, and `write_geojson()` emits a Leaflet-ready
`FeatureCollection` — `Point` geometry + class/score/review_status in `properties`. Drop it
straight into a Leaflet `L.geoJSON()` layer; no conversion needed.

**You must pass `xtf` and/or `nav`.** Without them the pipeline still returns boxes and scores,
but the coordinates are degenerate — fine for a UI smoke test, useless for the demo.

**Known accuracy caveat — quote this, don't overstate it.** The tow-fish *position* recovered from
XTF headers agrees with the recorded navigation to **< 1 m**. But the *detection* position also
depends on the across-track scale, and the two nav paths resolve that differently: on the
`run_geotag --selftest` fixture, path A (ping-index CSV) and path B (XTF headers + nav altitude)
place the same detection **~122 m apart** (across-track 21.7 m vs 56.3 m). Prefer the ping-index
path when a ping index exists. For the map this means: **the pin is a search area, not a survey
fix** — which is exactly how a debris report should be read.

---

## 3. The three seams you fill

All three are currently docstring-only stubs. Nothing else in the ML path needs changing.

### 3a. `backend/detections/models.py`
Mirror the record above. Suggested:

```python
class DetectionJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4)
    source_file = models.CharField(max_length=255)
    status = models.CharField(max_length=32, default="queued")   # queued|running|done|failed
    celery_task_id = models.CharField(max_length=128, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Detection(models.Model):
    detection_id  = models.UUIDField(primary_key=True)
    job           = models.ForeignKey(DetectionJob, related_name="detections", on_delete=models.CASCADE)
    ping_id       = models.CharField(max_length=255)
    timestamp     = models.DateTimeField()
    latitude      = models.FloatField()
    longitude     = models.FloatField()
    class_label   = models.CharField(max_length=64)
    confidence_score = models.FloatField()
    bounding_geometry = models.JSONField()
    review_status = models.CharField(max_length=32)   # auto_confirmed|pending_review|analyst_*
    source_file   = models.CharField(max_length=255)
    side          = models.CharField(max_length=16, blank=True)
    across_track_m = models.FloatField(null=True)
```

Keep field names identical to the JSON so the serializer is a straight `fields = "__all__"`.

### 3b. `backend/detections/tasks.py` — the actual wiring

```python
from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from ml.inference.pipeline import run_pipeline

MODEL = "ml/models/exported/best_detector.onnx"

@shared_task(bind=True)
def run_inference_job(self, job_id, image_paths, source_file, xtf=None, nav=None):
    job = DetectionJob.objects.get(id=job_id)
    job.status = "running"; job.save(update_fields=["status"])
    layer = get_channel_layer()

    for i, path in enumerate(image_paths):          # one tile at a time → live push
        report = run_pipeline(path, source_file, model_path=MODEL, xtf=xtf, nav=nav)
        recs = report["detections"]

        Detection.objects.bulk_create([
            Detection(job=job, **{k: v for k, v in r.items() if k != "job_id"}) for r in recs
        ])
        async_to_sync(layer.group_send)(f"job_{job_id}", {
            "type": "detection.partial", "tile_index": i, "detections": recs,
        })

    job.status = "done"; job.save(update_fields=["status"])
    return {"job_id": str(job_id), "total": job.detections.count()}
```

That is the whole model integration. Everything below `run_pipeline` — preprocessing, YOLO,
confidence calibration, shadow check, XTF parsing, geodesy — is already built and tested.

**Big uploaded logs:** `run_pipeline` handles one image. For a full waterfall, tile first —
copy the tiling loop from `ml/scripts/run_aurora_survey.py` (640 px window, stride 512), then
feed each tile through the loop above. That script also maps tile boxes back to full-image
pixels; reuse it rather than rewriting.

### 3c. `views.py` / `serializers.py` / `consumers.py`
- `POST /api/jobs/` — accept the upload, create `DetectionJob`, `run_inference_job.delay(...)`, return `job_id`
- `GET /api/detections/?job_id=` — serialize `Detection` rows (shape already matches the contract)
- `PATCH /api/detections/<id>/` — review queue: set `review_status` to `analyst_confirmed` / `analyst_rejected`
- `consumers.py` — join group `job_<id>`, forward `detection.partial` events to the browser

The frontend already builds against mock arrays of this exact shape, so no frontend change is
needed when you swap the mock for the real endpoint.

---

## 4. Rules

1. **Do not change `docs/api_contract.md` or `backend/reporting/schema.py`.** They are frozen —
   the frontend, the report exporters and the ML side all depend on them.
2. **Do not set `preprocess=False`.** The shipped model was trained on Lee+CLAHE input; skipping
   it silently degrades every prediction.
3. **Do not lower `detector_conf` below 0.10 or raise it.** Module 2's per-class gate is the tuned
   cut; the detector threshold is deliberately loose.
4. If a detection looks wrong, check it against `python -m ml.inference.pipeline` on the same
   image first — that isolates ML bugs from Django bugs.

---

## 5. Where to read more

- `docs/api_contract.md` — the frozen interface
- `docs/PROJECT_RECORD.html` §10 — what each module does and why (theory + physics + diagrams)
- `docs/PROJECT_RECORD.html` §11 — the end-to-end architecture diagram
- `ml/scripts/run_aurora_survey.py` — worked example: full transect → tiles → geotagged report
