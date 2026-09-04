# DRISHTI -- Locked Research/Prototype Contract

Lock this on Day 0-1 (per coach review, Risk 3) so the prototype team can build
against realistic mock data immediately instead of waiting on the trained model.

## Model input (what ml/inference/detector.py expects)
A single preprocessed grayscale tile: fixed size (e.g. 640x640), float32, normalized.

## Model output -> Detection object (before geotagging)
```json
{
  "class_label": "net",
  "confidence_raw": 0.87,
  "bbox": [x_min, y_min, x_max, y_max],
  "mask_polygon": [[x1,y1], [x2,y2], "..."]
}
```

## Final API response (after confidence filtering + geotagging) --
matches backend/reporting/schema.py exactly:
```json
{
  "detection_id": "uuid",
  "job_id": "uuid",
  "ping_id": "string",
  "timestamp": "ISO-8601",
  "latitude": 0.0,
  "longitude": 0.0,
  "class_label": "net",
  "confidence_score": 85.0,
  "bounding_geometry": { "bbox": [0,0,0,0], "mask_polygon": ["..."] },
  "review_status": "pending_review",
  "source_file": "string"
}
```

## WebSocket event (real-time push, see backend/detections/consumers.py)
```json
{ "type": "detection.partial", "tile_index": 3, "detections": ["... final API response objects ..."] }
```

Frontend builds against mock arrays of this exact shape until Day 5's model swap.
