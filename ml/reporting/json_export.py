"""
Writes the anomaly report as JSON: a job envelope plus the list of detection
records (schema.py). Also emits a GeoJSON FeatureCollection for direct use on
the Leaflet map.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def build_report(job_id: str, source_file: str, records: list[dict]) -> dict:
    return {
        "job_id": job_id,
        "source_file": source_file,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "detection_count": len(records),
        "class_counts": _class_counts(records),
        "detections": records,
    }


def _class_counts(records: Iterable[dict]) -> dict:
    out: dict[str, int] = {}
    for r in records:
        out[r["class_label"]] = out.get(r["class_label"], 0) + 1
    return out


def write_json(path: str | Path, job_id: str, source_file: str, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_report(job_id, source_file, records), indent=2))
    return path


def to_geojson(records: list[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
                "properties": {
                    "detection_id": r["detection_id"],
                    "class_label": r["class_label"],
                    "confidence_score": r["confidence_score"],
                    "review_status": r["review_status"],
                    "ping_id": r["ping_id"],
                    "timestamp": r["timestamp"],
                    "width_m": r["bounding_geometry"].get("width_m"),
                    "height_m": r["bounding_geometry"].get("height_m"),
                },
            }
            for r in records
        ],
    }


def write_geojson(path: str | Path, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_geojson(records), indent=2))
    return path
