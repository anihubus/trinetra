"""
Canonical structured-report schema - single source of truth, must match
docs/api_contract.md. One record per detected hazard.

Fields
------
detection_id      uuid4
job_id            uuid4 (one processing run over one sonar log)
ping_id           str  - "<source_file>#<ping number>"  (audit pointer back to the ping)
timestamp         ISO-8601 UTC of that ping
latitude          float, decimal degrees
longitude         float, decimal degrees
class_label       str  - one of the DRISHTI classes
confidence_score  float 0-100, AFTER Module 2 calibration
bounding_geometry {"bbox": [x_min, y_min, x_max, y_max] pixels,
                   "mask_polygon": [] , "width_m": float, "height_m": float}
review_status     auto_confirmed | pending_review | analyst_confirmed | analyst_rejected
source_file       str
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

REVIEW_STATES = ("auto_confirmed", "pending_review", "analyst_confirmed", "analyst_rejected")

# above this calibrated confidence a detection is auto-confirmed; below it needs a human
AUTO_CONFIRM_THRESHOLD = 80.0
REVIEW_FLOOR = 30.0          # below this we drop the detection entirely


def iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def review_status(confidence_score: float) -> str:
    if confidence_score >= AUTO_CONFIRM_THRESHOLD:
        return "auto_confirmed"
    return "pending_review"


@dataclass
class DetectionRecord:
    latitude: float
    longitude: float
    class_label: str
    confidence_score: float
    bbox: list[float]
    source_file: str
    ping_number: int
    timestamp: Optional[str] = None
    job_id: str = ""
    mask_polygon: list = field(default_factory=list)
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    across_track_m: Optional[float] = None
    side: Optional[str] = None
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def ping_id(self) -> str:
        return f"{self.source_file}#{self.ping_number}"

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "job_id": self.job_id,
            "ping_id": self.ping_id,
            "timestamp": self.timestamp,
            "latitude": round(self.latitude, 7),
            "longitude": round(self.longitude, 7),
            "class_label": self.class_label,
            "confidence_score": round(self.confidence_score, 1),
            "bounding_geometry": {
                "bbox": [round(v, 1) for v in self.bbox],
                "mask_polygon": self.mask_polygon,
                "width_m": self.width_m,
                "height_m": self.height_m,
            },
            "across_track_m": self.across_track_m,
            "side": self.side,
            "review_status": review_status(self.confidence_score),
            "source_file": self.source_file,
        }


def new_job_id() -> str:
    return str(uuid.uuid4())


# flat column order for CSV export
CSV_COLUMNS = (
    "detection_id", "job_id", "ping_id", "timestamp",
    "latitude", "longitude", "class_label", "confidence_score",
    "review_status", "side", "across_track_m", "width_m", "height_m",
    "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "source_file",
)


def record_to_csv_row(rec: dict) -> dict:
    bg = rec["bounding_geometry"]
    bx = bg["bbox"] + [None] * (4 - len(bg["bbox"]))
    return {
        "detection_id": rec["detection_id"],
        "job_id": rec["job_id"],
        "ping_id": rec["ping_id"],
        "timestamp": rec["timestamp"],
        "latitude": rec["latitude"],
        "longitude": rec["longitude"],
        "class_label": rec["class_label"],
        "confidence_score": rec["confidence_score"],
        "review_status": rec["review_status"],
        "side": rec.get("side"),
        "across_track_m": rec.get("across_track_m"),
        "width_m": bg.get("width_m"),
        "height_m": bg.get("height_m"),
        "bbox_x_min": bx[0], "bbox_y_min": bx[1], "bbox_x_max": bx[2], "bbox_y_max": bx[3],
        "source_file": rec["source_file"],
    }
