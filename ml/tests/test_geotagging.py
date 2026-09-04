"""Unit tests for the geotagging math - known inputs -> known outputs."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.geotagging.coordinate_projection import (
    SwathGeometry, slant_to_ground, column_to_across_track,
    _destination, project_detection,
)
from ml.reporting.schema import review_status, DetectionRecord

M_PER_DEG_LAT = 111_320.0


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_slant_to_ground():
    assert slant_to_ground(10.0, 0.0) == 10.0
    assert approx(slant_to_ground(5.0, 3.0), 4.0)          # 3-4-5
    assert slant_to_ground(2.0, 5.0) == 0.0                # inside nadir gap


def test_destination_cardinpoints():
    # 111.32 m north of the equator ~ 0.001 deg latitude
    p = _destination(0.0, 0.0, 0.0, M_PER_DEG_LAT / 1000)
    assert approx(p.lat, 0.001, 1e-4) and approx(p.lon, 0.0, 1e-4)
    # due east
    p = _destination(0.0, 0.0, 90.0, M_PER_DEG_LAT / 1000)
    assert approx(p.lon, 0.001, 1e-4) and approx(p.lat, 0.0, 1e-4)


def test_column_side_and_range():
    sw = SwathGeometry(max_slant_range_m=50.0, altitude_m=0.0, nadir_frac=0.5, port_is_left=True)
    # far left column -> port, near the full 50 m
    g, side = column_to_across_track(0, 2048, sw)
    assert side == "port" and 48 <= g <= 50
    # far right -> starboard
    g, side = column_to_across_track(2047, 2048, sw)
    assert side == "starboard" and 48 <= g <= 50
    # centre column -> ~0 m
    g, side = column_to_across_track(1024, 2048, sw)
    assert g < 1.0


def test_project_detection_starboard_east():
    """Fish at (10, -7), heading North; a detection on the right of the image
    must land east of the track."""
    class Fix:
        lat, lon, heading, ping = 10.0, -7.0, 0.0, 42

    sw = SwathGeometry(max_slant_range_m=50.0, altitude_m=5.0)
    # column 3/4 across a 2048-wide image -> starboard, ~25 m slant -> ~24.5 m ground
    proj = project_detection(
        (1536 - 10, 1000, 1536 + 10, 1040), img_w=2048, img_h=4000,
        ping_for_row=lambda _k: Fix(), swath=sw, row_to_ping=lambda r: r,
    )
    assert proj.side == "starboard"
    assert proj.longitude > -7.0          # east of the track
    assert approx(proj.latitude, 10.0, 1e-3)
    assert 20 <= proj.across_track_m <= 26


def test_review_status_bands():
    assert review_status(91.0) == "auto_confirmed"
    assert review_status(55.0) == "pending_review"


def test_record_to_dict_shape():
    rec = DetectionRecord(
        latitude=1.23456789, longitude=2.3456789, class_label="shipwreck",
        confidence_score=87.3, bbox=[10, 20, 30, 40], source_file="DATA0001.xtf",
        ping_number=99, job_id="job-1",
    ).to_dict()
    for key in ("detection_id", "job_id", "ping_id", "latitude", "longitude",
                "class_label", "confidence_score", "bounding_geometry",
                "review_status", "source_file"):
        assert key in key and key in rec
    assert rec["ping_id"] == "DATA0001.xtf#99"
    assert rec["latitude"] == 1.2345679       # rounded to 7 dp
    assert rec["review_status"] == "auto_confirmed"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
