"""
End-to-end inference wire-up:

    sonar image  ->  detect (Module 1)
                 ->  NMS + per-class gate + calibrate + shadow check (Module 2)
                 ->  geotag + report record (Module 3)
                 ->  JSON / CSV / GeoJSON

This is the single function backend/detections/tasks.py calls per uploaded image.
Runnable standalone:

    python -m ml.inference.pipeline --image path.tif --source-file DATA0000106.H-PU \\
        --xtf .../DATA0000106.H-PU.xtf --nav .../navigation.csv --out report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ml.inference.detector import SonarDetector
from ml.inference.confidence_filter import ConfidenceFilter
from ml.inference.preprocess import despeckle_clahe
from ml.geotagging.run_geotag import geotag, DEFAULT_MAX_SLANT_M
from ml.geotagging.metadata_parser import NavigationTable
from ml.geotagging.xtf_reader import XtfNav
from ml.reporting.json_export import write_json, write_geojson
from ml.reporting.csv_export import write_csv
from ml.reporting.schema import new_job_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = _ROOT / "ml" / "models" / "checkpoints" / "best_detector.pt"
DEFAULT_CALIBRATOR = _ROOT / "ml" / "models" / "exported" / "calibrator.pkl"


def _resolve_geometry(xtf: Optional[Path], nav: Optional[Path]) -> tuple[float, float]:
    """(altitude_m, max_slant_range_m) from the nav source, with fallbacks."""
    altitude = 5.0
    max_slant = DEFAULT_MAX_SLANT_M
    nav_tbl = NavigationTable.from_csv(nav) if nav and nav.exists() else None
    if xtf and xtf.exists():
        xn = XtfNav.from_file(xtf, max_pings=6000)
        if xn.median_slant_range:
            max_slant = xn.median_slant_range
        mid_t = sorted(p.t for p in xn.pings if p.t)
        if mid_t and nav_tbl:
            a = nav_tbl.at(mid_t[len(mid_t) // 2]).altitude
            if a:
                altitude = a
    elif nav_tbl:
        a = nav_tbl.at(nav_tbl.fixes[len(nav_tbl.fixes) // 2].t).altitude
        if a:
            altitude = a
    return altitude, max_slant


def run_pipeline(
    image_path: str | Path,
    source_file: str,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    calibrator_path: str | Path = DEFAULT_CALIBRATOR,
    xtf: Optional[Path] = None,
    ping_index: Optional[Path] = None,
    nav: Optional[Path] = None,
    detector_conf: float = 0.10,
    preprocess: bool = True,        # best_detector.pt is the Lee+CLAHE-trained model
) -> dict:
    """Run the full pipeline on one sonar image. Returns the report dict."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    # 0. serve-time preprocessing - MUST match how the model was trained
    if preprocess:
        gray = despeckle_clahe(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 1. detect (low threshold - Module 2's per-class gate does the real cut)
    detector = SonarDetector(str(model_path), conf_threshold=detector_conf)
    raw = detector.detect(img if preprocess else str(image_path))
    logger.info(f"detector: {len(raw)} raw detections @ conf>={detector_conf}")

    # 2. confidence + noise filter, with real geometry for the shadow check
    altitude, max_slant = _resolve_geometry(xtf, nav)
    sonar_meta = {
        "altitude": altitude, "max_range": max_slant,
        "image_height": img_h, "image_width": img_w,
    }
    cal = str(calibrator_path) if Path(calibrator_path).exists() else None
    cf = ConfidenceFilter(calibrator_path=cal)
    scored = cf.filter(raw, sonar_metadata=sonar_meta, image=gray)
    n_pen = sum(1 for d in scored if d.get("shadow_penalty", 0) > 0)
    logger.info(f"confidence filter: {len(raw)} -> {len(scored)} kept "
                f"(altitude {altitude:.1f} m, slant {max_slant:.0f} m, {n_pen} shadow-penalised)")

    # 3. geotag -> report records
    dets_for_geo = [
        {"class_label": d["class_label"], "confidence_score": d["confidence_score"], "bbox": d["bbox"]}
        for d in scored
    ]
    records = geotag(dets_for_geo, source_file, img_w, img_h,
                     xtf=xtf, ping_index=ping_index, nav=nav)
    logger.info(f"geotag: {len(records)} report records")

    job_id = records[0]["job_id"] if records else new_job_id()
    return {
        "job_id": job_id,
        "source_file": source_file,
        "image": image_path.name,
        "image_size": [img_w, img_h],
        "geometry": {"altitude_m": round(altitude, 2), "max_slant_range_m": round(max_slant, 1)},
        "detector_raw": len(raw),
        "kept": len(records),
        "detections": records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--source-file", default="")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    ap.add_argument("--xtf", type=Path)
    ap.add_argument("--ping-index", type=Path)
    ap.add_argument("--nav", type=Path)
    ap.add_argument("--no-preprocess", dest="preprocess", action="store_false",
                    help="skip Lee+CLAHE (only for a raw-trained model)")
    ap.set_defaults(preprocess=True)
    ap.add_argument("--out", type=Path, default=Path("report"))
    args = ap.parse_args()

    rep = run_pipeline(
        args.image, args.source_file or args.image.stem,
        model_path=args.model, calibrator_path=args.calibrator,
        xtf=args.xtf, ping_index=args.ping_index, nav=args.nav,
        preprocess=args.preprocess,
    )
    recs = rep["detections"]
    write_json(args.out.with_suffix(".json"), rep["job_id"], rep["source_file"], recs)
    write_geojson(args.out.with_suffix(".geojson"), recs)
    write_csv(args.out.with_suffix(".csv"), recs)
    print(f"\n{rep['detector_raw']} raw -> {rep['kept']} geotagged  "
          f"(alt {rep['geometry']['altitude_m']} m, slant {rep['geometry']['max_slant_range_m']} m)")
    for r in recs:
        print(f"  {r['class_label']:18s} {r['confidence_score']:5.1f}%  "
              f"({r['latitude']:.6f}, {r['longitude']:.6f})  {r['side']} "
              f"{r['across_track_m']}m  {r['ping_id']}  [{r['review_status']}]")
    print(f"\n-> {args.out}.{{json,geojson,csv}}")


if __name__ == "__main__":
    main()
