"""
Full-transect run: tile a real AURORA side-scan TIF, run the whole DRISHTI
pipeline over it, and produce one geotagged report.

    TIF  -> overlapping 640 px tiles
         -> detector per tile, boxes mapped back to full-image pixels
         -> Module 2 (per-class gate + calibrate + shadow check, real geometry)
         -> Module 3 geotag + JSON / CSV / GeoJSON

AURORA has real navigation but no target labels, so this is a qualitative
end-to-end check on unseen real data - it proves the pipeline runs and places
detections on the map; it does not score detection accuracy.

    python ml/scripts/run_aurora_survey.py --tif <file> [--limit-rows N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.inference.detector import SonarDetector
from ml.inference.confidence_filter import ConfidenceFilter
from ml.inference.preprocess import despeckle_clahe
from ml.inference.pipeline import _resolve_geometry, DEFAULT_MODEL, DEFAULT_CALIBRATOR
from ml.geotagging.run_geotag import geotag
from ml.reporting.json_export import write_json, write_geojson
from ml.reporting.csv_export import write_csv
from ml.reporting.schema import new_job_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AURORA = ROOT / "ml" / "data" / "raw" / "AURORA-SSS" / "side-scan-sonar"


def tile_offsets(w: int, h: int, tile: int, stride: int):
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    if xs[-1] != w - tile and w > tile:
        xs.append(w - tile)
    if ys[-1] != h - tile and h > tile:
        ys.append(h - tile)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0


def detect_tiled(detector: SonarDetector, img_bgr: np.ndarray,
                 tile: int = 640, stride: int = 512) -> list[dict]:
    h, w = img_bgr.shape[:2]
    offs = list(tile_offsets(w, h, tile, stride))
    out: list[dict] = []
    for k, (x0, y0) in enumerate(offs, 1):
        crop = img_bgr[y0:y0 + tile, x0:x0 + tile]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue
        for d in detector.detect(crop):
            x1, y1, x2, y2 = d["bbox"]
            d["bbox"] = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
            d["mask_polygon"] = []
            out.append(d)
        if k % 20 == 0 or k == len(offs):
            logger.info(f"  tiles {k}/{len(offs)}  running total {len(out)} raw dets")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path,
                    default=AURORA / "tif" / "DATA0000106-SS.H-PU_xtf-CH12.TIF")
    ap.add_argument("--source-file", default="DATA0000106.H-PU")
    ap.add_argument("--xtf", type=Path,
                    default=AURORA / "xtf" / "xtf-navigation" / "DATA0000106.H-PU.xtf")
    ap.add_argument("--nav", type=Path, default=AURORA / "navigation.csv")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--limit-rows", type=int, default=0, help="crop the TIF to the first N rows (speed)")
    ap.add_argument("--preprocess", action="store_true",
                    help="apply Lee+CLAHE before detection (use with the preprocessed model)")
    ap.add_argument("--out", type=Path, default=AURORA / "_aurora_run" / "report")
    args = ap.parse_args()

    img = cv2.imread(str(args.tif), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {args.tif}")
    if args.limit_rows and img.shape[0] > args.limit_rows:
        img = img[: args.limit_rows]
    H, W = img.shape[:2]
    if args.preprocess:
        img = cv2.cvtColor(despeckle_clahe(img), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.info(f"{args.tif.name}  {W}x{H}px{'  [preprocessed]' if args.preprocess else ''}")

    t0 = time.time()
    detector = SonarDetector(str(args.model), conf_threshold=0.10)
    raw = detect_tiled(detector, img, args.tile, args.stride)
    logger.info(f"detector: {len(raw)} raw detections over {(time.time()-t0):.0f}s")
    logger.info(f"  raw class mix: {dict(Counter(d['class_label'] for d in raw))}")

    altitude, max_slant = _resolve_geometry(args.xtf if args.xtf.exists() else None,
                                            args.nav if args.nav.exists() else None)
    meta = {"altitude": altitude, "max_range": max_slant, "image_height": H, "image_width": W}
    cal = str(args.calibrator) if args.calibrator.exists() else None
    cf = ConfidenceFilter(calibrator_path=cal, use_shadow_check=True)   # real transect -> shadow ON
    scored = cf.filter(raw, sonar_metadata=meta, image=gray)
    n_pen = sum(1 for d in scored if d.get("shadow_penalty", 0) > 0)
    logger.info(f"Module 2: {len(raw)} -> {len(scored)} kept  "
                f"(alt {altitude:.1f}m, slant {max_slant:.0f}m, {n_pen} shadow-penalised)")

    dets = [{"class_label": d["class_label"], "confidence_score": d["confidence_score"], "bbox": d["bbox"]}
            for d in scored]
    records = geotag(dets, args.source_file, W, H, xtf=args.xtf if args.xtf.exists() else None,
                     nav=args.nav if args.nav.exists() else None)
    job = records[0]["job_id"] if records else new_job_id()
    write_json(args.out.with_suffix(".json"), job, args.source_file, records)
    write_geojson(args.out.with_suffix(".geojson"), records)
    write_csv(args.out.with_suffix(".csv"), records)

    print("\n" + "=" * 68)
    print(f"  AURORA end-to-end run  -  {args.tif.name}")
    print("=" * 68)
    print(f"  raw detections (tiled) : {len(raw)}")
    print(f"  after Module 2         : {len(records)}")
    print(f"  class mix              : {dict(Counter(r['class_label'] for r in records))}")
    print(f"  review status          : {dict(Counter(r['review_status'] for r in records))}")
    if records:
        lats = [r['latitude'] for r in records]; lons = [r['longitude'] for r in records]
        print(f"  geographic extent      : lat {min(lats):.5f}..{max(lats):.5f}  "
              f"lon {min(lons):.5f}..{max(lons):.5f}")
        print("\n  first 12 records:")
        for r in records[:12]:
            print(f"    {r['class_label']:18s} {r['confidence_score']:5.1f}%  "
                  f"({r['latitude']:.6f},{r['longitude']:.6f})  {r['side']:9s} "
                  f"{r['across_track_m']:6.1f}m  {r['ping_id']}  [{r['review_status']}]")
    print(f"\n  -> {args.out}.{{json,geojson,csv}}")
    print("=" * 68)


if __name__ == "__main__":
    main()
