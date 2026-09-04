"""
Geotagging driver - turns one sonar image's detections into a geotagged report.

    detections (Module 1/2 output)  +  nav source  ->  JSON + CSV + GeoJSON

Nav source, in priority order:
  1. --xtf FILE           real ping headers (lat/lon/heading/altitude/slant range)
  2. --ping-index CSV      SONARWIZ-style (file,ping) -> lat/lon/heading
     + --nav CSV           navigation.csv for altitude
  3. --nav CSV only        treat image rows as evenly spaced along the track span

Detections JSON: [{ "class_label", "confidence_score", "bbox": [x1,y1,x2,y2] }, ...]
(bbox in pixels of the full sonar image; confidence_score already 0-100 from Module 2)

Self-test:  python -m backend.geotagging.run_geotag --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.geotagging.coordinate_projection import SwathGeometry, project_detection
from ml.geotagging.metadata_parser import NavigationTable, PingIndex
from ml.geotagging.xtf_reader import XtfNav
from ml.reporting.csv_export import write_csv
from ml.reporting.json_export import write_json, write_geojson
from ml.reporting.schema import DetectionRecord, new_job_id, iso, REVIEW_FLOOR

AURORA = Path(__file__).resolve().parents[2] / "ml" / "data" / "raw" / "AURORA-SSS" / "side-scan-sonar"
DEFAULT_MAX_SLANT_M = 50.0     # AURORA JC125/M87 swath edge, approx; overridden by XTF


def _row_to_ping_fn(img_h, ping_lo, ping_hi):
    """Linear image-row -> ping number when the exact per-row mapping is unknown."""
    span = max(ping_hi - ping_lo, 1)
    return lambda row: ping_lo + (row / max(img_h - 1, 1)) * span


def geotag(
    detections: list[dict],
    source_file: str,
    img_w: int,
    img_h: int,
    *,
    xtf: Path | None = None,
    ping_index: Path | None = None,
    nav: Path | None = None,
    max_slant_m: float = DEFAULT_MAX_SLANT_M,
    altitude_m: float | None = None,
) -> list[dict]:
    job_id = new_job_id()

    nav_tbl = NavigationTable.from_csv(nav) if nav else None
    xtf_nav = XtfNav.from_file(xtf) if xtf else None
    pidx = PingIndex.from_csv(ping_index) if ping_index else None

    # resolve swath geometry: slant range from XTF if present, altitude from the
    # nav CSV joined at the survey's mid timestamp (XTF export altitude is unreliable)
    alt = altitude_m
    slant = max_slant_m
    if xtf_nav and xtf_nav.pings:
        slant = xtf_nav.median_slant_range or slant
        if alt is None and nav_tbl:
            mid_t = [p.t for p in xtf_nav.pings if p.t]
            if mid_t:
                alt = nav_tbl.at(sorted(mid_t)[len(mid_t) // 2]).altitude
    if alt is None and nav_tbl:
        alt = nav_tbl.at(nav_tbl.fixes[len(nav_tbl.fixes) // 2].t).altitude
    swath = SwathGeometry(max_slant_range_m=slant, altitude_m=alt or 0.0)

    # choose the ping->fix provider and the row->ping mapping
    if xtf_nav and xtf_nav.pings:
        provider = xtf_nav
        p_lo, p_hi = xtf_nav.pings[0].ping, xtf_nav.pings[-1].ping
        row_to_ping = _row_to_ping_fn(img_h, p_lo, p_hi)
    elif pidx:
        f = pidx.match_file(source_file) or (pidx.files()[0] if pidx.files() else source_file)
        p_lo, p_hi = pidx.ping_span.get(f, (0, img_h))
        provider = pidx
        row_to_ping = _row_to_ping_fn(img_h, p_lo, p_hi)
        source_file = f
    elif nav_tbl:
        # no ping index: map rows evenly across the whole recorded track
        t0, t1 = nav_tbl._ts[0], nav_tbl._ts[-1]

        class _NavProvider:
            def at_ping(self, _file, key):
                fix = nav_tbl.at(t0 + (key / max(img_h - 1, 1)) * (t1 - t0))
                return type("R", (), {"lat": fix.lat, "lon": fix.lon,
                                      "heading": fix.heading, "ping": int(key)})
        provider = _NavProvider()
        row_to_ping = lambda row: row
    else:
        raise SystemExit("need one of --xtf / --ping-index / --nav")

    records: list[dict] = []
    for d in detections:
        conf = float(d["confidence_score"])
        if conf < REVIEW_FLOOR:
            continue
        x1, y1, x2, y2 = d["bbox"]
        proj = project_detection(
            (x1, y1, x2, y2), img_w, img_h,
            ping_for_row=lambda key: provider.at_ping(source_file, key),
            swath=swath, row_to_ping=row_to_ping,
        )
        ping_t = None
        if xtf_nav:
            ping_t = xtf_nav.at_ping(source_file, proj.ping).t
        elif pidx:
            pr = pidx.at_ping(source_file, proj.ping)
            ping_t = pr.t
        rec = DetectionRecord(
            latitude=proj.latitude, longitude=proj.longitude,
            class_label=d["class_label"], confidence_score=conf,
            bbox=[x1, y1, x2, y2], source_file=source_file, ping_number=proj.ping,
            timestamp=iso(ping_t), job_id=job_id,
            width_m=proj.width_m, height_m=proj.height_m,
            across_track_m=proj.across_track_m, side=proj.side,
        )
        records.append(rec.to_dict())
    return records


def _selftest():
    nav = AURORA / "navigation.csv"
    idx = AURORA / "side-scan-sonar-index.csv"
    if not nav.exists():
        print("AURORA fixture not found:", nav); return 1
    print("nav rows :", sum(1 for _ in nav.open()))
    nt = NavigationTable.from_csv(nav)
    print(f"nav span : {nt.span_seconds/3600:.2f} h   first fix "
          f"({nt.fixes[0].lat:.5f},{nt.fixes[0].lon:.5f})  alt~{nt.fixes[len(nt.fixes)//2].altitude}")

    src = "DATA0000106.H-PU"
    img_w, img_h = 2048, 15627
    fake = [
        {"class_label": "shipwreck", "confidence_score": 91.0, "bbox": [1500, 4200, 1620, 4360]},
        {"class_label": "submarine_pipeline", "confidence_score": 78.0, "bbox": [300, 9000, 340, 9400]},
        {"class_label": "mine_cylinder", "confidence_score": 65.0, "bbox": [1040, 12010, 1060, 12040]},
        {"class_label": "shipwreck", "confidence_score": 22.0, "bbox": [10, 10, 30, 30]},  # dropped
    ]
    out = AURORA.parent / "_geotag_selftest"

    print("\n--- path A: ping-index CSV + navigation CSV ---")
    recs = geotag(fake, src, img_w, img_h, ping_index=idx if idx.exists() else None, nav=nav)
    write_json(out / "report.json", recs[0]["job_id"] if recs else new_job_id(), src, recs)
    write_geojson(out / "report.geojson", recs)
    write_csv(out / "report.csv", recs)
    for r in recs:
        print(f"  {r['class_label']:18s} {r['confidence_score']:5.1f}%  "
              f"({r['latitude']:.6f}, {r['longitude']:.6f})  {r['side']} "
              f"{r['across_track_m']}m  ping {r['ping_id']}  [{r['review_status']}]")

    xtf = AURORA / "xtf" / "xtf-navigation" / "DATA0000106.H-PU.xtf"
    if xtf.exists():
        print("\n--- path B: XTF ping headers + navigation CSV (altitude) ---")
        recs_x = geotag(fake, src, img_w, img_h, xtf=xtf, nav=nav)
        for r in recs_x:
            print(f"  {r['class_label']:18s} {r['confidence_score']:5.1f}%  "
                  f"({r['latitude']:.6f}, {r['longitude']:.6f})  {r['side']} "
                  f"{r['across_track_m']}m  ping {r['ping_id']}  [{r['review_status']}]")
        if recs and recs_x:
            import math
            dm = math.hypot(recs[0]["latitude"] - recs_x[0]["latitude"],
                            recs[0]["longitude"] - recs_x[0]["longitude"]) * 111_320
            print(f"\n  A vs B first-detection position agreement: {dm:.1f} m")

    print(f"\nreports -> {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--detections", type=Path, help="detections JSON")
    ap.add_argument("--source-file", default="")
    ap.add_argument("--img-size", type=int, nargs=2, metavar=("W", "H"))
    ap.add_argument("--xtf", type=Path)
    ap.add_argument("--ping-index", type=Path)
    ap.add_argument("--nav", type=Path)
    ap.add_argument("--max-slant-m", type=float, default=DEFAULT_MAX_SLANT_M)
    ap.add_argument("--out", type=Path, default=Path("report"))
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    if not (args.detections and args.img_size):
        ap.error("--detections and --img-size are required (or use --selftest)")
    dets = json.loads(args.detections.read_text())
    w, h = args.img_size
    recs = geotag(dets, args.source_file or args.detections.stem, w, h,
                  xtf=args.xtf, ping_index=args.ping_index, nav=args.nav,
                  max_slant_m=args.max_slant_m)
    job = recs[0]["job_id"] if recs else new_job_id()
    write_json(args.out.with_suffix(".json"), job, args.source_file, recs)
    write_geojson(args.out.with_suffix(".geojson"), recs)
    write_csv(args.out.with_suffix(".csv"), recs)
    print(f"{len(recs)} detections geotagged -> {args.out}.{{json,geojson,csv}}")


if __name__ == "__main__":
    main()
