"""
Turns a detection's pixel position in a side-scan waterfall image into a
real-world (latitude, longitude), plus its size in metres.

Geometry
--------
    row  -> which ping   -> tow-fish (lat, lon, heading) from the nav source
    col  -> across-track slant range -> ground range  (sqrt(R^2 - altitude^2))
    port  = left of track  (bearing = heading - 90)
    starboard = right of track (bearing = heading + 90)
    target = geodesic point `ground_range` metres from the fish along that bearing

A side-scan image is two channels laid side by side: nadir (directly under the
fish) is the centre column; range increases toward both edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

_EARTH_R = 6_371_000.0  # metres


@dataclass(frozen=True)
class SwathGeometry:
    """Across-track scale of one image. `max_slant_range_m` is the range at the image edge."""
    max_slant_range_m: float
    altitude_m: float = 0.0
    nadir_frac: float = 0.5          # column fraction that sits directly under the fish
    port_is_left: bool = True        # port channel on the left half of the image


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float


def _destination(lat: float, lon: float, bearing_deg: float, dist_m: float) -> GeoPoint:
    """Forward geodesic on a sphere - fine for the < 1 km offsets in side-scan."""
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    dr = dist_m / _EARTH_R
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2),
    )
    return GeoPoint(math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180)


def slant_to_ground(slant_range_m: float, altitude_m: float) -> float:
    """Flat-seabed slant -> ground range. Returns 0 inside the nadir gap."""
    return math.sqrt(max(slant_range_m ** 2 - altitude_m ** 2, 0.0))


def column_to_across_track(col: float, img_w: int, swath: SwathGeometry) -> tuple[float, str]:
    """
    Column -> (signed ground-range offset in metres, 'port'|'starboard').
    Offset is the perpendicular distance from the track line.
    """
    frac = col / max(img_w - 1, 1)
    rel = frac - swath.nadir_frac                     # -0.5 .. +0.5 ish
    side_span = swath.nadir_frac if rel < 0 else (1.0 - swath.nadir_frac)
    slant = abs(rel) / max(side_span, 1e-6) * swath.max_slant_range_m
    ground = slant_to_ground(slant, swath.altitude_m)
    left = rel < 0
    is_port = left if swath.port_is_left else (not left)
    return ground, ("port" if is_port else "starboard")


@dataclass(frozen=True)
class ProjectedDetection:
    latitude: float
    longitude: float
    across_track_m: float
    side: str                    # 'port' | 'starboard'
    width_m: float
    height_m: float
    ping: int
    fish_lat: float
    fish_lon: float
    fish_heading: float


def project_detection(
    bbox_xyxy_px: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    ping_for_row,                 # callable: row(px) -> object with .lat .lon .heading .ping
    swath: SwathGeometry,
    row_to_ping: Optional[callable] = None,
) -> ProjectedDetection:
    """
    Project one detection box.

    ping_for_row(row_px) must return a record exposing .lat .lon .heading and,
    ideally, .ping (int). row_to_ping maps an image row to a ping number when the
    caller knows the mapping; otherwise the row index itself is passed through.
    """
    x1, y1, x2, y2 = bbox_xyxy_px
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    row_key = row_to_ping(cy) if row_to_ping else cy
    fix = ping_for_row(row_key)

    ground, side = column_to_across_track(cx, img_w, swath)
    bearing = fix.heading + (-90.0 if side == "port" else 90.0)
    target = _destination(fix.lat, fix.lon, bearing, ground)

    # size in metres: full swath edge-to-edge covers 2 * max ground range
    ground_edge = slant_to_ground(swath.max_slant_range_m, swath.altitude_m)
    m_per_px_x = (2.0 * ground_edge) / max(img_w, 1)
    # along-track metres/px needs vehicle speed * ping interval; approximate with
    # the across-track scale (square-ish pixels after slant correction).
    m_per_px_y = m_per_px_x

    return ProjectedDetection(
        latitude=round(target.lat, 7),
        longitude=round(target.lon, 7),
        across_track_m=round(ground, 2),
        side=side,
        width_m=round((x2 - x1) * m_per_px_x, 2),
        height_m=round((y2 - y1) * m_per_px_y, 2),
        ping=int(getattr(fix, "ping", row_key) or 0),
        fish_lat=round(fix.lat, 7),
        fish_lon=round(fix.lon, 7),
        fish_heading=round(fix.heading, 2),
    )
