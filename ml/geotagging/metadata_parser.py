"""
Parses sonar-log metadata (coordinate files / ping index / navigation channel)
into two lookup tables the projection step needs:

  NavigationTable  - time  -> interpolated (lat, lon, heading, altitude, depth)
  PingIndex        - (file, ping number) -> (lat, lon, heading, time)

Both are populated from plain CSV files (the hackathon-demo path). The stretch
path, xtf_reader.py, produces the same records straight from XTF ping headers.

Reference fixtures: AURORA side-scan-sonar dataset
  navigation.csv            : date,time,lat,lon,heading,roll,pitch,depth,altitude,speed   (no header)
  side-scan-sonar-index.csv : Data;Time;Ping Number;File;Latitude;Longitude;Heading;...   (';' header)
"""

from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# navigation.txt header: Mission Date Time NorthDeg EastDeg HeadingDeg RollDeg PitchDeg Depth Altitude Speed
NAV_COLUMNS = ("date", "time", "lat", "lon", "heading", "roll", "pitch", "depth", "altitude", "speed")


@dataclass(frozen=True)
class NavFix:
    """One navigation sample."""
    t: float                 # POSIX seconds (UTC)
    lat: float
    lon: float
    heading: float           # degrees, 0..360 clockwise from North
    altitude: Optional[float] = None   # metres above seabed
    depth: Optional[float] = None       # metres below surface


def _parse_dt(date_s: str, time_s: str) -> float:
    """AURORA uses DD/MM/YYYY and HH:MM:SS[.fff]. Returns POSIX seconds (UTC)."""
    date_s, time_s = date_s.strip(), time_s.strip()
    for dfmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%y%m%d"):
        for tfmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H%M%S"):
            try:
                dt = datetime.strptime(f"{date_s} {time_s}", f"{dfmt} {tfmt}")
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    raise ValueError(f"unparseable datetime: {date_s!r} {time_s!r}")


def _circ_lerp(a: float, b: float, f: float) -> float:
    """Interpolate a heading in degrees the short way around the circle."""
    d = ((b - a + 180.0) % 360.0) - 180.0
    return (a + d * f) % 360.0


class NavigationTable:
    """Time-indexed navigation, with interpolation between samples."""

    def __init__(self, fixes: list[NavFix]):
        self.fixes = sorted(fixes, key=lambda x: x.t)
        self._ts = [x.t for x in self.fixes]
        if not self.fixes:
            raise ValueError("NavigationTable is empty")

    @classmethod
    def from_csv(cls, path: str | Path, has_header: bool = False) -> "NavigationTable":
        path = Path(path)
        fixes: list[NavFix] = []
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        if has_header:
            rows = rows[1:]
        for r in rows:
            if len(r) < 5 or not r[0].strip():
                continue
            try:
                rec = dict(zip(NAV_COLUMNS, (c.strip() for c in r)))
                fixes.append(NavFix(
                    t=_parse_dt(rec["date"], rec["time"]),
                    lat=float(rec["lat"]), lon=float(rec["lon"]),
                    heading=float(rec["heading"]) % 360.0,
                    altitude=float(rec["altitude"]) if rec.get("altitude") else None,
                    depth=float(rec["depth"]) if rec.get("depth") else None,
                ))
            except (ValueError, KeyError):
                continue
        return cls(fixes)

    def at(self, t: float) -> NavFix:
        """Interpolated fix at POSIX time t (clamped to the recorded span)."""
        ts = self._ts
        if t <= ts[0]:
            return self.fixes[0]
        if t >= ts[-1]:
            return self.fixes[-1]
        i = bisect.bisect_right(ts, t)
        a, b = self.fixes[i - 1], self.fixes[i]
        span = b.t - a.t
        f = 0.0 if span <= 0 else (t - a.t) / span
        return NavFix(
            t=t,
            lat=a.lat + (b.lat - a.lat) * f,
            lon=a.lon + (b.lon - a.lon) * f,
            heading=_circ_lerp(a.heading, b.heading, f),
            altitude=_lerp_opt(a.altitude, b.altitude, f),
            depth=_lerp_opt(a.depth, b.depth, f),
        )

    @property
    def span_seconds(self) -> float:
        return self._ts[-1] - self._ts[0]


def _lerp_opt(a: Optional[float], b: Optional[float], f: float) -> Optional[float]:
    if a is None or b is None:
        return a if b is None else b
    return a + (b - a) * f


@dataclass(frozen=True)
class PingRecord:
    file: str
    ping: int
    lat: float
    lon: float
    heading: float
    t: Optional[float] = None


class PingIndex:
    """(file, ping number) -> position, from a SONARWIZ-style index CSV."""

    def __init__(self, records: list[PingRecord]):
        # keep per-file, sorted by ping, for nearest-ping lookup
        self._by_file: dict[str, list[PingRecord]] = {}
        for rec in records:
            self._by_file.setdefault(rec.file, []).append(rec)
        for recs in self._by_file.values():
            recs.sort(key=lambda x: x.ping)

    @classmethod
    def from_csv(cls, path: str | Path, delimiter: str = ";") -> "PingIndex":
        path = Path(path)
        records: list[PingRecord] = []
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                try:
                    row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                    t = None
                    if row.get("Data") and row.get("Time"):
                        try:
                            t = _parse_dt(row["Data"], row["Time"])
                        except ValueError:
                            t = None
                    records.append(PingRecord(
                        file=row["File"],
                        ping=int(float(row["Ping Number"])),
                        lat=float(row["Latitude"]),
                        lon=float(row["Longitude"]),
                        heading=float(row["Heading"]) % 360.0,
                        t=t,
                    ))
                except (ValueError, KeyError, TypeError):
                    continue
        return cls(records)

    def files(self) -> list[str]:
        return sorted(self._by_file)

    def match_file(self, name: str) -> Optional[str]:
        """Loose match: a TIF like 'DATA0000106-SS.H-PU_xtf-CH12.TIF' -> 'DATA0000106.H-PU'."""
        stem = Path(name).stem.upper()
        for f in self._by_file:
            key = f.upper().replace("-SS", "").replace(".", "").replace("-", "")
            probe = stem.replace("-SS", "").replace(".", "").replace("-", "")
            if key and (key in probe or probe.startswith(key[:12])):
                return f
        return None

    def at_ping(self, file: str, ping: float) -> PingRecord:
        """Nearest recorded ping to `ping` within `file` (ping counts can have gaps)."""
        recs = self._by_file.get(file)
        if recs is None:
            recs = self._by_file.get(self.match_file(file) or "")
        if not recs:
            raise KeyError(f"no ping records for file {file!r}")
        pings = [r.ping for r in recs]
        i = bisect.bisect_left(pings, ping)
        cands = [c for c in (i - 1, i, i + 1) if 0 <= c < len(recs)]
        best = min(cands, key=lambda c: abs(recs[c].ping - ping))
        return recs[best]

    @property
    def ping_span(self) -> dict[str, tuple[int, int]]:
        return {f: (recs[0].ping, recs[-1].ping) for f, recs in self._by_file.items()}
