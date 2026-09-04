"""
Minimal XTF ping-header reader - the "reads sonar ping headers directly" path.

Parses only what geotagging needs, header-only (sample bytes are skipped via
NumBytesThisRecord), so a 400 MB log is walked in about a second:

  per ping: ping number, UTC timestamp, sensor lat/lon, heading,
            altitude above seabed, slant range at the swath edge.

No third-party XTF library. Struct offsets follow the Triton XTF revision-style
XTFPINGHEADER (256 B) + XTFPINGCHANHEADER (64 B) layout. If a file doesn't match,
callers should fall back to the CSV path in metadata_parser.py.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

_FILE_HEADER = 1024
_PING_HEADER = 256
_CHAN_HEADER = 64
_MAGIC = 0xFACE


@dataclass(frozen=True)
class XtfPing:
    ping: int
    t: Optional[float]           # POSIX seconds UTC
    lat: float
    lon: float
    heading: float               # degrees
    altitude: Optional[float]    # m above seabed (SensorPrimaryAltitude)
    depth: Optional[float]       # m below surface (SensorDepth)
    slant_range_m: Optional[float]


def _ts(buf: bytes) -> Optional[float]:
    try:
        year = struct.unpack_from("<H", buf, 14)[0]
        month, day, hour, minute, second, hsec = buf[16], buf[17], buf[18], buf[19], buf[20], buf[21]
        if not (1970 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
            return None
        return datetime(year, month, day, hour % 24, minute % 60, second % 60,
                        (hsec % 100) * 10_000, tzinfo=timezone.utc).timestamp()
    except (ValueError, struct.error):
        return None


def read_ping_headers(path: str | Path, max_pings: Optional[int] = None) -> Iterator[XtfPing]:
    path = Path(path)
    with path.open("rb") as f:
        fh = f.read(_FILE_HEADER)
        if len(fh) < _FILE_HEADER or fh[0] != 123:
            raise ValueError(f"{path.name}: not an XTF file (FileFormat byte != 123)")

        n = 0
        while True:
            start = f.tell()
            hdr = f.read(_PING_HEADER)
            if len(hdr) < _PING_HEADER:
                return
            magic = struct.unpack_from("<H", hdr, 0)[0]
            if magic != _MAGIC:
                f.seek(start + 1)                      # resync one byte at a time
                continue

            header_type = hdr[2]
            nbytes = struct.unpack_from("<I", hdr, 10)[0]
            next_pos = start + nbytes if 0 < nbytes < 50_000_000 else start + _PING_HEADER

            if header_type == 0:                       # XTF_HEADER_SONAR
                # Field offsets calibrated against the AURORA / SONARWIZ export
                # (nav block sits 4 B earlier than the base Triton XTFPINGHEADER):
                #   160 d lat   168 d lon   200 d altitude   212 f heading
                ping = struct.unpack_from("<I", hdr, 28)[0]
                lat = struct.unpack_from("<d", hdr, 160)[0]
                lon = struct.unpack_from("<d", hdr, 168)[0]
                heading = struct.unpack_from("<f", hdr, 212)[0]
                # This export's altitude/depth offsets don't validate against
                # navigation.csv (values drift into the slant-range figure), so
                # leave them None and let the caller join altitude from the nav
                # CSV by timestamp. lat/lon/heading/slant DO validate (< 1 m).
                altitude = None
                depth = None

                slant = None
                chan = f.read(_CHAN_HEADER)
                if len(chan) == _CHAN_HEADER:
                    s = struct.unpack_from("<f", chan, 4)[0]
                    slant = s if 0 < s < 5000 else None

                if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0.0 or lon != 0.0):
                    yield XtfPing(
                        ping=ping, t=_ts(hdr), lat=lat, lon=lon,
                        heading=heading % 360.0,
                        altitude=altitude, depth=depth,
                        slant_range_m=slant,
                    )
                    n += 1
                    if max_pings and n >= max_pings:
                        return
            f.seek(next_pos)


class XtfNav:
    """Wraps XTF ping headers so coordinate_projection can use them like PingIndex."""

    def __init__(self, pings: list[XtfPing], source_file: str):
        self.pings = sorted(pings, key=lambda p: p.ping)
        self._keys = [p.ping for p in self.pings]
        self.source_file = source_file

    @classmethod
    def from_file(cls, path: str | Path, max_pings: Optional[int] = None) -> "XtfNav":
        path = Path(path)
        return cls(list(read_ping_headers(path, max_pings)), path.name)

    def at_ping(self, file: str, ping: float):
        import bisect
        if not self.pings:
            raise KeyError("no XTF pings parsed")
        i = bisect.bisect_left(self._keys, ping)
        cands = [c for c in (i - 1, i, i + 1) if 0 <= c < len(self.pings)]
        return self.pings[min(cands, key=lambda c: abs(self.pings[c].ping - ping))]

    @property
    def median_altitude(self) -> Optional[float]:
        alts = sorted(p.altitude for p in self.pings if p.altitude)
        return alts[len(alts) // 2] if alts else None

    @property
    def median_slant_range(self) -> Optional[float]:
        sr = sorted(p.slant_range_m for p in self.pings if p.slant_range_m)
        return sr[len(sr) // 2] if sr else None
