"""
Writes the anomaly report as a flat CSV - one row per detected hazard,
columns per schema.CSV_COLUMNS. Suitable for QGIS / Excel / Google Earth import.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .schema import CSV_COLUMNS, record_to_csv_row


def write_csv(path: str | Path, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in records:
            w.writerow(record_to_csv_row(r))
    return path
