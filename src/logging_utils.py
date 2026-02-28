from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Any

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def append_row_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    ensure_parent(csv_path)
    file_exists = csv_path.exists()

    # stable column order: existing header first, then any new keys
    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        cols = list(header)
        for k in row.keys():
            if k not in cols:
                cols.append(k)
    else:
        cols = list(row.keys())

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()
        # fill missing keys with empty
        w.writerow({c: row.get(c, "") for c in cols})