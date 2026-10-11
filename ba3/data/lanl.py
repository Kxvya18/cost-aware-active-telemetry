from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class TelemetryEvent:
    time: int
    modality: str
    source: str
    destination: str
    user: str = ""
    weight: float = 1.0


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")


def iter_redteam(path: str | Path) -> Iterator[tuple[int, str, str, str]]:
    """Yield de-duplicated red-team events as time, user, source, destination."""
    seen: set[tuple[int, str, str, str]] = set()
    with _open_text(Path(path)) as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            event = (int(row[0]), row[1], row[2], row[3])
            if event not in seen:
                seen.add(event)
                yield event


def resolve_modality_path(data_dir: str | Path, stem: str) -> Path:
    root = Path(data_dir)
    disk_stem = "flows" if stem == "flow" else stem
    for path in (root / f"{disk_stem}.full.gz", root / f"{disk_stem}.txt.gz"):
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No LANL {stem} file found below {root}")
