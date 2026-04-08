from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

MANIFEST_COLUMNS = [
    "source_rel_path",
    "audio_16k_rel_path",
    "spectrogram_rel_path",
    "embedding_rel_path",
    "duration_sec",
    "sr",
    "spectrogram_shape",
    "embedding_shape",
    "model",
    "input_length_sec",
    "embedding_key",
]


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Index rows by source_rel_path."""
    return {r["source_rel_path"]: r for r in read_manifest_rows(path) if r.get("source_rel_path")}


def write_manifest(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})
