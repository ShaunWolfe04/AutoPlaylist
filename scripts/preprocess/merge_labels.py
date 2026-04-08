from __future__ import annotations

import csv
from pathlib import Path


def load_playlist_labels(labels_csv: Path) -> dict[str, dict[str, str]]:
    """
    Load music_labeler output: filename,song,artist,study,drive,workout (no header).
    Keys are basenames (first column).
    """
    by_file: dict[str, dict[str, str]] = {}
    with labels_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            filename, song, artist, study, drive, workout = row[:6]
            by_file[filename.strip()] = {
                "label_filename": filename.strip(),
                "label_song": song.strip(),
                "label_artist": artist.strip(),
                "label_study": study.strip(),
                "label_drive": drive.strip(),
                "label_workout": workout.strip(),
            }
    return by_file


def merge_manifest_with_labels(
    manifest_rows: list[dict[str, str]],
    labels_csv: Path,
) -> list[dict[str, str]]:
    by_base = load_playlist_labels(labels_csv)
    merged: list[dict[str, str]] = []
    for row in manifest_rows:
        src = row.get("source_rel_path", "")
        base = Path(src.replace("\\", "/")).name
        label = by_base.get(base, {})
        out = {**row, **label}
        merged.append(out)
    return merged


def write_merged_manifest(
    manifest_path: Path,
    labels_csv: Path,
    output_path: Path,
    extra_fieldnames: list[str] | None = None,
) -> None:
    from preprocess.manifest import MANIFEST_COLUMNS, read_manifest_rows

    rows = read_manifest_rows(manifest_path)
    merged = merge_manifest_with_labels(rows, labels_csv)
    fieldnames = list(MANIFEST_COLUMNS)
    if extra_fieldnames:
        fieldnames.extend(extra_fieldnames)
    else:
        fieldnames.extend(
            [
                "label_filename",
                "label_song",
                "label_artist",
                "label_study",
                "label_drive",
                "label_workout",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in merged:
            writer.writerow(row)
