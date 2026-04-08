#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/preprocess_pipeline.py` from repo root without package install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import soundfile as sf

from preprocess.embed_musicnn import CANONICAL_FEATURE_KEY, extract_musicnn_embedding
from preprocess.manifest import MANIFEST_COLUMNS, read_manifest_rows, write_manifest
from preprocess.merge_labels import write_merged_manifest
from preprocess.paths import track_id
from preprocess.resample import TARGET_SR, dest_path_for_resample, iter_audio_files, resample_to_16k_mono
from preprocess.spectrogram import dest_path_spectrogram, save_spectrogram_npz


def _duration_if_exists(audio_path: Path) -> float:
    if not audio_path.exists():
        return 0.0
    try:
        return float(sf.info(str(audio_path)).duration)
    except OSError:
        return 0.0


def _embedding_rel_path(tid: str) -> str:
    return (Path("embeddings") / f"{tid}.npz").as_posix()


def _spectrogram_shape_from_npz(path: Path) -> str:
    data = np.load(path, allow_pickle=True)
    lm = data["log_mel"]
    return f"{lm.shape[0]}x{lm.shape[1]}"


def _row_defaults() -> dict[str, str]:
    return {col: "" for col in MANIFEST_COLUMNS}


def _parse_preprocess_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="16 kHz audio → log-mel spectrogram (MusiCNN-style) → MusiCNN embeddings.",
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Root folder of raw audio.")
    p.add_argument("--output-root", type=Path, required=True, help="e.g. processed/")
    p.add_argument(
        "--stage",
        choices=("resample", "spectrogram", "embed", "all", "both"),
        default="all",
        help="resample: 16 kHz WAV only; spectrogram: log-mel .npz only (needs WAV); "
        "embed: MusiCNN only; all|both: WAV + spectrogram + embedding (default).",
    )
    p.add_argument("--force", action="store_true", help="Rebuild outputs even if they exist.")
    p.add_argument(
        "--model",
        default="MSD_musicnn",
        help="MusiCNN checkpoint name (e.g. MSD_musicnn, MTT_musicnn).",
    )
    p.add_argument(
        "--input-length",
        type=float,
        default=3.0,
        help="Patch length in seconds passed to MusiCNN (default 3).",
    )
    p.add_argument(
        "--overlap-sec",
        type=float,
        default=None,
        help="If set, patch overlap in seconds (MusiCNN input_overlap); omit for non-overlapping patches.",
    )
    return p.parse_args(argv)


def _parse_merge_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join manifest.csv with playlist_labels.csv on track basename.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def _effective_stage(stage: str) -> str:
    return "all" if stage == "both" else stage


def run_preprocess(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.csv"
    stage = _effective_stage(args.stage)

    do_resample = stage in ("all", "resample")
    do_spectrogram = stage in ("all", "spectrogram")
    do_embed = stage in ("all", "embed")

    overlap: float | bool = args.overlap_sec if args.overlap_sec is not None else False

    existing = {r["source_rel_path"]: r for r in read_manifest_rows(manifest_path) if r.get("source_rel_path")}
    updated_rows: dict[str, dict[str, str]] = {}

    for k, row in existing.items():
        merged = _row_defaults()
        merged.update(row)
        updated_rows[k] = merged

    for src_abs, rel in iter_audio_files(input_dir):
        audio_dest = dest_path_for_resample(output_root, rel)
        audio_rel = audio_dest.resolve().relative_to(output_root).as_posix()
        duration_sec = _duration_if_exists(audio_dest)
        tid = track_id(rel)
        spec_dest = dest_path_spectrogram(output_root, rel)
        spec_rel = spec_dest.resolve().relative_to(output_root).as_posix()
        emb_rel = _embedding_rel_path(tid)
        emb_abs = output_root / emb_rel

        if rel not in updated_rows:
            updated_rows[rel] = _row_defaults()

        row = updated_rows[rel]
        row["source_rel_path"] = rel

        if do_resample:
            ran, d = resample_to_16k_mono(src_abs, audio_dest, force=args.force)
            if d > 0:
                duration_sec = d
            if ran:
                print(f"[resample] {rel} -> {audio_rel}")
            elif stage == "resample":
                print(f"[resample skip] {rel}")
            row["audio_16k_rel_path"] = audio_rel
            row["duration_sec"] = str(duration_sec)
            row["sr"] = str(TARGET_SR)

        if not do_resample:
            row.setdefault("audio_16k_rel_path", audio_rel)
            if duration_sec <= 0:
                duration_sec = _duration_if_exists(audio_dest)
            if row.get("duration_sec", "") == "" and duration_sec > 0:
                row["duration_sec"] = str(duration_sec)
            if not row.get("sr"):
                row["sr"] = str(TARGET_SR)

        if do_spectrogram:
            if not audio_dest.exists():
                print(f"[spectrogram skip] missing 16k WAV for {rel}", file=sys.stderr)
            elif spec_dest.exists() and not args.force:
                row["spectrogram_rel_path"] = spec_rel
                row["spectrogram_shape"] = _spectrogram_shape_from_npz(spec_dest)
                if stage == "spectrogram":
                    print(f"[spectrogram skip] {rel}")
            else:
                save_spectrogram_npz(audio_dest, spec_dest, source_rel_path=rel)
                row["spectrogram_rel_path"] = spec_rel
                row["spectrogram_shape"] = _spectrogram_shape_from_npz(spec_dest)
                print(f"[spectrogram] {rel} {row['spectrogram_shape']}")

        if do_embed:
            if not audio_dest.exists():
                print(f"[embed skip] missing 16k file for {rel}", file=sys.stderr)
                continue
            if emb_abs.exists() and not args.force:
                info = np.load(emb_abs, allow_pickle=True)
                emb = info["embedding"]
                shape = f"{emb.shape[0]}x{emb.shape[1]}"
                row["embedding_rel_path"] = emb_rel
                row["embedding_shape"] = shape
                row["model"] = args.model
                row["input_length_sec"] = str(args.input_length)
                row["embedding_key"] = CANONICAL_FEATURE_KEY
                if not row.get("spectrogram_shape") and spec_dest.exists():
                    row["spectrogram_rel_path"] = spec_rel
                    row["spectrogram_shape"] = _spectrogram_shape_from_npz(spec_dest)
                if not row.get("duration_sec"):
                    row["duration_sec"] = str(_duration_if_exists(audio_dest))
                row["sr"] = str(TARGET_SR)
                print(f"[embed skip] {rel}")
                continue

            meta = extract_musicnn_embedding(
                audio_dest,
                model=args.model,
                input_length=args.input_length,
                input_overlap=overlap,
            )
            emb = meta["embedding"]
            taggram = meta["taggram"]
            emb_abs.parent.mkdir(parents=True, exist_ok=True)
            tags_joined = "\n".join(meta["tags"])
            np.savez_compressed(
                emb_abs,
                embedding=emb,
                taggram=taggram,
                tags_joined=np.asarray(tags_joined),
                feature_key=np.asarray(meta["feature_key"]),
                model=np.asarray(meta["model"]),
            )

            if duration_sec <= 0:
                duration_sec = _duration_if_exists(audio_dest)

            row["audio_16k_rel_path"] = audio_rel
            row["embedding_rel_path"] = emb_rel
            row["duration_sec"] = str(duration_sec)
            row["sr"] = str(TARGET_SR)
            row["embedding_shape"] = meta["embedding_shape"]
            row["model"] = meta["model"]
            row["input_length_sec"] = str(meta["input_length_sec"])
            row["embedding_key"] = meta["feature_key"]
            if spec_dest.exists():
                row["spectrogram_rel_path"] = spec_rel
                row["spectrogram_shape"] = _spectrogram_shape_from_npz(spec_dest)
            print(f"[embed] {rel} shape={meta['embedding_shape']}")

        if stage == "resample" and not do_spectrogram:
            row.setdefault("spectrogram_rel_path", "")
            row.setdefault("spectrogram_shape", "")
            row.setdefault("embedding_rel_path", "")
            row.setdefault("embedding_shape", "")
            row.setdefault("model", "")
            row.setdefault("input_length_sec", "")
            row.setdefault("embedding_key", "")

    ordered = [updated_rows[k] for k in sorted(updated_rows.keys())]
    write_manifest(manifest_path, ordered)
    print(f"Wrote {manifest_path} ({len(ordered)} tracks)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "merge-labels":
        margs = _parse_merge_args(argv[1:])
        write_merged_manifest(margs.manifest, margs.labels, margs.output)
        print(f"Wrote {margs.output}")
        return 0
    pargs = _parse_preprocess_args(argv)
    return run_preprocess(pargs)


if __name__ == "__main__":
    raise SystemExit(main())
