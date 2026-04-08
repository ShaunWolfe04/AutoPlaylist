from __future__ import annotations

from pathlib import Path
from typing import Iterator

import librosa
import numpy as np
import soundfile as sf

from preprocess.paths import AUDIO_EXTENSIONS, normalize_rel_path

TARGET_SR = 16000


def iter_audio_files(input_dir: Path) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, relative posix path) for each audio file under input_dir."""
    input_dir = input_dir.resolve()
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        rel = normalize_rel_path(path, input_dir)
        yield path, rel


def resample_to_16k_mono(
    source: Path,
    dest: Path,
    *,
    force: bool = False,
) -> tuple[bool, float]:
    """
    Load audio at TARGET_SR mono, write PCM_16 WAV.

    Returns (did_run, duration_sec). If skipped (exists and not force), did_run is False
    and duration_sec is from the existing file if readable, else 0.0.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        try:
            info = sf.info(str(dest))
            return False, float(info.duration)
        except OSError:
            return False, 0.0

    y, _ = librosa.load(str(source), sr=TARGET_SR, mono=True)
    # Peak normalize to avoid clipping if upstream is hot; keep headroom.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    sf.write(str(dest), y, TARGET_SR, subtype="PCM_16")
    return True, float(len(y) / TARGET_SR)


def dest_path_for_resample(output_root: Path, rel_posix: str) -> Path:
    """Mirror tree under output_root/audio_16k with .wav extension."""
    rel = Path(rel_posix)
    return output_root / "audio_16k" / rel.with_suffix(".wav")
