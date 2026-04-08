from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

# Extensions we attempt to load with librosa/soundfile.
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma"})


def normalize_rel_path(path: Path, base: Path) -> str:
    """Relative path using forward slashes (for manifests and stable IDs)."""
    return PurePosixPath(path.resolve().relative_to(base.resolve())).as_posix()


def track_id(source_rel_path: str) -> str:
    """Stable short id from normalized relative path."""
    normalized = source_rel_path.replace("\\", "/").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]
