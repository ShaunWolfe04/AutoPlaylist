from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

# MusiCNN expects librosa at SR=16000 (see musicnn.configuration); keep in sync with resample.TARGET_SR.
TARGET_SR = 16000

# Intermediate layer before tag logits: shape (num_patches, 200) for MSD_musicnn.
CANONICAL_FEATURE_KEY = "penultimate"


def _configure_tensorflow_for_musicnn() -> None:
    """TensorFlow 2.16+ defaults to Keras 3; musicnn needs legacy v1 layers."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _padded_wav_if_short(audio_path: Path, input_length_sec: float) -> tuple[Path, bool]:
    """
    musicnn's batching assumes at least one patch; very short files can leave `batch` unset.
    Return path to read (possibly a temp padded wav) and whether a temp file was created.
    """
    min_samples = int(np.ceil(input_length_sec * TARGET_SR))
    y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    if len(y) >= min_samples:
        return audio_path, False
    y = np.pad(y, (0, min_samples - len(y)), mode="constant")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    sf.write(str(tmp_path), y, TARGET_SR, subtype="PCM_16")
    return tmp_path, True


def extract_musicnn_embedding(
    audio_path: Path,
    *,
    model: str = "MSD_musicnn",
    input_length: float = 3.0,
    input_overlap: float | bool = False,
) -> dict[str, Any]:
    """
    Run MusiCNN extractor; return canonical penultimate embedding (T, C), taggram, and tag names.

    input_overlap: False for non-overlapping patches (library default), or overlap in seconds.
    """
    _configure_tensorflow_for_musicnn()
    from musicnn.extractor import extractor as musicnn_extractor

    path_for_nn, is_temp = _padded_wav_if_short(audio_path, input_length)
    try:
        taggram, tags, features = musicnn_extractor(
            str(path_for_nn),
            model=model,
            input_length=input_length,
            input_overlap=input_overlap,
            extract_features=True,
        )
    finally:
        if is_temp:
            try:
                path_for_nn.unlink()
            except OSError:
                pass

    emb = np.asarray(features[CANONICAL_FEATURE_KEY], dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    taggram_arr = np.asarray(taggram, dtype=np.float32)

    return {
        "embedding": emb,
        "embedding_shape": f"{emb.shape[0]}x{emb.shape[1]}",
        "taggram": taggram_arr,
        "tags": list(tags),
        "feature_key": CANONICAL_FEATURE_KEY,
        "model": model,
        "input_length_sec": input_length,
    }
