from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from preprocess.paths import track_id
from preprocess.resample import TARGET_SR

# Match musicnn.configuration + musicnn.extractor.batch_data (MusiCNN input).
FFT_HOP = 256
FFT_SIZE = 512
N_MELS = 96
LOG_OFFSET = 10_000.0


def log_mel_spectrogram_musicnn(audio_16k_wav: Path) -> np.ndarray:
    """
    Full-track log-mel spectrogram in the same convention as MusiCNN: time × n_mels,
    log10(10000 * mel + 1), using librosa melspectrogram with MusiCNN's STFT/mel settings.
    """
    audio, sr = librosa.load(str(audio_16k_wav), sr=TARGET_SR, mono=True)
    if sr != TARGET_SR:
        raise ValueError(f"Expected {TARGET_SR} Hz audio, got {sr}")
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        hop_length=FFT_HOP,
        n_fft=FFT_SIZE,
        n_mels=N_MELS,
    ).T
    mel = mel.astype(np.float32)
    return np.log10(LOG_OFFSET * mel + 1.0).astype(np.float32)


def dest_path_spectrogram(output_root: Path, source_rel_posix: str) -> Path:
    tid = track_id(source_rel_posix)
    return output_root / "spectrograms" / f"{tid}.npz"


def save_spectrogram_npz(
    audio_16k_wav: Path,
    dest_npz: Path,
    *,
    source_rel_path: str,
) -> None:
    log_mel = log_mel_spectrogram_musicnn(audio_16k_wav)
    dest_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest_npz,
        log_mel=log_mel,
        sr=np.int32(TARGET_SR),
        hop_length=np.int32(FFT_HOP),
        n_fft=np.int32(FFT_SIZE),
        n_mels=np.int32(N_MELS),
        source_rel_path=np.asarray(source_rel_path),
    )
