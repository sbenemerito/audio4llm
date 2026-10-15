"""Audio file helpers built on soundfile (no torchaudio I/O anywhere).

Arrays are float32 shaped (channels, samples), which is what Demucs expects.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf


def load(path: str | Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    y = data.T.copy()
    if mono:
        y = y.mean(axis=0, keepdims=True)
    if sr is not None and sr != file_sr:
        y = resample(y, file_sr, sr)
        file_sr = sr
    return y, file_sr


def load_mono(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    y, rate = load(path, sr=sr, mono=True)
    return y[0], rate


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import soxr

    if y.ndim == 1:
        return soxr.resample(y, sr_in, sr_out).astype(np.float32)
    return soxr.resample(y.T, sr_in, sr_out).T.astype(np.float32).copy()


def save(path: str | Path, y: np.ndarray, sr: int, subtype: str = "FLOAT") -> Path:
    """Write (channels, samples) or (samples,) audio atomically."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    data = y.T if y.ndim == 2 else y
    sf.write(str(tmp), data, sr, subtype=subtype)
    os.replace(tmp, path)
    return path


def duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate
