from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def load_audio(path: str | Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Load an audio file as mono float32 numpy array."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    waveform, sr = librosa.load(path.as_posix(), sr=None, mono=True)
    if target_sr and sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


__all__ = ['load_audio', 'resample_audio']
