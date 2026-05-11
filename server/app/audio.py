"""Audio I/O helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def save_wav(audio: np.ndarray, sample_rate: int, out_dir: str, name: str) -> str:
    """Write float32 mono audio to a 16-bit PCM WAV; return its path."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.wav"
    pcm = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), pcm, sample_rate, subtype="PCM_16")
    return str(path)
