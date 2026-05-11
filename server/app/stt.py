"""faster-whisper transcription for non-patient (context-feed) speech."""
from __future__ import annotations

import threading

import numpy as np


class Transcriber:
    """Lazily loads faster-whisper; transcribes float32 mono 16 kHz audio."""

    def __init__(self, model_size: str, compute_type: str, device: str) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._device = "cpu" if device == "cpu" else device
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # heavy import, deferred

        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        with self._lock:
            self._ensure_loaded()
            segments, _info = self._model.transcribe(
                audio.astype(np.float32, copy=False),
                language="en",
                vad_filter=False,
                beam_size=1,
            )
            return " ".join(s.text.strip() for s in segments).strip()
