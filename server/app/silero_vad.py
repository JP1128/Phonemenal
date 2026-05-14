"""Silero VAD wrapper exposing the same is_speech(frame) contract as webrtcvad.

NOTE: This file backs slide 2 of ``client/app/how-it-works/page.tsx``. If
you change the VAD backend, threshold, or segmentation parameters here,
update the matching slide. See CLAUDE.md ("Documentation invariant").


Loads the bundled silero-vad PyTorch model once and runs it on 30 ms PCM
frames at 16 kHz. The model itself prefers 32 ms windows internally, so we
zero-pad / truncate as needed and let it score each window. A minimum
probability of ``speech_threshold`` is required to call a frame "speech."
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import torch


class SileroVad:
    SAMPLE_RATE = 16000

    def __init__(self, speech_threshold: float = 0.5) -> None:
        from silero_vad import load_silero_vad  # heavy import deferred

        self._model = load_silero_vad(onnx=False)
        self._model.eval()
        self._threshold = float(speech_threshold)
        self._lock = threading.Lock()

    @torch.inference_mode()
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError(f"SileroVad expects 16 kHz, got {sample_rate}")
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return False
        # Silero needs at least ~32 ms; pad if shorter.
        min_samples = self.SAMPLE_RATE * 32 // 1000
        if samples.size < min_samples:
            samples = np.pad(samples, (0, min_samples - samples.size))
        with self._lock:
            tensor = torch.from_numpy(samples)
            prob = self._model(tensor, self.SAMPLE_RATE).item()
        return prob >= self._threshold


VadLike = SileroVad  # structural type alias; webrtcvad.Vad satisfies it too


def make_vad(backend: str, *, threshold: float, aggressiveness: int) -> Callable:
    """Return an object exposing ``is_speech(frame, sample_rate) -> bool``."""
    if backend == "silero":
        return SileroVad(speech_threshold=threshold)
    if backend == "webrtc":
        import webrtcvad

        v = webrtcvad.Vad(aggressiveness)
        # Adapt webrtcvad's signature to our callable.
        class _WebrtcAdapter:
            def is_speech(self, frame: bytes, sample_rate: int) -> bool:
                return v.is_speech(frame, sample_rate)
        return _WebrtcAdapter()
    raise ValueError(f"Unknown VAD backend: {backend!r}")
