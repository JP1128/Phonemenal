"""Voice activity detection + utterance segmentation.

The active VAD backend is plug-replaceable: ``webrtcvad`` (fast, lossy on
non-speech transients) or ``silero`` (the default; better at suppressing
keyboard/mouse/paper noise). Both expose ``is_speech(frame, sample_rate)``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


class VadBackend(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


@dataclass
class Utterance:
    start_ms: int           # absolute ms from session start
    end_ms: int
    audio: np.ndarray       # float32 mono in [-1, 1], 16 kHz


class UtteranceSegmenter:
    """Streams int16 PCM in, yields Utterance objects on silence boundaries.

    Sub-frames are scored by ``vad`` (webrtcvad-style or Silero). Silence
    longer than ``silence_close_ms`` closes the active utterance. A pre-roll
    buffer keeps audio from before speech onset so consonants aren't clipped.
    """

    def __init__(
        self,
        vad: VadBackend,
        sample_rate: int,
        frame_ms: int,
        silence_close_ms: int,
        min_utterance_ms: int,
        max_utterance_ms: int,
        pre_roll_ms: int,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be one of {10, 20, 30}")
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.frame_ms = frame_ms
        self._vad = vad
        self._silence_close_frames = max(1, silence_close_ms // frame_ms)
        self._min_frames = max(1, min_utterance_ms // frame_ms)
        self._max_frames = max(self._min_frames + 1, max_utterance_ms // frame_ms)
        self._pre_roll: deque[bytes] = deque(maxlen=max(1, pre_roll_ms // frame_ms))

        self._raw = bytearray()
        self._absolute_frame_index = 0  # frames consumed (each = frame_ms)

        self._in_speech = False
        self._active_frames: list[bytes] = []
        self._active_start_frame = 0
        self._silence_run = 0

    # ------------------------------------------------------------------ ingest

    def append_pcm16(self, raw: bytes) -> None:
        if raw:
            self._raw.extend(raw)

    def utterances(self) -> Iterator[Utterance]:
        bytes_per_frame = self.frame_samples * 2
        while len(self._raw) >= bytes_per_frame:
            frame = bytes(self._raw[:bytes_per_frame])
            del self._raw[:bytes_per_frame]
            yield from self._step(frame)

    def flush(self) -> Iterator[Utterance]:
        """Emit any in-flight utterance (called on disconnect / stop)."""
        if self._in_speech and len(self._active_frames) >= self._min_frames:
            yield self._close_active()
        self._reset_active()

    # ------------------------------------------------------------------- core

    def _step(self, frame: bytes) -> Iterator[Utterance]:
        try:
            is_speech = self._vad.is_speech(frame, self.sample_rate)
        except Exception:
            # Treat backend hiccups as silence rather than crashing the stream.
            is_speech = False
        self._absolute_frame_index += 1

        if not self._in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                # Start new utterance, prepending pre-roll for clean onset.
                self._in_speech = True
                self._active_frames = list(self._pre_roll)
                self._active_start_frame = (
                    self._absolute_frame_index - len(self._active_frames)
                )
                self._silence_run = 0
                self._pre_roll.clear()
            return

        self._active_frames.append(frame)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self._silence_close_frames:
            yield self._close_active()
            return

        if len(self._active_frames) >= self._max_frames:
            yield self._close_active()

    def _close_active(self) -> Utterance:
        # Trim trailing silence, but keep a little for natural prosody.
        keep_silence = max(1, self._silence_close_frames // 2)
        trimmed = self._active_frames
        if self._silence_run > keep_silence:
            trimmed = trimmed[: -(self._silence_run - keep_silence)]

        n_frames = len(trimmed)
        too_short = n_frames < self._min_frames
        start_frame = self._active_start_frame
        end_frame = start_frame + n_frames

        self._reset_active()

        if too_short:
            # Return a sentinel-ish utterance that the caller can ignore.
            return Utterance(
                start_ms=start_frame * self.frame_ms,
                end_ms=end_frame * self.frame_ms,
                audio=np.zeros(0, dtype=np.float32),
            )

        pcm = b"".join(trimmed)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return Utterance(
            start_ms=start_frame * self.frame_ms,
            end_ms=end_frame * self.frame_ms,
            audio=samples,
        )

    def _reset_active(self) -> None:
        self._in_speech = False
        self._active_frames = []
        self._silence_run = 0
        self._pre_roll.clear()
