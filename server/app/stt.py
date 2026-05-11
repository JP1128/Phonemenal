"""faster-whisper transcription, plus a phoneme-alignment sanity gate.

The gate addresses the empirical finding that the wav2vec2 phoneme
distribution doesn't contain enough additional signal to *correct*
Whisper's near-misses, but it does reliably catch Whisper's worst
failure mode: hallucinated long transcriptions for short utterances.
A clip with 1 s of audio can't legitimately phonemize as 30 phonemes,
so when Whisper proposes that, forced alignment of those phonemes
against the audio scores catastrophically low.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

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


@dataclass
class GatedTranscription:
    text: str                 # Whisper's transcription, unchanged
    align_score: float        # mean forced-align log-prob of `text` vs audio
    phoneme_density: float    # phonemes per second of audio
    flagged: bool             # True when the gate suspects hallucination
    reason: str | None        # "low_alignment" / "phoneme_density" / None


class GatedTranscriber:
    """Whisper output gated by phoneme-alignment plausibility.

    Two signals are checked against the audio's wav2vec2 phoneme
    distribution:

      * Mean alignment log-prob of Whisper's text. Hallucinated
        transcriptions don't actually fit the audio, so the alignment
        score collapses far below the typical -1 to -1.5 range.
      * Phoneme density (phonemes per second of audio). Normal speech
        runs ~10-15 phonemes/s; a Whisper hallucination of 30+
        phonemes for 1 s of audio is physically impossible.

    Either signal alone is enough to flag a clip. The intent is *not*
    to correct the transcription — empirically the phoneme pipeline
    can't do that — but to refuse to confidently surface a
    transcription the audio doesn't support. Critical for medical
    settings where confident-wrong is worse than uncertain.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        extractor,
        token_to_id: dict[str, int],
        blank_id: int,
        sample_rate: int = 16000,
        min_alignment_score: float = -1.0,
        max_phoneme_density: float = 25.0,
    ) -> None:
        self._transcriber = transcriber
        self._extractor = extractor
        self._token_to_id = token_to_id
        self._blank_id = blank_id
        self._sample_rate = sample_rate
        self._min_alignment = min_alignment_score
        self._max_density = max_phoneme_density

    def transcribe_with_gate(
        self,
        audio: np.ndarray,
        probs: np.ndarray | None = None,
    ) -> GatedTranscription:
        """Run Whisper + alignment gate. ``probs`` may be pre-computed
        wav2vec2 softmax output to avoid duplicating extraction work
        when the caller already ran the phoneme model.
        """
        from .alignment import alignment_score, forced_align, text_to_phonemes

        text = self._transcriber.transcribe(audio).strip()
        if not text or audio.size == 0:
            return GatedTranscription(
                text=text, align_score=0.0, phoneme_density=0.0,
                flagged=False, reason=None,
            )

        phonemes = text_to_phonemes(text)
        if not phonemes:
            return GatedTranscription(
                text=text, align_score=0.0, phoneme_density=0.0,
                flagged=False, reason=None,
            )

        duration_s = max(audio.size / self._sample_rate, 1e-3)
        density = len(phonemes) / duration_s

        if probs is None:
            probs = self._extractor.extract(audio, self._sample_rate)
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        aligned = forced_align(
            log_probs, phonemes, self._token_to_id, self._blank_id
        )
        score = alignment_score(aligned)

        reason: str | None = None
        if density > self._max_density:
            reason = "phoneme_density"
        elif score < self._min_alignment:
            reason = "low_alignment"
        return GatedTranscription(
            text=text,
            align_score=score,
            phoneme_density=density,
            flagged=reason is not None,
            reason=reason,
        )
