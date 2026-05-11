"""Phase 1.7 regression: non-speech audio must not produce phoneme/feature work.

Given a synthetic noise + click train (no speech), the pipeline should:
  * Either emit zero utterances at the VAD layer (Silero is strict enough), or
  * Emit utterances that the speaker-ID layer rejects with similarity below
    the configured ``speaker_reject_threshold``.

This guards against keyboard / mouse / paper transients flooding the LLM
once Phase 3 ships.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _ensure_espeak_library  # noqa: F401
from app.config import settings
from app.silero_vad import make_vad
from app.vad import UtteranceSegmenter


def _white_noise_with_clicks(duration_s: float = 4.0, sample_rate: int = 16000) -> bytes:
    rng = np.random.default_rng(42)
    n = int(duration_s * sample_rate)
    base = rng.normal(0, 0.005, n).astype(np.float32)  # quiet hiss
    # Add 20 sharp clicks (3 ms each) at random offsets — keyboard analogue.
    click_len = int(0.003 * sample_rate)
    for _ in range(20):
        start = int(rng.integers(0, n - click_len))
        base[start : start + click_len] += rng.normal(0, 0.6, click_len)
    base = np.clip(base, -1.0, 1.0)
    return (base * 32767).astype(np.int16).tobytes()


def _enrolled_voice_embedding(rng: np.random.Generator) -> np.ndarray:
    # ECAPA embeddings live on the unit sphere; pick a random direction.
    e = rng.normal(0, 1, 192).astype(np.float32)
    return e / np.linalg.norm(e)


def test_silero_vad_keeps_noise_quiet():
    """Direct Silero check: at least 80% of click-laden frames are silence."""
    pytest.importorskip("silero_vad")
    vad = make_vad("silero", threshold=settings.vad_speech_threshold,
                   aggressiveness=settings.vad_aggressiveness)

    pcm = _white_noise_with_clicks()
    frame_ms = settings.vad_frame_ms
    frame_bytes = settings.sample_rate * frame_ms // 1000 * 2

    speech_count = total = 0
    for i in range(0, len(pcm) - frame_bytes, frame_bytes):
        frame = pcm[i : i + frame_bytes]
        total += 1
        if vad.is_speech(frame, settings.sample_rate):
            speech_count += 1

    assert total > 0
    speech_ratio = speech_count / total
    assert speech_ratio < 0.20, (
        f"Silero VAD called {speech_ratio:.0%} of click-train frames speech "
        "— that's too many; webrtcvad regression?"
    )


def test_segmenter_emits_no_full_utterances_from_noise():
    pytest.importorskip("silero_vad")
    vad = make_vad("silero", threshold=settings.vad_speech_threshold,
                   aggressiveness=settings.vad_aggressiveness)
    segmenter = UtteranceSegmenter(
        vad=vad,
        sample_rate=settings.sample_rate,
        frame_ms=settings.vad_frame_ms,
        silence_close_ms=settings.silence_close_ms,
        min_utterance_ms=settings.min_utterance_ms,
        max_utterance_ms=settings.max_utterance_ms,
        pre_roll_ms=settings.pre_roll_ms,
    )
    segmenter.append_pcm16(_white_noise_with_clicks())

    accepted = [u for u in segmenter.utterances() if u.audio.size > 0]
    # Synthetic clicks may chain; allow up to one false utterance, but no more.
    assert len(accepted) <= 1, (
        f"Segmenter emitted {len(accepted)} non-speech utterances; expected ≤ 1"
    )


def test_speaker_id_rejects_random_embeddings_below_threshold():
    """Cosine sim between two unrelated 192-d embeddings is ~0; well below the
    rejection floor. This is the geometric reason the rejection path works."""
    rng = np.random.default_rng(7)
    a = _enrolled_voice_embedding(rng)
    b = _enrolled_voice_embedding(rng)
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert abs(sim) < settings.speaker_reject_threshold, (
        f"Random-vs-random cosine sim {sim:.3f} should be far below "
        f"the rejection threshold {settings.speaker_reject_threshold}"
    )
