"""Runtime configuration for the Phonemer server."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16000

    # VAD / segmenter
    vad_backend: str = os.environ.get("PHONEMER_VAD", "silero")  # "silero" or "webrtc"
    vad_aggressiveness: int = 3          # webrtcvad only: 0 (least) .. 3 (most aggressive)
    vad_speech_threshold: float = 0.5    # silero only: 0..1, higher = stricter
    vad_frame_ms: int = 30               # webrtcvad accepts 10/20/30; silero uses internal 32 ms
    silence_close_ms: int = 250          # silence required to close an utterance
    min_utterance_ms: int = 400          # drop blips shorter than this
    max_utterance_ms: int = 15000        # force-cut runaway utterances
    pre_roll_ms: int = 200               # keep audio before speech onset

    # Phase 1.7 non-speech rejection
    speaker_reject_threshold: float = 0.15  # below this cosine sim to every
                                            # enrolled voice → reject the utterance

    # Calibration quality gate. Mean alignment log-prob below this is treated
    # as "the model couldn't fit this audio to the intended phonemes" (e.g.
    # silence, throat-clear, wrong transcript). Such utterances are dropped
    # before they pollute the confusion matrix. -3.5 ≈ avg per-frame prob 0.03.
    min_alignment_score: float = -3.5

    # Phoneme model
    phoneme_model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    top_k_phonemes: int = 10

    # Speaker ID
    speaker_model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_model_dir: str = os.environ.get(
        "PHONEMER_SPEAKER_DIR", "data/models/spkrec-ecapa"
    )
    speaker_match_threshold: float = 0.55  # cosine sim above => same speaker

    # STT (non-patient context)
    stt_model_size: str = os.environ.get("PHONEMER_WHISPER_SIZE", "base.en")
    stt_compute_type: str = os.environ.get("PHONEMER_WHISPER_COMPUTE", "int8")

    # Storage
    db_path: str = os.environ.get("PHONEMER_DB_PATH", "data/phonemer.db")
    audio_dir: str = os.environ.get("PHONEMER_AUDIO_DIR", "data/audio")

    # LLM (Phase 3)
    translation_model: str = os.environ.get(
        "PHONEMER_TRANSLATION_MODEL", "claude-sonnet-4-6"
    )
    analysis_model: str = os.environ.get(
        "PHONEMER_ANALYSIS_MODEL", "claude-opus-4-7"
    )
    translation_max_tokens: int = 1024
    translation_recent_utterances: int = 20

    # Runtime
    device: str = os.environ.get("PHONEMER_DEVICE", "cpu")
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)


settings = Settings()
