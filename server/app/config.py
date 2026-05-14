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
    # Silence required to close an utterance. Tuned for the worst-case
    # dysarthric speaker the system aims to support. Earlier defaults of
    # 800/1500/2000 ms all still let VAD split mid-sentence on more
    # severe speakers — M01 reading "The islands are sparsely populated"
    # produced two utterances 2-3 s apart, captured separately as "The
    # iris, iris." and "fartybodyweeding". 3000 ms is the floor for that
    # population; tune per-session via PHONEMER_SILENCE_CLOSE_MS=3500
    # if a particular patient still splits. Trade-off: listener
    # turn-taking gets ~1 s less responsive, which is acceptable because
    # the clinician naturally waits on the patient anyway. We did not
    # add per-speaker-role thresholds because VAD runs before
    # speaker-ID — coupling the two subsystems introduces more failure
    # modes than the slow-listener cost it would avoid.
    silence_close_ms: int = int(os.environ.get("PHONEMER_SILENCE_CLOSE_MS", "3000"))
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

    # Live safety-gate threshold for the patient transcription path
    # (separate from the calibration drop above). Tightened from -1.0 to
    # -0.85 after F03 "A roll of wire lay near the wall" → "I need a
    # little wired lay Diego mall" (score -0.900) showed fluent
    # LM-hallucinations slipping under the old threshold. Combined with
    # n-best rescoring (which raises the typical picked-score), -0.85
    # flags the borderline cases without blowing up FP rate.
    gate_min_alignment_score: float = float(
        os.environ.get("PHONEMER_GATE_MIN_ALIGN", "-0.85")
    )
    # How many Whisper candidates to rescore against the wav2vec2
    # posterior before picking the final transcription. 1 = greedy
    # (legacy behavior). >= 2 enables *diverse* beam search (one beam
    # per group + diversity penalty + early_stopping), which gives
    # genuinely different candidates instead of N near-identical ones.
    # Default 3 is chosen to fit inside the frontend's 12s
    # pending-translation idle timer on whisper-small.en CPU
    # (~4-7s per utterance warm). Bump to 5 if you have GPU or are
    # willing to extend the idle timer.
    gate_n_best: int = int(os.environ.get("PHONEMER_GATE_N_BEST", "3"))

    # Under-production density check. Density is measured over the
    # VOICED portion of the audio (via wav2vec2 blank-frame ratio),
    # not total duration — that fix avoids false-flagging fluent
    # speakers like FC02 who pause heavily on prompted reading
    # (2.3 s of silence in a 4 s clip dropped the old total-duration
    # density to 5.8 ph/s and flagged a clean transcription).
    # With voiced-duration density, natural speech sits at 10-15
    # ph/s, slow dysarthric phonation at 5-8 ph/s, and real
    # under-production (Whisper bailed mid-utterance while audio
    # kept phonating) drops to 2-4 ph/s. 4.0 is the floor that
    # flags real under-production without false-flagging slow-but-
    # correct dysarthric speech.
    gate_min_phoneme_density: float = float(
        os.environ.get("PHONEMER_GATE_MIN_DENSITY", "4.0")
    )
    gate_density_check_min_duration_s: float = float(
        os.environ.get("PHONEMER_GATE_DENSITY_MIN_DURATION", "2.0")
    )

    # UI-level low-confidence threshold. When overall_confidence
    # (= exp(align_score)) drops below this, the listener UI gets the
    # same red-ring treatment as a flagged transcription. Catches
    # cases where the phoneme alignment gate didn't trip but the
    # per-word confidences are uniformly low — usually a sign the
    # patient was effortful and Whisper produced fluent-but-shallow
    # text. 0.5 ≈ exp(-0.69), so this kicks in around align_score
    # = -0.7 even when the gate threshold is -0.85.
    ui_low_confidence_threshold: float = float(
        os.environ.get("PHONEMER_LOW_CONFIDENCE_THRESHOLD", "0.55")
    )

    # Phoneme model
    phoneme_model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    top_k_phonemes: int = 10

    # Speaker ID
    speaker_model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_model_dir: str = os.environ.get(
        "PHONEMER_SPEAKER_DIR", "data/models/spkrec-ecapa"
    )
    speaker_match_threshold: float = 0.55  # cosine sim above => same speaker

    # STT (non-patient context — listener speech uses plain faster-whisper)
    stt_model_size: str = os.environ.get("PHONEMER_WHISPER_SIZE", "base.en")
    stt_compute_type: str = os.environ.get("PHONEMER_WHISPER_COMPUTE", "int8")

    # Patient adapter — if a directory with a PEFT LoRA exists at this path,
    # the patient transcription path uses HuggingFace Whisper + adapter
    # instead of plain faster-whisper. Empty = always use plain Whisper.
    # The directory should contain ``adapter_model.safetensors``,
    # ``adapter_config.json``, and ``BASE_MODEL.txt`` (output of
    # train_patient_adapter.py).
    patient_adapter_dir: str = os.environ.get(
        "PHONEMER_PATIENT_ADAPTER_DIR", "data/adapters/patient"
    )

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
