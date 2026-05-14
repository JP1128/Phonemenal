"""faster-whisper transcription, plus a phoneme-alignment sanity gate.

NOTE: This file backs slides 5 and 6 of ``client/app/how-it-works/page.tsx``.
If you change the base model name, the LoRA wiring, or any of the gate
thresholds here, also update the matching slide descriptions / threshold
values on that page in the same commit. See CLAUDE.md ("Documentation
invariant") for the full rule.

The gate addresses the empirical finding that the wav2vec2 phoneme
distribution doesn't contain enough additional signal to *correct*
Whisper's near-misses, but it does reliably catch Whisper's worst
failure mode: hallucinated long transcriptions for short utterances.
A clip with 1 s of audio can't legitimately phonemize as 30 phonemes,
so when Whisper proposes that, forced alignment of those phonemes
against the audio scores catastrophically low.

Two transcriber implementations:

  * ``Transcriber`` — faster-whisper (CTranslate2), used for non-patient
    speech (listener / caregiver). Fast, no per-speaker customization.
  * ``AdaptedTranscriber`` — HuggingFace Whisper + PEFT LoRA adapter,
    used for the patient when an adapter is available. ~3x slower than
    faster-whisper but produces patient-specialized output (-32% to
    -47% relative WER on TORGO dysarthric speakers vs naive Whisper).

Both share the ``transcribe(audio, initial_prompt="")`` interface so
``GatedTranscriber`` can wrap either.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("phonemer.stt")


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

    def transcribe(self, audio: np.ndarray, initial_prompt: str = "") -> str:
        """Transcribe audio. ``initial_prompt`` (optional) is fed to
        faster-whisper as conversational context — recent listener turns,
        active topics, patient personal vocabulary. Whisper uses it as a
        soft prior on what to expect; doesn't override clear acoustic
        evidence but biases borderline calls."""
        if audio.size == 0:
            return ""
        with self._lock:
            self._ensure_loaded()
            kwargs: dict = dict(
                language="en",
                vad_filter=False,
                beam_size=1,
            )
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            segments, _info = self._model.transcribe(
                audio.astype(np.float32, copy=False),
                **kwargs,
            )
            return " ".join(s.text.strip() for s in segments).strip()


class AdaptedTranscriber:
    """HuggingFace Whisper + per-patient PEFT LoRA adapter.

    Loads the base model and the patient's adapter at first use, then runs
    inference on the patient-specialized model. Uses HF transformers' Whisper
    (not faster-whisper / CTranslate2) because PEFT adapters are trained
    against the HF model and can't be applied to the CT2 binary. Tradeoff:
    ~3x slower inference than faster-whisper, but ships the patient
    personalization gain (-32 to -47% relative WER on TORGO dysarthric
    speakers vs naive Whisper).
    """

    def __init__(
        self,
        adapter_dir: str,
        device: str = "cpu",
        base_model: str | None = None,
    ) -> None:
        path = Path(adapter_dir)
        if base_model is None:
            base_marker = path / "BASE_MODEL.txt"
            if base_marker.exists():
                base_model = base_marker.read_text().strip()
            else:
                base_model = "openai/whisper-small.en"
        self._adapter_dir = str(path)
        self._base_model = base_model
        self._device = device
        self._processor = None
        self._model = None
        self._lock = threading.Lock()
        # PEFT supports multiple adapters on a single base model. We load
        # each speaker's LoRA delta on demand and swap with set_adapter()
        # — base Whisper is shared so memory cost per extra adapter is
        # only the LoRA weights (~30 MB) rather than a full model.
        self._loaded_adapters: dict[str, str] = {}  # name -> dir
        self._active_adapter: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # heavy, deferred
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info(
            "Loading patient adapter %s on top of %s (device=%s)",
            self._adapter_dir, self._base_model, self._device,
        )
        self._processor = WhisperProcessor.from_pretrained(self._base_model)
        base = WhisperForConditionalGeneration.from_pretrained(self._base_model)
        # Whisper's generation config sets forced_decoder_ids that conflict
        # with custom prompting; clear them so initial_prompt is the only
        # decoder priming signal.
        base.config.forced_decoder_ids = None
        base.config.suppress_tokens = []
        self._model = (
            PeftModel.from_pretrained(
                base, self._adapter_dir, adapter_name="default"
            )
            .to(self._device)
            .eval()
        )
        self._loaded_adapters["default"] = self._adapter_dir
        self._active_adapter = "default"
        self._torch = torch
        logger.info("Patient adapter ready")

    def _ensure_adapter(self, adapter_dir: str) -> None:
        """Make ``adapter_dir`` the active adapter. Caller holds the lock.

        Idempotent — no-op if already active. Loads the LoRA delta on
        first request for a given dir, otherwise swaps via PEFT
        ``set_adapter``. Falls back to the default adapter on any error so
        a missing/broken speaker adapter doesn't block transcription.
        """
        # Already active?
        for name, path in self._loaded_adapters.items():
            if path == adapter_dir:
                if self._active_adapter != name:
                    self._model.set_adapter(name)
                    self._active_adapter = name
                return
        # Not yet seen — try to load.
        new_name = f"adapter_{len(self._loaded_adapters)}"
        try:
            self._model.load_adapter(adapter_dir, adapter_name=new_name)
            self._model.set_adapter(new_name)
            self._loaded_adapters[new_name] = adapter_dir
            self._active_adapter = new_name
            logger.info("Loaded patient adapter %s as %s", adapter_dir, new_name)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to load adapter %s (%s) — using default", adapter_dir, e
            )
            if self._active_adapter != "default":
                self._model.set_adapter("default")
                self._active_adapter = "default"

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str = "",
        adapter_dir: str | None = None,
    ) -> str:
        """Match ``Transcriber.transcribe`` interface.

        ``adapter_dir`` lets a caller pin a specific per-speaker adapter
        for this single transcription. When omitted the default adapter
        (the one passed to ``__init__``) is used.
        """
        candidates = self.transcribe_n_best(
            audio, initial_prompt=initial_prompt, adapter_dir=adapter_dir, n_best=1,
        )
        return candidates[0] if candidates else ""

    def transcribe_n_best(
        self,
        audio: np.ndarray,
        initial_prompt: str = "",
        adapter_dir: str | None = None,
        n_best: int = 1,
    ) -> list[str]:
        """Generate ``n_best`` candidate transcriptions for rescoring.

        Used by ``GatedTranscriber`` to rerank candidates against the
        wav2vec2 phoneme posterior. The greedy (n_best=1) path is what
        ``transcribe()`` returns; n_best>=2 trades inference time for
        the chance to recover an acoustically faithful candidate when
        Whisper's top beam picked an LM-fluent hallucination.

        For n_best>=2 we use *diverse beam search* (one beam per group +
        diversity penalty) instead of plain beam search. Empirically,
        plain beam search on confident decodes returns the same string
        n times — paying 5x compute for zero rescoring signal. Diverse
        beam search forces the groups apart so the rescoring loop
        actually has alternatives to choose between.

        ``early_stopping=True`` lets generation halt as soon as every
        beam has emitted EOS, which materially shortens decode on
        short utterances.

        Returns up to ``n_best`` strings ordered by the model's
        score (best first).
        """
        if audio.size == 0:
            return [""]
        n_best = max(1, int(n_best))
        with self._lock:
            self._ensure_loaded()
            if adapter_dir:
                self._ensure_adapter(adapter_dir)
            elif self._active_adapter != "default":
                self._model.set_adapter("default")
                self._active_adapter = "default"
            torch = self._torch  # local alias for the inference_mode context
            with torch.inference_mode():
                feats = self._processor.feature_extractor(
                    audio.astype(np.float32, copy=False),
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(self._device)
                gen_kwargs: dict
                if n_best <= 1:
                    gen_kwargs = dict(
                        max_new_tokens=128,
                        num_beams=1,
                        do_sample=False,
                    )
                else:
                    # Plain (deterministic) beam search. We previously
                    # used multinomial sampling at temp=0.8 here for
                    # candidate diversity, but that made the same audio
                    # produce *different transcripts on different runs*
                    # — clinically unacceptable. (M01 "The islands are
                    # sparsely populated" came out as "the island is...
                    # sparley populated" one run and "the island has
                    # hardly evacuated" the next, on byte-identical
                    # audio.) Beam search is deterministic and gives
                    # diversity naturally when Whisper is uncertain
                    # (different beams find different paths through
                    # the decoder); when Whisper is confident the
                    # beams collapse to the same string and the
                    # rescoring is a harmless no-op. Diverse beam
                    # search (num_beam_groups + diversity_penalty)
                    # would be nicer here but requires
                    # ``trust_remote_code=True`` in transformers 5.8
                    # and isn't worth the supply-chain cost.
                    gen_kwargs = dict(
                        max_new_tokens=128,
                        num_beams=n_best,
                        num_return_sequences=n_best,
                        do_sample=False,
                        early_stopping=True,
                    )
                # Selectively disable Whisper's quality-fallback
                # checks. By default Whisper aborts a segment when:
                #   * compression_ratio > 2.4 (looks like repetition)
                #   * mean logprob < -1.0 (low decode confidence)
                #   * no_speech_prob > 0.6 (looks like silence)
                # The first two trigger spuriously on effortful
                # dysarthric audio — patient speech *is* low-logprob
                # and sometimes repetitive — and Whisper would drop
                # half the sentence ("The islands are sparsely
                # populated" → "The island is...."). So those two
                # stay off.
                #
                # ``no_speech_threshold`` is left ON (default 0.6).
                # Disabling it produced the regression where ambient
                # mic noise during VAD-merged silence windows got
                # hallucinated into phantom utterances. Whisper's
                # no-speech detector is well-calibrated for actual
                # silence vs. articulated speech, even on patient
                # voices, so it's the right gate to keep.
                gen_kwargs["compression_ratio_threshold"] = None
                gen_kwargs["logprob_threshold"] = None
                # gen_kwargs["no_speech_threshold"] left at default
                if initial_prompt and initial_prompt.strip():
                    tok = self._processor.tokenizer
                    prev_id = tok.convert_tokens_to_ids("<|startofprev|>")
                    # Whisper's decoder context is 224 tokens; reserve room
                    # for the prev marker plus generation.
                    word_ids = tok(
                        " " + initial_prompt.strip(),
                        add_special_tokens=False,
                    ).input_ids[:200]
                    gen_kwargs["prompt_ids"] = torch.tensor(
                        [prev_id] + word_ids, device=self._device
                    )
                out = self._model.generate(feats, **gen_kwargs)
                texts = self._processor.tokenizer.batch_decode(
                    out, skip_special_tokens=True
                )
                return [t.strip() for t in texts]


@dataclass
class GatedTranscription:
    text: str                 # Whisper's transcription, unchanged
    align_score: float        # mean forced-align log-prob of `text` vs audio
    phoneme_density: float    # phonemes per second of VOICED audio (see below)
    flagged: bool             # True when the gate suspects hallucination
    reason: str | None        # "low_alignment" / "phoneme_density" / None
    # voiced_fraction: estimated fraction of audio frames the wav2vec2
    # head thinks are phonation rather than silence (1 - blank-frame
    # ratio). phoneme_density is computed over the voiced duration —
    # phonemes / (audio_duration × voiced_fraction). Without this,
    # fluent speakers who pause heavily for prompted reading (e.g.
    # FC02 reading TIMIT with 2 s of silence between words inside a
    # 4 s clip) trip the underproduction check despite saying every
    # intended word at a normal phonation rate.
    voiced_fraction: float = 1.0
    # Per-word breakdown of `align_score`. Same length as `words`. The
    # score for each word is the mean alignment log-prob of *that
    # word's* phonemes against the audio — a real "did the audio
    # support this word" signal, not a uniform default. Words whose
    # phonemes were dropped by the vocab filter (rare) get -inf.
    words: list[str] = field(default_factory=list)
    per_word_scores: list[float] = field(default_factory=list)


class GatedTranscriber:
    """Whisper output gated by phoneme-alignment plausibility.

    Three signals are checked against the audio's wav2vec2 phoneme
    distribution:

      * Mean alignment log-prob of Whisper's text. Hallucinated
        transcriptions don't actually fit the audio, so the alignment
        score collapses far below the typical -1 to -1.5 range.
      * Phoneme density too high (over-production). Normal speech
        runs ~10-15 phonemes/s; a Whisper hallucination of 30+
        phonemes for 1 s of audio is physically impossible.
      * Phoneme density too low (under-production). The opposite
        failure mode: Whisper drops most of the audio and emits a
        short transcription that doesn't account for what was said
        ("...about my grandfather" → "You will soon..."). The
        alignment score is silent on this because it can find the
        few proposed phonemes somewhere in the audio. Only checked
        on utterances longer than ``density_check_min_duration_s``
        so we don't false-flag legitimately short single-word
        stimuli.

    Any single signal alone is enough to flag a clip. The intent is
    not to correct the transcription — empirically the phoneme
    pipeline can't do that — but to refuse to confidently surface a
    transcription the audio doesn't support.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        extractor,
        token_to_id: dict[str, int],
        blank_id: int,
        sample_rate: int = 16000,
        min_alignment_score: float = -0.85,
        max_phoneme_density: float = 25.0,
        # Threshold measured over VOICED duration (not total audio).
        # Natural speech runs 10-15 ph/s during phonation; slow dysarthric
        # phonation lands 5-8 ph/s; under-production typically drops to
        # 2-4 ph/s because Whisper emits short text against many seconds
        # of voiced audio. 4.0 is the floor that flags under-production
        # without false-flagging slow-but-correct dysarthric speech.
        # Per-patient overrides arrive at call time via ``adapter_dir``'s
        # calibration.json — see ``transcribe_with_gate``.
        min_phoneme_density: float = 4.0,
        density_check_min_duration_s: float = 2.0,
        n_best: int = 3,
    ) -> None:
        # Threshold note: tightened from -1.0 → -0.85 after the
        # diagnostic on F03 "A roll of wire lay near the wall" → "I need
        # a little wired lay Diego mall" (score -0.900) revealed that
        # fluent LM-hallucinations with phonetically-plausible words
        # could land just below the old threshold. Combined with N-best
        # rescoring (which raises the *typical* score by picking the
        # best beam), -0.85 is meant to flag the borderline cases the
        # old gate let through. Tunable via PHONEMER_GATE_MIN_ALIGN.
        self._transcriber = transcriber
        self._extractor = extractor
        self._token_to_id = token_to_id
        self._blank_id = blank_id
        self._sample_rate = sample_rate
        self._min_alignment = min_alignment_score
        self._max_density = max_phoneme_density
        self._min_density = min_phoneme_density
        self._min_density_duration = density_check_min_duration_s
        self._n_best = max(1, int(n_best))
        # Cache: adapter_dir → {min_alignment_score, min_phoneme_density,
        # ui_low_confidence_threshold} loaded from that adapter's
        # ``calibration.json``. Empty dict means "no calibration file" and
        # the global defaults above apply. The cache is keyed by string
        # path so swapping between adapters mid-session doesn't reread.
        self._calibration_cache: dict[str, dict] = {}

    def _load_calibration(self, adapter_dir: str) -> dict:
        """Return the ``thresholds`` block from ``{adapter_dir}/calibration.json``.

        Cached by adapter_dir string. Returns empty dict on miss / error
        — callers fall back to the instance defaults. The calibration
        file is produced by ``scripts/compute_calibration.py`` after
        adapter training.
        """
        cached = self._calibration_cache.get(adapter_dir)
        if cached is not None:
            return cached
        thresholds: dict = {}
        try:
            from pathlib import Path
            import json as _json
            cal_path = Path(adapter_dir) / "calibration.json"
            if cal_path.exists():
                data = _json.loads(cal_path.read_text())
                thresholds = dict(data.get("thresholds") or {})
                logger.info(
                    "loaded per-patient calibration from %s: %s",
                    cal_path, thresholds,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "failed to load calibration.json from %s: %s",
                adapter_dir, e,
            )
        self._calibration_cache[adapter_dir] = thresholds
        return thresholds

    def get_calibration_thresholds(self, adapter_dir: str | None) -> dict:
        """Public accessor so ``_process_patient`` can read the same
        per-patient thresholds the gate is using (for the UI-level
        ``ui_low_confidence_threshold`` override) without re-parsing
        the file or reaching into the gate's private cache."""
        if not adapter_dir:
            return {}
        return self._load_calibration(adapter_dir)

    def transcribe_with_gate(
        self,
        audio: np.ndarray,
        probs: np.ndarray | None = None,
        initial_prompt: str = "",
        adapter_dir: str | None = None,
    ) -> GatedTranscription:
        """Run Whisper + alignment gate. ``probs`` may be pre-computed
        wav2vec2 softmax output to avoid duplicating extraction work
        when the caller already ran the phoneme model. ``initial_prompt``
        is passed through to Whisper as conversational/vocabulary
        context. ``adapter_dir``, if set and the wrapped transcriber is
        an ``AdaptedTranscriber``, swaps the active LoRA for this call.

        When ``self._n_best > 1`` and the wrapped transcriber supports
        it (``AdaptedTranscriber``), this generates that many beam
        candidates and reranks by phoneme-alignment quality before
        applying the gate. The picked candidate is the one that fits
        the audio best, not the one Whisper's LM head liked most.
        """
        from .alignment import (
            alignment_score, forced_align, text_to_word_phonemes,
        )

        if audio.size == 0:
            return GatedTranscription(
                text="", align_score=0.0, phoneme_density=0.0,
                flagged=False, reason=None,
            )

        # Per-patient threshold override. When the active adapter's
        # directory carries a ``calibration.json`` (produced by
        # ``scripts/compute_calibration.py`` after training), use the
        # patient-specific thresholds for *this* call instead of the
        # global defaults. Without this the gate treats M01's
        # naturally-low alignment scores the same as FC02's — over-
        # flagging the dysarthric patient and under-flagging the
        # control. Falls back to instance defaults when the file is
        # missing or unreadable.
        min_align = self._min_alignment
        min_density = self._min_density
        # Resolve calibration source: explicit ``adapter_dir`` wins; else
        # fall back to the wrapped AdaptedTranscriber's default adapter
        # (the one loaded at init from ``settings.patient_adapter_dir``).
        # That way mic streams without an explicit adapter swap still see
        # patient-specific thresholds.
        cal_source = adapter_dir
        if not cal_source and isinstance(self._transcriber, AdaptedTranscriber):
            cal_source = getattr(self._transcriber, "_adapter_dir", None)
        if cal_source:
            cal = self._load_calibration(cal_source)
            if cal:
                min_align = float(cal.get("min_alignment_score", min_align))
                min_density = float(cal.get("min_phoneme_density", min_density))

        # Compute the phoneme posterior once — both the rescoring loop
        # and the final gate decision align against the same probs.
        if probs is None:
            probs = self._extractor.extract(audio, self._sample_rate)
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))

        # Gather candidate transcripts. When the wrapped transcriber is
        # the LoRA-adapted HF model and n_best > 1, we get a beam list;
        # otherwise we fall back to a single greedy decode.
        candidates: list[str] = []
        if self._n_best > 1 and isinstance(
            self._transcriber, AdaptedTranscriber
        ):
            try:
                candidates = self._transcriber.transcribe_n_best(
                    audio,
                    initial_prompt=initial_prompt,
                    adapter_dir=adapter_dir,
                    n_best=self._n_best,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "n-best generation failed (%s); falling back to greedy", e
                )
                candidates = []

        if not candidates:
            if adapter_dir is not None and isinstance(
                self._transcriber, AdaptedTranscriber
            ):
                candidates = [self._transcriber.transcribe(
                    audio, initial_prompt=initial_prompt, adapter_dir=adapter_dir,
                ).strip()]
            else:
                candidates = [self._transcriber.transcribe(
                    audio, initial_prompt=initial_prompt,
                ).strip()]

        # Rescore: phonemize each candidate per-word, force-align against
        # the wav2vec2 posterior, pick the highest mean log-prob. We
        # carry the per-word phoneme grouping through so the chosen
        # candidate can be sliced into per-word scores at the end.
        best_text = ""
        best_score: float = -float("inf")
        best_phonemes: list[str] = []
        best_word_pairs: list[tuple[str, list[str]]] = []
        best_kept_word_idx: list[int] = []
        best_aligned_tokens: list = []  # list[AlignedToken]

        for cand in candidates:
            cand = (cand or "").strip()
            if not cand:
                continue
            word_pairs = text_to_word_phonemes(cand)
            if not word_pairs:
                continue
            # Flatten phonemes + remember which word each came from, so
            # the alignment can be sliced back to per-word scores.
            flat: list[str] = []
            word_indices: list[int] = []
            for wi, (_, ps) in enumerate(word_pairs):
                flat.extend(ps)
                word_indices.extend([wi] * len(ps))
            if not flat:
                continue
            keep_mask = [p in self._token_to_id for p in flat]
            kept_word_indices = [
                w for w, k in zip(word_indices, keep_mask) if k
            ]
            aligned = forced_align(
                log_probs, flat, self._token_to_id, self._blank_id
            )
            score = alignment_score(aligned)
            if score > best_score:
                best_score = score
                best_text = cand
                best_phonemes = flat
                best_word_pairs = word_pairs
                best_kept_word_idx = kept_word_indices
                best_aligned_tokens = aligned

        if not best_text:
            # Every candidate was empty or non-phonemizable — return
            # the first non-empty raw text (or "") with no gate signal.
            fallback = next((c.strip() for c in candidates if c and c.strip()), "")
            return GatedTranscription(
                text=fallback, align_score=0.0, phoneme_density=0.0,
                flagged=False, reason=None,
            )

        # Slice the alignment back to per-word scores. forced_align may
        # have dropped phonemes that fall outside the wav2vec2 vocab,
        # which is why we tracked kept_word_indices alongside the kept
        # phoneme stream.
        words = [w for w, _ in best_word_pairs]
        per_word_buckets: list[list[float]] = [[] for _ in words]
        if len(best_aligned_tokens) == len(best_kept_word_idx):
            for tok, wi in zip(best_aligned_tokens, best_kept_word_idx):
                per_word_buckets[wi].append(float(tok.score))
        per_word_scores = [
            float(np.mean(bucket)) if bucket else float("-inf")
            for bucket in per_word_buckets
        ]

        duration_s = max(audio.size / self._sample_rate, 1e-3)

        # Voiced duration via energy-based pause detection. Earlier
        # attempt used the wav2vec2 blank-frame ratio, but that failed:
        # CTC blanks fire *between* phonemes by design (~50% of frames
        # during continuous speech are blank), so the ratio drastically
        # over-counts "silence" and flipped FC02 "but he always
        # answers banana oil" from a 5.81-ph/s false-low to a 42-ph/s
        # false-high. Real pause detection lives in
        # ``features._pause_stats`` — runs of low-RMS frames longer
        # than 80 ms. Reused here so the gate sees the same pause
        # picture as the prosody diagnostic.
        from .features import _pause_stats
        _, pause_total_ms = _pause_stats(audio, self._sample_rate)
        voiced_duration_s = max(duration_s - pause_total_ms / 1000.0, 0.05)
        voiced_fraction = voiced_duration_s / duration_s
        density = len(best_phonemes) / voiced_duration_s

        reason: str | None = None
        if density > self._max_density:
            reason = "phoneme_density"
        elif (
            duration_s >= self._min_density_duration
            and density < min_density
        ):
            # Under-production: long-form audio whose proposed
            # transcription has too few phonemes to plausibly account
            # for the duration. Only checked above min_density_duration
            # so single-word stimuli aren't false-flagged. Density is
            # measured over voiced (non-blank) duration via wav2vec2's
            # blank-frame ratio so heavily-paused fluent speech doesn't
            # false-flag. Threshold (``min_density``) is per-patient
            # when ``calibration.json`` is present.
            reason = "underproduction"
        elif best_score < min_align:
            reason = "low_alignment"
        return GatedTranscription(
            text=best_text,
            align_score=best_score,
            phoneme_density=density,
            flagged=reason is not None,
            reason=reason,
            words=words,
            per_word_scores=per_word_scores,
            voiced_fraction=voiced_fraction,
        )
