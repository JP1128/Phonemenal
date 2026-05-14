"""Compare per-patient LoRA adapter against vanilla Whisper on the eval split.

Both branches use the same HuggingFace pipeline (NOT faster-whisper) so the
comparison isolates the adapter effect — different decoders would confound
that.

Reports WER and a per-prompt-set breakout, mirroring torgo_eval.py's columns
so results are directly comparable to the journal's baseline numbers.

Usage:
    cd server
    .venv/bin/python -m scripts.evaluate_adapter --speaker-id F03
    .venv/bin/python -m scripts.evaluate_adapter --speaker-id M01 --limit 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from scripts.splits import filter_by_split  # noqa: E402
from scripts.torgo_eval import levenshtein  # noqa: E402


def _build_normalizer():
    """Return Whisper's official EnglishTextNormalizer.

    Expands contractions (``"don't" → "do not"``), normalizes numbers /
    possessives / casing, and strips punctuation. Both vanilla and adapted
    hypotheses go through the same normalizer (and so does the reference)
    so neither side is favored — WER measures recognition error, not
    formatting drift.
    """
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    # The normalizer reads contraction/spelling tables; pass an empty mapping
    # for English-specific extras (we don't need domain replacements here).
    return EnglishTextNormalizer({})


def _normalize_words(text: str, normalizer) -> list[str]:
    """Whisper-normalize a string and split into a token list for WER."""
    text = normalizer(text or "").strip()
    return [w for w in text.split() if w]

logger = logging.getLogger("evaluate_adapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


TARGET_SR = 16000


def _load_audio(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception as exc:
        logger.warning("audio load failed for %s: %s", path, exc)
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32, copy=False)


def _estimate_phoneme_rate(audio: np.ndarray, extractor, pad_id: int) -> float:
    """Estimate phonemes per second over the *full* clip (including silence).

    Uses CTC argmax + run-collapse on wav2vec2 output; non-blank runs are
    counted as distinct phonemes. Includes silence in the denominator so
    short isolated-word clips (mostly silence padding) get low rates and
    are protected from over-aggressive stretching by the duration gate.
    """
    if audio.size < TARGET_SR // 4:
        return float("nan")
    probs = extractor.extract(audio, TARGET_SR)
    arg = probs.argmax(axis=-1)
    n_phon = 0
    last = -1
    for x in arg:
        if x != last:
            if x != pad_id:
                n_phon += 1
            last = int(x)
    duration_s = max(audio.size / TARGET_SR, 0.1)
    return n_phon / duration_s


def _normalize_rate(
    audio: np.ndarray,
    extractor,
    pad_id: int,
    target_rate: float = 5.0,
    min_duration_s: float = 2.5,
    max_speedup: float = 2.5,
    min_speedup: float = 1.1,
) -> tuple[np.ndarray, float]:
    """Time-stretch slow speech toward ``target_rate`` phonemes/second.

    Returns ``(audio, applied_speedup)``. Speed factor of 1.0 means unchanged.
    Pitch is preserved (librosa phase vocoder). Heuristics:

    * Only stretch clips at least ``min_duration_s`` seconds — short clips are
      mostly silence padding around isolated-word stimuli, where time-stretch
      hurts more than helps.
    * Cap the speedup at ``max_speedup`` so an unusual-but-correct slow clip
      isn't crushed.
    * Ignore tiny adjustments (< ``min_speedup``).
    """
    duration_s = audio.size / TARGET_SR
    if duration_s < min_duration_s:
        return audio, 1.0
    rate = _estimate_phoneme_rate(audio, extractor, pad_id)
    if not np.isfinite(rate) or rate <= 0:
        return audio, 1.0
    speedup = target_rate / rate
    speedup = max(1.0, min(speedup, max_speedup))
    if speedup < min_speedup:
        return audio, 1.0
    # Prefer pyrubberband (rubberband C++ library) — much higher audio quality
    # than librosa's phase vocoder at 1.5x+ speedups. librosa fallback is
    # acceptable for milder stretches but introduces artifacts that hurt
    # Whisper for the kinds of speedups severe dysarthria needs (~2x).
    try:
        import pyrubberband as pyrb

        stretched = pyrb.time_stretch(audio, TARGET_SR, speedup)
    except ImportError:
        import librosa

        stretched = librosa.effects.time_stretch(audio, rate=speedup)
    return stretched.astype(np.float32, copy=False), float(speedup)


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_bias_token_ids(
    processor, speaker_id: str, conn, top_n: int
) -> set[int]:
    """Collect Whisper-tokenizer ids for the patient's top-N cal-split words.

    Each cal-split word is BPE-tokenized (with leading-space variants, since
    Whisper's BPE is whitespace-aware: ``"rock"`` and ``" rock"`` are
    different ids). The returned set is consumed by ``_VocabBiasLogitsProcessor``
    to nudge — not constrain — Whisper's distribution toward patient vocab
    at every decoding step. Unlike ``<|startofprev|>`` prompting, this does
    NOT change Whisper's expectation of utterance length, so it doesn't
    blow up isolated-word recognition.
    """
    import re
    from collections import Counter

    from app.db import list_torgo_clips
    from scripts.splits import filter_by_split

    clips = list_torgo_clips(conn, speaker_id=speaker_id, limit=1_000_000)
    cal = [
        c for c in filter_by_split(clips, "cal")
        if (c.get("intended_text") or "").strip()
        and c.get("prompt_set") != "non-word"
    ]
    counter: Counter[str] = Counter()
    for c in cal:
        for w in re.findall(r"[a-zA-Z']+", c["intended_text"].lower()):
            if len(w) >= 2:
                counter[w] += 1
    if not counter:
        return set()

    top_words = [w for w, _ in counter.most_common(top_n)]
    tok = processor.tokenizer
    ids: set[int] = set()
    for w in top_words:
        # Both forms — Whisper BPE uses leading-space tokens for word boundaries.
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            sub_ids = tok(variant, add_special_tokens=False).input_ids
            ids.update(sub_ids)
    return ids


class _VocabBiasLogitsProcessor:
    """Add a constant boost to a fixed set of token ids at every decoding step.

    Conforms to the transformers ``LogitsProcessor`` ABI without inheriting
    from it (avoids tight coupling to internal class shape). Boost magnitude
    of +1.5 to +3.0 nudges decoding toward target tokens without overpowering
    acoustic evidence — gentle prior, not a constraint.
    """

    def __init__(self, vocab_ids: set[int], boost: float):
        self.vocab_ids = torch.tensor(sorted(vocab_ids), dtype=torch.long)
        self.boost = float(boost)

    def __call__(self, input_ids, scores):
        ids = self.vocab_ids.to(scores.device)
        scores[:, ids] = scores[:, ids] + self.boost
        return scores


@torch.inference_mode()
def _transcribe(
    model,
    processor,
    audio: np.ndarray,
    device: str,
    beam_size: int = 5,
    logits_processor=None,
    n_return: int = 1,
    no_repeat_ngram_size: int = 0,
) -> tuple[list[str], list[float]]:
    """Decode and return ``n_return`` candidate strings with Whisper log-probs.

    Always returns a list (length 1 when ``n_return`` == 1) so callers can
    treat single-best and N-best uniformly. ``no_repeat_ngram_size`` is the
    standard Whisper safeguard against the "side → sighsigh..." repetition
    collapse on short clips when beam>1.
    """
    inputs = processor.feature_extractor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt"
    )
    feats = inputs.input_features.to(device)
    gen_kwargs: dict = dict(
        max_new_tokens=128,
        num_return_sequences=n_return,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if n_return > 1:
        # Diverse beam search: standard beam + no_repeat_ngram collapses to a
        # single solution returned N times, so ranking has nothing to choose
        # from. Group beams force distinct candidates with a small diversity
        # penalty. transformers 5 routed group beam search through a community
        # remote-code repo (transformers-community/group-beam-search), which
        # requires trust_remote_code=True.
        gen_kwargs["num_beams"] = max(beam_size, n_return * 2)
        gen_kwargs["num_beam_groups"] = n_return
        gen_kwargs["diversity_penalty"] = 0.5
        gen_kwargs["trust_remote_code"] = True
    else:
        gen_kwargs["num_beams"] = max(beam_size, n_return)
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor
    out = model.generate(feats, **gen_kwargs)
    seqs = out.sequences  # [n_return, T]
    # ``sequences_scores`` is length-normalized log-prob from beam search,
    # already comparable across hypotheses of different length.
    scores_t = getattr(out, "sequences_scores", None)
    if scores_t is None:
        scores = [0.0] * seqs.shape[0]
    else:
        scores = scores_t.detach().cpu().tolist()
    texts = [
        processor.tokenizer.decode(s, skip_special_tokens=True).strip()
        for s in seqs
    ]
    return texts, scores


def _rerank(
    audio: np.ndarray,
    candidates: list[str],
    whisper_scores: list[float],
    extractor,
    token_to_id: dict[str, int],
    blank_id: int,
    alpha: float,
) -> str:
    """Re-rank Whisper N-best by combined Whisper + phoneme-alignment score.

    The phoneme alignment is the same machinery already used in stt.py's
    GatedTranscriber: phonemize each candidate, force-align against the audio's
    wav2vec2 distribution, mean log-prob is the alignment score. Combined
    score is min-max normalized within this clip's candidate set so the
    Whisper / alignment scores are on a comparable scale before weighting.
    """
    from app.alignment import alignment_score, forced_align, text_to_phonemes

    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    probs = extractor.extract(audio, 16000)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))

    align_scores: list[float] = []
    for text in candidates:
        if not text:
            align_scores.append(float("-inf"))
            continue
        phonemes = text_to_phonemes(text)
        if not phonemes:
            align_scores.append(float("-inf"))
            continue
        aligned = forced_align(log_probs, phonemes, token_to_id, blank_id)
        align_scores.append(alignment_score(aligned))

    def _norm(xs: list[float]) -> list[float]:
        finite = [x for x in xs if x != float("-inf") and x != float("inf")]
        if not finite:
            return [0.0] * len(xs)
        lo, hi = min(finite), max(finite)
        if hi - lo < 1e-9:
            return [0.5 if x not in (float("-inf"), float("inf")) else 0.0 for x in xs]
        return [
            (x - lo) / (hi - lo) if x not in (float("-inf"), float("inf")) else 0.0
            for x in xs
        ]

    w_norm = _norm(whisper_scores)
    a_norm = _norm(align_scores)
    combined = [alpha * w + (1 - alpha) * a for w, a in zip(w_norm, a_norm)]
    best = int(np.argmax(combined))
    return candidates[best]


def _wer(reference: str, hypothesis: str, normalizer) -> float | None:
    ref = _normalize_words(reference, normalizer)
    hyp = _normalize_words(hypothesis, normalizer)
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def _summarize(label: str, rows: list[dict], by: str) -> dict:
    """Group rows by ``rows[i][by]`` and report mean WER per bucket."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r[label] is not None:
            buckets[r[by]].append(r[label])
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WER comparison: vanilla Whisper vs per-speaker LoRA adapter."
    )
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument(
        "--eval-speaker-id",
        default=None,
        help="Eval-data speaker. Defaults to --speaker-id. Set to a different "
        "speaker for cross-speaker sanity checks (e.g. run F03's adapter "
        "on M01's audio to confirm the adapter learned patient-specific, "
        "not generic, patterns).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Whisper beam size during decoding. 1 (greedy) is the safe default "
        "on TORGO. Beam=5 hallucinates long completions on short isolated-word "
        "clips (every beam pursues a parallel hallucination), which "
        "catastrophically inflates WER on the intelligibility prompt set. "
        "Worth re-trying once we move to a stronger base model (small/medium) "
        "and longer-form audio.",
    )
    parser.add_argument(
        "--bias-vocab",
        type=int,
        default=0,
        help="When > 0, boost the logits of all tokens that appear in the "
        "patient's top-N cal-split words at every decoding step. Biases "
        "decoding toward the patient's vocabulary universe without hard-"
        "constraining it. Applied equally to vanilla and adapted. Vocab is "
        "deterministic from --speaker-id (NOT --eval-speaker-id).",
    )
    parser.add_argument(
        "--bias-strength",
        type=float,
        default=2.0,
        help="Logit boost added to vocab tokens at each step. ~+2 nudges; "
        "+5 forces.",
    )
    parser.add_argument(
        "--rerank-n",
        type=int,
        default=0,
        help="When > 0, generate this many candidates per clip and re-rank by "
        "phoneme-alignment score against the audio. Combined score = "
        "rerank_alpha * whisper_logprob + (1-rerank_alpha) * align_score. "
        "Implements the abandoned phoneme pipeline as a re-ranker on top of "
        "the adapter. Implies beam_size = max(beam_size, rerank_n).",
    )
    parser.add_argument(
        "--rerank-alpha",
        type=float,
        default=0.5,
        help="Weight on Whisper score in re-ranking (0 = phoneme alignment "
        "decides; 1 = Whisper decides). 0.5 splits influence equally.",
    )
    parser.add_argument(
        "--normalize-rate",
        action="store_true",
        help="Time-stretch slow speech toward normal pace (~5 phon/sec gross) "
        "before feeding to Whisper. Helps severe dysarthric audio that's "
        "outside Whisper's training distribution. Only applied to clips >= "
        "2.5s (avoids hurting short isolated-word stimuli).",
    )
    parser.add_argument(
        "--rate-target",
        type=float,
        default=5.0,
        help="Target gross phoneme rate in phon/sec (control speakers' median "
        "on TIMIT is ~5.0). Used only when --normalize-rate is set.",
    )
    parser.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=3,
        help="Block repeating n-grams during beam decoding to prevent the "
        "'side → sighsigh...' collapse. Set to 0 to disable.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Path to the saved adapter. Default: data/adapters/<speaker_id>/.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model. Default: read from <adapter-dir>/BASE_MODEL.txt, "
        "fall back to openai/whisper-small.en.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap eval clips. 0 = all eval-split clips.",
    )
    parser.add_argument("--db", default=settings.db_path)
    parser.add_argument(
        "--per-clip",
        action="store_true",
        help="Print per-clip vanilla/adapted hypotheses alongside intended text.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Write a JSON summary to this path (in addition to stdout).",
    )
    args = parser.parse_args()

    adapter_dir = Path(
        args.adapter_dir or f"data/adapters/{args.speaker_id}"
    ).resolve()
    if not adapter_dir.exists():
        raise SystemExit(f"adapter dir missing: {adapter_dir}")

    base_model = args.base_model
    if base_model is None:
        marker = adapter_dir / "BASE_MODEL.txt"
        base_model = marker.read_text().strip() if marker.exists() else "openai/whisper-small.en"
    device = _device()
    logger.info("device=%s base=%s adapter=%s", device, base_model, adapter_dir)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    # Vanilla side: identical base model, no adapter — isolates adapter delta.
    processor = WhisperProcessor.from_pretrained(base_model)
    vanilla = WhisperForConditionalGeneration.from_pretrained(base_model).to(device)
    vanilla.config.forced_decoder_ids = None
    vanilla.config.suppress_tokens = []
    vanilla.eval()

    # Adapted side: load base then attach the LoRA delta.
    adapted_base = WhisperForConditionalGeneration.from_pretrained(base_model)
    adapted = PeftModel.from_pretrained(adapted_base, str(adapter_dir)).to(device)
    adapted.config.forced_decoder_ids = None
    adapted.config.suppress_tokens = []
    adapted.eval()

    eval_speaker = args.eval_speaker_id or args.speaker_id
    if eval_speaker != args.speaker_id:
        logger.info(
            "cross-speaker eval: adapter=%s, audio=%s",
            args.speaker_id,
            eval_speaker,
        )

    conn = connect(args.db)
    all_clips = list_torgo_clips(
        conn, speaker_id=eval_speaker, limit=1_000_000
    )
    eval_clips = [
        c for c in filter_by_split(all_clips, "eval")
        if (c.get("intended_text") or "").strip()
        and c.get("prompt_set") != "non-word"
    ]
    if args.limit:
        eval_clips = eval_clips[: args.limit]
    if not eval_clips:
        raise SystemExit(
            f"no eval-split clips with intended_text for speaker {eval_speaker!r}"
        )
    extractor = None
    align_token_to_id = None
    align_blank_id = None
    if args.rerank_n > 0 or args.normalize_rate:
        from app.config import settings as _settings
        from app.phonemes import PhonemeExtractor

        logger.info("loading phoneme extractor (rerank=%s, normalize_rate=%s)…",
                    args.rerank_n > 0, args.normalize_rate)
        extractor = PhonemeExtractor(_settings.phoneme_model_name, device=device)
        align_token_to_id = extractor.processor.tokenizer.get_vocab()
        align_blank_id = extractor.processor.tokenizer.pad_token_id or 0

    logits_processors = None
    if args.bias_vocab > 0:
        from transformers.generation import LogitsProcessorList

        bias_ids = _build_bias_token_ids(
            processor, args.speaker_id, conn, args.bias_vocab
        )
        if bias_ids:
            logger.info(
                "vocab biasing: top-%d cal words → %d tokens (boost=%+.1f)",
                args.bias_vocab, len(bias_ids), args.bias_strength,
            )
            logits_processors = LogitsProcessorList(
                [_VocabBiasLogitsProcessor(bias_ids, args.bias_strength)]
            )

    logger.info("evaluating on %d eval clips (beam_size=%d)", len(eval_clips), args.beam_size)
    normalizer = _build_normalizer()

    rows: list[dict] = []
    started = time.monotonic()
    speedups: list[float] = []
    for i, clip in enumerate(eval_clips, 1):
        audio = _load_audio(clip["wav_path"])
        if audio is None or audio.size == 0:
            continue
        if args.normalize_rate and extractor is not None:
            # Only stretch on prompts likely to be connected speech. Isolated
            # word stimuli get hurt: a 3s clip of "rock" stretched to 1.2s at
            # 2.5x is incomprehensible. The duration gate alone wasn't enough
            # because severe dysarthria often makes single words exceed 2.5s.
            ps = (clip.get("prompt_set") or "").lower()
            multi_word = len((clip.get("intended_text") or "").split()) >= 3
            if ps != "intelligibility" and multi_word:
                audio, sp = _normalize_rate(
                    audio, extractor, align_blank_id,
                    target_rate=args.rate_target,
                )
                speedups.append(sp)
        intended = clip["intended_text"].strip()
        n_ret = max(1, args.rerank_n)
        v_texts, v_scores = _transcribe(
            vanilla, processor, audio, device, args.beam_size,
            logits_processors, n_return=n_ret,
            no_repeat_ngram_size=args.no_repeat_ngram,
        )
        a_texts, a_scores = _transcribe(
            adapted, processor, audio, device, args.beam_size,
            logits_processors, n_return=n_ret,
            no_repeat_ngram_size=args.no_repeat_ngram,
        )
        if args.rerank_n > 0 and extractor is not None:
            v_text = _rerank(
                audio, v_texts, v_scores, extractor,
                align_token_to_id, align_blank_id, args.rerank_alpha,
            )
            a_text = _rerank(
                audio, a_texts, a_scores, extractor,
                align_token_to_id, align_blank_id, args.rerank_alpha,
            )
        else:
            v_text = v_texts[0]
            a_text = a_texts[0]
        v_wer = _wer(intended, v_text, normalizer)
        a_wer = _wer(intended, a_text, normalizer)
        row = {
            "intended": intended,
            "vanilla_text": v_text,
            "adapted_text": a_text,
            "vanilla_wer": v_wer,
            "adapted_wer": a_wer,
            "prompt_set": clip.get("prompt_set") or "?",
            "duration_ms": int(clip.get("duration_ms") or 0),
        }
        rows.append(row)
        if args.per_clip:
            print(
                f"[{i:>3}/{len(eval_clips)}] "
                f"v={v_wer if v_wer is None else f'{v_wer:.2f}'}  "
                f"a={a_wer if a_wer is None else f'{a_wer:.2f}'}  "
                f"intended={intended[:40]!r}  "
                f"vanilla={v_text[:40]!r}  "
                f"adapted={a_text[:40]!r}"
            )
        if i % 25 == 0:
            elapsed = time.monotonic() - started
            logger.info("progress: %d/%d (%.1fs)", i, len(eval_clips), elapsed)

    # Aggregate
    v_all = [r["vanilla_wer"] for r in rows if r["vanilla_wer"] is not None]
    a_all = [r["adapted_wer"] for r in rows if r["adapted_wer"] is not None]
    v_mean = float(np.mean(v_all)) if v_all else float("nan")
    a_mean = float(np.mean(a_all)) if a_all else float("nan")
    delta_rel = (
        (a_mean - v_mean) / v_mean if v_mean and not np.isnan(v_mean) else float("nan")
    )

    print()
    print(f"== {args.speaker_id} eval (n={len(rows)}) ==")
    print(f"vanilla WER:  {v_mean:.3f}")
    print(f"adapted WER:  {a_mean:.3f}")
    print(f"Δ relative:   {delta_rel:+.1%}")

    print("\nby prompt_set:")
    v_by = _summarize("vanilla_wer", rows, "prompt_set")
    a_by = _summarize("adapted_wer", rows, "prompt_set")
    keys = sorted(set(v_by) | set(a_by))
    print(f"{'prompt_set':<15} {'vanilla':>9} {'adapted':>9} {'Δ rel':>8}")
    for k in keys:
        v = v_by.get(k, float("nan"))
        a = a_by.get(k, float("nan"))
        d = (a - v) / v if v else float("nan")
        print(f"{k:<15} {v:>9.3f} {a:>9.3f} {d:>+8.1%}")

    summary = {
        "speaker_id": args.speaker_id,
        "eval_speaker_id": eval_speaker,
        "beam_size": args.beam_size,
        "n_clips": len(rows),
        "vanilla_wer": v_mean,
        "adapted_wer": a_mean,
        "delta_relative": delta_rel,
        "by_prompt_set": {
            k: {"vanilla": v_by.get(k), "adapted": a_by.get(k)} for k in keys
        },
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        logger.info("wrote summary → %s", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
