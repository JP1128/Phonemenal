"""Offline evaluation harness for the Phonemer pipeline against TORGO.

Computes the metrics that answer "is the model listening, and how much
does our calibrated pipeline beat naive ASR?":

  * PER (phoneme error rate): greedy CTC decode of wav2vec2 logits vs.
    espeak phonemization of the intended text. Controls should land
    around 0.05-0.20; dysarthria patients vary widely by intelligibility.
  * Alignment score: mean per-token log-prob from forced alignment.
    Correlates with "did the model hear what was said?" — clips below
    -3.5 are usually silence / throat-clear / mismatched transcript.
  * Whisper WER: faster-whisper transcription vs. intended text. The
    "naive ASR baseline" Phonemer needs to beat. Skipped by default
    (slow); pass --with-whisper to include.

Outputs a per-speaker × per-condition summary, plus an optional verbose
per-clip line. The non-word DDK clips (prompt_set='non-word') are
excluded by default since they aren't lexical content.

Usage:
    cd server
    python -m scripts.torgo_eval --speakers FC01,F01 --limit 50
    python -m scripts.torgo_eval --condition control --with-whisper --limit 100
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _ensure_espeak_library  # noqa: F401
from app.alignment import alignment_score, forced_align, text_to_phonemes  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import connect, list_torgo_clips  # noqa: E402
from app.distortion import build_profile  # noqa: E402
from app.phonemes import FRAME_STRIDE_SAMPLES, PhonemeExtractor  # noqa: E402
from app.vocab import normalize_phonemes  # noqa: E402
from scripts.splits import filter_by_split  # noqa: E402


# CTC special tokens we never want to decode as actual phonemes.
_CTC_SPECIALS = {"<pad>", "<s>", "</s>", "<unk>", "|", " "}


@dataclass
class ClipResult:
    speaker_id: str
    condition: str
    prompt_set: str
    intended_text: str
    duration_ms: int
    per: float
    n_phonemes: int
    align_score: float
    whisper_text: str | None = None
    whisper_wer: float | None = None
    llm_text: str | None = None
    llm_wer: float | None = None
    llm_confidence: float | None = None


@dataclass
class SpeakerStats:
    n: int = 0
    per_values: list[float] = field(default_factory=list)
    align_values: list[float] = field(default_factory=list)
    whisper_wers: list[float] = field(default_factory=list)
    llm_wers: list[float] = field(default_factory=list)


# ============================================================ metrics


def levenshtein(a: list[str], b: list[str]) -> int:
    """Standard token-level edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def greedy_ctc_decode(probs: np.ndarray, id_to_token: dict[int, str]) -> list[str]:
    """Argmax per frame, collapse runs, drop blanks/specials. Returns phonemes.

    The output is run through the same ``normalize_phonemes`` step the
    intended-text side uses, so length marks (``iː`` vs ``i``) and other
    surface differences don't artificially inflate PER.
    """
    ids = probs.argmax(axis=-1).tolist()
    raw: list[str] = []
    last = -1
    for i in ids:
        if i == last:
            continue
        last = i
        sym = id_to_token.get(int(i), "")
        if not sym or sym in _CTC_SPECIALS:
            continue
        raw.append(sym)
    return normalize_phonemes(raw)


def normalize_words(text: str) -> list[str]:
    """Lowercase + strip non-alphanumerics for WER comparison."""
    keep = []
    for w in text.lower().split():
        cleaned = "".join(c for c in w if c.isalnum() or c == "'")
        if cleaned:
            keep.append(cleaned)
    return keep


# ============================================================ pipeline


def _load_audio(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != settings.sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)
    return audio


def build_profile_summary(
    extractor: PhonemeExtractor,
    cal_clips: list[dict],
    token_to_id: dict[str, int],
    blank_id: int,
    vocab_symbols: list[str],
) -> str:
    """Build an in-memory distortion profile summary from the cal split.

    Mirrors what main.py's calibration_build does, but without persisting
    anything to the patient_profiles table — TORGO speakers shouldn't pollute
    the user's calibration history. Returns the prompt-ready summary string
    that goes into the LLM's cached system block.
    """
    import json as _json
    from app.llm import summarize_profile_for_prompt

    aligned: list[dict] = []
    for clip in cal_clips:
        text = (clip.get("intended_text") or "").strip()
        if not text or clip.get("prompt_set") == "non-word":
            continue
        audio = _load_audio(clip["wav_path"])
        if audio is None:
            continue
        probs = extractor.extract(audio, settings.sample_rate)
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        intended = text_to_phonemes(text)
        words_ph = [text_to_phonemes(w) for w in text.split()]
        a = forced_align(log_probs, intended, token_to_id, blank_id)
        if alignment_score(a) < settings.min_alignment_score:
            continue
        produced = extractor.top_k_per_frame(probs, settings.top_k_phonemes)
        aligned.append(
            {
                "intended_tokens": intended,
                "intended_words_phonemes": words_ph,
                "aligned": a,
                "produced": produced,
            }
        )

    if not aligned:
        return ""
    profile = build_profile(aligned, vocab_symbols=vocab_symbols)
    return summarize_profile_for_prompt(_json.loads(profile.to_json()))


def llm_translate_clip(
    top_k: list,
    profile_summary: str,
    frame_stride_ms: int,
) -> tuple[str, float]:
    """One-shot LLM translation for an isolated TORGO clip (no context).

    Returns (primary_text, overall_confidence). Caches profile_summary in
    the system block so repeated calls within ~5 minutes hit the prompt
    cache at ~0.1× cost.
    """
    from app.llm import TranslationContext, translate

    frames = [
        {
            "timestamp_ms": i * frame_stride_ms,
            "phonemes": [{"symbol": s, "prob": p} for s, p in frame],
        }
        for i, frame in enumerate(top_k)
    ]
    ctx = TranslationContext(phoneme_frames=frames)
    result = translate(
        model=settings.translation_model,
        profile_summary=profile_summary,
        context=ctx,
        max_tokens=settings.translation_max_tokens,
    )
    return result.primary, result.overall_confidence


def evaluate_clip(
    extractor: PhonemeExtractor,
    transcriber,
    clip: dict,
    id_to_token: dict[int, str],
    token_to_id: dict[str, int],
    blank_id: int,
    profile_summary: str | None = None,
    frame_stride_ms: int = 20,
) -> ClipResult | None:
    text = (clip.get("intended_text") or "").strip()
    if not text:
        return None
    audio = _load_audio(clip["wav_path"])
    if audio is None or audio.size == 0:
        return None

    probs = extractor.extract(audio, settings.sample_rate)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))

    intended_phonemes = text_to_phonemes(text)
    decoded = greedy_ctc_decode(probs, id_to_token)
    per_dist = levenshtein(intended_phonemes, decoded)
    per = per_dist / max(len(intended_phonemes), 1)

    aligned = forced_align(log_probs, intended_phonemes, token_to_id, blank_id)
    score = alignment_score(aligned)

    whisper_text = None
    whisper_wer = None
    if transcriber is not None:
        whisper_text = transcriber.transcribe(audio).strip()
        ref_words = normalize_words(text)
        hyp_words = normalize_words(whisper_text)
        if ref_words:
            whisper_wer = levenshtein(ref_words, hyp_words) / len(ref_words)

    llm_text = None
    llm_wer = None
    llm_conf = None
    if profile_summary is not None:
        top_k = extractor.top_k_per_frame(probs, settings.top_k_phonemes)
        try:
            llm_text, llm_conf = llm_translate_clip(
                top_k, profile_summary, frame_stride_ms
            )
        except Exception as e:  # noqa: BLE001
            llm_text = f"<error: {type(e).__name__}>"
        ref_words = normalize_words(text)
        hyp_words = normalize_words(llm_text)
        if ref_words:
            llm_wer = levenshtein(ref_words, hyp_words) / len(ref_words)

    return ClipResult(
        speaker_id=clip["speaker_id"],
        condition=clip.get("condition") or "?",
        prompt_set=clip.get("prompt_set") or "?",
        intended_text=text,
        duration_ms=int(clip.get("duration_ms") or 0),
        per=per,
        n_phonemes=len(intended_phonemes),
        align_score=score,
        whisper_text=whisper_text,
        whisper_wer=whisper_wer,
        llm_text=llm_text,
        llm_wer=llm_wer,
        llm_confidence=llm_conf,
    )


# ============================================================ reporting


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, pct))


def _print_per_clip(
    rows: Iterable[ClipResult], with_whisper: bool, with_llm: bool
) -> None:
    base = f"{'speaker':<7} {'cond':<10} {'dur_s':>5} {'PER':>5} {'align':>6}"
    if with_whisper:
        base += f" {'WERw':>5}"
    if with_llm:
        base += f" {'WERl':>5}"
    header = base + "  intended | hypotheses"
    print(header)
    print("-" * len(header))
    for r in rows:
        score_str = (
            "  -inf" if r.align_score == float("-inf") else f"{r.align_score:>6.2f}"
        )
        line = (
            f"{r.speaker_id:<7} {r.condition:<10} {r.duration_ms/1000:>5.2f} "
            f"{r.per:>5.2f} {score_str}"
        )
        hyps: list[str] = []
        if with_whisper:
            wer_str = "    -" if r.whisper_wer is None else f"{r.whisper_wer:>5.2f}"
            line += f" {wer_str}"
            hyps.append(f"whisper={(r.whisper_text or '')[:40]!r}")
        if with_llm:
            wer_str = "    -" if r.llm_wer is None else f"{r.llm_wer:>5.2f}"
            line += f" {wer_str}"
            hyps.append(f"llm={(r.llm_text or '')[:40]!r}")
        tail = f"  {r.intended_text[:40]!r}"
        if hyps:
            tail += " | " + " ".join(hyps)
        print(line + tail)


def _print_summary(rows: list[ClipResult], with_whisper: bool, with_llm: bool) -> None:
    """Two summaries: per (speaker, condition) and per (speaker, prompt_set).

    The prompt-set breakout matters because isolated single-word stimuli have
    very different recognition profiles than TIMIT sentences — averaging
    across both hides which conditions the model actually struggles with.
    """
    print()
    _print_grouped(
        rows,
        group_key=lambda r: (r.speaker_id, r.condition),
        header_cols=("speaker", "cond"),
        with_whisper=with_whisper,
        with_llm=with_llm,
        title="summary by speaker × condition",
    )
    print()
    _print_grouped(
        rows,
        group_key=lambda r: (r.speaker_id, r.prompt_set),
        header_cols=("speaker", "set"),
        with_whisper=with_whisper,
        with_llm=with_llm,
        title="summary by speaker × prompt_set",
    )


def _print_grouped(
    rows: list[ClipResult],
    *,
    group_key,
    header_cols: tuple[str, str],
    with_whisper: bool,
    with_llm: bool,
    title: str,
) -> None:
    buckets: dict[tuple[str, str], SpeakerStats] = {}
    for r in rows:
        key = group_key(r)
        b = buckets.setdefault(key, SpeakerStats())
        b.n += 1
        b.per_values.append(r.per)
        if r.align_score != float("-inf"):
            b.align_values.append(r.align_score)
        if r.whisper_wer is not None:
            b.whisper_wers.append(r.whisper_wer)
        if r.llm_wer is not None:
            b.llm_wers.append(r.llm_wer)

    print(f"== {title} ==")
    c1, c2 = header_cols
    base = f"{c1:<7} {c2:<15} {'n':>4} {'PER_mean':>9} {'PER_p50':>8} {'align':>7}"
    extra = ""
    if with_whisper:
        extra += f"  {'WERw_mean':>9}"
    if with_llm:
        extra += f"  {'WERl_mean':>9} {'WERl_p50':>9}"
    header = base + extra
    print(header)
    print("-" * len(header))
    for key, s in sorted(buckets.items()):
        a, b = key
        per_mean = float(np.mean(s.per_values)) if s.per_values else float("nan")
        per_med = _percentile(s.per_values, 50)
        align = float(np.mean(s.align_values)) if s.align_values else float("nan")
        line = (
            f"{a:<7} {b:<15} {s.n:>4} "
            f"{per_mean:>9.3f} {per_med:>8.3f} {align:>7.2f}"
        )
        if with_whisper:
            wer_mean = float(np.mean(s.whisper_wers)) if s.whisper_wers else float("nan")
            line += f"  {wer_mean:>9.3f}"
        if with_llm:
            llm_mean = float(np.mean(s.llm_wers)) if s.llm_wers else float("nan")
            llm_med = _percentile(s.llm_wers, 50) if s.llm_wers else float("nan")
            line += f"  {llm_mean:>9.3f} {llm_med:>9.3f}"
        print(line)


# ============================================================ entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phonemer against TORGO.")
    parser.add_argument(
        "--speakers",
        default=None,
        help="Comma-separated speaker ids (e.g. FC01,F01). Default: all.",
    )
    parser.add_argument(
        "--condition",
        choices=("control", "dysarthria"),
        default=None,
        help="Filter to one condition.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max clips per speaker (deterministic by id order). 0 = all.",
    )
    parser.add_argument(
        "--split",
        choices=("cal", "eval", "all"),
        default="eval",
        help="Held-out partition to evaluate on. Default: eval (the 20%% the "
        "calibration step never sees).",
    )
    parser.add_argument(
        "--with-whisper",
        action="store_true",
        help="Also run faster-whisper as a baseline. Slower (~0.5s per clip on CPU).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run the calibrated Phonemer LLM (Sonnet) per clip and "
        "compute its WER. Builds a profile from the cal split for each "
        "speaker first. Costs roughly $0.003 per eval clip; requires "
        "ANTHROPIC_API_KEY.",
    )
    parser.add_argument(
        "--max-cal-clips",
        type=int,
        default=50,
        help="Cap calibration clips per speaker when building --with-llm "
        "profiles. Each clip is ~1s of CPU work; 50 keeps profile build "
        "under a minute per speaker.",
    )
    parser.add_argument(
        "--include-non-word",
        action="store_true",
        help="Don't skip prompt_set='non-word' clips (DDK stimuli).",
    )
    parser.add_argument(
        "--per-clip",
        action="store_true",
        help="Print per-clip metrics in addition to the summary.",
    )
    parser.add_argument("--db", default=settings.db_path)
    args = parser.parse_args()

    extractor = PhonemeExtractor(settings.phoneme_model_name, device=settings.device)
    token_to_id = extractor.processor.tokenizer.get_vocab()
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    blank_id = extractor.processor.tokenizer.pad_token_id or 0
    vocab_symbols = list(token_to_id.keys())
    frame_stride_ms = int(1000 * FRAME_STRIDE_SAMPLES / settings.sample_rate)

    transcriber = None
    if args.with_whisper:
        from app.stt import Transcriber

        transcriber = Transcriber(
            model_size=settings.stt_model_size,
            compute_type=settings.stt_compute_type,
            device=settings.device,
        )

    if args.with_llm:
        from app.llm import is_configured

        if not is_configured():
            raise SystemExit(
                "--with-llm requires anthropic + ANTHROPIC_API_KEY. "
                "Set up server/.env."
            )
        if args.split == "all":
            raise SystemExit(
                "--with-llm requires --split=eval (or cal); 'all' would "
                "calibrate and evaluate on the same clips."
            )

    conn = connect(args.db)
    speakers: list[str | None]
    if args.speakers:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:
        speakers = [None]

    all_rows: list[ClipResult] = []
    started = time.monotonic()
    for spk in speakers:
        all_clips = list_torgo_clips(
            conn, speaker_id=spk, condition=args.condition, limit=1_000_000
        )
        if not args.include_non_word:
            all_clips = [c for c in all_clips if c.get("prompt_set") != "non-word"]

        # Build a per-speaker profile from the cal split (in memory; never
        # written to patient_profiles). Required for --with-llm so each
        # speaker's translation is fair-tested against their own profile.
        profile_summary: str | None = None
        if args.with_llm:
            cal_clips = filter_by_split(all_clips, "cal")
            if args.max_cal_clips:
                cal_clips = cal_clips[: args.max_cal_clips]
            if not cal_clips:
                print(f"[skip] {spk}: no cal-split clips for profile")
                continue
            print(f"[profile] {spk}: building from {len(cal_clips)} cal clips…")
            profile_summary = build_profile_summary(
                extractor, cal_clips, token_to_id, blank_id, vocab_symbols
            )
            if not profile_summary:
                print(f"[warn] {spk}: empty profile; will translate without one")

        # Eval clips: the held-out split (or whatever the user picked).
        clips = filter_by_split(all_clips, args.split)
        if args.limit:
            clips = clips[: args.limit]
        if not clips:
            print(f"[skip] {spk or 'all'}: no clips after filters")
            continue

        per_clip_rows: list[ClipResult] = []
        for clip in clips:
            r = evaluate_clip(
                extractor,
                transcriber,
                clip,
                id_to_token,
                token_to_id,
                blank_id,
                profile_summary=profile_summary,
                frame_stride_ms=frame_stride_ms,
            )
            if r is not None:
                per_clip_rows.append(r)

        if args.per_clip:
            _print_per_clip(per_clip_rows, args.with_whisper, args.with_llm)
            print()
        all_rows.extend(per_clip_rows)

    if not all_rows:
        print("no clips evaluated")
        return 1

    _print_summary(all_rows, args.with_whisper, args.with_llm)
    elapsed = time.monotonic() - started
    print(f"\n{len(all_rows)} clips in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
