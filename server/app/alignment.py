"""Text→phoneme conversion + CTC forced alignment.

Used by the calibration pipeline to map intended phonemes onto audio
frames so the distortion model can compare what the patient *meant* to
say with what the wav2vec2 logits *show* they said.

text_to_phonemes(...)   espeak-ng via phonemizer, normalized to model vocab.
forced_align(...)       torchaudio CTC alignment given audio + intended IPA.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .vocab import normalize_phonemes

# Suppress phonemizer's "words count mismatch on X% of the lines"
# warnings. Triggered every time we phonemize a word that espeak
# segments into >1 internal "word" (contractions, possessives, etc.).
# Noise — we don't care about espeak's internal word counting, only
# its phoneme output.
#
# Belt and suspenders: setLevel(CRITICAL) so all WARNING/ERROR
# records get filtered, propagate=False so the records don't bubble
# to the root logger (which is what uvicorn formats and prints), and
# a final no-op Filter that drops every record even if a downstream
# library re-enables the logger.
_phon_logger = logging.getLogger("phonemizer")
_phon_logger.setLevel(logging.CRITICAL)
_phon_logger.propagate = False
_phon_logger.addFilter(lambda _record: False)


# Lazy, shared EspeakBackend. The high-level ``phonemize()`` helper from
# the phonemizer package re-initializes a backend on every call —
# ~700 ms of espeak init overhead per invocation, which dominates the
# rescoring loop (3 candidates × 700 ms = 2.1 s of pure init per
# utterance). Holding a single persistent backend drops that to ~0 ms.
# Initialised on first use because espeak's library path is set up by
# ``app/__init__.py``, which may not yet be on sys.path at import time.
_BACKEND_LOCK = threading.Lock()
_BACKEND_CACHE: dict[str, Any] = {}


def _get_backend(language: str):
    with _BACKEND_LOCK:
        be = _BACKEND_CACHE.get(language)
        if be is None:
            from phonemizer.backend import EspeakBackend  # heavy import deferred
            be = EspeakBackend(
                language, preserve_punctuation=False, with_stress=False,
            )
            _BACKEND_CACHE[language] = be
        return be


@dataclass
class AlignedToken:
    symbol: str       # phoneme in model vocabulary
    start_frame: int  # inclusive
    end_frame: int    # exclusive
    score: float      # average log-prob across the segment


def alignment_score(aligned: list[AlignedToken]) -> float:
    """Mean per-token log-probability across the alignment, or ``-inf`` when empty.

    Used as a "did the model actually hear this?" gate before adding the clip
    to the confusion matrix. Fitted-but-implausible alignments (silence, wrong
    transcript, throat-clear) score around -8 or worse; clean speech reads
    closer to -1. The default cutoff in ``settings.min_alignment_score`` is
    -3.5 (≈ avg per-frame phoneme prob 0.03) — strict enough to drop obvious
    garbage but permissive enough to keep heavily distorted patient speech.
    """
    if not aligned:
        return float("-inf")
    return float(sum(t.score for t in aligned) / len(aligned))


def text_to_phonemes(text: str, language: str = "en-us") -> list[str]:
    """Phonemize a sentence and normalize into the wav2vec2 vocab.

    Without an explicit phone separator, espeak emits ``"ælfə"`` for ``alpha``
    as a single concatenated chunk — the model's tokenizer only knows
    individual phoneme symbols, so forced alignment then drops every token.
    Forcing a phone-level separator gives ``["æ", "l", "f", "ə"]`` that maps
    cleanly into the vocab.
    """
    from phonemizer.separator import Separator

    be = _get_backend(language)
    with _BACKEND_LOCK:  # phonemizer backends are not thread-safe
        raw = be.phonemize(
            [text],
            separator=Separator(phone=" ", syllable="", word=" | "),
            strip=True,
        )[0]
    raw = raw.replace("|", " ")
    return normalize_phonemes(raw)


def text_to_word_phonemes(
    text: str, language: str = "en-us"
) -> list[tuple[str, list[str]]]:
    """Per-word phonemization, returning ``(word, [phonemes])`` pairs.

    Phonemizes each whitespace-separated word *as its own input* rather
    than relying on espeak's ``|`` word separator inside one call.
    Reason: in connected speech, espeak fuses short function words into
    the preceding chunk ("for the" → ``"f ɚ ð ə"`` as a single chunk),
    so the chunk count no longer matches ``text.split()`` and per-word
    aggregation drifts. Phonemizing words individually preserves a
    strict 1-to-1 mapping from input words to phoneme lists.

    Words that phonemize to nothing after vocab normalization (rare;
    typically punctuation-only tokens) are omitted from the output.
    """
    from phonemizer.separator import Separator

    src_words = text.split()
    if not src_words:
        return []
    be = _get_backend(language)
    with _BACKEND_LOCK:
        raw_list = be.phonemize(
            src_words,
            separator=Separator(phone=" ", syllable="", word=""),
            strip=True,
        )
    pairs: list[tuple[str, list[str]]] = []
    for word, raw in zip(src_words, raw_list):
        normalized = normalize_phonemes(raw)
        if not normalized:
            continue
        pairs.append((word, normalized))
    return pairs


def forced_align(
    log_probs: np.ndarray,
    intended_tokens: list[str],
    token_to_id: dict[str, int],
    blank_id: int,
) -> list[AlignedToken]:
    """Greedy CTC forced alignment.

    Args:
      log_probs: ``[T, V]`` log-softmax probabilities from wav2vec2.
      intended_tokens: target phoneme sequence in the model's vocab.
      token_to_id: tokenizer's vocab dict.
      blank_id: id of the CTC blank token (typically `<pad>` for wav2vec2).

    Falls back to torchaudio's `forced_align` when available; otherwise
    runs a pure-Python Viterbi over the CTC trellis. Tokens missing from
    the vocab are dropped after a warning.

    Returns one AlignedToken per intended phoneme, in order.
    """
    target_ids = [token_to_id[t] for t in intended_tokens if t in token_to_id]
    if not target_ids:
        return []

    # Try torchaudio's optimized kernel.
    try:
        import torchaudio.functional as F  # type: ignore[attr-defined]

        emissions = torch.from_numpy(log_probs).unsqueeze(0)  # [1, T, V]
        targets = torch.tensor([target_ids], dtype=torch.int64)
        alignments, scores = F.forced_align(
            emissions, targets, blank=blank_id
        )
        return _segments_from_alignments(
            alignments[0].tolist(), scores[0].tolist(), intended_tokens, token_to_id
        )
    except (ImportError, AttributeError, RuntimeError):
        # Older torchaudio or unsupported op → fall back to Python Viterbi.
        return _viterbi_align(log_probs, target_ids, intended_tokens, blank_id)


# ============================================================ helpers


def _segments_from_alignments(
    frame_ids: list[int],
    frame_scores: list[float],
    intended_tokens: list[str],
    token_to_id: dict[str, int],
) -> list[AlignedToken]:
    """Collapse per-frame label assignments into one segment per target token."""
    keep = [t for t in intended_tokens if t in token_to_id]
    out: list[AlignedToken] = []
    cur_token_idx = -1
    seg_start = 0
    seg_scores: list[float] = []
    last_id: int | None = None

    for frame, (lbl, score) in enumerate(zip(frame_ids, frame_scores)):
        if lbl == 0:  # blank → just continue current segment
            seg_scores.append(score)
            continue
        if lbl != last_id:
            cur_token_idx += 1
            if out and seg_scores:
                out[-1].end_frame = frame
                out[-1].score = float(np.mean(seg_scores))
            seg_start = frame
            seg_scores = [score]
            if cur_token_idx < len(keep):
                out.append(
                    AlignedToken(
                        symbol=keep[cur_token_idx],
                        start_frame=seg_start,
                        end_frame=frame + 1,
                        score=score,
                    )
                )
            last_id = lbl
        else:
            seg_scores.append(score)
            if out:
                out[-1].end_frame = frame + 1

    if out and seg_scores:
        out[-1].score = float(np.mean(seg_scores))
    return out


def _viterbi_align(
    log_probs: np.ndarray,
    target_ids: list[int],
    intended_tokens: list[str],
    blank_id: int,
) -> list[AlignedToken]:
    """Pure-Python CTC forced-alignment Viterbi (slow but correct)."""
    T, V = log_probs.shape
    # Insert blanks between targets: [b, t1, b, t2, b, ...]
    expanded = [blank_id]
    for tid in target_ids:
        expanded.extend([tid, blank_id])
    L = len(expanded)
    if T < (L + 1) // 2:
        return []  # not enough frames to fit the sequence

    NEG_INF = -1e30
    trellis = np.full((T, L), NEG_INF, dtype=np.float64)
    backptr = np.zeros((T, L), dtype=np.int32)

    trellis[0, 0] = log_probs[0, expanded[0]]
    if L > 1:
        trellis[0, 1] = log_probs[0, expanded[1]]

    for t in range(1, T):
        for s in range(L):
            stay = trellis[t - 1, s]
            from_prev = trellis[t - 1, s - 1] if s > 0 else NEG_INF
            from_skip = (
                trellis[t - 1, s - 2]
                if s > 1 and expanded[s] != blank_id and expanded[s] != expanded[s - 2]
                else NEG_INF
            )
            best = max(stay, from_prev, from_skip)
            backptr[t, s] = (
                0 if best == stay else (1 if best == from_prev else 2)
            )
            trellis[t, s] = best + log_probs[t, expanded[s]]

    # Best terminal state
    end_state = L - 1 if trellis[T - 1, L - 1] >= trellis[T - 1, L - 2] else L - 2
    path = [(T - 1, end_state)]
    t, s = T - 1, end_state
    while t > 0:
        step = backptr[t, s]
        s = s - step
        t -= 1
        path.append((t, s))
    path.reverse()

    # Walk path → segments per non-blank target.
    segments: dict[int, tuple[int, int, list[float]]] = {}
    target_idx_by_state = {2 * i + 1: i for i in range(len(target_ids))}
    for t, s in path:
        if s in target_idx_by_state:
            ti = target_idx_by_state[s]
            score = float(log_probs[t, expanded[s]])
            if ti not in segments:
                segments[ti] = (t, t + 1, [score])
            else:
                start, _, scores = segments[ti]
                scores.append(score)
                segments[ti] = (start, t + 1, scores)

    out: list[AlignedToken] = []
    keep = [t for t in intended_tokens if any(target_ids)]
    for ti in range(len(target_ids)):
        if ti not in segments:
            continue
        start, end, scores = segments[ti]
        out.append(
            AlignedToken(
                symbol=keep[ti] if ti < len(keep) else "?",
                start_frame=start,
                end_frame=end,
                score=float(np.mean(scores)),
            )
        )
    return out
