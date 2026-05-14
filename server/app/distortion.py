"""Patient distortion profile (Layer 4).

Builds a per-patient model from calibration utterances:

  * Confusion matrix ``P(produced | intended)`` — softmax-normalized counts of
    which model phonemes the patient actually produces when attempting each
    intended phoneme.
  * Positional rules — separate confusion matrices for word-initial,
    word-medial, and word-final positions.
  * Cluster simplification — frequency of each consonant cluster being
    dropped or simplified (e.g. /str/ → /s/).
  * Syllable deletion — fraction of multi-syllable words missing a syllable.

The profile is JSON-serializable and stored in ``patient_profiles``.

Phase 3.3 (LLM Opus analysis) layers on top of this with deeper pattern
inference; for now we ship the empirical statistics.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np

from .alignment import AlignedToken


# Vowels in the wav2vec2 espeak inventory we use to detect syllables.
VOWEL_SET = set("aeiouæɑɐɒəɛɜɪɨɔøœuʊʌyɯɤɵ".replace(" ", ""))

# Special tokens emitted by wav2vec2 / CTC that aren't actual phonemes the
# speaker produced. ``<pad>`` is the CTC blank — it dominates probability mass
# during silence frames that sit at alignment-segment boundaries, and counting
# it as the "produced phoneme" would push every confusion onto the same row.
_NON_PHONEME_TOKENS = frozenset({"<pad>", "<s>", "</s>", "<unk>", "|", " "})


@dataclass
class ConfusionMatrix:
    rows: list[str]                          # intended phonemes
    cols: list[str]                          # produced phonemes
    counts: list[list[int]] = field(default_factory=list)

    def to_probabilities(self) -> list[list[float]]:
        out: list[list[float]] = []
        for row in self.counts:
            total = sum(row) or 1
            out.append([c / total for c in row])
        return out


@dataclass
class PatientDistortionProfile:
    n_calibration_utterances: int
    intended_phoneme_counts: dict[str, int]
    confusion_overall: ConfusionMatrix
    confusion_initial: ConfusionMatrix
    confusion_medial: ConfusionMatrix
    confusion_final: ConfusionMatrix
    cluster_simplifications: dict[str, dict[str, int]]
    syllable_deletion_rate: float
    fatigue_priors: dict[str, float]
    # Median voice-quality features over the patient's calibration set.
    # Used at runtime to threshold distress/emergency detection on the
    # delta from this patient's normal speech, instead of absolute
    # values — dysarthric speech has elevated jitter/shimmer at rest.
    prosody_baseline: dict[str, float] | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_calibration_utterances": self.n_calibration_utterances,
                "intended_phoneme_counts": self.intended_phoneme_counts,
                "confusion_overall": asdict(self.confusion_overall),
                "confusion_initial": asdict(self.confusion_initial),
                "confusion_medial": asdict(self.confusion_medial),
                "confusion_final": asdict(self.confusion_final),
                "cluster_simplifications": self.cluster_simplifications,
                "syllable_deletion_rate": self.syllable_deletion_rate,
                "fatigue_priors": self.fatigue_priors,
                "prosody_baseline": self.prosody_baseline,
            },
            ensure_ascii=False,
        )


# ============================================================ builder


def build_profile(
    aligned_utterances: list[dict],
    vocab_symbols: list[str],
    features_list: list[dict] | None = None,
) -> PatientDistortionProfile:
    """Aggregate alignment data from many calibration utterances into a profile.

    Each item in ``aligned_utterances`` is a dict with:
      - intended_tokens: list[str]              (full phoneme sequence)
      - aligned: list[AlignedToken]             (alignment of intended → frames)
      - produced: list[list[tuple[str, float]]] (top-k phonemes per frame)
      - intended_words_phonemes: list[list[str]] (words split into phonemes)

    ``features_list`` is the matching per-utterance prosody output of
    ``features.extract`` (jitter, shimmer, pause_count, pause_total_ms,
    duration_ms). Median values become the patient's prosody baseline, so
    runtime distress detection can fire on the delta from this patient's
    normal voice quality instead of absolute thresholds that don't fit
    dysarthric speech.
    """
    overall = _new_matrix(vocab_symbols)
    initial = _new_matrix(vocab_symbols)
    medial = _new_matrix(vocab_symbols)
    final = _new_matrix(vocab_symbols)

    intended_counts: Counter[str] = Counter()
    cluster_simp: dict[str, Counter[str]] = defaultdict(Counter)
    syllable_total = 0
    syllable_missing = 0
    n_utt = 0

    for utt in aligned_utterances:
        intended: list[str] = utt["intended_tokens"]
        aligned: list[AlignedToken] = utt["aligned"]
        produced: list[list[tuple[str, float]]] = utt["produced"]
        words: list[list[str]] = utt.get("intended_words_phonemes", [])
        if not aligned or not produced:
            continue
        n_utt += 1

        # Tag positions per-token for positional confusion matrices.
        positions = _tag_positions(words, intended)

        for token, position in zip(aligned, positions):
            intended_counts[token.symbol] += 1
            top = _aggregate_segment_top(produced, token.start_frame, token.end_frame)
            if not top:
                continue
            produced_sym = top[0][0]
            _bump(overall, token.symbol, produced_sym)
            if position == "initial":
                _bump(initial, token.symbol, produced_sym)
            elif position == "final":
                _bump(final, token.symbol, produced_sym)
            else:
                _bump(medial, token.symbol, produced_sym)

        # Cluster simplification: detect dropped consonants in clusters.
        for word in words:
            for cluster in _consonant_clusters(word):
                produced_count = _count_aligned_in_cluster(aligned, cluster)
                if produced_count < len(cluster):
                    dropped_key = "".join(cluster)
                    survived = "".join(cluster[:produced_count])
                    cluster_simp[dropped_key][survived or "<deleted>"] += 1

        # Syllable deletion: count vowel nuclei in intended vs aligned.
        for word in words:
            intended_syllables = sum(1 for p in word if _is_vowel(p))
            aligned_set = set(t.symbol for t in aligned)
            produced_syllables = sum(
                1 for p in word if _is_vowel(p) and p in aligned_set
            )
            syllable_total += intended_syllables
            syllable_missing += max(0, intended_syllables - produced_syllables)

    deletion_rate = syllable_missing / syllable_total if syllable_total else 0.0

    # Fatigue priors are computed online during sessions; seed them flat here.
    fatigue_priors = {"fresh": 1.0, "mild": 1.1, "moderate": 1.25, "severe": 1.5}

    prosody_baseline: dict[str, float] | None = None
    if features_list:
        jitters = [float(f.get("jitter", 0) or 0) for f in features_list]
        shimmers = [float(f.get("shimmer", 0) or 0) for f in features_list]
        pause_ratios = [
            float(f.get("pause_total_ms", 0) or 0)
            / max(1.0, float(f.get("duration_ms", 1) or 1))
            for f in features_list
        ]
        pause_counts = [float(f.get("pause_count", 0) or 0) for f in features_list]
        if jitters:
            prosody_baseline = {
                "jitter_p50": _median(jitters),
                "shimmer_p50": _median(shimmers),
                "pause_ratio_p50": _median(pause_ratios),
                "pause_count_p50": _median(pause_counts),
                "n_samples": float(len(jitters)),
            }

    return PatientDistortionProfile(
        n_calibration_utterances=n_utt,
        intended_phoneme_counts=dict(intended_counts),
        confusion_overall=overall,
        confusion_initial=initial,
        confusion_medial=medial,
        confusion_final=final,
        cluster_simplifications={k: dict(v) for k, v in cluster_simp.items()},
        syllable_deletion_rate=deletion_rate,
        fatigue_priors=fatigue_priors,
        prosody_baseline=prosody_baseline,
    )


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float((s[n // 2 - 1] + s[n // 2]) / 2.0)


# ============================================================ utilities


def _new_matrix(symbols: list[str]) -> ConfusionMatrix:
    return ConfusionMatrix(
        rows=symbols,
        cols=symbols,
        counts=[[0] * len(symbols) for _ in range(len(symbols))],
    )


def _bump(m: ConfusionMatrix, intended: str, produced: str) -> None:
    if intended not in m.rows or produced not in m.cols:
        return
    i = m.rows.index(intended)
    j = m.cols.index(produced)
    m.counts[i][j] += 1


def _is_vowel(p: str) -> bool:
    return any(c in VOWEL_SET for c in p)


def _consonant_clusters(word_phonemes: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    cur: list[str] = []
    for p in word_phonemes:
        if _is_vowel(p):
            if len(cur) >= 2:
                out.append(cur)
            cur = []
        else:
            cur.append(p)
    if len(cur) >= 2:
        out.append(cur)
    return out


def _tag_positions(words: list[list[str]], intended: list[str]) -> list[str]:
    """Return one tag per intended phoneme: 'initial' / 'medial' / 'final'."""
    out = ["medial"] * len(intended)
    cursor = 0
    for word in words:
        if not word:
            continue
        if cursor < len(out):
            out[cursor] = "initial"
        if cursor + len(word) - 1 < len(out):
            out[cursor + len(word) - 1] = (
                "final" if len(word) > 1 else "initial"
            )
        cursor += len(word)
    return out


def _aggregate_segment_top(
    produced: list[list[tuple[str, float]]],
    start_frame: int,
    end_frame: int,
) -> list[tuple[str, float]]:
    """Sum probabilities across the aligned frames for each phoneme; pick top.

    CTC special tokens (blank, sentinels) are excluded — they aren't phonemes
    the speaker produced, just emission artifacts at silence boundaries.
    """
    accum: Counter[str] = Counter()
    for frame in produced[start_frame:end_frame]:
        for sym, prob in frame:
            if sym in _NON_PHONEME_TOKENS:
                continue
            accum[sym] += prob
    return accum.most_common(5)


def _count_aligned_in_cluster(
    aligned: list[AlignedToken], cluster: list[str]
) -> int:
    """Count how many of the consonants in ``cluster`` survived into alignment."""
    seen = [t.symbol for t in aligned]
    surviving = 0
    cursor = 0
    for ph in cluster:
        try:
            cursor = seen.index(ph, cursor) + 1
            surviving += 1
        except ValueError:
            return surviving
    return surviving
