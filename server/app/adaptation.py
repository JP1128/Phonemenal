"""Phase 8 — Continuous Learning & Adaptation.

The pieces:

  * ``ingest_feedback(...)``             — turn one confirmed/corrected
    translation into (intended, produced) phoneme pairs and persist them
    to ``online_observations``.

  * ``effective_profile_summary(...)``   — combine the baseline calibration
    profile with EMA-decayed online observations and produce a short
    text summary for the Sonnet translator's cached system prompt.

  * ``detect_shift(...)``                — flag when the recent diagonal mass
    diverges from the baseline by more than ``SIGMA_THRESHOLD`` σ. The
    spec calls this out as a possible neurological-event signal.

  * ``rebuild_intelligibility_snapshots(...)`` — weekly buckets of diagonal
    mass + confirmed-rate + avg confidence for the /progress page.

The math:

  Baseline confusion matrix     C0[i,j]   (integer counts from calibration)
  Online weighted observation   ΔC[i,j] = w · exp(-age_days / τ)
                                          summed over recent obs
  Effective row probability     P_eff[i,j] ∝ C0[i,j] + α · ΔC[i,j]

  α (= ``ONLINE_BLEND``) controls how aggressively recent feedback shifts
  the baseline. τ (= ``DECAY_DAYS``) is the half-life of an observation.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .alignment import AlignedToken, forced_align, text_to_phonemes
from .db import (
    list_online_observations,
    list_translations_with_feedback,
    write_intelligibility_snapshot,
    write_online_observations,
)

logger = logging.getLogger("phonemer.adaptation")


ONLINE_BLEND = 0.05            # α — EMA aggressiveness
DECAY_DAYS = 14.0              # τ — observation half-life
SIGMA_THRESHOLD = 2.0          # neurological-shift alert
MAX_RECENT_OBS_DAYS = 60.0     # how far back to consider for the effective profile
TOP_CONFUSIONS_FOR_PROMPT = 24


# ============================================================ ingest


def ingest_feedback(
    *,
    conn: sqlite3.Connection,
    extractor,                 # PhonemeExtractor
    utterance_id: int,
    intended_text: str,
    confirmed: bool,
    corrected_text: str | None,
    calibration_id: int | None,
) -> int:
    """Align the (corrected or primary) text against the utterance's phoneme
    distributions and persist the resulting (intended, produced) pairs.

    Returns the number of pair observations written.
    """
    # 1. Pull the persisted phoneme distributions for the utterance.
    cur = conn.execute(
        "SELECT timestamp_ms, distribution_json FROM phoneme_frames"
        " WHERE utterance_id = ? ORDER BY timestamp_ms",
        (utterance_id,),
    )
    frames_raw = cur.fetchall()
    if not frames_raw:
        logger.info(
            "no phoneme_frames for utterance %s — skipping adaptation", utterance_id
        )
        return 0

    vocab = extractor.processor.tokenizer.get_vocab()
    blank_id = extractor.processor.tokenizer.pad_token_id or 0

    # Reconstruct full per-frame probability vectors over the model vocab.
    V = len(vocab)
    probs = np.full((len(frames_raw), V), 1e-12, dtype=np.float32)
    for f_idx, (_, dist_json) in enumerate(frames_raw):
        try:
            dist = json.loads(dist_json)
        except json.JSONDecodeError:
            continue
        for entry in dist:
            tid = vocab.get(entry["symbol"])
            if tid is not None:
                probs[f_idx, tid] = float(entry["prob"])
    log_probs = np.log(probs)

    # 2. Phonemize the (corrected or primary) text and force-align.
    intended_tokens = text_to_phonemes(corrected_text or intended_text)
    if not intended_tokens:
        return 0
    aligned = forced_align(
        log_probs=log_probs,
        intended_tokens=intended_tokens,
        token_to_id=vocab,
        blank_id=blank_id,
    )
    if not aligned:
        return 0

    # 3. For each aligned segment, the dominant produced phoneme is the
    #    argmax of the summed probabilities across that segment.
    pairs: list[tuple[str, str]] = []
    inv_vocab = {v: k for k, v in vocab.items()}
    for tok in aligned:
        if tok.start_frame >= tok.end_frame:
            continue
        segment = probs[tok.start_frame : tok.end_frame].sum(axis=0)
        produced_id = int(np.argmax(segment))
        produced_sym = inv_vocab.get(produced_id, "")
        if not produced_sym:
            continue
        pairs.append((tok.symbol, produced_sym))

    source = "corrected" if not confirmed else "confirmed"
    weight = 1.0 if confirmed else 1.5  # corrections carry slightly more weight
    n = write_online_observations(
        conn,
        calibration_id=calibration_id,
        utterance_id=utterance_id,
        pairs=pairs,
        source=source,
        weight=weight,
    )
    logger.info(
        "ingest_feedback utt=%s pairs=%d source=%s",
        utterance_id,
        n,
        source,
    )
    return n


# ============================================================ effective profile


def effective_profile_summary(
    conn: sqlite3.Connection,
    calibration_id: int | None,
) -> str | None:
    """Build the LLM prompt block: baseline + decayed online observations
    + top confusions + simple recent-trend stats. Returns ``None`` if no
    baseline exists (caller should fall back to "no profile yet")."""
    baseline = _load_baseline(conn, calibration_id)
    if baseline is None:
        return None

    rows = baseline.get("confusion_overall", {}).get("rows") or []
    cols = baseline.get("confusion_overall", {}).get("cols") or []
    counts = baseline.get("confusion_overall", {}).get("counts") or []
    if not rows or not counts:
        return None

    # Apply decayed online observations on top of baseline counts.
    now = time.time()
    cutoff = now - MAX_RECENT_OBS_DAYS * 86400
    obs = list_online_observations(conn, calibration_id, since_ts=cutoff)

    sym_to_idx = {s: i for i, s in enumerate(rows)}
    accum = np.array(counts, dtype=np.float64)
    online_total_weight = 0.0
    for o in obs:
        i = sym_to_idx.get(o["intended"])
        j = sym_to_idx.get(o["produced"])
        if i is None or j is None:
            continue
        age_days = max(0.0, (now - o["observed_at"]) / 86400.0)
        decay = math.exp(-age_days / DECAY_DAYS)
        w = float(o["weight"]) * decay
        accum[i, j] += ONLINE_BLEND * w * 100.0  # scaled into "count" units
        online_total_weight += w

    diag_mass, top_confusions = _matrix_stats(accum, rows, cols, k=TOP_CONFUSIONS_FOR_PROMPT)

    n_baseline_utt = baseline.get("n_calibration_utterances", 0)
    syll_rate = baseline.get("syllable_deletion_rate", 0.0)
    cluster = baseline.get("cluster_simplifications") or {}

    lines: list[str] = [
        f"calibration utterances: {n_baseline_utt}",
        f"online observations applied: {len(obs)} (effective weight {online_total_weight:.2f})",
        f"effective diagonal probability mass: {diag_mass:.2f}",
        f"syllable deletion rate (baseline): {syll_rate:.2f}",
        "top phoneme confusions (intended → produced, weight) — effective:",
    ]
    for intended, produced, w in top_confusions:
        lines.append(f"  {intended} → {produced}  (w={w:.2f})")

    if cluster:
        lines.append("cluster simplifications:")
        for src, mapping in list(cluster.items())[:8]:
            survivors = ", ".join(
                f"{k}:{v}"
                for k, v in sorted(mapping.items(), key=lambda kv: -kv[1])[:3]
            )
            lines.append(f"  {src} → {{{survivors}}}")

    insights = baseline.get("opus_insights") or {}
    if insights.get("summary"):
        lines.append(f"opus summary: {insights['summary']}")

    return "\n".join(lines)


def _load_baseline(
    conn: sqlite3.Connection, calibration_id: int | None
) -> dict | None:
    if calibration_id is not None:
        cur = conn.execute(
            "SELECT profile_json FROM patient_profiles"
            " WHERE calibration_id = ? ORDER BY version DESC LIMIT 1",
            (calibration_id,),
        )
    else:
        cur = conn.execute(
            "SELECT profile_json FROM patient_profiles ORDER BY id DESC LIMIT 1"
        )
    row = cur.fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def _matrix_stats(
    accum: np.ndarray,
    rows: list[str],
    cols: list[str],
    k: int,
) -> tuple[float, list[tuple[str, str, float]]]:
    diag = 0.0
    total = 0.0
    for i in range(accum.shape[0]):
        for j in range(accum.shape[1]):
            v = accum[i, j]
            if v <= 0:
                continue
            total += v
            if i == j:
                diag += v
    diag_mass = diag / total if total else 0.0

    flat: list[tuple[str, str, float]] = []
    for i in range(accum.shape[0]):
        for j in range(accum.shape[1]):
            if i == j or accum[i, j] <= 0:
                continue
            flat.append((rows[i], cols[j], float(accum[i, j])))
    flat.sort(key=lambda t: t[2], reverse=True)
    return diag_mass, flat[:k]


# ============================================================ shift detect


def detect_shift(
    conn: sqlite3.Connection,
    calibration_id: int | None,
    window_obs: int = 50,
) -> dict | None:
    """Compare diagonal-mass over the last ``window_obs`` observations
    against the running mean from earlier observations. Returns a
    ``{"flagged": bool, "z_score": float, ...}`` dict or ``None`` if
    not enough data."""
    obs = list_online_observations(conn, calibration_id, since_ts=None)
    if len(obs) < window_obs * 2:
        return None

    def diag_rate(seq: list[dict]) -> float:
        if not seq:
            return 0.0
        n_match = sum(1 for o in seq if o["intended"] == o["produced"])
        return n_match / len(seq)

    recent = obs[-window_obs:]
    earlier = obs[:-window_obs]

    earlier_chunks = [
        earlier[i : i + window_obs]
        for i in range(0, len(earlier), window_obs)
        if len(earlier[i : i + window_obs]) == window_obs
    ]
    if not earlier_chunks:
        return None
    earlier_rates = [diag_rate(c) for c in earlier_chunks]
    mu = float(np.mean(earlier_rates))
    sigma = float(np.std(earlier_rates)) or 0.05  # avoid div-by-zero
    cur_rate = diag_rate(recent)
    z = (cur_rate - mu) / sigma

    flagged = abs(z) >= SIGMA_THRESHOLD and cur_rate < mu  # only flag drops
    return {
        "flagged": bool(flagged),
        "z_score": round(z, 3),
        "current_diagonal_rate": round(cur_rate, 3),
        "baseline_mean": round(mu, 3),
        "baseline_sigma": round(sigma, 3),
        "n_observations": len(obs),
    }


# ============================================================ snapshots


def rebuild_intelligibility_snapshots(
    conn: sqlite3.Connection,
    calibration_id: int | None = None,
) -> int:
    """Recompute weekly snapshots (diagonal mass, confirmed rate, avg conf)
    from the translations + observations history. Returns rows written.
    """
    feedback_rows = list_translations_with_feedback(conn)
    if not feedback_rows:
        return 0

    obs = list_online_observations(conn, calibration_id, since_ts=None)
    obs_by_week: dict[float, list[dict]] = defaultdict(list)
    for o in obs:
        obs_by_week[_week_start(o["observed_at"])].append(o)

    fb_by_week: dict[float, list[dict]] = defaultdict(list)
    for r in feedback_rows:
        try:
            fb = json.loads(r["feedback"])
        except (json.JSONDecodeError, TypeError):
            continue
        ts = float(fb.get("at", r["created_at"] or 0.0))
        fb_by_week[_week_start(ts)].append(
            {
                "confirmed": bool(fb.get("confirmed")),
                "overall_confidence": float(r["overall_confidence"] or 0.0),
            }
        )

    weeks = sorted(set(obs_by_week.keys()) | set(fb_by_week.keys()))
    written = 0
    for week_start in weeks:
        week_obs = obs_by_week.get(week_start, [])
        week_fb = fb_by_week.get(week_start, [])
        if not week_obs and not week_fb:
            continue
        if week_obs:
            n_match = sum(1 for o in week_obs if o["intended"] == o["produced"])
            diag_mass = n_match / len(week_obs)
        else:
            diag_mass = 0.0
        if week_fb:
            confirmed_rate = sum(1 for f in week_fb if f["confirmed"]) / len(week_fb)
            avg_conf = float(np.mean([f["overall_confidence"] for f in week_fb]))
        else:
            confirmed_rate = 0.0
            avg_conf = 0.0
        write_intelligibility_snapshot(
            conn,
            calibration_id=calibration_id,
            week_start=week_start,
            n_utterances=max(len(week_obs), len(week_fb)),
            diagonal_mass=diag_mass,
            confirmed_rate=confirmed_rate,
            avg_overall_confidence=avg_conf,
        )
        written += 1
    return written


def _week_start(ts: float) -> float:
    """Round a Unix timestamp down to the most recent Monday 00:00 UTC."""
    t = time.gmtime(ts)
    # Monday = 0 in struct_time.tm_wday
    day_seconds = (
        t.tm_wday * 86400 + t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
    )
    return float(int(ts - day_seconds))
