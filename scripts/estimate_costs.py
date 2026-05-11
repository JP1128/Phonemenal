"""One-off: estimate Claude API cost per utterance and per calibration.

Builds realistic translate() and analyze_calibration() prompts by reusing
app.llm's formatters, measures their byte lengths, converts to a token
estimate (chars/3.7 for prose, chars/4.0 for JSON-y schema), then applies
the Sonnet 4.6 / Opus 4.7 rate card.

Run:  python scripts/estimate_costs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from app.llm import (  # noqa: E402
    _SUBMIT_TRANSLATION_TOOL,
    _SYSTEM_PREFIX,
    _ANALYZE_TOOL,
    _format_phoneme_frames,
    _format_profile_block,
    _format_user_block,
    summarize_profile_for_prompt,
    TranslationContext,
)

# ---------- prompt fixtures (representative, not extreme) ----------

# A "lived-in" profile after calibration: 20-ish top confusions + a few
# cluster simplifications + an opus summary. summarize_profile_for_prompt
# is exactly what the runtime feeds into the cached block.
SAMPLE_ROWS = ["aa", "ae", "b", "d", "k", "p", "s", "t", "th", "z"]
SAMPLE_PROFILE = {
    "n_calibration_utterances": 30,
    "syllable_deletion_rate": 0.18,
    "confusion_overall": {
        "rows": SAMPLE_ROWS,
        "cols": SAMPLE_ROWS,
        "counts": [
            [40, 5, 0, 0, 0, 0, 0, 1, 0, 0],
            [3, 35, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 28, 4, 0, 6, 0, 0, 0, 0],
            [0, 0, 5, 22, 0, 0, 0, 6, 0, 0],
            [0, 0, 0, 0, 30, 0, 2, 4, 0, 0],
            [0, 0, 7, 0, 0, 25, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 33, 2, 1, 4],
            [0, 0, 0, 4, 3, 0, 2, 27, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 1, 18, 0],
            [0, 0, 0, 0, 0, 0, 4, 0, 0, 22],
        ],
    },
    "cluster_simplifications": {
        "str": {"s": 12, "t": 4},
        "skr": {"s": 8, "k": 3},
        "spl": {"s": 10, "p": 2},
        "thr": {"th": 7, "t": 5, "r": 2},
    },
    "opus_insights": {
        "summary": (
            "Patient shows systematic cluster simplification, "
            "stop voicing confusion, and mild fricative weakening "
            "consistent with apraxia of speech."
        ),
    },
}

PROFILE_SUMMARY = summarize_profile_for_prompt(SAMPLE_PROFILE)


def make_translation_ctx(n_frames: int) -> TranslationContext:
    """Build a TranslationContext with n_frames synthetic top-3 frames."""
    frames = []
    for i in range(n_frames):
        ts = i * 20
        frames.append(
            {
                "timestamp_ms": ts,
                "phonemes": [
                    {"symbol": "aa", "prob": 0.48},
                    {"symbol": "ae", "prob": 0.27},
                    {"symbol": "ah", "prob": 0.11},
                ],
            }
        )
    return TranslationContext(
        phoneme_frames=frames,
        recent_utterances=[
            {"speaker": "nurse", "text": "How are you feeling this morning?"},
            {"speaker": "patient", "text": "tired"},
            {"speaker": "nurse", "text": "Are you in any pain?"},
        ]
        * 4,  # 12 lines, ~typical mid-conversation
        active_topics=["pain assessment", "morning routine", "medication"],
        pending_questions=[{"type": "yes_no", "text": "Are you in any pain?"}],
        fatigue_phase="mid",
        setting="hospital_room",
        listener_role="nurse",
    )


# ---------- size measurement ----------


def tok(chars: int, density: float = 3.7) -> int:
    """char→token estimate. 3.7 for English prose, ~4.0 for JSON-heavy text."""
    return int(round(chars / density))


def measure_translate(n_frames: int) -> dict:
    ctx = make_translation_ctx(n_frames)
    profile_block = _format_profile_block(PROFILE_SUMMARY, ctx)
    user_block = _format_user_block(ctx)
    tool_json = json.dumps(_SUBMIT_TRANSLATION_TOOL)

    sys_prefix_t = tok(len(_SYSTEM_PREFIX))
    profile_block_t = tok(len(profile_block))
    tool_t = tok(len(tool_json), density=4.0)
    user_t = tok(len(user_block))

    # Anthropic caches everything before the breakpoint: tools + system
    # blocks (the breakpoint here is on the profile block).
    cached_prefix_t = tool_t + sys_prefix_t + profile_block_t
    return {
        "n_frames": n_frames,
        "tool_tokens": tool_t,
        "sys_prefix_tokens": sys_prefix_t,
        "profile_block_tokens": profile_block_t,
        "cached_prefix_tokens": cached_prefix_t,
        "user_tokens": user_t,
        "user_chars": len(user_block),
    }


def measure_analyze() -> dict:
    examples = [
        {"intended": "I need water", "intended_phonemes": ["ai", "n", "iy", "d", "w", "ao", "t", "er"]},
        {"intended": "please call the nurse", "intended_phonemes": ["p", "l", "iy", "z"]},
    ] * 5
    empirical = {
        "n_calibration_utterances": SAMPLE_PROFILE["n_calibration_utterances"],
        "syllable_deletion_rate": SAMPLE_PROFILE["syllable_deletion_rate"],
        "intended_phoneme_counts": {p: 25 for p in SAMPLE_ROWS},
        "cluster_simplifications": SAMPLE_PROFILE["cluster_simplifications"],
    }
    user_text = (
        "## Empirical confusion summary\n"
        f"{json.dumps(empirical, ensure_ascii=False)[:8000]}\n\n"
        "## Sample utterances\n"
        f"{json.dumps(examples, ensure_ascii=False)[:8000]}\n\n"
        "Submit your analysis via the submit_distortion_insights tool."
    )
    sys_text = (
        "You are Phonemer's calibration analyst. Given empirical "
        "phoneme confusion data from a patient with a brain-injury "
        "speech impediment, identify systematic distortion patterns. "
        "Be precise; over-claiming patterns is worse than under-claiming."
    )
    return {
        "tool_tokens": tok(len(json.dumps(_ANALYZE_TOOL)), density=4.0),
        "sys_tokens": tok(len(sys_text)),
        "user_tokens": tok(len(user_text)),
    }


# ---------- pricing (USD per 1M tokens) ----------

# Sonnet 4.6 — input/output/cache rates per Anthropic pricing.
SONNET_IN = 3.00
SONNET_OUT = 15.00
SONNET_CACHE_WRITE_5M = 3.75  # 1.25× input
SONNET_CACHE_READ = 0.30      # 0.10× input

# Opus 4.7
OPUS_IN = 15.00
OPUS_OUT = 75.00
OPUS_CACHE_WRITE_5M = 18.75
OPUS_CACHE_READ = 1.50


def usd(tokens: int, rate_per_million: float) -> float:
    return tokens * rate_per_million / 1_000_000


# ---------- report ----------


def main() -> None:
    print("=" * 64)
    print("Phonemer — Claude API cost estimate")
    print("=" * 64)

    # Translation per utterance, three utterance lengths.
    print("\n[translate()]  Sonnet 4.6, per utterance")
    print(f"  cached prefix  ≈  (tool schema + system prefix + profile block)")
    print(f"  output         ≈  ~120 tokens (tool_use JSON, primary + 3 alts + scalars)")
    print()
    print(f"  {'utterance':<14}  {'cached':>7}  {'volatile':>9}  {'cold $':>9}  {'warm $':>9}")
    OUTPUT_TOK = 120
    for n_frames, label in [(50, "1 s short"), (100, "2 s typical"), (200, "4 s long")]:
        m = measure_translate(n_frames)
        cold = (
            usd(m["cached_prefix_tokens"], SONNET_CACHE_WRITE_5M)
            + usd(m["user_tokens"], SONNET_IN)
            + usd(OUTPUT_TOK, SONNET_OUT)
        )
        warm = (
            usd(m["cached_prefix_tokens"], SONNET_CACHE_READ)
            + usd(m["user_tokens"], SONNET_IN)
            + usd(OUTPUT_TOK, SONNET_OUT)
        )
        print(
            f"  {label:<14}  {m['cached_prefix_tokens']:>7}  "
            f"{m['user_tokens']:>9}  ${cold:>7.5f}  ${warm:>7.5f}"
        )

    # Per-minute and per-hour for the typical case.
    typ = measure_translate(100)
    typ_cold = (
        usd(typ["cached_prefix_tokens"], SONNET_CACHE_WRITE_5M)
        + usd(typ["user_tokens"], SONNET_IN)
        + usd(OUTPUT_TOK, SONNET_OUT)
    )
    typ_warm = (
        usd(typ["cached_prefix_tokens"], SONNET_CACHE_READ)
        + usd(typ["user_tokens"], SONNET_IN)
        + usd(OUTPUT_TOK, SONNET_OUT)
    )
    # Conversational pace: ~6 utterances/min (one every 10 s incl. pauses).
    UTT_PER_MIN = 6
    # First utt cold, rest warm (cache holds for 5 min).
    cost_per_min = typ_cold + (UTT_PER_MIN - 1) * typ_warm
    print(f"\n  ≈ {UTT_PER_MIN} utt/min  →  ${cost_per_min:.4f}/min  →  ${cost_per_min*60:.2f}/hr")
    print(f"  (1st utt cold, rest warm; cache TTL 5 min keeps prefix cached during active use)")

    # Calibration analysis — one-off Opus pass.
    print("\n[analyze_calibration()]  Opus 4.7 + adaptive thinking, one-off")
    a = measure_analyze()
    in_tok = a["tool_tokens"] + a["sys_tokens"] + a["user_tokens"]
    # Adaptive thinking: ranges widely; show low / mid / high envelopes.
    out_only = 500  # tool_use JSON
    print(f"  input tokens    ≈ {in_tok}  (tool {a['tool_tokens']} + system {a['sys_tokens']} + user {a['user_tokens']})")
    print(f"  output (no think) ≈ {out_only}")
    print()
    print(f"  {'thinking':<14}  {'output $':>10}")
    for thinking in [0, 2000, 6000, 12000]:
        cost = (
            usd(in_tok, OPUS_IN)
            + usd(out_only + thinking, OPUS_OUT)
        )
        label = f"{thinking} tok" if thinking else "off"
        print(f"  {label:<14}  ${cost:>8.4f}")

    print("\nNotes")
    print("  • char→token uses 3.7 (prose) / 4.0 (JSON schema). Real counts")
    print("    may vary ±15%; rerun with messages.count_tokens for exact.")
    print("  • Sonnet 4.6 pricing assumed: $3 in / $15 out / $0.30 cache-read")
    print("    / $3.75 cache-write (5-min TTL).")
    print("  • Opus 4.7 pricing assumed: $15 in / $75 out. Thinking tokens")
    print("    bill as output. Adaptive thinking length is workload-dependent.")
    print("  • Cache holds ~5 min; idle gaps >5 min force a re-write each time.")


if __name__ == "__main__":
    main()
