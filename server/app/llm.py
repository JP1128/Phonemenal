"""Layer 5 — LLM contextual correction.

Two surfaces:
  * ``translate()``       — Sonnet 4.6, real-time per-utterance translation.
                            Tool-use schema returns ranked candidates (LLM
                            IS the beam search; no separate beam_search module).
  * ``analyze_calibration()`` — Opus 4.7 + adaptive thinking, offline analysis
                            of calibration data to enrich the distortion profile.

The system prompt (patient profile + situational context) is marked
``cache_control: ephemeral`` so repeated translations within ~5 minutes
read it from cache at ~0.1× cost.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("phonemer.llm")


# ============================================================ schemas


@dataclass
class TranslationContext:
    """Everything Sonnet needs for one utterance, sans the static prefix."""
    phoneme_frames: list[dict]                  # [{timestamp_ms, phonemes:[{symbol, prob}]}]
    recent_utterances: list[dict] = field(default_factory=list)
    active_topics: list[str] = field(default_factory=list)
    pending_questions: list[dict] = field(default_factory=list)
    fatigue_phase: str | None = None
    setting: str | None = None
    listener_role: str | None = None


@dataclass
class TranslationResult:
    primary: str
    alternatives: list[str]
    per_word_confidence: list[float]
    emergency: bool
    context_hints: list[str]
    overall_confidence: float
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# Tool schema given to Sonnet — forces structured ranked candidates.
_SUBMIT_TRANSLATION_TOOL = {
    "name": "submit_translation",
    "description": (
        "Submit your interpretation of what the patient just said. "
        "Provide the most likely sentence as `primary`, then 3 alternatives "
        "in `alternatives` (at least 2 of which must IGNORE the conversational "
        "context entirely — context-free candidates serve as a safety net "
        "against context-driven hallucination). Always include per-word "
        "confidence and an emergency flag."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "primary": {
                "type": "string",
                "description": "Most likely sentence the patient intended.",
            },
            "alternatives": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Top-3 alternative interpretations. At least 2 must be "
                    "context-free (ignore active topics / pending questions)."
                ),
            },
            "per_word_confidence": {
                "type": "array",
                "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "description": "Confidence (0..1) for each word in `primary`.",
            },
            "overall_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Aggregate confidence in `primary`.",
            },
            "emergency": {
                "type": "boolean",
                "description": (
                    "True if the utterance signals pain, distress, "
                    "breathing trouble, choking, or any medical emergency. "
                    "False positives are acceptable; missed emergencies are not."
                ),
            },
            "context_hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Short phrases describing how active context shaped "
                    "the translation, if at all."
                ),
            },
        },
        "required": [
            "primary",
            "alternatives",
            "per_word_confidence",
            "overall_confidence",
            "emergency",
            "context_hints",
        ],
        "additionalProperties": False,
    },
}


_SYSTEM_PREFIX = (
    "You are Phonemer's contextual correction layer for a brain-injury speech "
    "translator. The patient's motor system distorts how they produce phonemes, "
    "so you receive PHONEME PROBABILITY DISTRIBUTIONS (not text) for what they "
    "actually produced, plus a learned distortion profile and conversational "
    "context.\n\n"
    "Your job: infer what the patient INTENDED to say and submit ranked "
    "candidates via the `submit_translation` tool.\n\n"
    "Hard rules:\n"
    "  - Context is a prior, never a constraint. At least 2 of your 3 "
    "    alternatives MUST ignore active topics and pending questions.\n"
    "  - Transparency over confidence: a confident wrong translation is far "
    "    more dangerous than 'I'm not sure'. If phoneme evidence is too thin "
    "    for any sensible candidate, set overall_confidence < 0.3 and pick a "
    "    short hedged primary like 'unable to interpret'.\n"
    "  - Emergency detection has a low bar. Any plausible sign of pain, "
    "    distress, breathing trouble, or choking → emergency: true.\n"
    "  - Use the patient's distortion profile to interpret which produced "
    "    phonemes likely correspond to which intended phonemes. Read confusion "
    "    matrices in the direction P(produced | intended).\n"
    "  - Keep `primary` to a short, natural English sentence. Single words and "
    "    fragments are fine when that's all the patient produced.\n"
)


# ============================================================ client


_client_lock = threading.Lock()
_client = None


def _get_client():
    """Lazy singleton; raises a clean error if the API key is unset."""
    global _client
    with _client_lock:
        if _client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic package not installed; pip install anthropic"
                ) from e
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
                    "and fill in your Anthropic API key."
                )
            _client = anthropic.Anthropic(api_key=api_key)
        return _client


def is_configured() -> bool:
    """Whether the Anthropic SDK is importable and an API key is present."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ============================================================ translate


def translate(
    *,
    model: str,
    profile_summary: str,
    context: TranslationContext,
    max_tokens: int = 1024,
) -> TranslationResult:
    """Run one Sonnet inference for a single patient utterance.

    The system prompt holds the static prefix + patient distortion profile
    + situational context, all marked cacheable. The user message holds the
    volatile per-utterance content.
    """
    client = _get_client()

    system = [
        {"type": "text", "text": _SYSTEM_PREFIX},
        {
            "type": "text",
            "text": _format_profile_block(profile_summary, context),
            "cache_control": {"type": "ephemeral"},
        },
    ]

    user_text = _format_user_block(context)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_text}],
        tools=[_SUBMIT_TRANSLATION_TOOL],
        tool_choice={"type": "tool", "name": "submit_translation"},
    )

    parsed = _extract_tool_call(response)
    return TranslationResult(
        primary=parsed.get("primary", ""),
        alternatives=list(parsed.get("alternatives", []))[:3],
        per_word_confidence=list(parsed.get("per_word_confidence", [])),
        emergency=bool(parsed.get("emergency", False)),
        context_hints=list(parsed.get("context_hints", [])),
        overall_confidence=float(parsed.get("overall_confidence", 0.0)),
        cache_read_tokens=int(getattr(response.usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_tokens=int(
            getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        ),
        input_tokens=int(response.usage.input_tokens or 0),
        output_tokens=int(response.usage.output_tokens or 0),
    )


# ============================================================ analyze


_ANALYZE_TOOL = {
    "name": "submit_distortion_insights",
    "description": (
        "Submit deeper insights about this patient's distortion patterns, "
        "based on the calibration data and the empirical confusion matrix."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "systematic_substitutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "intended": {"type": "string"},
                        "produced": {"type": "string"},
                        "context": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["intended", "produced", "confidence"],
                    "additionalProperties": False,
                },
            },
            "cluster_simplification_rules": {
                "type": "array",
                "items": {"type": "string"},
            },
            "fatigue_signals": {
                "type": "array",
                "items": {"type": "string"},
            },
            "vocabulary_recommendations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence_caveats": {"type": "string"},
        },
        "required": ["summary", "systematic_substitutions"],
        "additionalProperties": False,
    },
}


def analyze_calibration(
    *,
    model: str,
    empirical_profile: dict,
    examples: list[dict],
) -> dict:
    """Run an Opus pass over calibration data and return structured insights.

    ``examples`` is a small subset of (intended_text, produced_top_phonemes)
    pairs — keep it under ~20 to stay well inside the cache window.
    """
    client = _get_client()

    system = [
        {
            "type": "text",
            "text": (
                "You are Phonemer's calibration analyst. Given empirical "
                "phoneme confusion data from a patient with a brain-injury "
                "speech impediment, identify systematic distortion patterns. "
                "Be precise; over-claiming patterns is worse than under-claiming."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    user_text = (
        "## Empirical confusion summary\n"
        f"{json.dumps(empirical_profile, ensure_ascii=False)[:8000]}\n\n"
        "## Sample utterances\n"
        f"{json.dumps(examples, ensure_ascii=False)[:8000]}\n\n"
        "Submit your analysis via the submit_distortion_insights tool."
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user_text}],
        tools=[_ANALYZE_TOOL],
        tool_choice={"type": "tool", "name": "submit_distortion_insights"},
    )

    return _extract_tool_call(response)


# ============================================================ formatting


def _format_profile_block(profile_summary: str, ctx: TranslationContext) -> str:
    parts = ["# Patient distortion profile\n", profile_summary or "(no profile yet)"]
    parts.append("\n\n# Situational context")
    parts.append(f"setting: {ctx.setting or 'unspecified'}")
    if ctx.listener_role:
        parts.append(f"listener_role: {ctx.listener_role}")
    return "\n".join(parts)


def _format_user_block(ctx: TranslationContext) -> str:
    out: list[str] = []
    out.append("# Phoneme distributions for the current utterance")
    out.append("Top-K probabilities per 20 ms frame, in order:\n")
    out.append(_format_phoneme_frames(ctx.phoneme_frames))

    if ctx.fatigue_phase:
        out.append(f"\n# Fatigue phase: {ctx.fatigue_phase}")

    if ctx.recent_utterances:
        out.append("\n# Recent conversation (newest last)")
        for utt in ctx.recent_utterances:
            who = utt.get("speaker", "?")
            text = utt.get("text") or utt.get("transcript") or ""
            out.append(f"  - {who}: {text}")

    if ctx.active_topics:
        out.append(
            "\n# Active topics (vocabulary boost, but do NOT let this override "
            "phoneme evidence): " + ", ".join(ctx.active_topics)
        )

    if ctx.pending_questions:
        out.append("\n# Pending questions awaiting a patient response")
        for q in ctx.pending_questions:
            out.append(f"  - [{q.get('type', 'open')}] {q.get('text', '')}")

    out.append("\n\nCall submit_translation now.")
    return "\n".join(out)


def _format_phoneme_frames(frames: list[dict]) -> str:
    """Compact representation: one frame per line, top-3 only, prob to 2 dp."""
    lines: list[str] = []
    for f in frames[:200]:  # cap at 200 frames (~4 s) per request
        ts = f.get("timestamp_ms", 0)
        phs = f.get("phonemes", [])[:3]
        chunk = " ".join(f"{p['symbol']}:{p['prob']:.2f}" for p in phs)
        lines.append(f"  {ts:>5}ms  {chunk}")
    if len(frames) > 200:
        lines.append(f"  …({len(frames) - 200} more frames truncated)")
    return "\n".join(lines)


def _extract_tool_call(response) -> dict[str, Any]:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    raise RuntimeError(
        f"Expected a tool_use block, got stop_reason={response.stop_reason}"
    )


# ============================================================ profile compaction


def summarize_profile_for_prompt(profile_json: dict) -> str:
    """Compress a full PatientDistortionProfile dict into a short prompt block.

    Sending the raw 392x392 confusion matrix would dominate the cache; instead
    we surface the top-K confusions by mass plus high-level statistics.
    """
    if not profile_json:
        return "(no calibration profile yet — use phoneme evidence directly)"

    rows = profile_json.get("confusion_overall", {}).get("rows") or []
    cols = profile_json.get("confusion_overall", {}).get("cols") or []
    counts = profile_json.get("confusion_overall", {}).get("counts") or []

    diag_total = 0
    grand_total = 0
    confusions: list[tuple[str, str, int]] = []
    for i, row in enumerate(counts):
        for j, n in enumerate(row):
            if n == 0:
                continue
            grand_total += n
            if i == j:
                diag_total += n
            else:
                confusions.append((rows[i], cols[j], n))
    diag_mass = (diag_total / grand_total) if grand_total else 0.0
    confusions.sort(key=lambda t: t[2], reverse=True)

    lines: list[str] = [
        f"calibration utterances: {profile_json.get('n_calibration_utterances', 0)}",
        f"diagonal probability mass: {diag_mass:.2f}",
        f"syllable deletion rate: {profile_json.get('syllable_deletion_rate', 0.0):.2f}",
        "top phoneme confusions (intended → produced, weight):",
    ]
    for intended, produced, count in confusions[:20]:
        lines.append(f"  {intended} → {produced}  (n={count})")

    cluster = profile_json.get("cluster_simplifications") or {}
    if cluster:
        lines.append("cluster simplifications:")
        for src, mapping in list(cluster.items())[:8]:
            survivors = ", ".join(
                f"{k}:{v}" for k, v in sorted(mapping.items(), key=lambda kv: -kv[1])[:3]
            )
            lines.append(f"  {src} → {{{survivors}}}")

    insights = profile_json.get("opus_insights") or {}
    if insights.get("summary"):
        lines.append(f"opus summary: {insights['summary']}")
    return "\n".join(lines)
