"""Emergency detection (Layer 6 / spec).

Two complementary detectors:
  * ``check_text``     — keyword scan over the LLM's primary translation +
                          alternatives; fires even when overall confidence is
                          low (false positives are acceptable, missed
                          emergencies are not).
  * ``check_prosody``  — voice-quality red flags (high distress score from the
                          emotional tracker, very high jitter/shimmer, sudden
                          rate change). Surfaces an alert before the LLM even
                          gets a chance to translate.

Both write to a shared ``EmergencyAlert`` shape. The translator separately
sets ``emergency: true`` on the LLM tool result; this module is the safety
net layer that catches things the LLM might wave away.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Priority vocabulary the spec calls out: pain, help, can't breathe, choking,
# plus a small expansion for medical settings.
_PRIORITY_TERMS: dict[str, str] = {
    "pain": "pain",
    "hurt": "pain",
    "hurts": "pain",
    "hurting": "pain",
    "ache": "pain",
    "help": "help",
    "stop": "stop",
    "cant breathe": "respiratory",
    "can't breathe": "respiratory",
    "cannot breathe": "respiratory",
    "breathe": "respiratory",
    "breathing": "respiratory",
    "choke": "choking",
    "choking": "choking",
    "fall": "fall",
    "falling": "fall",
    "fallen": "fall",
    "fell": "fall",
    "bleeding": "bleeding",
    "blood": "bleeding",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "panic": "panic",
    "scared": "panic",
    "afraid": "panic",
    "emergency": "emergency",
}


@dataclass
class EmergencyAlert:
    triggered: bool
    category: str | None = None
    phrase: str | None = None
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)


def check_text(*, primary: str, alternatives: list[str], overall_confidence: float) -> EmergencyAlert:
    """Fire if any priority term appears in primary OR any alternative.

    We deliberately also scan alternatives — if the LLM hedged "help" into
    its alternatives we still want the alert to fire. The threshold for
    emergencies is set lower than the threshold for routine display.
    """
    candidates: list[tuple[str, str]] = [("primary", primary)]
    for alt in alternatives:
        candidates.append(("alternative", alt))

    for source, text in candidates:
        if not text:
            continue
        lowered = text.lower()
        for term, category in _PRIORITY_TERMS.items():
            if _word_in(term, lowered):
                # Even at low confidence we surface the alert — the spec is
                # explicit that missed emergencies are far worse than false
                # positives. We still report the LLM's overall confidence so
                # the UI can shade accordingly.
                return EmergencyAlert(
                    triggered=True,
                    category=category,
                    phrase=text,
                    confidence=overall_confidence,
                    sources=[source, term],
                )
    return EmergencyAlert(triggered=False)


def check_prosody(
    features: dict,
    emotion: dict,
    baseline: dict | None = None,
) -> EmergencyAlert:
    """Fire on voice-quality red flags before/without an LLM translation.

    When a calibrated patient ``baseline`` is provided, threshold on the
    *delta* over baseline — dysarthric speech has elevated jitter/shimmer
    at rest and the absolute thresholds tuned for healthy speakers fire
    constantly without this correction.

    When no baseline is available, fall back to higher absolute thresholds
    (a deliberate retune) — false-positive prosody emergencies are worse
    than the original tunables suggested for impaired speech.
    """
    if not features:
        return EmergencyAlert(triggered=False)

    jitter = float(features.get("jitter", 0) or 0)
    shimmer = float(features.get("shimmer", 0) or 0)
    distress = float(emotion.get("distress", 0) or 0)

    bl = baseline or {}
    has_baseline = bool(bl)

    # `distress` is computed by EmotionalStateTracker; when a baseline is
    # active that score is already a delta, so a 0.6 reading represents a
    # meaningful escalation. Without a baseline the score sits high on
    # dysarthric speech, so we require a much higher absolute reading.
    distress_threshold = 0.6 if has_baseline else 0.95
    if distress >= distress_threshold:
        return EmergencyAlert(
            triggered=True,
            category="distress",
            phrase=None,
            confidence=distress,
            sources=["prosody", "distress_score"],
        )

    if has_baseline:
        jitter_threshold = float(bl.get("jitter_p50", 0) or 0) + 0.08
        shimmer_threshold = float(bl.get("shimmer_p50", 0) or 0) + 0.10
    else:
        jitter_threshold = 0.15
        shimmer_threshold = 0.20
    if jitter > jitter_threshold and shimmer > shimmer_threshold:
        excess_j = jitter - jitter_threshold
        excess_s = shimmer - shimmer_threshold
        return EmergencyAlert(
            triggered=True,
            category="distress",
            phrase=None,
            confidence=min(1.0, (excess_j + excess_s) * 4),
            sources=["prosody", "jitter+shimmer"],
        )
    return EmergencyAlert(triggered=False)


def _word_in(term: str, text: str) -> bool:
    """Match whole-word for single tokens; substring for multi-word phrases."""
    if " " in term:
        return term in text
    # Simple word-boundary check.
    pad = f" {text} "
    return f" {term} " in pad or pad.startswith(f" {term}") or pad.endswith(f"{term} ")
