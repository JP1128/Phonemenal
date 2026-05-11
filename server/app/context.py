"""Six-layer context window (Phase 4).

Each layer is a small in-memory class with explicit decay rules. The
``ContextManager`` aggregates them per-session and exposes
``assemble_for_llm()`` which returns the dict the Sonnet translator's
``TranslationContext`` consumes.

Spec rules baked in:
  * Utterance relevance — exponential decay, half-life 14 s.
  * Topic relevance — half-life 4 min, floor 0.1.
  * Pending question urgency — bell curve, peak ~5 s, expires ~30 s.
  * Vocabulary boost — decays with parent topic but lingers 1.3× longer.
  * Context is a prior, not a constraint — we surface weights, not bans.
  * All speech feeds context, but only direct address creates a response
    expectation. We mark the latter via PendingQuestionQueue.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterable


# ============================================================ tunables


UTTERANCE_HALFLIFE_S = 14.0
TOPIC_HALFLIFE_S = 240.0
TOPIC_RELEVANCE_FLOOR = 0.1
VOCAB_LINGER_MULTIPLIER = 1.3
PENDING_PEAK_S = 5.0
PENDING_EXPIRY_S = 30.0


def _now() -> float:
    return time.time()


def _decay(weight: float, age_s: float, halflife_s: float) -> float:
    if age_s <= 0:
        return weight
    return weight * (0.5 ** (age_s / halflife_s))


# ============================================================ utterance log


@dataclass
class LoggedUtterance:
    speaker: str
    role: str | None
    is_patient: bool
    text: str          # translated text for patient, transcript for others
    raw_text: str | None = None
    timestamp: float = field(default_factory=_now)


class UtteranceLog:
    """Chronological record. Last N kept verbatim; older folded into topics."""

    def __init__(self, max_recent: int = 20) -> None:
        self.max_recent = max_recent
        self._buf: list[LoggedUtterance] = []

    def append(self, utt: LoggedUtterance) -> None:
        self._buf.append(utt)
        if len(self._buf) > self.max_recent * 2:
            # Trim, keep the most recent 2x the LLM window for safety.
            self._buf = self._buf[-(self.max_recent * 2):]

    def recent(self, n: int | None = None) -> list[LoggedUtterance]:
        n = n or self.max_recent
        return self._buf[-n:]

    def with_relevance_now(self, n: int | None = None) -> list[dict]:
        now = _now()
        out: list[dict] = []
        for u in self.recent(n):
            age = now - u.timestamp
            rel = _decay(1.0, age, UTTERANCE_HALFLIFE_S)
            out.append(
                {
                    "speaker": u.speaker if not u.is_patient else "patient",
                    "role": u.role,
                    "text": u.text,
                    "age_s": round(age, 1),
                    "relevance": round(rel, 3),
                }
            )
        return out


# ============================================================ topic tracker


_KEYWORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")
_STOPWORDS = {
    "have", "your", "with", "this", "that", "from", "they", "them",
    "what", "when", "where", "would", "could", "should", "about",
    "their", "there", "here", "thing", "going", "doing", "really",
    "still", "again", "right", "feel", "feeling", "felt", "very",
    "much", "more", "less", "okay", "alright", "tell", "told", "good",
    "well", "down", "back", "just", "want", "need", "know",
    "yourself", "myself", "today", "yesterday", "tomorrow",
}


@dataclass
class Topic:
    keyword: str
    weight: float
    last_observed: float = field(default_factory=_now)


class TopicTracker:
    """Keyword-based topic accumulator with exponential decay + floor."""

    def __init__(self, max_topics: int = 20) -> None:
        self.max_topics = max_topics
        self._topics: dict[str, Topic] = {}

    def observe(self, text: str, weight: float = 1.0) -> None:
        now = _now()
        seen: set[str] = set()
        for tok in _KEYWORD_RE.findall(text.lower()):
            if tok in _STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            t = self._topics.get(tok)
            if t is None:
                self._topics[tok] = Topic(keyword=tok, weight=weight, last_observed=now)
            else:
                # Re-observation reinforces (capped). Don't grow without bound.
                t.weight = min(5.0, t.weight + weight)
                t.last_observed = now

    def active(self, k: int = 8) -> list[dict]:
        now = _now()
        ranked: list[tuple[str, float]] = []
        for keyword, t in self._topics.items():
            age = now - t.last_observed
            decayed = _decay(t.weight, age, TOPIC_HALFLIFE_S)
            decayed = max(decayed, TOPIC_RELEVANCE_FLOOR)
            ranked.append((keyword, decayed))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        return [{"keyword": k, "weight": round(v, 3)} for k, v in ranked[:k]]

    def vocabulary_boost(self, k: int = 12) -> dict[str, float]:
        """Per-keyword multiplier the LLM can use to shade its prior.

        Vocabulary lingers slightly longer than the parent topic — the spec
        calls for a 1.3× multiplier on the half-life.
        """
        now = _now()
        out: dict[str, float] = {}
        for keyword, t in self._topics.items():
            age = now - t.last_observed
            decayed = _decay(t.weight, age, TOPIC_HALFLIFE_S * VOCAB_LINGER_MULTIPLIER)
            if decayed >= TOPIC_RELEVANCE_FLOOR:
                out[keyword] = round(decayed, 3)
        # Keep the highest-weight K to cap the prompt size.
        if len(out) > k:
            top = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:k]
            out = dict(top)
        return out


# ============================================================ questions


# Light-weight question-type heuristics. The LLM already disambiguates well;
# this is just to give it a structured prior + an urgency curve.
_YES_NO_RE = re.compile(r"^(do|does|did|are|is|can|could|will|would|should|have|has)\b", re.I)
_NUMERIC_RE = re.compile(r"\b(scale|rate|how (much|many|long))\b", re.I)


def classify_question(text: str) -> str:
    if not text.strip().endswith("?") and not text.lower().startswith(
        ("how", "what", "where", "when", "why", "who")
    ):
        return ""
    if _NUMERIC_RE.search(text):
        return "numeric"
    if _YES_NO_RE.match(text.strip()):
        return "yes_no"
    return "open"


@dataclass
class PendingQuestion:
    text: str
    qtype: str         # yes_no / numeric / open
    asked_at: float = field(default_factory=_now)
    answered: bool = False


class PendingQuestionQueue:
    """Non-patient questions awaiting a patient response.

    Urgency is a bell curve: low immediately, peaks at ~5 s, decays past ~30 s.
    """

    def __init__(self) -> None:
        self._queue: list[PendingQuestion] = []

    def observe_question(self, text: str) -> None:
        qtype = classify_question(text)
        if not qtype:
            return
        self._queue.append(PendingQuestion(text=text.strip(), qtype=qtype))

    def mark_answered(self) -> None:
        # When the patient produces an utterance, mark the most recent
        # unanswered question as answered. Simple and good enough.
        for q in reversed(self._queue):
            if not q.answered:
                q.answered = True
                return

    def active(self) -> list[dict]:
        now = _now()
        out: list[dict] = []
        for q in self._queue[-8:]:  # cap
            age = now - q.asked_at
            if age > PENDING_EXPIRY_S or q.answered:
                continue
            # Bell curve: gaussian centered on PENDING_PEAK_S.
            sigma = PENDING_EXPIRY_S / 4.0
            urgency = math.exp(-((age - PENDING_PEAK_S) ** 2) / (2 * sigma * sigma))
            out.append(
                {
                    "text": q.text,
                    "type": q.qtype,
                    "age_s": round(age, 1),
                    "urgency": round(urgency, 3),
                }
            )
        return out


# ============================================================ speaker registry


# Vocabulary cues for role inference. Crude but good enough as a prior; the
# LLM can override based on the actual content.
_MEDICAL_CUES = re.compile(
    r"\b(scale|pain|medic\w*|doctor|nurse|prescrib|symptom|breathe|swallow|"
    r"appointment|therapy|swelling|x-ray|scan|blood|pressure|mri|ct\b)\b",
    re.I,
)
_FAMILY_CUES = re.compile(
    r"\b(love|honey|dad|mom|mommy|daddy|brother|sister|grandma|grandpa|"
    r"cousin|uncle|aunt|sweet\w*)\b",
    re.I,
)


@dataclass
class TrackedSpeaker:
    label: str
    transcript_buffer: list[str] = field(default_factory=list)
    inferred_role: str | None = None
    role_confidence: float = 0.0


class SpeakerRegistry:
    """Tracks within-session speakers + infers a role from their vocabulary."""

    def __init__(self) -> None:
        self._speakers: dict[str, TrackedSpeaker] = {}

    def observe(self, label: str, transcript: str) -> None:
        if label == "patient":
            return
        sp = self._speakers.setdefault(label, TrackedSpeaker(label=label))
        sp.transcript_buffer.append(transcript)
        joined = " ".join(sp.transcript_buffer[-20:])
        med = len(_MEDICAL_CUES.findall(joined))
        fam = len(_FAMILY_CUES.findall(joined))
        if med == 0 and fam == 0:
            sp.inferred_role = None
            sp.role_confidence = 0.0
            return
        if med >= fam:
            sp.inferred_role = "medical_professional"
            sp.role_confidence = round(med / max(1, med + fam), 2)
        else:
            sp.inferred_role = "family_member"
            sp.role_confidence = round(fam / max(1, med + fam), 2)

    def role_for(self, label: str) -> str | None:
        sp = self._speakers.get(label)
        return sp.inferred_role if sp else None

    def snapshot(self) -> list[dict]:
        return [
            {
                "label": s.label,
                "role": s.inferred_role,
                "role_confidence": s.role_confidence,
            }
            for s in self._speakers.values()
        ]


# ============================================================ emotional state


@dataclass
class EmotionalSignal:
    frustration: float = 0.0
    distress: float = 0.0
    last_updated: float = field(default_factory=_now)


class EmotionalStateTracker:
    """Inferred from prosody only (per spec rule — never from word content).

    We use jitter, shimmer, pause counts, and abandoned-utterance rate.
    """

    def __init__(self) -> None:
        self._state = EmotionalSignal()
        self._abandoned_window: list[float] = []

    def observe_features(self, features: dict) -> None:
        jitter = float(features.get("jitter", 0) or 0)
        shimmer = float(features.get("shimmer", 0) or 0)
        pauses = int(features.get("pause_count", 0) or 0)
        pause_total = int(features.get("pause_total_ms", 0) or 0)
        duration = max(1, int(features.get("duration_ms", 1) or 1))

        # Distress: high jitter + shimmer (voice-quality breakdown).
        distress = min(1.0, jitter * 6.0 + shimmer * 4.0)
        # Frustration: many pauses relative to utterance length.
        pause_ratio = pause_total / duration
        frustration = min(1.0, 0.4 * pauses / 5.0 + 0.6 * pause_ratio)

        s = self._state
        # EMA so single utterances don't whipsaw the gauge.
        s.distress = round(0.6 * s.distress + 0.4 * distress, 3)
        s.frustration = round(0.6 * s.frustration + 0.4 * frustration, 3)
        s.last_updated = _now()

    def observe_abandoned_utterance(self) -> None:
        self._abandoned_window.append(_now())
        # Keep only last 60 s.
        cutoff = _now() - 60
        self._abandoned_window = [t for t in self._abandoned_window if t >= cutoff]
        if len(self._abandoned_window) >= 3:
            self._state.frustration = min(1.0, self._state.frustration + 0.2)

    def snapshot(self) -> dict:
        return {
            "frustration": round(self._state.frustration, 3),
            "distress": round(self._state.distress, 3),
            "last_updated_s_ago": round(_now() - self._state.last_updated, 1),
        }


# ============================================================ metadata


@dataclass
class SessionMetadata:
    setting: str | None = None             # hospital / home / therapy
    listener_role_hint: str | None = None  # active speaker's inferred role
    time_of_day: str | None = None
    activity_context: str | None = None    # mealtime / medication / etc.
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "setting": self.setting,
            "listener_role_hint": self.listener_role_hint,
            "time_of_day": self.time_of_day,
            "activity_context": self.activity_context,
            "notes": self.notes,
        }


def _time_of_day_now() -> str:
    h = time.localtime().tm_hour
    if h < 5:
        return "night"
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    if h < 21:
        return "evening"
    return "night"


# ============================================================ aggregator


class ContextManager:
    """Per-session aggregator wiring all six layers together."""

    def __init__(self, session_id: int, max_recent: int = 20) -> None:
        self.session_id = session_id
        self.utterance_log = UtteranceLog(max_recent=max_recent)
        self.topics = TopicTracker()
        self.questions = PendingQuestionQueue()
        self.speakers = SpeakerRegistry()
        self.emotion = EmotionalStateTracker()
        self.metadata = SessionMetadata(time_of_day=_time_of_day_now())
        self._lock = Lock()

    # ----- ingest ------------------------------------------------------

    def observe_non_patient(self, speaker: str, transcript: str) -> None:
        if not transcript:
            return
        with self._lock:
            self.speakers.observe(speaker, transcript)
            self.topics.observe(transcript, weight=1.0)
            self.questions.observe_question(transcript)
            role = self.speakers.role_for(speaker)
            self.metadata.listener_role_hint = role
            self.utterance_log.append(
                LoggedUtterance(
                    speaker=speaker,
                    role=role,
                    is_patient=False,
                    text=transcript,
                )
            )

    def observe_patient(self, translated: str, raw: str | None = None) -> None:
        with self._lock:
            self.questions.mark_answered()
            self.topics.observe(translated, weight=0.5)  # quieter than listener
            self.utterance_log.append(
                LoggedUtterance(
                    speaker="patient",
                    role="patient",
                    is_patient=True,
                    text=translated,
                    raw_text=raw,
                )
            )

    def observe_features(self, features: dict) -> None:
        with self._lock:
            self.emotion.observe_features(features)

    def observe_abandoned_utterance(self) -> None:
        with self._lock:
            self.emotion.observe_abandoned_utterance()

    # ----- output ------------------------------------------------------

    def assemble_for_llm(self) -> dict:
        """Return a dict ready to be sliced into ``TranslationContext``."""
        with self._lock:
            return {
                "recent_utterances": self.utterance_log.with_relevance_now(),
                "active_topics": [t["keyword"] for t in self.topics.active()],
                "vocabulary_boost": self.topics.vocabulary_boost(),
                "pending_questions": [
                    {"text": q["text"], "type": q["type"], "urgency": q["urgency"]}
                    for q in self.questions.active()
                ],
                "speakers": self.speakers.snapshot(),
                "emotion": self.emotion.snapshot(),
                "session": self.metadata.to_dict(),
            }

    def whisper_prompt(self, max_chars: int = 220, vocab_text: str = "") -> str:
        """Assemble a short ``initial_prompt`` string for faster-whisper.

        Whisper accepts a free-form text prior that biases what it expects
        to hear next. We feed it the most recent listener turn (verbatim,
        natural language), the active topic keywords, and any
        caregiver-supplied personal vocabulary (family names, medications).
        Kept compact — Whisper's prompt budget is small and longer strings
        can pull the model into reproducing the prompt instead of
        transcribing.

        ``vocab_text`` is the personal-vocab block from
        ``vocabulary_prompt_block(db)``; empty when no terms are saved.
        """
        with self._lock:
            chunks: list[str] = []

            # Last non-patient utterance gives Whisper the conversational
            # backdrop. Patient turns are what we're transcribing now,
            # so we don't include them — Whisper might echo them back.
            recent = self.utterance_log.recent()
            for u in reversed(recent):
                if not u.is_patient and u.text:
                    chunks.append(u.text.strip())
                    break

            topics = [t["keyword"] for t in self.topics.active(k=6)]
            if topics:
                chunks.append("topics: " + ", ".join(topics))

            # Personal-vocab block from the DB-backed list. We only feed
            # the bare term tokens to Whisper, no boost numbers or kind
            # labels — those would just be noise to the ASR.
            if vocab_text:
                terms: list[str] = []
                for line in vocab_text.splitlines():
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    _, _, after = line.partition(":")
                    for raw in after.split(","):
                        term = raw.strip().split(" (boost")[0]
                        if term:
                            terms.append(term)
                if terms:
                    chunks.append("vocabulary: " + ", ".join(terms[:24]))

        prompt = " | ".join(chunks)
        if len(prompt) > max_chars:
            prompt = prompt[: max_chars - 1] + "…"
        return prompt

    def snapshot_for_ui(self) -> dict:
        """Identical structure returned to the debug panel — no truncation."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "recent": self.utterance_log.with_relevance_now(),
                "topics": self.topics.active(),
                "vocabulary_boost": self.topics.vocabulary_boost(),
                "pending_questions": self.questions.active(),
                "speakers": self.speakers.snapshot(),
                "emotion": self.emotion.snapshot(),
                "session": self.metadata.to_dict(),
            }
