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

import logging
import math
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterable

logger = logging.getLogger("phonemer.context")


# ============================================================ tunables


UTTERANCE_HALFLIFE_S = 14.0
TOPIC_HALFLIFE_S = 120.0
TOPIC_RELEVANCE_FLOOR = 0.1
VOCAB_LINGER_MULTIPLIER = 1.3
# Topics below this decayed weight don't appear in the UI's "active topics".
# A single mention starts at 1.0 and decays past this floor in roughly
# 1 half-life, so one-off words ("hello") drop out within ~2 minutes
# instead of lingering at the relevance floor forever.
TOPIC_DISPLAY_FLOOR = 0.5
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
    # function words / determiners (4+ chars; the regex already drops <4)
    "have", "having", "your", "yours", "with", "without", "this", "that",
    "those", "these", "from", "into", "onto", "upon", "they", "them",
    "their", "there", "here", "what", "when", "where", "while", "whom",
    "whose", "which", "would", "could", "should", "about", "above",
    "below", "across", "after", "before", "between", "during", "since",
    "though", "although", "however", "because", "than", "then", "until",
    "unless", "either", "neither", "both", "each", "some", "such",
    "very", "much", "more", "less", "most", "least", "many", "several",
    "even", "ever", "never", "always", "often", "sometimes", "anyway",
    "still", "again", "also", "only", "just", "quite", "rather",
    "almost", "nearly", "another", "other", "others", "same", "every",
    # courtesy / greetings / fillers
    "hello", "good", "goodbye", "please", "thank", "thanks", "welcome",
    "sure", "yeah", "alright", "okay", "right", "fine", "great",
    # generic verbs that rarely carry topical meaning
    "doing", "done", "going", "gone", "make", "makes", "made", "making",
    "take", "takes", "took", "taken", "taking", "give", "given", "gave",
    "giving", "gives", "come", "comes", "came", "coming", "look", "looks",
    "looked", "looking", "seem", "seems", "seemed", "seeming", "find",
    "found", "feel", "feels", "felt", "feeling", "tell", "told", "tells",
    "telling", "want", "wants", "wanted", "wanting", "need", "needs",
    "needed", "know", "knows", "known", "knowing", "said", "says",
    "saying", "talk", "talks", "talked", "talking", "think", "thinks",
    "thought", "thinking", "let", "lets", "letting", "keep", "kept",
    "keeps", "keeping", "really", "actually", "basically", "literally",
    "honestly", "probably", "perhaps", "maybe",
    # common nouns that are too generic to count as a topic on their own
    "thing", "things", "stuff", "kind", "kinds", "type", "types", "sort",
    "sorts", "lot", "lots", "today", "tonight", "yesterday", "tomorrow",
    "year", "years", "month", "months", "week", "weeks", "hour", "hours",
    "minute", "minutes", "second", "seconds", "moment", "moments",
    # pronouns / reflexives 4+ chars
    "yourself", "myself", "himself", "herself", "itself", "themselves",
    "ourselves",
    # state-of-being / aux 4+ chars
    "been", "being", "were", "have", "having",
    # generic adjectives
    "well", "down", "back", "long", "short", "true", "false",
}


@dataclass
class Topic:
    keyword: str
    weight: float
    last_observed: float = field(default_factory=_now)


# Small extra filter on top of POS-tagging — high-frequency nouns that
# pass NOUN/PROPN but carry no topical meaning. spaCy's `is_stop` already
# catches most function words, but generic nouns like "thing" / "stuff"
# need an explicit list because spaCy correctly identifies them as nouns.
_NLP_CONTENT_STOPWORDS = {
    "thing", "things", "stuff", "kind", "kinds", "sort", "sorts",
    "type", "types", "lot", "lots", "way", "ways", "side", "sides",
    "place", "places", "person", "people", "guy", "someone", "something",
    "anything", "everything", "nothing",
    "today", "tonight", "tomorrow", "yesterday",
    "year", "years", "month", "months", "week", "weeks",
    "hour", "hours", "minute", "minutes", "second", "seconds",
    "moment", "moments", "time", "times",
    "hello", "hi", "hey", "thanks", "thank",
}

# Lazy-loaded spaCy pipeline. Loaded once per process; reused across
# every TopicTracker. Falls back to the regex+stopword path if spaCy or
# the model isn't installed (`pip install spacy && python -m spacy
# download en_core_web_sm`).
_nlp_singleton: Any = None
_nlp_load_attempted = False


def _get_nlp() -> Any:
    global _nlp_singleton, _nlp_load_attempted
    if _nlp_load_attempted:
        return _nlp_singleton
    _nlp_load_attempted = True
    try:
        import spacy  # type: ignore[import-not-found]

        # Disable parser — we only need tokenizer + tagger + NER. ~3x faster
        # and the dependency graph isn't used for keyword extraction.
        _nlp_singleton = spacy.load("en_core_web_sm", disable=["parser"])
        logger.info("spaCy loaded for topic extraction (en_core_web_sm)")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "spaCy unavailable (%s) — topic extraction falls back to regex",
            e,
        )
        _nlp_singleton = None
    return _nlp_singleton


class TopicTracker:
    """Topic accumulator with exponential decay + display floor.

    Extracts keywords with spaCy POS tagging (NOUN / PROPN only) plus
    NER (PERSON / ORG / GPE / PRODUCT / EVENT / WORK_OF_ART) so meaningful
    nouns and named entities both surface as topics. Falls back to a
    regex+stopword scan if spaCy or its model isn't loaded.
    """

    # spaCy entity labels we want to capture as topical (the rest are
    # quantitative / temporal / numeric and not interesting as topics).
    _ENTITY_LABELS: frozenset[str] = frozenset(
        {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC"}
    )

    def __init__(self, max_topics: int = 20) -> None:
        self.max_topics = max_topics
        self._topics: dict[str, Topic] = {}

    def observe(self, text: str, weight: float = 1.0) -> None:
        if not text or not text.strip():
            return
        nlp = _get_nlp()
        if nlp is None:
            self._observe_regex(text, weight)
            return

        now = _now()
        seen: set[str] = set()
        doc = nlp(text)

        # Multi-token named entities first — preserves "Dr. Smith" /
        # "blood pressure cuff" as a single keyword instead of letting the
        # token loop split them apart.
        for ent in doc.ents:
            if ent.label_ not in self._ENTITY_LABELS:
                continue
            key = ent.text.strip().lower()
            if not key or key in seen:
                continue
            if key in _NLP_CONTENT_STOPWORDS:
                continue
            seen.add(key)
            self._reinforce(key, weight, now)

        # Single-token nouns / proper nouns. Use lemmas so "shoulders" and
        # "shoulder" merge.
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue
            if tok.pos_ not in ("NOUN", "PROPN"):
                continue
            lemma = (tok.lemma_ or tok.text).lower().strip()
            if len(lemma) < 3:
                continue
            if lemma in _NLP_CONTENT_STOPWORDS or lemma in seen:
                continue
            seen.add(lemma)
            self._reinforce(lemma, weight, now)

    def _observe_regex(self, text: str, weight: float) -> None:
        """Legacy fallback: regex tokenizer + handcrafted stopword set."""
        now = _now()
        seen: set[str] = set()
        for tok in _KEYWORD_RE.findall(text.lower()):
            if tok in _STOPWORDS or tok in seen:
                continue
            seen.add(tok)
            self._reinforce(tok, weight, now)

    def _reinforce(self, key: str, weight: float, now: float) -> None:
        t = self._topics.get(key)
        if t is None:
            self._topics[key] = Topic(keyword=key, weight=weight, last_observed=now)
        else:
            # Re-observation reinforces (capped) so a single repeated word
            # doesn't dominate forever.
            t.weight = min(5.0, t.weight + weight)
            t.last_observed = now

    def active(self, k: int = 8) -> list[dict]:
        """Topics that are still salient *for display*. Filters by
        TOPIC_DISPLAY_FLOOR so single-mention words ("hello") drop out
        after a half-life or two instead of sticking forever via the
        legacy relevance floor.
        """
        now = _now()
        ranked: list[tuple[str, float]] = []
        for keyword, t in self._topics.items():
            age = now - t.last_observed
            decayed = _decay(t.weight, age, TOPIC_HALFLIFE_S)
            if decayed < TOPIC_DISPLAY_FLOOR:
                continue
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

    When a patient ``baseline`` (median jitter/shimmer/pause_ratio over their
    own calibration utterances) is provided, we measure the *delta* from
    that baseline so dysarthric patients aren't pegged at "high distress"
    by the natural voice-quality signature of their disorder.
    """

    def __init__(self, baseline: dict | None = None) -> None:
        self._state = EmotionalSignal()
        self._abandoned_window: list[float] = []
        self._baseline = baseline or {}

    def observe_features(self, features: dict) -> None:
        jitter = float(features.get("jitter", 0) or 0)
        shimmer = float(features.get("shimmer", 0) or 0)
        pauses = int(features.get("pause_count", 0) or 0)
        pause_total = int(features.get("pause_total_ms", 0) or 0)
        duration = max(1, int(features.get("duration_ms", 1) or 1))

        # Subtract the patient's calibration baseline (clamped at 0). Without
        # this step a dysarthric speaker's normal jitter ~0.10 + shimmer
        # ~0.12 would push distress to ~1.0 every utterance. With baseline,
        # distress represents how much THIS utterance deviates from how the
        # patient normally sounds.
        bl = self._baseline
        jitter_d = max(0.0, jitter - float(bl.get("jitter_p50", 0) or 0))
        shimmer_d = max(0.0, shimmer - float(bl.get("shimmer_p50", 0) or 0))
        pause_ratio = pause_total / duration
        pause_ratio_d = max(
            0.0, pause_ratio - float(bl.get("pause_ratio_p50", 0) or 0)
        )

        distress = min(1.0, jitter_d * 6.0 + shimmer_d * 4.0)
        frustration = min(1.0, 0.4 * pauses / 5.0 + 0.6 * pause_ratio_d)

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

    def __init__(
        self,
        session_id: int,
        max_recent: int = 20,
        prosody_baseline: dict | None = None,
    ) -> None:
        self.session_id = session_id
        self.utterance_log = UtteranceLog(max_recent=max_recent)
        self.topics = TopicTracker()
        self.questions = PendingQuestionQueue()
        self.speakers = SpeakerRegistry()
        self.emotion = EmotionalStateTracker(baseline=prosody_baseline)
        self._prosody_baseline = prosody_baseline or {}
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

    def get_prosody_baseline(self) -> dict:
        return self._prosody_baseline

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
