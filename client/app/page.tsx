"use client";

import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { AccessibilityPanel } from "../components/accessibility-panel";
import { ThemeToggle } from "../components/theme-toggle";
import { Button } from "../components/button";
import { KeyboardHelp } from "../components/keyboard-help";
import {
  DemoBadge,
  LiveStatus,
  type LiveState,
} from "../components/live-status";
import { MicLevelMeter } from "../components/mic-level-meter";
import { PanelResizer } from "../components/panel-resizer";
import { PanelTab } from "../components/panel-tab";
import { ProfileSelector } from "../components/profile-selector";
import { TorgoDrawer } from "../components/torgo-drawer";
import { Waveform } from "../components/waveform";
import { useAccessibility } from "../lib/accessibility";
import {
  type CalibratedSpeaker,
  defaultProfile,
  loadProfile,
  parseCalibratedSpeakers,
  type Profile,
  reconcileProfile,
  saveProfile,
} from "../lib/profile";
import {
  MicPermissionState,
  queryMicPermission,
} from "../lib/audio-devices";
import { useReplayAudio } from "../lib/replay-audio";
import { playEmergencyChime } from "../lib/chime";
import type {
  AcousticFeatures,
  ContextSnapshot,
  EmergencyMessage,
  TranslationMessage,
  UtteranceMessage,
  WsMessage,
} from "../lib/messages";
import { usePhonemerSession } from "../lib/use-phonemer-session";

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";
const WS_URL = SERVER_HTTP.replace(/^http/, "ws") + "/ws/audio";

type Translated = {
  utteranceId: number;
  speaker: string;
  startMs: number;
  endMs: number;
  primary: string;
  alternatives: string[];
  perWordConfidence: number[];
  overallConfidence: number;
  emergency: boolean;
  contextHints: string[];
  // Phoneme-alignment safety gate. Flagged means Whisper produced text the
  // wav2vec2 phoneme distribution doesn't support (likely hallucination).
  flagged: boolean;
  flagReason: string | null;
  // Diagnostic-report fields. align_score is the mean forced-align log-prob
  // of Whisper's text against the wav2vec2 phoneme distribution; density is
  // phonemes-per-second of the proposed transcription. Both surface in the
  // copy-diagnostic report for TORGO clips.
  alignScore?: number;
  phonemeDensity?: number;
  feedback?: "confirmed" | "corrected";
  correctedText?: string;
  // Set only when this utterance came from a TORGO demo clip. Lets the UI
  // show the ground-truth prompt next to the model's transcription so the
  // listener can compare them directly.
  intendedText?: string;
  arrivedAt: number;
};

type EmergencyAlert = {
  utteranceId: number;
  phrase: string | null;
  category?: string;
  confidence: number;
  ts: number;
  dismissed: boolean;
};

const MAX_HISTORY = 100;

type ThreadFilter = "all" | "flagged" | "unconfirmed";

type SessionEvent = { type: "start" | "stop"; ts: number };

type ConversationEntry = {
  utteranceId: number;
  speaker: string;
  isPatient: boolean;
  text: string | null;
  ts: number;
};

const CONVERSATION_LOG_CAP = 100;

const TORGO_WIDTH_DEFAULT = 320;
const CONTEXT_WIDTH_DEFAULT = 320;
const SIDE_PANEL_MIN = 220;
const SIDE_PANEL_MAX = 640;

const STORAGE_KEYS = {
  torgoWidth: "phonemer.panel.torgoWidth",
  contextWidth: "phonemer.panel.contextWidth",
  showTorgo: "phonemer.panel.showTorgo",
  showContext: "phonemer.panel.showContext",
};

const clamp = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v));

function loadStoredNumber(key: string, fallback: number): number {
  if (typeof window === "undefined") return fallback;
  const raw = window.localStorage.getItem(key);
  if (raw == null) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function loadStoredBool(key: string, fallback: boolean): boolean {
  if (typeof window === "undefined") return fallback;
  const raw = window.localStorage.getItem(key);
  if (raw == null) return fallback;
  return raw === "true";
}

function formatRelativeTime(then: number, now: number): string {
  const diff = Math.max(0, now - then);
  if (diff < 5_000) return "just now";
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

function formatAbsoluteTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

// Word-level diff: returns each segment of the intended text annotated with
// whether the (normalized) word also appears in the heard text. Uses a
// multiset so repeated words are matched 1:1 (e.g. intended "the the" only
// fully matches heard "the the"). Order is ignored — adequate for short
// utterances and avoids the visual cost of true sequence alignment.
type IntendedSegment = { token: string; matched: boolean; isWord: boolean };

function diffIntendedWords(
  intended: string,
  heard: string,
): IntendedSegment[] {
  const norm = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s']/gu, " ")
      .split(/\s+/)
      .filter(Boolean);
  const heardCounts = new Map<string, number>();
  for (const w of norm(heard)) {
    heardCounts.set(w, (heardCounts.get(w) ?? 0) + 1);
  }
  // Split preserving whitespace so we can render the original spacing back.
  const tokens = intended.split(/(\s+)/);
  return tokens.map((tok) => {
    const isWord = /\p{L}|\p{N}/u.test(tok);
    if (!isWord) return { token: tok, matched: true, isWord: false };
    const key = norm(tok)[0] ?? "";
    const remaining = heardCounts.get(key) ?? 0;
    if (remaining > 0) {
      heardCounts.set(key, remaining - 1);
      return { token: tok, matched: true, isWord: true };
    }
    return { token: tok, matched: false, isWord: true };
  });
}

export default function HomePage() {
  const a11y = useAccessibility();

  const [history, setHistory] = useState<Translated[]>([]);
  const [emergency, setEmergency] = useState<EmergencyAlert | null>(null);
  const [emergencyLog, setEmergencyLog] = useState<EmergencyAlert[]>([]);
  // Count of emergencies that landed while the right panel was closed —
  // surfaced as a badge on the right edge tab so the user knows there's
  // unread alert content even after dismissing the banner.
  const [unreadEmergencyCount, setUnreadEmergencyCount] = useState(0);
  // Mic start/stop events, newest-first. Interleaved into the thread as
  // horizontal divider rows so a long page session reads as discrete
  // recordings instead of one undifferentiated stream.
  const [sessionEvents, setSessionEvents] = useState<SessionEvent[]>([]);
  // Client-side conversation log built from utterance + translation +
  // utterance_transcript messages. The server's per-WS snapshot resets
  // every time TorgoDrawer opens a new WS for a clip (force_patient=1),
  // so the right-panel "conversation" section can't trust snapshot.recent.
  // This log aggregates across all WS connections within a page lifetime.
  const [conversationLog, setConversationLog] = useState<ConversationEntry[]>(
    [],
  );
  // Refs read from inside the WS-message handler (which is memoized with []
  // deps) so they always see the latest state without re-creating the
  // callback. Mirrors the existing pendingUttRef pattern.
  const showContextRef = useRef(false);
  const [threadFilter, setThreadFilter] = useState<ThreadFilter>("all");
  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem("phonemer.thread.filter");
    if (raw === "all" || raw === "flagged" || raw === "unconfirmed") {
      setThreadFilter(raw);
    }
  }, []);
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("phonemer.thread.filter", threadFilter);
  }, [threadFilter]);

  // Patient display name. When set, replaces the "PATIENT" badge text on the
  // active card and thread rows. Set via the small editor that opens when
  // the active card's badge is clicked. Persists per browser.
  const [patientName, setPatientName] = useState<string | null>(null);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem("phonemer.patientName");
    if (raw) setPatientName(raw);
  }, []);
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (patientName) {
      window.localStorage.setItem("phonemer.patientName", patientName);
    } else {
      window.localStorage.removeItem("phonemer.patientName");
    }
  }, [patientName]);

  // Active patient profile + the set of TORGO speakers we have adapters
  // for. Together these drive the ProfileSelector dropdown and the
  // clips-drawer's scope. We refresh /adapters on mount, on window focus,
  // and every 30 s so a freshly-trained patient adapter shows up in the
  // dropdown without a manual reload.
  const [profile, setProfile] = useState<Profile>({ kind: "demo" });
  const [calibratedSpeakers, setCalibratedSpeakers] = useState<
    CalibratedSpeaker[]
  >([]);
  // Whether the very first /adapters fetch has finished. Until it has we
  // can't reconcile the saved profile, so the dropdown shows whatever was
  // saved (or "Demo TORGO") but the auto-refresher won't trample it.
  const profileSeededRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const savedOnce = loadProfile();

    const refresh = async () => {
      try {
        const r = await fetch(`${SERVER_HTTP}/adapters`);
        if (!r.ok) throw new Error(r.statusText);
        const data = await r.json();
        if (cancelled) return;
        const enriched = parseCalibratedSpeakers(data.speakers);
        setCalibratedSpeakers((prev) => {
          // Avoid re-rendering downstream consumers when the list didn't
          // actually change. Cheap shallow compare on id+label+kind.
          if (
            prev.length === enriched.length &&
            prev.every(
              (v, i) =>
                v.id === enriched[i].id &&
                v.label === enriched[i].label &&
                v.kind === enriched[i].kind,
            )
          ) {
            return prev;
          }
          return enriched;
        });
        if (!profileSeededRef.current) {
          profileSeededRef.current = true;
          setProfile(reconcileProfile(savedOnce, enriched));
        } else {
          // After the first reconcile we still want to recover if a saved
          // profile pointed at a (now-deleted) adapter — but we don't want
          // to overwrite a user's active selection mid-session.
          setProfile((cur) => reconcileProfile(cur, enriched));
        }
      } catch {
        if (cancelled) return;
        if (!profileSeededRef.current) {
          profileSeededRef.current = true;
          setProfile(savedOnce ?? defaultProfile([]));
        }
      }
    };

    refresh();
    const interval = setInterval(refresh, 30_000);
    const onFocus = () => {
      // Window focus = the user came back to this tab. Refresh
      // immediately so a training run that finished in another tab /
      // window shows up right away.
      refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => {
      cancelled = true;
      clearInterval(interval);
      window.removeEventListener("focus", onFocus);
    };
  }, []);
  // TorgoDrawer matches `clip.speaker_id` (TORGO ids only) against this
  // set to decide which clips to mark "Calibrated". Patient-trained
  // adapters never match any TORGO clip, so we filter them out here so
  // the drawer's calibrated check stays focused.
  const calibratedSpeakerSet = useMemo(
    () =>
      new Set(
        calibratedSpeakers
          .filter((s) => s.kind === "torgo")
          .map((s) => s.id),
      ),
    [calibratedSpeakers],
  );
  useEffect(() => {
    saveProfile(profile);
  }, [profile]);

  // Distraction-free mode: hides side/bottom panels and the header so only
  // the active card + thread + mic bar remain. Toggled by `F` or the small
  // exit pill that appears while active.
  const [focusMode, setFocusMode] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (window.localStorage.getItem("phonemer.focusMode") === "true") {
      setFocusMode(true);
    }
  }, []);
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("phonemer.focusMode", String(focusMode));
  }, [focusMode]);
  const [ctx, setCtx] = useState<ContextSnapshot | null>(null);
  const [editing, setEditing] = useState<{
    utteranceId: number;
    text: string;
  } | null>(null);
  const [translationError, setTranslationError] = useState<string | null>(null);
  const [demoActive, setDemoActive] = useState(false);
  // 30s ticker so relative times in the thread refresh ("2m ago" → "3m ago")
  // without forcing parents to re-render every second.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, []);
  // Smart-autoscroll bookkeeping. mainScrollRef points at the scroll viewport
  // for the active card + thread; scrolledAway flips when the user scrolls
  // away from the top (i.e. they're reading older rows), and unreadCount
  // counts new utterances that landed while they were away so we can show
  // a "↑ N new" pill instead of yanking them back.
  const mainScrollRef = useRef<HTMLElement | null>(null);
  const [scrolledAway, setScrolledAway] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);
  useEffect(() => {
    const el = mainScrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const away = el.scrollTop > 80;
      setScrolledAway(away);
      if (!away) setUnreadCount(0);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);
  const [expandedThreadIds, setExpandedThreadIds] = useState<Set<number>>(
    () => new Set(),
  );
  const toggleThreadExpand = useCallback((id: number) => {
    setExpandedThreadIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);
  // While a TORGO clip is streaming, this holds the clip's intended_text so
  // we can stamp it on every utterance the server emits during the stream.
  const demoIntendedRef = useRef<string | null>(null);
  // utterance_id -> intended_text, populated as utterances arrive during a
  // demo stream so the value sticks even after the stream closes.
  const intendedByUtteranceRef = useRef<Map<number, string>>(new Map());
  const replay = useReplayAudio();

  const [showTorgo, setShowTorgo] = useState(false);
  const [showContext, setShowContext] = useState(false);

  const [torgoWidth, setTorgoWidth] = useState(TORGO_WIDTH_DEFAULT);
  const [contextWidth, setContextWidth] = useState(CONTEXT_WIDTH_DEFAULT);

  useEffect(() => {
    setTorgoWidth(
      clamp(
        loadStoredNumber(STORAGE_KEYS.torgoWidth, TORGO_WIDTH_DEFAULT),
        SIDE_PANEL_MIN,
        SIDE_PANEL_MAX,
      ),
    );
    setContextWidth(
      clamp(
        loadStoredNumber(STORAGE_KEYS.contextWidth, CONTEXT_WIDTH_DEFAULT),
        SIDE_PANEL_MIN,
        SIDE_PANEL_MAX,
      ),
    );
    setShowTorgo(loadStoredBool(STORAGE_KEYS.showTorgo, false));
    setShowContext(loadStoredBool(STORAGE_KEYS.showContext, false));
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEYS.torgoWidth, String(torgoWidth));
  }, [torgoWidth]);
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(
      STORAGE_KEYS.contextWidth,
      String(contextWidth),
    );
  }, [contextWidth]);
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEYS.showTorgo, String(showTorgo));
  }, [showTorgo]);
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEYS.showContext, String(showContext));
  }, [showContext]);
  useEffect(() => {
    showContextRef.current = showContext;
    if (showContext) setUnreadEmergencyCount(0);
  }, [showContext]);

  const prevMicStateRef = useRef<string | null>(null);

  const onResizeTorgo = useCallback((dx: number) => {
    setTorgoWidth((w) => clamp(w + dx, SIDE_PANEL_MIN, SIDE_PANEL_MAX));
  }, []);
  const onResizeContext = useCallback((dx: number) => {
    setContextWidth((w) => clamp(w - dx, SIDE_PANEL_MIN, SIDE_PANEL_MAX));
  }, []);

  // Open the TORGO panel automatically when arriving via /demo redirect.
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      if (params.get("demo") === "1") setShowTorgo(true);
    }
  }, []);

  const [pendingUtt, setPendingUtt] = useState<UtteranceMessage | null>(null);
  const pendingUttRef = useRef<UtteranceMessage | null>(null);
  const utteranceMessagesRef = useRef<Map<number, UtteranceMessage>>(new Map());
  // Acoustic features per utterance, populated from `features` WS messages.
  // Looked up at "Why?" button time so the diagnostic report can include
  // jitter / shimmer / pauses / duration alongside the text comparison.
  const featuresByUtteranceRef = useRef<Map<number, AcousticFeatures>>(
    new Map(),
  );

  useEffect(() => {
    pendingUttRef.current = pendingUtt;
  }, [pendingUtt]);

  // Reset all per-patient in-memory state when the active profile flips.
  // Why this exists: the WebSocket tears down (and rebinds to the new
  // ?adapter_speaker=) via the URL-change effect in usePhonemerSession,
  // but the transcripts, conversation log, emergencies, etc. live in
  // this component and would otherwise linger from the previous patient
  // — which is actively misleading in a clinical context. We key on a
  // stable string so any change (demo → calibrated, or one calibrated
  // patient → another) triggers the reset. The first render writes the
  // key without clearing.
  const profileKey =
    profile.kind === "calibrated"
      ? `calibrated:${profile.speakerId}`
      : "demo";
  const prevProfileKeyRef = useRef<string>(profileKey);
  useEffect(() => {
    if (prevProfileKeyRef.current === profileKey) return;
    prevProfileKeyRef.current = profileKey;
    // Transcript surfaces
    setHistory([]);
    setPendingUtt(null);
    setEditing(null);
    setExpandedThreadIds(new Set());
    setUnreadCount(0);
    // Conversation / context / alerts
    setConversationLog([]);
    setSessionEvents([]);
    setCtx(null);
    setEmergency(null);
    setEmergencyLog([]);
    setUnreadEmergencyCount(0);
    setTranslationError(null);
    // UI overrides bound to the previous patient
    setPatientName(null);
    // Refs that mirror message state — clear so a re-emitted id from the
    // new session can't accidentally inherit prior intended_text/features.
    utteranceMessagesRef.current = new Map();
    featuresByUtteranceRef.current = new Map();
    intendedByUtteranceRef.current = new Map();
    demoIntendedRef.current = null;
    // Any in-flight replay belonged to the old patient's audio.
    replay.stop();
  }, [profileKey, replay.stop]);

  // Watchdog: if no translation arrives within 30 s, clear the pending
  // utterance so the "transcribing…" placeholder can't pin to the
  // screen forever. Common causes for this firing:
  //   * TORGO demo WS hard-closed after streaming, server still owed a
  //     translation but the channel is gone.
  //   * Server-side exception during patient processing that didn't
  //     emit a `translation_error` (e.g. PEFT adapter swap blew up).
  //   * Network blip that dropped the translation message in flight.
  useEffect(() => {
    if (!pendingUtt) return;
    const watchedId = pendingUtt.utterance_id;
    const timer = window.setTimeout(() => {
      setPendingUtt((prev) =>
        prev?.utterance_id === watchedId ? null : prev,
      );
    }, 30_000);
    return () => window.clearTimeout(timer);
  }, [pendingUtt]);

  // ============================================================ ws ingest

  const handleMessage = useCallback((msg: WsMessage) => {
    switch (msg.type) {
      case "utterance":
        // Only patient utterances trigger a translation message — listener /
        // caregiver utterances finish at this stage. Tracking non-patient
        // utterances in pendingUtt would leave inFlight stuck true forever
        // and pin the "transcribing…" placeholder to the screen.
        if (msg.is_patient) {
          setPendingUtt(msg);
        }
        utteranceMessagesRef.current.set(msg.utterance_id, msg);
        if (demoIntendedRef.current) {
          intendedByUtteranceRef.current.set(
            msg.utterance_id,
            demoIntendedRef.current,
          );
        }
        // Client-side conversation log: stub out the row, fill in text once
        // the corresponding translation / utterance_transcript arrives.
        setConversationLog((prev) => {
          if (prev.some((e) => e.utteranceId === msg.utterance_id)) return prev;
          const next: ConversationEntry = {
            utteranceId: msg.utterance_id,
            speaker: msg.speaker,
            isPatient: msg.is_patient,
            text: null,
            ts: Date.now(),
          };
          return [next, ...prev].slice(0, CONVERSATION_LOG_CAP);
        });
        break;
      case "utterance_transcript":
        setConversationLog((prev) =>
          prev.map((e) =>
            e.utteranceId === msg.utterance_id
              ? { ...e, text: msg.transcript }
              : e,
          ),
        );
        break;
      case "features":
        featuresByUtteranceRef.current.set(msg.utterance_id, msg.features);
        // Also write to the module-scoped lookup so the leaf WhyButton
        // (which doesn't receive the ref via props) can read it.
        featuresByUtteranceLookup.set(msg.utterance_id, msg.features);
        break;
      case "translation":
        applyTranslation(msg);
        // Drop pendingUtt for this id so inFlight goes false immediately
        // and the watchdog has nothing left to time out.
        setPendingUtt((prev) =>
          prev?.utterance_id === msg.utterance_id ? null : prev,
        );
        setTranslationError(null);
        break;
      case "translation_error":
        // Clear pendingUtt for THIS utterance so the "transcribing…"
        // placeholder doesn't pin to the screen forever after a server
        // failure for this id. The server emits this from the gated-
        // transcription path (Whisper + phoneme alignment); post-pivot
        // there is no LLM in the live loop, so the message reflects an
        // ASR / alignment failure.
        setPendingUtt((prev) =>
          prev?.utterance_id === msg.utterance_id ? null : prev,
        );
        setTranslationError(`Transcription error: ${msg.message}`);
        break;
      case "emergency":
        applyEmergency(msg);
        break;
      case "context":
        setCtx(msg.context);
        break;
      default:
        break;
    }
  }, []);

  const applyTranslation = (msg: TranslationMessage) => {
    // Fill the patient text into the client-side conversation log entry
    // that was stubbed out when the utterance message landed.
    setConversationLog((prev) =>
      prev.map((e) =>
        e.utteranceId === msg.utterance_id ? { ...e, text: msg.primary } : e,
      ),
    );
    setHistory((prev) => {
      const u = pendingUttRef.current;
      const matchingUtt =
        utteranceMessagesRef.current.get(msg.utterance_id) ?? u ?? null;
      const row: Translated = {
        utteranceId: msg.utterance_id,
        speaker: matchingUtt?.speaker ?? "patient",
        startMs: matchingUtt?.start_ms ?? 0,
        endMs: matchingUtt?.end_ms ?? 0,
        primary: msg.primary,
        alternatives: msg.alternatives,
        perWordConfidence: msg.per_word_confidence,
        overallConfidence: msg.overall_confidence,
        emergency: msg.emergency,
        contextHints: msg.context_hints,
        flagged: msg.flagged ?? false,
        flagReason: msg.reason ?? null,
        alignScore: msg.align_score,
        phonemeDensity: msg.phoneme_density,
        intendedText: intendedByUtteranceRef.current.get(msg.utterance_id),
        arrivedAt: Date.now(),
      };
      const existing = prev.findIndex((r) => r.utteranceId === row.utteranceId);
      let next: Translated[];
      if (existing >= 0) {
        next = [...prev];
        next[existing] = { ...next[existing], ...row };
      } else {
        next = [row, ...prev];
      }
      return next.slice(0, MAX_HISTORY);
    });
  };

  const applyEmergency = (msg: EmergencyMessage) => {
    const alert: EmergencyAlert = {
      utteranceId: msg.utterance_id,
      phrase: msg.phrase ?? null,
      category: msg.category,
      confidence: msg.confidence,
      ts: Date.now(),
      dismissed: false,
    };
    setEmergency(alert);
    setEmergencyLog((prev) => [alert, ...prev].slice(0, 50));
    if (!showContextRef.current) setUnreadEmergencyCount((c) => c + 1);
    playEmergencyChime();
  };

  // Live-mic WS URL. When a calibrated profile is active, append
  // `adapter_speaker=<id>` so the server pins the matching LoRA for the
  // session. The Demo profile is a bucket of uncalibrated speakers, so
  // we don't send the param and the server falls back to its default
  // adapter — matching the existing TORGO-clip behavior for
  // uncalibrated speakers. The hook reacts to URL changes by tearing
  // down the existing connection so the next mic start picks up the
  // new adapter binding.
  const wsUrl = useMemo(() => {
    if (profile.kind === "calibrated" && profile.speakerId) {
      return `${WS_URL}?adapter_speaker=${encodeURIComponent(
        profile.speakerId,
      )}`;
    }
    return WS_URL;
  }, [profile]);

  const session = usePhonemerSession({ url: wsUrl, onMessage: handleMessage });

  // Push a session event whenever the mic transitions in/out of "recording".
  // Connecting/idle/blocked transitions are ignored — only the actual start
  // and stop of audio capture become divider rows in the thread.
  useEffect(() => {
    const prev = prevMicStateRef.current;
    const curr = session.micState;
    if (prev !== null && prev !== curr) {
      if (curr === "recording" && prev !== "recording") {
        setSessionEvents((evs) =>
          [{ type: "start", ts: Date.now() } as SessionEvent, ...evs].slice(
            0,
            100,
          ),
        );
      } else if (prev === "recording" && curr !== "recording") {
        setSessionEvents((evs) =>
          [{ type: "stop", ts: Date.now() } as SessionEvent, ...evs].slice(
            0,
            100,
          ),
        );
      }
    }
    prevMicStateRef.current = curr;
  }, [session.micState]);

  // ============================================================ feedback

  const sendFeedback = useCallback(
    async (
      utteranceId: number,
      confirmed: boolean,
      correctedText: string | null,
    ) => {
      try {
        await fetch(`${SERVER_HTTP}/feedback/${utteranceId}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ confirmed, corrected_text: correctedText }),
        });
        setHistory((prev) =>
          prev.map((r) =>
            r.utteranceId === utteranceId
              ? {
                  ...r,
                  feedback: confirmed ? "confirmed" : "corrected",
                  correctedText: correctedText ?? undefined,
                }
              : r,
          ),
        );
      } catch (err) {
        console.warn("feedback failed", err);
      }
    },
    [],
  );

  const onConfirm = (id: number) => sendFeedback(id, true, null);
  const onStartEdit = (row: Translated) =>
    setEditing({
      utteranceId: row.utteranceId,
      // Seed the textarea from the latest accepted text: a prior
      // correction if there is one, otherwise the ASR primary. Without
      // this, a second edit silently throws away the user's earlier
      // correction when the textarea pops open.
      text:
        row.feedback === "corrected" && row.correctedText
          ? row.correctedText
          : row.primary,
    });
  const onCancelEdit = () => setEditing(null);
  const onSaveEdit = () => {
    if (!editing) return;
    const next = editing.text.trim();
    // An empty save would persist correctedText="" which then falls back
    // to the original in display but still flips the row to "Corrected"
    // status — confusing. Treat as a no-op cancel instead. Same goes
    // for a "save" that didn't change anything against the current
    // accepted text.
    const row = history.find((r) => r.utteranceId === editing.utteranceId);
    const current =
      row?.feedback === "corrected" && row.correctedText
        ? row.correctedText
        : (row?.primary ?? "");
    if (next.length === 0 || next === current.trim()) {
      setEditing(null);
      return;
    }
    sendFeedback(editing.utteranceId, false, next);
    setEditing(null);
  };

  // ============================================================ render

  const latest = history[0];
  const isLive = session.micState === "recording";

  // Keyboard shortcuts. Space toggles mic, Enter confirms latest, E starts a
  // correction on latest, Esc cancels an in-progress correction. Skip when a
  // form field has focus (otherwise Space/Enter/E would clobber typing).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName.toLowerCase();
      const inField =
        tag === "input" ||
        tag === "textarea" ||
        target?.isContentEditable === true;

      if (e.key === "Escape" && editing) {
        e.preventDefault();
        setEditing(null);
        return;
      }
      if (inField) return;

      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        if (session.micState === "recording") session.stop();
        else if (session.micState !== "connecting") session.start();
        return;
      }
      if (e.key === "Enter" && latest && !editing) {
        e.preventDefault();
        onConfirm(latest.utteranceId);
        return;
      }
      if ((e.key === "e" || e.key === "E") && latest && !editing) {
        e.preventDefault();
        setEditing({
          utteranceId: latest.utteranceId,
          text:
            latest.feedback === "corrected" && latest.correctedText
              ? latest.correctedText
              : latest.primary,
        });
        return;
      }
      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        setFocusMode((v) => !v);
        return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing, latest, session]);

  // Pipeline state for the live status indicator. "transcribing" = the server
  // emitted an utterance segment but its translation hasn't landed yet.
  // "listening" = mic (or demo clip) is active, no in-flight utterance.
  const inFlight =
    pendingUtt &&
    (!latest || pendingUtt.utterance_id !== latest.utteranceId);
  const liveState: LiveState = inFlight
    ? "transcribing"
    : isLive || demoActive
      ? "listening"
      : "idle";

  // Bump the unread counter when a brand-new latest utterance lands while the
  // user is scrolled into the thread. Skip the very first transition from
  // null → some-id (the user just got here, nothing to be "unread" about).
  const latestId = latest?.utteranceId ?? null;
  const prevLatestIdRef = useRef<number | null>(null);
  useEffect(() => {
    const prev = prevLatestIdRef.current;
    if (latestId !== null && prev !== null && latestId !== prev) {
      if (scrolledAway) setUnreadCount((c) => c + 1);
    }
    prevLatestIdRef.current = latestId;
  }, [latestId, scrolledAway]);

  const jumpToTop = useCallback(() => {
    mainScrollRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    setUnreadCount(0);
  }, []);

  // Replace the active card with a "transcribing…" placeholder while we're
  // waiting for a translation. Skip when the user is mid-correction so we
  // don't yank their textarea out from under them.
  const showPlaceholder = !!inFlight && !editing;

  return (
    <main
      className="h-screen flex flex-col overflow-hidden"
      style={{
        ...a11y.styleVars,
        background: "var(--phonemer-bg-color)",
        color: "var(--phonemer-text-color)",
      }}
    >
      {emergency && !emergency.dismissed && (
        <EmergencyBanner
          alert={emergency}
          onDismiss={() =>
            setEmergency((prev) => (prev ? { ...prev, dismissed: true } : null))
          }
        />
      )}

      {translationError && (
        <div className="px-6 py-2 bg-amber-900/40 border-b border-amber-500/30 text-amber-100 text-sm flex items-center gap-3">
          <span className="font-mono text-xs uppercase tracking-wider opacity-80">
            transcribe
          </span>
          <span className="flex-1">{translationError}</span>
          <button
            onClick={() => setTranslationError(null)}
            className="px-2 py-0.5 rounded-md bg-white/10 hover:bg-white/20 text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      {focusMode && (
        <button
          type="button"
          onClick={() => setFocusMode(false)}
          aria-label="Exit focus mode"
          title="Exit focus mode (press F)"
          className="fixed top-3 right-3 z-30 inline-flex items-center justify-center h-6 px-2 rounded-full bg-black/40 border border-white/10 text-[10px] font-mono uppercase tracking-wider whitespace-nowrap opacity-60 hover:opacity-100 transition-opacity backdrop-blur-md"
        >
          exit focus · F
        </button>
      )}
      {!focusMode && (
      <header className="shrink-0 h-12 px-4 flex items-center justify-between border-b border-white/5 bg-black/20">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-[10px] tracking-[0.3em] font-mono opacity-50 uppercase">
            phonemer
          </span>
          <ProfileSelector
            profile={profile}
            onChange={setProfile}
            calibratedSpeakers={calibratedSpeakers}
          />
        </div>
        <div className="flex items-center gap-3">
          <HeaderConnPill
            state={session.connState}
            onReconnect={session.reconnect}
          />
          <KeyboardHelp />
          <ThemeToggle />
          <AccessibilityPanel
            settings={a11y.settings}
            onChange={a11y.setSettings}
          />
          <AdminMenu
            focusMode={focusMode}
            onToggleFocusMode={() => setFocusMode((v) => !v)}
          />
        </div>
      </header>
      )}

      <div className="flex-1 flex min-h-0">
        {!focusMode && (showTorgo ? (
          <>
            <aside
              className="shrink-0 min-h-0"
              style={{ width: torgoWidth }}
            >
              <TorgoDrawer
                variant="inline"
                open={showTorgo}
                onClose={() => setShowTorgo(false)}
                onMessage={handleMessage}
                onStreamStart={(clip) => {
                  setDemoActive(true);
                  demoIntendedRef.current = clip.intended_text || null;
                }}
                onStreamEnd={() => {
                  setDemoActive(false);
                  demoIntendedRef.current = null;
                }}
                profile={profile}
                calibratedSpeakers={calibratedSpeakerSet}
              />
            </aside>
            <PanelResizer
              axis="x"
              onDrag={onResizeTorgo}
              ariaLabel="Resize clips panel"
            />
          </>
        ) : (
          <PanelTab
            side="left"
            label="clips"
            onClick={() => setShowTorgo(true)}
          />
        ))}

        <div className="flex-1 flex flex-col relative min-w-[280px]">
          {scrolledAway && unreadCount > 0 && (
            <button
              type="button"
              onClick={jumpToTop}
              className="absolute top-3 left-1/2 -translate-x-1/2 z-10 px-3 py-1.5 rounded-full bg-emerald-500/90 hover:bg-emerald-500 text-white text-xs font-medium shadow-lg shadow-black/40 transition-colors"
              aria-label={`Jump to ${unreadCount} new utterance${unreadCount > 1 ? "s" : ""}`}
            >
              ↑ {unreadCount} new
            </button>
          )}
          <section
            ref={mainScrollRef}
            className={`flex-1 overflow-y-auto ${
              !latest && !showPlaceholder
                ? "flex items-center justify-center px-8 md:px-12"
                : ""
            }`}
          >
            {showPlaceholder ? (
              <div className="max-w-3xl mx-auto px-6 md:px-10 py-12 space-y-10">
                <TranscribingPlaceholder
                  level={session.level}
                  demoActive={demoActive}
                />
                {history.length > 0 && (
                  <Thread
                    rows={history}
                    now={now}
                    filter={threadFilter}
                    onChangeFilter={setThreadFilter}
                    sessionEvents={sessionEvents}
                    patientName={patientName}
                    replayingId={replay.playingId}
                    onToggleReplay={replay.toggle}
                    expandedIds={expandedThreadIds}
                    onToggleExpand={toggleThreadExpand}
                    editing={editing}
                    onConfirm={onConfirm}
                    onStartEdit={onStartEdit}
                    onCancelEdit={onCancelEdit}
                    onSaveEdit={onSaveEdit}
                    onChangeEdit={(t) =>
                      setEditing((prev) =>
                        prev ? { ...prev, text: t } : null,
                      )
                    }
                  />
                )}
              </div>
            ) : !latest ? (
              <EmptyState
                liveState={liveState}
                level={session.level}
                isLive={isLive}
                demoActive={demoActive}
                hasCalibratedPatient={calibratedSpeakers.some(
                  (s) => s.kind === "patient",
                )}
              />
            ) : (
              <div className="max-w-3xl mx-auto px-6 md:px-10 py-12 space-y-10">
                <ActiveDisplay
                  row={latest}
                  editing={
                    editing?.utteranceId === latest.utteranceId ? editing : null
                  }
                  onConfirm={onConfirm}
                  onStartEdit={onStartEdit}
                  onCancelEdit={onCancelEdit}
                  onSaveEdit={onSaveEdit}
                  onChangeEdit={(t) =>
                    setEditing((prev) => (prev ? { ...prev, text: t } : null))
                  }
                  liveState={liveState}
                  level={session.level}
                  demoActive={demoActive}
                  replayingId={replay.playingId}
                  onToggleReplay={replay.toggle}
                  patientName={patientName}
                  onSetPatientName={setPatientName}
                />
                {history.length > 1 && (
                  <Thread
                    rows={history.slice(1)}
                    now={now}
                    filter={threadFilter}
                    onChangeFilter={setThreadFilter}
                    sessionEvents={sessionEvents}
                    patientName={patientName}
                    replayingId={replay.playingId}
                    onToggleReplay={replay.toggle}
                    expandedIds={expandedThreadIds}
                    onToggleExpand={toggleThreadExpand}
                    editing={editing}
                    onConfirm={onConfirm}
                    onStartEdit={onStartEdit}
                    onCancelEdit={onCancelEdit}
                    onSaveEdit={onSaveEdit}
                    onChangeEdit={(t) =>
                      setEditing((prev) =>
                        prev ? { ...prev, text: t } : null,
                      )
                    }
                  />
                )}
              </div>
            )}
          </section>

          <div className="relative shrink-0 px-6 py-3 border-t border-white/5 bg-black/20 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 min-w-0">
            {/* Level meter is decorative; lifted out of flex flow so the
                Start mic / Calibrate row is centered against the column,
                not against the (meter + buttons) group. */}
            <div className="hidden sm:flex absolute left-6 top-1/2 -translate-y-1/2 pointer-events-none">
              <MicLevelMeter level={session.level} active={isLive} />
            </div>
            <PrimaryControls
              micState={session.micState}
              level={session.level}
              errorMessage={session.errorMessage}
              onStart={session.start}
              onStop={session.stop}
            />
          </div>
        </div>

        {!focusMode && (showContext ? (
          <>
            <PanelResizer
              axis="x"
              onDrag={onResizeContext}
              ariaLabel="Resize context panel"
            />
            <aside
              className="shrink-0 min-h-0 flex flex-col border-l border-white/10 bg-black/20"
              style={{ width: contextWidth }}
            >
              <div className="shrink-0 px-4 py-2 border-b border-white/5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] tracking-[0.25em] font-mono opacity-50 uppercase">
                    context
                  </span>
                  <button
                    onClick={() => setShowContext(false)}
                    aria-label="Hide context panel"
                    title="hide"
                    className="w-5 h-5 flex items-center justify-center rounded text-xs opacity-50 hover:opacity-100 hover:bg-white/10 transition-colors"
                  >
                    ✕
                  </button>
                </div>
                <div className="mt-0.5 text-[10px] opacity-40 leading-snug">
                  session view — conversation, topics, emotional state
                </div>
              </div>
              <ContextPanel
                snapshot={ctx}
                emergencyLog={emergencyLog}
                conversationLog={conversationLog}
                patientName={patientName}
                now={now}
              />
            </aside>
          </>
        ) : (
          <PanelTab
            side="right"
            label="context"
            badge={unreadEmergencyCount}
            onClick={() => setShowContext(true)}
          />
        ))}
      </div>

    </main>
  );
}

// ============================================================ admin menu

// Overflow menu that holds admin / diagnostic routes. Keeps the main
// header focused on the active profile + connection status instead of
// mixing patient context with engineering tools. Modeled after
// ``KeyboardHelp`` (click-outside + Escape close).
function AdminMenu({
  focusMode,
  onToggleFocusMode,
}: {
  focusMode: boolean;
  onToggleFocusMode: () => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label="More tools"
        aria-haspopup="menu"
        aria-expanded={open}
        title="View options & diagnostics"
        className={`inline-flex items-center justify-center h-7 w-7 rounded-full text-base font-mono leading-none transition-colors border ${
          open
            ? "bg-white/10 border-white/20 text-white"
            : "bg-transparent border-transparent text-white/70 hover:text-white hover:bg-white/[0.06]"
        }`}
      >
        ⋯
      </button>
      {open && (
        <div
          role="menu"
          aria-label="View options and diagnostics"
          className="absolute top-9 right-0 z-40 w-56 rounded-lg shadow-2xl bg-neutral-950/95 backdrop-blur-md border border-white/10 overflow-hidden"
        >
          <div className="px-3 py-2 text-[10px] tracking-[0.25em] font-mono opacity-50 uppercase">
            view
          </div>
          <button
            type="button"
            role="menuitemcheckbox"
            aria-checked={focusMode}
            onClick={() => {
              onToggleFocusMode();
              setOpen(false);
            }}
            className="w-full text-left px-3 py-2 text-sm hover:bg-white/[0.06] text-white/80 hover:text-white flex items-center justify-between gap-2"
          >
            <span>
              Focus mode
              <span className="block text-[10px] opacity-50 leading-snug mt-0.5">
                Hide chrome, just the transcript
              </span>
            </span>
            <span
              className={`shrink-0 inline-flex items-center justify-center h-5 px-1.5 rounded-full text-[10px] font-mono uppercase tracking-wider border ${
                focusMode
                  ? "bg-emerald-400/15 border-emerald-400/40 text-emerald-200"
                  : "border-white/15 text-white/40"
              }`}
            >
              {focusMode ? "on" : "F"}
            </span>
          </button>
          <div className="px-3 py-2 text-[10px] tracking-[0.25em] font-mono opacity-50 uppercase border-t border-white/5">
            diagnostics
          </div>
          <a
            href="/audio"
            role="menuitem"
            className="block px-3 py-2 text-sm hover:bg-white/[0.06] text-white/80 hover:text-white"
          >
            Diagnose audio
            <div className="text-[10px] opacity-50 leading-snug mt-0.5">
              Check mic capture, VAD, levels
            </div>
          </a>
          <a
            href="/debug"
            role="menuitem"
            className="block px-3 py-2 text-sm hover:bg-white/[0.06] text-white/80 hover:text-white border-t border-white/5"
          >
            Debug view
            <div className="text-[10px] opacity-50 leading-snug mt-0.5">
              Raw pipeline + phoneme stream
            </div>
          </a>
        </div>
      )}
    </div>
  );
}

// ============================================================ empty state

function EmptyState({
  level,
  isLive,
  liveState,
  demoActive,
  hasCalibratedPatient,
}: {
  level: number;
  isLive: boolean;
  liveState: LiveState;
  demoActive: boolean;
  hasCalibratedPatient: boolean;
}) {
  const showCalibrateHint =
    !hasCalibratedPatient && !isLive && !demoActive;
  return (
    <div className="text-center max-w-2xl mx-auto w-full">
      <div className="space-y-3">
        <h1
          className="font-medium tracking-tight"
          style={{ fontSize: "calc(var(--phonemer-base-size) * 2.4)" }}
        >
          Phonemenal is listening
        </h1>
        <p className="opacity-50 text-base">
          {isLive || demoActive
            ? "The next thing said will appear here in large text."
            : "Press the microphone below to begin."}
        </p>
      </div>

      {(liveState !== "idle" || demoActive) && (
        <div className="mt-8 flex items-center justify-center gap-2">
          {demoActive && <DemoBadge />}
          <LiveStatus state={liveState} level={level} />
        </div>
      )}

      {showCalibrateHint && (
        <div className="mt-12 inline-flex flex-col items-center gap-2 px-5 py-4 rounded-lg border border-white/10 bg-neutral-900/30 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
          <div className="text-xs uppercase tracking-[0.25em] font-mono opacity-50">
            new patient?
          </div>
          <div className="text-sm opacity-80 max-w-sm">
            Calibrate first — recording 145 short prompts lets Phonemenal
            adapt to a patient's speech and dramatically improves
            transcription accuracy.
          </div>
          <a
            href="/calibrate"
            className="mt-1 inline-flex items-center gap-1 px-4 py-1.5 rounded-md text-sm font-medium"
            style={{
              background: "var(--phonemer-accent)",
              color: "#0a0a0a",
            }}
          >
            Start calibration →
          </a>
        </div>
      )}
    </div>
  );
}

// ============================================================ primary controls

function PrimaryControls({
  micState,
  level,
  errorMessage,
  onStart,
  onStop,
}: {
  micState: string;
  level: number;
  errorMessage?: string;
  onStart: () => void | Promise<void>;
  onStop: () => void | Promise<void>;
}) {
  const [perm, setPerm] = useState<MicPermissionState>("unknown");
  useEffect(() => {
    queryMicPermission().then(setPerm);
  }, []);

  const isLive = micState === "recording";
  const isBusy = micState === "connecting";
  const isBlocked = micState === "blocked";

  let label = "Start mic";
  if (isBusy) label = "Starting…";
  else if (isLive) label = "Stop mic";
  else if (isBlocked) label = "Mic blocked";
  else if (micState === "error") label = "Retry mic";

  // Custom intent classes for the mic button so its colors track micState
  // (live/busy/blocked) without losing the canonical lg height. We bypass
  // the standard Button intent and use the size + shape classes directly.
  const micClass = isLive
    ? "bg-rose-500/15 hover:bg-rose-500/25 border-rose-400/40 text-rose-100"
    : isBusy
      ? "bg-amber-500/10 border-amber-400/30 text-amber-200 cursor-wait"
      : isBlocked
        ? "bg-white/[0.02] border-rose-500/30 text-rose-300 cursor-not-allowed"
        : "bg-white/[0.04] hover:bg-white/[0.09] border-white/15 hover:border-white/30";

  return (
    <div className="flex flex-col items-center gap-2 min-w-0">
      <div className="flex flex-wrap items-center justify-center gap-3 min-w-0">
        <button
          disabled={isBusy || isBlocked}
          onClick={() => (isLive ? onStop() : onStart())}
          className={`inline-flex items-center justify-center gap-2 whitespace-nowrap shrink-0 font-medium transition-all duration-200 disabled:cursor-not-allowed h-11 px-6 text-sm rounded-full border ${micClass}`}
        >
          <MicIndicator state={micState} level={level} />
          {label}
        </button>
      </div>

      {errorMessage && (
        <div className="text-[11px] text-rose-300/90 max-w-xs leading-tight">
          {errorMessage}
        </div>
      )}
      {perm === "denied" && !errorMessage && (
        <div className="text-[11px] text-rose-300/70 max-w-xs leading-tight">
          Microphone access is blocked in browser settings.
        </div>
      )}
    </div>
  );
}

function MicIndicator({ state, level }: { state: string; level: number }) {
  if (state === "recording") {
    const v = Math.min(1, Math.pow(level * 6, 0.7));
    return (
      <span className="relative inline-flex w-2.5 h-2.5">
        <span
          className="absolute inset-0 rounded-full bg-rose-400"
          style={{
            transform: `scale(${1 + v * 0.6})`,
            opacity: 0.55 + v * 0.45,
            transition: "transform 80ms, opacity 80ms",
          }}
        />
        <span
          className="absolute inset-0 rounded-full bg-rose-300"
          style={{ animation: "mic-pulse 1.4s ease-in-out infinite" }}
        />
        <style jsx>{`
          @keyframes mic-pulse {
            0%, 100% { opacity: 0.4; transform: scale(1); }
            50% { opacity: 0.9; transform: scale(1.4); }
          }
        `}</style>
      </span>
    );
  }
  if (state === "connecting") {
    return (
      <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-300 animate-pulse" />
    );
  }
  return (
    <span className="inline-block w-2.5 h-2.5 rounded-full bg-white/30" />
  );
}

// ============================================================ active display

function ActiveDisplay({
  row,
  editing,
  onConfirm,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onChangeEdit,
  liveState,
  level,
  demoActive,
  replayingId,
  onToggleReplay,
  patientName,
  onSetPatientName,
}: {
  row: Translated;
  editing: { utteranceId: number; text: string } | null;
  onConfirm: (id: number) => void;
  onStartEdit: (row: Translated) => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  onChangeEdit: (t: string) => void;
  liveState: LiveState;
  level: number;
  demoActive: boolean;
  replayingId: number | null;
  onToggleReplay: (utteranceId: number) => void;
  patientName: string | null;
  onSetPatientName: (name: string | null) => void;
}) {
  // A corrected row should display the human-edited text as the source
  // of truth — the listener confirmed by typing it, so the big text
  // should reflect that. Per-word confidence shading is suppressed in
  // that case because perWordConfidence is keyed to the ASR primary
  // words; lining it up with arbitrary user-typed words would be wrong.
  const isCorrected =
    row.feedback === "corrected" && !!row.correctedText;
  const words = useMemo(
    () => row.primary.split(/\s+/).filter(Boolean),
    [row.primary],
  );

  return (
    <div className="max-w-5xl mx-auto w-full space-y-8">
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-2 flex-wrap">
          <ReplayButton
            playing={replayingId === row.utteranceId}
            onClick={() => onToggleReplay(row.utteranceId)}
          />
          <Waveform
            utteranceId={row.utteranceId}
            className="text-white/60 w-16 h-4"
          />
          <EditableSpeakerBadge
            speaker={row.speaker}
            patientName={patientName}
            onSetPatientName={onSetPatientName}
          />
          {demoActive && <DemoBadge />}
          {liveState !== "idle" && (
            <LiveStatus state={liveState} level={level} />
          )}
        </div>
        {row.intendedText && (
          <div className="text-sm opacity-80">
            <span className="text-[10px] tracking-[0.25em] font-mono uppercase opacity-60 mr-2">
              intended
            </span>
            <DiffedIntended intended={row.intendedText} heard={row.primary} />
          </div>
        )}
        {!editing ? (
          // When the phoneme-alignment gate flags this clip, frame the text
          // in a red ring + below-text hint. The text stays visible because
          // hiding it would defeat the medical-safety story (the listener
          // needs to see the candidate to judge it), but the visual treatment
          // makes it clear the system isn't standing behind the words.
          <div
            className={
              row.flagged
                ? "rounded-2xl px-6 py-5 ring-2 ring-rose-500/80 bg-rose-950/15"
                : ""
            }
          >
            {isCorrected ? (
              <div
                className="leading-snug font-medium text-neutral-100"
                style={{ fontSize: "calc(var(--phonemer-base-size) * 2.6)" }}
              >
                {row.correctedText}
              </div>
            ) : (
              <div
                className="leading-snug font-medium flex flex-wrap justify-center gap-x-3 gap-y-2"
                style={{ fontSize: "calc(var(--phonemer-base-size) * 2.6)" }}
              >
                {words.map((w, i) => (
                  <ShadedWord
                    key={i}
                    word={w}
                    conf={row.perWordConfidence[i] ?? row.overallConfidence}
                  />
                ))}
              </div>
            )}
            {row.flagged && (
              <div className="mt-3 text-xs text-rose-300/90 tracking-wide uppercase">
                uncertain — phoneme alignment doesn&apos;t support this
                {row.flagReason ? ` (${row.flagReason.replace("_", " ")})` : ""}
              </div>
            )}
          </div>
        ) : (
          <textarea
            autoFocus
            value={editing.text}
            onChange={(e) => onChangeEdit(e.target.value)}
            className="w-full max-w-3xl mx-auto rounded-lg p-4 text-2xl bg-black/40 border border-white/15 focus:outline-none focus:border-teal-400/60"
            rows={3}
          />
        )}
        {!editing && <MetaStrip row={row} />}
      </div>

      <>
        {row.feedback && !editing && (
          <div className="flex justify-center">
            <span
              className={`text-xs uppercase tracking-wider font-mono ${
                row.feedback === "confirmed"
                  ? "text-emerald-300"
                  : "text-amber-300"
              }`}
            >
              {row.feedback === "confirmed" ? "Confirmed" : "Corrected"}
            </span>
          </div>
        )}
        <RowActions
          row={row}
          text={
            row.feedback === "corrected" && row.correctedText
              ? row.correctedText
              : row.primary
          }
          isEditing={!!editing}
          onConfirm={onConfirm}
          onStartEdit={onStartEdit}
          onCancelEdit={onCancelEdit}
          onSaveEdit={onSaveEdit}
        />
      </>
    </div>
  );
}

function ShadedWord({ word, conf }: { word: string; conf: number }) {
  // Per-word confidence is exp(mean phoneme log-prob), measured against
  // the audio's wav2vec2 posterior. For dysarthric speech the baseline
  // is ~0.3-0.6 even on *correctly* transcribed words — articulation is
  // imprecise, so the phoneme alignment is proportionally less peaky.
  // The signal answers "did the audio support this word?" not "is this
  // word right?", and the two correlate only loosely on severely
  // dysarthric audio. Whisper's LM head sometimes lands on the correct
  // word despite very weak acoustic evidence (e.g. "immediately" at 12%
  // on an M01 reading), while hallucinations can have moderate acoustic
  // support (e.g. "looted" at 38% substituted for "losing"). At a 30%
  // underline threshold the alarm therefore flagged the *only correctly
  // transcribed word* while leaving the wrong words plain. Reserved
  // red+underline for catastrophic cases only (<10% ≈ avg per-frame
  // phoneme prob below 10%) where the model is clearly guessing; the
  // broad yellow band covers normal dysarthric uncertainty.
  const color = conf >= 0.75 ? "#e5e7eb" : conf >= 0.1 ? "#fbbf24" : "#fb7185";
  // Use longhand textDecorationLine instead of the shorthand textDecoration
  // so React doesn't warn about mixing shorthand + textDecorationColor.
  const decorationLine = conf < 0.1 ? "underline" : "none";
  return (
    <span
      title={`${(conf * 100).toFixed(0)}% confidence`}
      style={{
        color,
        textDecorationLine: decorationLine,
        textUnderlineOffset: 6,
        textDecorationColor: "rgba(251, 113, 133, 0.5)",
      }}
    >
      {word}
    </span>
  );
}

function MetaStrip({ row }: { row: Translated }) {
  const conf = row.overallConfidence;
  const confColor =
    conf >= 0.8 ? "#10b981" : conf >= 0.5 ? "#f59e0b" : "#ef4444";
  const hasAlts = row.alternatives.length > 0;
  const hasCtx = row.contextHints.length > 0;
  const Sep = () => (
    <span aria-hidden className="opacity-30">
      ·
    </span>
  );
  return (
    <div className="flex items-center justify-center gap-2 flex-wrap text-xs opacity-90">
      <span className="font-mono">
        <span style={{ color: confColor }}>
          {(conf * 100).toFixed(0)}%
        </span>
        <span className="opacity-60"> conf</span>
      </span>
      {hasAlts && (
        <>
          <Sep />
          <span className="max-w-md truncate">
            <span className="opacity-60 mr-1 font-mono">alt:</span>
            {row.alternatives.map((alt, i) => (
              <span key={i}>
                {i > 0 && (
                  <span aria-hidden className="opacity-30 mx-1">
                    ·
                  </span>
                )}
                <span className="opacity-90">&ldquo;{alt}&rdquo;</span>
              </span>
            ))}
          </span>
        </>
      )}
      {hasCtx && (
        <>
          <Sep />
          <span>
            <span className="opacity-60 mr-1 font-mono">topic:</span>
            <span className="opacity-90">
              {row.contextHints.join(" · ")}
            </span>
          </span>
        </>
      )}
    </div>
  );
}

function SpeakerBadge({
  speaker,
  patientName,
}: {
  speaker: string;
  patientName?: string | null;
}) {
  const isPatient = speaker === "patient" || speaker === "PATIENT";
  const label =
    isPatient && patientName
      ? patientName
      : isPatient
        ? "patient"
        : speaker;
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-[10px] tracking-[0.25em] font-mono"
      style={{
        background: isPatient
          ? "rgba(245, 158, 11, 0.12)"
          : "rgba(56, 189, 248, 0.12)",
        color: isPatient ? "#fde68a" : "#bae6fd",
      }}
    >
      {label.toUpperCase()}
    </span>
  );
}

function EditableSpeakerBadge({
  speaker,
  patientName,
  onSetPatientName,
}: {
  speaker: string;
  patientName: string | null;
  onSetPatientName: (name: string | null) => void;
}) {
  const isPatient = speaker === "patient" || speaker === "PATIENT";
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!editing) return;
    setDraft(patientName ?? "");
    const onDocClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setEditing(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setEditing(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [editing, patientName]);

  if (!isPatient) {
    // Non-patient speakers aren't renamed via this control.
    return <SpeakerBadge speaker={speaker} />;
  }

  const label = patientName ? patientName : "patient";
  const onSave = () => {
    const trimmed = draft.trim();
    onSetPatientName(trimmed.length > 0 ? trimmed : null);
    setEditing(false);
  };

  return (
    <div ref={ref} className="relative inline-block">
      <button
        type="button"
        onClick={() => setEditing((v) => !v)}
        title="Click to rename patient"
        aria-label="Rename patient"
        className="inline-block px-2 py-0.5 rounded text-[10px] tracking-[0.25em] font-mono hover:opacity-80 transition-opacity cursor-pointer"
        style={{
          background: "rgba(245, 158, 11, 0.12)",
          color: "#fde68a",
        }}
      >
        {label.toUpperCase()}
      </button>
      {editing && (
        <div className="absolute top-7 left-1/2 -translate-x-1/2 z-30 w-56 p-3 rounded-lg bg-neutral-950/95 backdrop-blur-md border border-white/10 shadow-2xl space-y-2">
          <label className="block text-[10px] tracking-[0.25em] font-mono opacity-50 uppercase">
            patient name
          </label>
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                onSave();
              }
            }}
            placeholder="patient"
            className="w-full px-2 py-1 text-sm rounded-md bg-black/40 border border-white/10 focus:outline-none focus:border-teal-400/40"
          />
          <div className="flex gap-2 justify-end">
            {patientName && (
              <button
                type="button"
                onClick={() => {
                  onSetPatientName(null);
                  setEditing(false);
                }}
                className="px-2 py-1 rounded-md text-xs border border-white/10 hover:bg-white/5"
              >
                clear
              </button>
            )}
            <button
              type="button"
              onClick={onSave}
              className="px-2 py-1 rounded-md text-xs bg-emerald-600/80 hover:bg-emerald-500 text-white"
            >
              save
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================ thread

type ThreadEditState = { utteranceId: number; text: string } | null;

function Thread({
  rows,
  now,
  filter,
  onChangeFilter,
  sessionEvents,
  patientName,
  replayingId,
  onToggleReplay,
  expandedIds,
  onToggleExpand,
  editing,
  onConfirm,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onChangeEdit,
}: {
  rows: Translated[];
  now: number;
  filter: ThreadFilter;
  onChangeFilter: (f: ThreadFilter) => void;
  sessionEvents: SessionEvent[];
  patientName: string | null;
  replayingId: number | null;
  onToggleReplay: (utteranceId: number) => void;
  expandedIds: Set<number>;
  onToggleExpand: (utteranceId: number) => void;
  editing: ThreadEditState;
  onConfirm: (id: number) => void;
  onStartEdit: (row: Translated) => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  onChangeEdit: (t: string) => void;
}) {
  const [search, setSearch] = useState("");
  const trimmed = search.trim().toLowerCase();
  const counts = {
    all: rows.length,
    flagged: rows.filter((r) => r.flagged).length,
    unconfirmed: rows.filter((r) => !r.feedback).length,
  };
  const visible = rows.filter((row) => {
    if (filter === "flagged" && !row.flagged) return false;
    if (filter === "unconfirmed" && row.feedback) return false;
    if (trimmed) {
      const text = (
        row.feedback === "corrected" && row.correctedText
          ? row.correctedText
          : row.primary
      ).toLowerCase();
      if (!text.includes(trimmed)) return false;
    }
    return true;
  });

  let emptyHint: string | null = null;
  if (visible.length === 0) {
    if (trimmed) emptyHint = `no rows match “${search.trim()}”`;
    else if (filter !== "all") emptyHint = `no rows match the ${filter} filter`;
    else emptyHint = "no earlier utterances yet";
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <span className="text-[10px] tracking-[0.25em] font-mono opacity-40 uppercase">
          earlier
        </span>
        <div className="relative flex-1 min-w-[100px] max-w-[260px]">
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="search…"
            aria-label="Search earlier utterances"
            className="w-full pl-2 pr-6 py-1 text-xs rounded-md bg-black/20 border border-white/10 placeholder:opacity-40 focus:outline-none focus:border-teal-400/40"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              aria-label="Clear search"
              className="absolute right-1 top-1/2 -translate-y-1/2 w-4 h-4 flex items-center justify-center rounded-full opacity-50 hover:opacity-100 hover:bg-white/10"
            >
              ✕
            </button>
          )}
        </div>
        <div className="flex items-center gap-1 ml-auto">
          {(["all", "flagged", "unconfirmed"] as ThreadFilter[]).map((f) => {
            const active = filter === f;
            const count = counts[f];
            return (
              <button
                key={f}
                type="button"
                onClick={() => onChangeFilter(f)}
                aria-pressed={active}
                className={`px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-wider transition-colors ${
                  active
                    ? "bg-white/10 text-white border border-white/20"
                    : "border border-white/10 opacity-60 hover:opacity-100 hover:bg-white/[0.04]"
                }`}
              >
                {f}
                <span className="ml-1 opacity-60 tabular-nums">{count}</span>
              </button>
            );
          })}
        </div>
      </div>
      {emptyHint ? (
        <div className="text-[11px] opacity-40 italic py-4 text-center">
          {emptyHint}
        </div>
      ) : (
        <ul className="divide-y divide-white/5 border-y border-white/5">
          {interleaveSessionEvents(visible, sessionEvents).map((item) =>
            item.kind === "row" ? (
              <ThreadRow
                key={item.row.utteranceId}
                row={item.row}
                now={now}
                patientName={patientName}
                playing={replayingId === item.row.utteranceId}
                onToggleReplay={() => onToggleReplay(item.row.utteranceId)}
                expanded={expandedIds.has(item.row.utteranceId)}
                onToggleExpand={() => onToggleExpand(item.row.utteranceId)}
                editing={editing}
                onConfirm={onConfirm}
                onStartEdit={onStartEdit}
                onCancelEdit={onCancelEdit}
                onSaveEdit={onSaveEdit}
                onChangeEdit={onChangeEdit}
              />
            ) : (
              <SessionDivider key={`ev-${item.event.ts}`} event={item.event} />
            ),
          )}
        </ul>
      )}
    </div>
  );
}

type ThreadItem =
  | { kind: "row"; row: Translated }
  | { kind: "event"; event: SessionEvent };

// Walk the (already newest-first) row list and drop in any session events
// whose ts falls between consecutive rows. Events newer than every row land
// above the first row; events older than every row land below the last.
function interleaveSessionEvents(
  rows: Translated[],
  events: SessionEvent[],
): ThreadItem[] {
  if (events.length === 0) return rows.map((row) => ({ kind: "row", row }));
  const ev = [...events].sort((a, b) => b.ts - a.ts);
  const out: ThreadItem[] = [];
  let i = 0;
  for (const row of rows) {
    while (i < ev.length && ev[i].ts > row.arrivedAt) {
      out.push({ kind: "event", event: ev[i] });
      i++;
    }
    out.push({ kind: "row", row });
  }
  while (i < ev.length) {
    out.push({ kind: "event", event: ev[i] });
    i++;
  }
  return out;
}

function SessionDivider({ event }: { event: SessionEvent }) {
  const verb = event.type === "start" ? "started recording" : "stopped recording";
  return (
    <li
      className="flex items-center gap-3 py-2 opacity-70"
      role="separator"
      aria-label={`${verb} at ${formatAbsoluteTime(event.ts)}`}
    >
      <span className="flex-1 h-px bg-white/10" />
      <span
        className="text-[10px] font-mono uppercase tracking-[0.25em] opacity-80"
        title={formatAbsoluteTime(event.ts)}
      >
        {verb} · {formatAbsoluteTime(event.ts)}
      </span>
      <span className="flex-1 h-px bg-white/10" />
    </li>
  );
}

function ThreadRow({
  row,
  now,
  patientName,
  playing,
  onToggleReplay,
  expanded,
  onToggleExpand,
  editing,
  onConfirm,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onChangeEdit,
}: {
  row: Translated;
  now: number;
  patientName: string | null;
  playing: boolean;
  onToggleReplay: () => void;
  expanded: boolean;
  onToggleExpand: () => void;
  editing: ThreadEditState;
  onConfirm: (id: number) => void;
  onStartEdit: (row: Translated) => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  onChangeEdit: (t: string) => void;
}) {
  const isEditing = editing?.utteranceId === row.utteranceId;
  const showActions = expanded || isEditing;
  const c = row.overallConfidence;
  const tone = row.flagged
    ? "#fb7185"
    : c >= 0.8
      ? "#34d399"
      : c >= 0.5
        ? "#fbbf24"
        : "#fb7185";
  const text =
    row.feedback === "corrected" && row.correctedText
      ? row.correctedText
      : row.primary;
  const stop = (e: React.SyntheticEvent) => e.stopPropagation();
  return (
    <li
      className={`flex items-start gap-3 py-3 transition-colors ${
        showActions ? "bg-white/[0.02]" : ""
      }`}
    >
      <span className="shrink-0 mt-0.5" onClick={stop}>
        <ReplayButton playing={playing} onClick={onToggleReplay} />
      </span>
      <span className="shrink-0 mt-1">
        <SpeakerBadge speaker={row.speaker} patientName={patientName} />
      </span>
      <div
        className={`flex-1 min-w-0 ${isEditing ? "" : "cursor-pointer"}`}
        onClick={() => {
          if (!isEditing) onToggleExpand();
        }}
      >
        {!isEditing ? (
          <div
            className={`text-base leading-snug ${
              row.flagged ? "text-rose-200/80" : "opacity-90"
            }`}
          >
            {text}
          </div>
        ) : (
          <textarea
            autoFocus
            value={editing!.text}
            onChange={(e) => onChangeEdit(e.target.value)}
            onClick={stop}
            className="w-full rounded-md p-2 text-base bg-black/40 border border-white/15 focus:outline-none focus:border-teal-400/60"
            rows={2}
          />
        )}
        <div className="mt-1 flex items-center gap-2 text-[10px] font-mono">
          <span style={{ color: tone }} className="tabular-nums">
            {(c * 100).toFixed(0)}%
          </span>
          {row.flagged && (
            <span className="uppercase tracking-wider text-rose-400/80">
              uncertain
            </span>
          )}
          {row.feedback && (
            <span
              className="uppercase tracking-wider"
              style={{
                color: row.feedback === "confirmed" ? "#34d399" : "#fbbf24",
              }}
            >
              {row.feedback}
            </span>
          )}
          <Waveform
            utteranceId={row.utteranceId}
            className="ml-auto text-white/50"
          />
          <span
            className="opacity-50 tabular-nums"
            title={formatAbsoluteTime(row.arrivedAt)}
          >
            {formatRelativeTime(row.arrivedAt, now)}
          </span>
        </div>
        {row.intendedText && (
          <div className="mt-1 text-xs opacity-70 truncate">
            <span className="font-mono uppercase tracking-wider opacity-60 mr-1">
              intended
            </span>
            <DiffedIntended intended={row.intendedText} heard={row.primary} />
          </div>
        )}
        {showActions && (
          <div onClick={stop} className="mt-2">
            <RowActions
              row={row}
              text={text}
              isEditing={isEditing}
              size="sm"
              onConfirm={onConfirm}
              onStartEdit={onStartEdit}
              onCancelEdit={onCancelEdit}
              onSaveEdit={onSaveEdit}
            />
          </div>
        )}
      </div>
    </li>
  );
}

function RowActions({
  row,
  text,
  isEditing,
  size = "md",
  onConfirm,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
}: {
  row: Translated;
  text: string;
  isEditing: boolean;
  size?: "sm" | "md";
  onConfirm: (id: number) => void;
  onStartEdit: (row: Translated) => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
}) {
  if (isEditing) {
    return (
      <div className="flex flex-wrap gap-2 justify-center">
        <Button onClick={onSaveEdit} size={size} intent="primary">
          Save correction
        </Button>
        <Button onClick={onCancelEdit} size={size} intent="secondary">
          Cancel
        </Button>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap gap-2 justify-center">
      <Button
        onClick={() => onConfirm(row.utteranceId)}
        size={size}
        intent="primary"
      >
        Looks right
      </Button>
      <Button
        onClick={() => onStartEdit(row)}
        size={size}
        intent="warning"
      >
        Edit
      </Button>
      <CopyButton text={text} size={size} />
      {row.intendedText && (
        <WhyButton row={row} size={size} />
      )}
    </div>
  );
}

function WhyButton({
  row,
  size,
}: {
  row: Translated;
  size: "sm" | "md";
}) {
  const [copied, setCopied] = useState(false);
  const onCopy = () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    const features = featuresByUtteranceLookup.get(row.utteranceId);
    const report = buildDiagnosticReport(row, features);
    navigator.clipboard
      .writeText(report)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  };
  return (
    <Button
      onClick={onCopy}
      title={
        copied
          ? "Diagnostic report copied!"
          : "Copy a structured diagnostic (intended vs heard, confidence, alignment, prosody) to paste into a chat session"
      }
      size={size}
      intent={copied ? "primary" : "secondary"}
    >
      {copied ? "Copied" : "Why?"}
    </Button>
  );
}

// Module-scoped accessor for the per-utterance features map. Set by
// HomePage on mount so the leaf WhyButton component (which doesn't get
// passed the ref) can still look up prosody for its row.
const featuresByUtteranceLookup = new Map<number, AcousticFeatures>();

// One-word annotation for a "seconds per word" pace value. Used by the
// audio-capture diagnostic so the reader doesn't have to remember the
// typical-dysarthric-rate band by heart.
function paceTypicalNote(secPerWord: number): string {
  if (secPerWord < 0.3) return "  (impossibly fast)";
  if (secPerWord < 0.5) return "  (fast)";
  if (secPerWord <= 1.0) return "  (typical dysarthric)";
  if (secPerWord <= 1.5) return "  (slow)";
  return "  (very slow)";
}

function buildDiagnosticReport(
  row: Translated,
  features: AcousticFeatures | undefined,
): string {
  const intended = (row.intendedText ?? "").trim();
  const heard = row.primary.trim();
  const refWords = intended.toLowerCase().split(/\s+/).filter(Boolean);
  const hypWords = heard.toLowerCase().split(/\s+/).filter(Boolean);
  const wer =
    refWords.length === 0
      ? hypWords.length === 0
        ? 0
        : 1
      : wordEditDistance(refWords, hypWords) / refWords.length;
  const wordOps = wordDiffOps(refWords, hypWords);

  const lines: string[] = [];
  lines.push("# Phonemenal transcription diagnostic");
  lines.push("");
  lines.push(`Speaker: ${row.speaker}`);
  lines.push(`Intended: "${intended}"`);
  lines.push(`Heard:    "${heard}"`);
  lines.push(`WER: ${(wer * 100).toFixed(1)}%`);

  // Audio capture sanity. On dysarthric speech the DEFAULT assumption
  // is MIS-ARTICULATED — patients with dysarthria say what they're
  // trying to say, and ASR error is the dominant failure mode this
  // app exists to mitigate. STOPPED is reserved for cases where the
  // physical audio can't fit the intended word count even at very
  // fast articulation; TRUNCATED for cases where VAD obviously cut.
  //
  //   TRUNCATED — pace_required (audio_duration / intended_words) is
  //     faster than any natural speech. Indicates VAD cut, not ASR.
  //
  //   STOPPED — speaking_time (speech after subtracting pauses) is
  //     too short for the intended word count even at very fast
  //     articulation (< 0.2 s/word ≈ 300 WPM). Rare in practice.
  //
  //   MIS-ARTICULATED — default for WER > 30%. Patient articulated
  //     the full sentence; Whisper transcribed it incorrectly or
  //     partially. Whisper's token count (`spoken_words`) is NOT
  //     trusted as a measure of what the patient said — for severely
  //     dysarthric speech, Whisper often emits fewer garbled tokens
  //     than the actual word count would suggest, and that's a
  //     transcription artifact, not patient behaviour.
  if (refWords.length > 0 && features?.duration_ms) {
    const durationMs = features.duration_ms;
    const pauseTotalMs = features.pause_total_ms ?? 0;
    const speakingMs = Math.max(0, durationMs - pauseTotalMs);
    const spokenWords = hypWords.length;
    const paceRequired = durationMs / 1000 / refWords.length;
    const paceSpoken =
      spokenWords > 0 ? speakingMs / 1000 / spokenWords : null;
    const speakingPerIntended = speakingMs / 1000 / refWords.length;

    lines.push("");
    lines.push("## Audio capture");
    lines.push(`  audio_duration:  ${durationMs} ms`);
    lines.push(
      `  speaking_time:   ${speakingMs} ms  (after ${pauseTotalMs} ms of internal pauses)`,
    );
    lines.push(`  intended_words:  ${refWords.length}`);
    lines.push(`  spoken_words:    ${spokenWords}  (Whisper token count — not trusted as patient-word count)`);
    if (paceSpoken !== null)
      lines.push(
        `  pace_spoken:     ${paceSpoken.toFixed(2)} s/word${paceTypicalNote(paceSpoken)}`,
      );
    lines.push(`  pace_required:   ${paceRequired.toFixed(2)} s/word`);

    // Failure-mode classification. TRUNCATED and STOPPED only fire on
    // explicit physical evidence; everything else high-WER is
    // MIS-ARTICULATED (the case this app actually targets).
    if (paceRequired < 0.3) {
      lines.push(
        `  >> TRUNCATED — audio_duration can't physically contain the intended sentence at any natural rate (typical dysarthric: 0.7-1.0 s/word, fast: 0.3 s/word). VAD likely cut mid-sentence; this is a capture problem, not an ASR error.`,
      );
    } else if (speakingPerIntended < 0.2) {
      lines.push(
        `  >> STOPPED — speaking_time (${speakingMs} ms) is too short to contain ${refWords.length} words even at very fast articulation (< 0.2 s/word ≈ 300 WPM). Patient likely physically did not say the full sentence.`,
      );
    } else if (wer > 0.3) {
      lines.push(
        `  >> MIS-ARTICULATED — patient had enough speaking time for all ${refWords.length} words at ${speakingPerIntended.toFixed(2)} s/word. The high WER reflects Whisper transcribing the dysarthric articulation incorrectly, not the patient saying less than intended. This is the failure mode the per-patient LoRA + safety gate are designed to handle.`,
      );
    }
  }

  lines.push("");
  lines.push("## Word-level diff (ref vs hyp)");
  for (const op of wordOps) {
    if (op.kind === "match") lines.push(`  =  ${op.ref}`);
    else if (op.kind === "sub") lines.push(`  ~  ${op.ref}  →  ${op.hyp}`);
    else if (op.kind === "del") lines.push(`  -  ${op.ref}    (missed)`);
    else if (op.kind === "ins") lines.push(`  +  ${op.hyp}    (added)`);
  }
  lines.push("");
  lines.push("## Per-word confidence (heard)");
  hypWords.forEach((w, i) => {
    const c = row.perWordConfidence[i] ?? row.overallConfidence;
    lines.push(`  ${(c * 100).toFixed(0).padStart(3)}%  ${w}`);
  });
  lines.push(
    `  overall confidence: ${(row.overallConfidence * 100).toFixed(0)}%`,
  );
  lines.push("");
  lines.push("## Phoneme-alignment safety gate");
  lines.push(`  flagged: ${row.flagged ? "yes" : "no"}`);
  if (row.flagReason) lines.push(`  reason: ${row.flagReason}`);
  if (row.alignScore !== undefined)
    lines.push(`  alignment_score (mean log-prob): ${row.alignScore.toFixed(3)}`);
  if (row.phonemeDensity !== undefined)
    lines.push(
      `  phoneme_density: ${row.phonemeDensity.toFixed(2)} phonemes/sec`,
    );

  if (features) {
    lines.push("");
    lines.push("## Acoustic features (prosody)");
    lines.push(`  jitter:        ${features.jitter.toFixed(3)}`);
    lines.push(`  shimmer:       ${features.shimmer.toFixed(3)}`);
    lines.push(`  pause_count:   ${features.pause_count}`);
    lines.push(`  pause_total:   ${features.pause_total_ms} ms`);
    lines.push(`  duration:      ${features.duration_ms} ms`);
    if (typeof features.pitch_hz_mean === "number")
      lines.push(`  pitch_mean:    ${features.pitch_hz_mean.toFixed(1)} Hz`);
    if (typeof features.pitch_hz_std === "number")
      lines.push(`  pitch_std:     ${features.pitch_hz_std.toFixed(1)} Hz`);
  }

  if (row.alternatives.length > 0) {
    lines.push("");
    lines.push("## Alternatives the model considered");
    row.alternatives.forEach((alt) => lines.push(`  - "${alt}"`));
  }

  if (row.contextHints.length > 0) {
    lines.push("");
    lines.push(
      `## Active topics at the time: ${row.contextHints.join(", ")}`,
    );
  }

  return lines.join("\n");
}

function wordEditDistance(a: string[], b: string[]): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(0),
  );
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) dp[i][j] = dp[i - 1][j - 1];
      else
        dp[i][j] =
          1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

type WordOp =
  | { kind: "match"; ref: string }
  | { kind: "sub"; ref: string; hyp: string }
  | { kind: "del"; ref: string }
  | { kind: "ins"; hyp: string };

function wordDiffOps(ref: string[], hyp: string[]): WordOp[] {
  const m = ref.length;
  const n = hyp.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(0),
  );
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (ref[i - 1] === hyp[j - 1]) dp[i][j] = dp[i - 1][j - 1];
      else
        dp[i][j] =
          1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  // Backtrack to produce ops in order.
  const ops: WordOp[] = [];
  let i = m;
  let j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && ref[i - 1] === hyp[j - 1]) {
      ops.unshift({ kind: "match", ref: ref[i - 1] });
      i--;
      j--;
    } else if (
      i > 0 &&
      j > 0 &&
      dp[i][j] === dp[i - 1][j - 1] + 1
    ) {
      ops.unshift({ kind: "sub", ref: ref[i - 1], hyp: hyp[j - 1] });
      i--;
      j--;
    } else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
      ops.unshift({ kind: "del", ref: ref[i - 1] });
      i--;
    } else {
      ops.unshift({ kind: "ins", hyp: hyp[j - 1] });
      j--;
    }
  }
  return ops;
}

function CopyButton({
  text,
  size = "md",
}: {
  text: string;
  size?: "sm" | "md";
}) {
  const [copied, setCopied] = useState(false);
  const onCopy = () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  };
  return (
    <Button
      onClick={onCopy}
      aria-label={copied ? "Copied to clipboard" : "Copy text"}
      title={copied ? "Copied!" : "Copy text"}
      size={size}
      intent={copied ? "primary" : "secondary"}
      icon
      shape="pill"
    >
      {copied ? <CheckIcon size={size} /> : <CopyIcon size={size} />}
    </Button>
  );
}

function CopyIcon({ size }: { size: "sm" | "md" }) {
  const px = size === "sm" ? 12 : 14;
  return (
    <svg
      width={px}
      height={px}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon({ size }: { size: "sm" | "md" }) {
  const px = size === "sm" ? 12 : 14;
  return (
    <svg
      width={px}
      height={px}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function TranscribingPlaceholder({
  level,
  demoActive,
}: {
  level: number;
  demoActive: boolean;
}) {
  return (
    <div className="max-w-5xl mx-auto w-full space-y-8">
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {demoActive && <DemoBadge />}
          <LiveStatus state="transcribing" level={level} />
        </div>
        <div
          aria-busy="true"
          aria-label="Transcribing the last utterance"
          className="rounded-2xl px-8 py-12 max-w-3xl mx-auto border-2 border-dashed border-amber-400/30 bg-amber-950/10"
        >
          <div className="opacity-60 text-xs font-mono uppercase tracking-[0.25em]">
            transcribing your last utterance
          </div>
        </div>
      </div>
    </div>
  );
}

function DiffedIntended({
  intended,
  heard,
}: {
  intended: string;
  heard: string;
}) {
  const segments = diffIntendedWords(intended, heard);
  return (
    <span className="italic">
      &ldquo;
      {segments.map((s, i) =>
        s.matched || !s.isWord ? (
          <span key={i}>{s.token}</span>
        ) : (
          <span
            key={i}
            className="text-rose-300 line-through decoration-rose-400/60"
            title="missed by the model"
          >
            {s.token}
          </span>
        ),
      )}
      &rdquo;
    </span>
  );
}

function ReplayButton({
  playing,
  onClick,
}: {
  playing: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={playing ? "Stop replay" : "Replay audio"}
      title={playing ? "stop" : "replay audio"}
      className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-[10px] transition-colors ${
        playing
          ? "bg-rose-500/20 text-rose-200 border border-rose-400/40"
          : "bg-white/[0.04] hover:bg-white/[0.1] border border-white/10 opacity-70 hover:opacity-100"
      }`}
    >
      {playing ? "■" : "▶"}
    </button>
  );
}

// ============================================================ context panel

function ContextPanel({
  snapshot,
  emergencyLog,
  conversationLog,
  patientName,
  now,
}: {
  snapshot: ContextSnapshot | null;
  emergencyLog: EmergencyAlert[];
  conversationLog: ConversationEntry[];
  patientName: string | null;
  now: number;
}) {
  const sections: StackedPanelDef[] = [];

  if (emergencyLog.length > 0) {
    sections.push({
      key: "emergencies",
      label: "emergencies",
      count: emergencyLog.length,
      render: () => (
        <ul className="space-y-2">
          {emergencyLog.map((alert, i) => (
            <li
              key={`${alert.utteranceId}-${i}`}
              className="rounded-md bg-rose-950/40 border border-rose-500/30 px-2 py-1.5"
            >
              <div className="flex items-center gap-2 text-[10px] font-mono opacity-80">
                <span className="text-rose-300 uppercase tracking-wider">
                  {alert.category ?? "alert"}
                </span>
                <span className="ml-auto tabular-nums">
                  {(alert.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="text-xs text-rose-100 mt-0.5">
                {alert.phrase ?? "Distress detected"}
              </div>
              <div
                className="text-[10px] opacity-50 font-mono mt-0.5"
                title={formatAbsoluteTime(alert.ts)}
              >
                {formatRelativeTime(alert.ts, now)}
              </div>
            </li>
          ))}
        </ul>
      ),
    });
  }

  sections.push(
    {
      key: "conversation",
      label: "conversation",
      count: conversationLog.length,
      render: () =>
        conversationLog.length === 0 ? (
          <div className="opacity-50">no utterances yet</div>
        ) : (
          <ul className="space-y-1.5">
            {conversationLog.slice(0, 24).map((u) => {
              const label = u.isPatient
                ? patientName || "patient"
                : u.speaker;
              return (
                <li key={u.utteranceId} className="flex gap-2">
                  <span
                    className="shrink-0 w-16 truncate font-mono opacity-60"
                    style={{ color: u.isPatient ? "#fde68a" : "#bae6fd" }}
                    title={
                      u.isPatient ? "patient" : `speaker: ${u.speaker}`
                    }
                  >
                    {label}
                  </span>
                  <span className="opacity-90">
                    {u.text ?? <span className="opacity-50">…</span>}
                  </span>
                </li>
              );
            })}
          </ul>
        ),
    },
    {
      key: "topics",
      label: "active topics",
      count: snapshot?.topics.length ?? 0,
      render: () =>
        !snapshot || snapshot.topics.length === 0 ? (
          <div className="opacity-50">no topics tracked yet</div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {snapshot.topics.slice(0, 10).map((t) => (
              <span
                key={t.keyword}
                className="px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-200"
                style={{ opacity: 0.4 + Math.min(0.6, t.weight / 2) }}
                title={`weight ${t.weight}`}
              >
                {t.keyword}
              </span>
            ))}
          </div>
        ),
    },
  );

  if (snapshot && snapshot.pending_questions.length > 0) {
    // Storage key kept as "unanswered" so user's collapse state survives
    // the rename. The label and subtitle now make clear that this is a
    // listener-side reminder, not something the transcription pipeline
    // acts on (the most recent listener turn is already in Whisper's
    // initial_prompt, and the LLM correction layer was removed in the
    // architecture pivot).
    sections.push({
      key: "unanswered",
      label: "open questions",
      count: snapshot.pending_questions.length,
      render: () => (
        <div>
          <p className="text-[10px] opacity-40 italic leading-snug mb-2">
            For your reference — does not affect transcription.
          </p>
          <ul className="space-y-1">
            {snapshot.pending_questions.map((q, i) => (
              <li key={i} className="bg-white/5 rounded-md px-2 py-1">
                <span className="opacity-60 mr-1">[{q.type}]</span>
                {q.text}
              </li>
            ))}
          </ul>
        </div>
      ),
    });
  }

  if (snapshot) {
    sections.push({
      key: "emotion",
      label: "emotional state",
      render: () => (
        <div className="space-y-1">
          <Bar
            label="frustration"
            value={snapshot.emotion.frustration}
            color="#f59e0b"
          />
          <Bar
            label="distress"
            value={snapshot.emotion.distress}
            color="#ef4444"
          />
        </div>
      ),
    });
  }

  return (
    <StackedPanels sections={sections} storagePrefix="phonemer.section" />
  );
}

type StackedPanelDef = {
  key: string;
  label: string;
  count?: number | null;
  defaultOpen?: boolean;
  render: () => React.ReactNode;
};

const STACKED_MIN_WEIGHT = 60;
const STACKED_DEFAULT_WEIGHT = 100;

function StackedPanels({
  sections,
  storagePrefix,
}: {
  sections: StackedPanelDef[];
  storagePrefix: string;
}) {
  const [openMap, setOpenMap] = useState<Record<string, boolean>>({});
  // Flex weights in pixel-equivalents — at any instant the container
  // distributes its flex pool proportionally to these. Adjusting two
  // adjacent weights by ±delta therefore moves the boundary by ~delta px.
  const [weights, setWeights] = useState<Record<string, number>>({});

  // Hydrate any newly seen section keys from localStorage. We never overwrite
  // an existing in-memory value, so user adjustments survive when conditional
  // sections (e.g. "unanswered") appear/disappear.
  const sectionKeys = sections.map((s) => s.key).join(",");
  useEffect(() => {
    setOpenMap((prev) => {
      const next: Record<string, boolean> = { ...prev };
      for (const s of sections) {
        if (next[s.key] === undefined) {
          next[s.key] = loadStoredBool(
            `${storagePrefix}.${s.key}.open`,
            s.defaultOpen ?? true,
          );
        }
      }
      return next;
    });
    setWeights((prev) => {
      const next: Record<string, number> = { ...prev };
      for (const s of sections) {
        if (next[s.key] === undefined) {
          next[s.key] = loadStoredNumber(
            `${storagePrefix}.${s.key}.weight`,
            STACKED_DEFAULT_WEIGHT,
          );
        }
      }
      return next;
    });
    // sectionKeys captures the meaningful change; sections is recreated each
    // render so we exclude it from deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectionKeys, storagePrefix]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    for (const [key, val] of Object.entries(openMap)) {
      window.localStorage.setItem(`${storagePrefix}.${key}.open`, String(val));
    }
  }, [openMap, storagePrefix]);
  useEffect(() => {
    if (typeof window === "undefined") return;
    for (const [key, val] of Object.entries(weights)) {
      window.localStorage.setItem(
        `${storagePrefix}.${key}.weight`,
        String(val),
      );
    }
  }, [weights, storagePrefix]);

  const toggle = useCallback((key: string) => {
    setOpenMap((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const makeResizer = useCallback(
    (aboveKey: string, belowKey: string) => (delta: number) => {
      setWeights((prev) => {
        const a = prev[aboveKey] ?? STACKED_DEFAULT_WEIGHT;
        const b = prev[belowKey] ?? STACKED_DEFAULT_WEIGHT;
        const newA = a + delta;
        const newB = b - delta;
        if (newA < STACKED_MIN_WEIGHT || newB < STACKED_MIN_WEIGHT) return prev;
        return { ...prev, [aboveKey]: newA, [belowKey]: newB };
      });
    },
    [],
  );

  const isOpen = (s: StackedPanelDef) =>
    openMap[s.key] ?? (s.defaultOpen ?? true);

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {sections.map((s, i) => {
        const open = isOpen(s);
        const weight = weights[s.key] ?? STACKED_DEFAULT_WEIGHT;

        // Resizer above this section iff this section is open AND there's an
        // earlier open section in the array (closed sections in between are
        // skipped — their fixed-height headers don't participate in the flex
        // pool, so dragging here adjusts the two open sections directly).
        let prevOpenKey: string | null = null;
        if (open) {
          for (let j = i - 1; j >= 0; j--) {
            if (isOpen(sections[j])) {
              prevOpenKey = sections[j].key;
              break;
            }
          }
        }

        const sectionBorder = prevOpenKey
          ? ""
          : "border-t border-white/10 first:border-t-0";

        return (
          <Fragment key={s.key}>
            {prevOpenKey && (
              <PanelResizer
                axis="y"
                onDrag={makeResizer(prevOpenKey, s.key)}
                ariaLabel={`Resize ${s.label} section`}
              />
            )}
            <div
              className={`flex flex-col ${sectionBorder} ${
                open ? "min-h-0" : "shrink-0"
              }`}
              style={
                open ? { flex: `${weight} 1 0`, minHeight: 0 } : undefined
              }
            >
              <button
                type="button"
                onClick={() => toggle(s.key)}
                aria-expanded={open}
                className="shrink-0 w-full h-7 flex items-center gap-2 px-3 bg-white/[0.02] hover:bg-white/[0.06] transition-colors text-left"
              >
                <SectionChevron open={open} />
                <span className="text-[10px] tracking-[0.25em] font-mono opacity-70 uppercase">
                  {s.label}
                </span>
                {s.count != null && (
                  <span className="ml-auto text-[10px] font-mono opacity-40 tabular-nums">
                    {s.count}
                  </span>
                )}
              </button>
              {open && (
                <div className="flex-1 min-h-0 overflow-y-auto px-3 pt-2 pb-3 text-xs">
                  {s.render()}
                </div>
              )}
            </div>
          </Fragment>
        );
      })}
    </div>
  );
}

function SectionChevron({ open }: { open: boolean }) {
  return (
    <svg
      width="8"
      height="8"
      viewBox="0 0 8 8"
      fill="none"
      style={{
        transform: open ? "rotate(90deg)" : "rotate(0)",
        transition: "transform 180ms",
        opacity: 0.6,
        flexShrink: 0,
      }}
    >
      <path
        d="M2 1 L6 4 L2 7"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Bar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="opacity-60 w-20 text-[11px]">{label}</span>
      <div className="flex-1 h-1 bg-white/10 rounded">
        <div
          className="h-1 rounded"
          style={{
            width: `${Math.max(0, Math.min(1, value)) * 100}%`,
            background: color,
          }}
        />
      </div>
      <span className="opacity-60 w-10 text-right tabular-nums text-[11px]">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

// ============================================================ misc UI

function PanelHeader({
  label,
  subtitle,
  onHide,
}: {
  label: string;
  subtitle?: string;
  onHide: () => void;
}) {
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] tracking-[0.25em] font-mono opacity-50 uppercase">
          {label}
        </span>
        <button
          onClick={onHide}
          aria-label={`Hide ${label} panel`}
          title="hide"
          className="w-6 h-6 flex items-center justify-center rounded text-xs opacity-50 hover:opacity-100 hover:bg-white/10 transition-colors"
        >
          ✕
        </button>
      </div>
      {subtitle && (
        <div className="mt-1 text-[10px] opacity-40 leading-snug">
          {subtitle}
        </div>
      )}
    </div>
  );
}

function ConnStateDot({ state }: { state: string }) {
  const color =
    state === "open"
      ? "#10b981"
      : state === "connecting"
        ? "#f59e0b"
        : state === "error"
          ? "#ef4444"
          : "#6b7280";
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full"
      style={{ background: color, boxShadow: `0 0 6px ${color}77` }}
    />
  );
}

function HeaderConnPill({
  state,
  onReconnect,
}: {
  state: string;
  onReconnect?: () => void;
}) {
  const label =
    state === "open"
      ? "live"
      : state === "connecting"
        ? "connecting"
        : state === "closed"
          ? "offline"
          : state;
  const canRetry =
    !!onReconnect && (state === "closed" || state === "error");

  const base =
    "inline-flex items-center justify-center gap-1.5 h-6 px-2 rounded-full text-[10px] font-mono uppercase tracking-wider whitespace-nowrap shrink-0";

  if (canRetry) {
    return (
      <button
        type="button"
        onClick={onReconnect}
        title={`websocket ${state} — click to reconnect`}
        className={`${base} border border-white/10 hover:bg-white/[0.06] text-white/85 transition-colors`}
      >
        <ConnStateDot state={state} />
        <span>{label}</span>
        <span className="opacity-50">· retry</span>
      </button>
    );
  }

  return (
    <span
      title={`websocket ${state}`}
      className={`${base} text-white/60`}
    >
      <ConnStateDot state={state} />
      <span>{label}</span>
    </span>
  );
}

function EmergencyBanner({
  alert,
  onDismiss,
}: {
  alert: EmergencyAlert;
  onDismiss: () => void;
}) {
  return (
    <div
      className="px-6 py-3 flex items-center gap-4"
      style={{
        background: "linear-gradient(90deg, #b91c1c 0%, #991b1b 100%)",
        color: "#fff5f5",
        animation: "emergency-pulse 1.6s ease-in-out infinite",
      }}
    >
      <span className="text-2xl">⚠</span>
      <div className="flex-1 min-w-0">
        <div className="text-xs font-mono uppercase tracking-[0.25em] opacity-80">
          Emergency · {alert.category ?? "alert"}
        </div>
        <div className="text-base font-semibold truncate">
          {alert.phrase ?? "Distress detected — check on the patient"}
        </div>
      </div>
      <span className="text-xs opacity-80 font-mono tabular-nums">
        {(alert.confidence * 100).toFixed(0)}%
      </span>
      <button
        onClick={onDismiss}
        className="px-3 py-1 rounded-md bg-white/15 hover:bg-white/25 text-sm"
      >
        Dismiss
      </button>
      <style jsx>{`
        @keyframes emergency-pulse {
          0%, 100% { box-shadow: inset 0 0 0 rgba(255, 0, 0, 0); }
          50% { box-shadow: inset 0 0 80px rgba(255, 60, 60, 0.45); }
        }
      `}</style>
    </div>
  );
}
