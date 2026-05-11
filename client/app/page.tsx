"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AccessibilityPanel } from "../components/accessibility-panel";
import { ActionMenu } from "../components/action-menu";
import { PipelineViz } from "../components/pipeline-viz";
import { TorgoDrawer } from "../components/torgo-drawer";
import { useAccessibility } from "../lib/accessibility";
import {
  MicPermissionState,
  queryMicPermission,
} from "../lib/audio-devices";
import { playEmergencyChime } from "../lib/chime";
import type {
  ContextSnapshot,
  EmergencyMessage,
  PhonemeFrame,
  PhonemeWindowMessage,
  TranslationMessage,
  UtteranceMessage,
  WsMessage,
} from "../lib/messages";
import { usePhonemerSession } from "../lib/use-phonemer-session";
import { buildVizFromLive } from "../lib/viz-from-ws";
import type { VizUtterance } from "../lib/viz-types";

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
  feedback?: "confirmed" | "corrected";
  correctedText?: string;
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

const MAX_HISTORY = 8;

export default function HomePage() {
  const a11y = useAccessibility();

  const [history, setHistory] = useState<Translated[]>([]);
  const [emergency, setEmergency] = useState<EmergencyAlert | null>(null);
  const [ctx, setCtx] = useState<ContextSnapshot | null>(null);
  const [editing, setEditing] = useState<{
    utteranceId: number;
    text: string;
  } | null>(null);
  const [translationError, setTranslationError] = useState<string | null>(null);

  const [showPipeline, setShowPipeline] = useState(false);
  const [showTorgo, setShowTorgo] = useState(false);
  const [showContext, setShowContext] = useState(false);

  // Open the TORGO panel automatically when arriving via /demo redirect.
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      if (params.get("demo") === "1") setShowTorgo(true);
    }
  }, []);

  const [pendingUtt, setPendingUtt] = useState<UtteranceMessage | null>(null);
  const pendingUttRef = useRef<UtteranceMessage | null>(null);
  const framesByUtteranceRef = useRef<Map<number, PhonemeFrame[]>>(new Map());
  const utteranceMessagesRef = useRef<Map<number, UtteranceMessage>>(new Map());
  const [vizUtterance, setVizUtterance] = useState<VizUtterance | null>(null);
  const [vizRunKey, setVizRunKey] = useState(0);

  useEffect(() => {
    pendingUttRef.current = pendingUtt;
  }, [pendingUtt]);

  // ============================================================ ws ingest

  const handleMessage = useCallback((msg: WsMessage) => {
    switch (msg.type) {
      case "utterance":
        setPendingUtt(msg);
        utteranceMessagesRef.current.set(msg.utterance_id, msg);
        framesByUtteranceRef.current.set(msg.utterance_id, []);
        break;
      case "phoneme_window":
        accumulateFrames(msg);
        break;
      case "translation":
        applyTranslation(msg);
        setTranslationError(null);
        break;
      case "translation_error":
        if (
          msg.message.toLowerCase().includes("credit") ||
          msg.message.toLowerCase().includes("billing")
        ) {
          setTranslationError(
            "Anthropic API has no credits. Add credits at console.anthropic.com → Plans & Billing.",
          );
        } else if (msg.error === "AuthenticationError") {
          setTranslationError(
            "ANTHROPIC_API_KEY is invalid. Check the key in .env and restart the server.",
          );
        } else {
          setTranslationError(`Translation error: ${msg.message}`);
        }
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

  const accumulateFrames = (msg: PhonemeWindowMessage) => {
    const list = framesByUtteranceRef.current.get(msg.utterance_id) ?? [];
    list.push(...msg.frames);
    framesByUtteranceRef.current.set(msg.utterance_id, list);
  };

  const applyTranslation = (msg: TranslationMessage) => {
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
        arrivedAt: performance.now(),
      };
      const existing = prev.findIndex((r) => r.utteranceId === row.utteranceId);
      let next: Translated[];
      if (existing >= 0) {
        next = [...prev];
        next[existing] = { ...next[existing], ...row };
      } else {
        next = [row, ...prev];
      }

      const frames = framesByUtteranceRef.current.get(msg.utterance_id) ?? [];
      if (matchingUtt) {
        try {
          const viz = buildVizFromLive({
            utterance: matchingUtt,
            frames,
            translation: msg,
          });
          setVizUtterance(viz);
          setVizRunKey((k) => k + 1);
        } catch (err) {
          console.warn("buildVizFromLive failed", err);
        }
      }
      return next.slice(0, MAX_HISTORY);
    });
  };

  const applyEmergency = (msg: EmergencyMessage) => {
    setEmergency({
      utteranceId: msg.utterance_id,
      phrase: msg.phrase ?? null,
      category: msg.category,
      confidence: msg.confidence,
      ts: performance.now(),
      dismissed: false,
    });
    playEmergencyChime();
  };

  const session = usePhonemerSession({ url: WS_URL, onMessage: handleMessage });

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
    setEditing({ utteranceId: row.utteranceId, text: row.primary });
  const onCancelEdit = () => setEditing(null);
  const onSaveEdit = () => {
    if (!editing) return;
    sendFeedback(editing.utteranceId, false, editing.text.trim());
    setEditing(null);
  };

  // ============================================================ render

  const latest = history[0];
  const isLive = session.micState === "recording";

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
            LLM
          </span>
          <span className="flex-1">{translationError}</span>
          <button
            onClick={() => setTranslationError(null)}
            className="px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      <ActionMenu
        showPipeline={showPipeline}
        showTorgo={showTorgo}
        showContext={showContext}
        onTogglePipeline={() => setShowPipeline((v) => !v)}
        onToggleTorgo={() => setShowTorgo((v) => !v)}
        onToggleContext={() => setShowContext((v) => !v)}
      />

      <div className="flex-1 flex min-h-0">
        {showTorgo && (
          <aside className="w-80 shrink-0 min-h-0">
            <TorgoDrawer
              variant="inline"
              open={showTorgo}
              onClose={() => setShowTorgo(false)}
              onMessage={handleMessage}
            />
          </aside>
        )}

        <div className="flex-1 min-w-0 flex flex-col overflow-y-auto">
          <section className="flex-1 px-8 md:px-12 py-16 flex items-center justify-center">
            {latest ? (
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
                primaryControl={
                  <PrimaryControls
                    micState={session.micState}
                    level={session.level}
                    errorMessage={session.errorMessage}
                    onStart={session.start}
                    onStop={session.stop}
                  />
                }
                connState={session.connState}
              />
            ) : (
              <EmptyState
                micState={session.micState}
                level={session.level}
                errorMessage={session.errorMessage}
                onStart={session.start}
                onStop={session.stop}
                connState={session.connState}
                isLive={isLive}
              />
            )}
          </section>

          {showPipeline && (
            <section className="px-8 md:px-12 pb-10 border-t border-white/5">
              <div className="max-w-5xl mx-auto pt-8">
                <PanelHeader
                  label="pipeline"
                  onHide={() => setShowPipeline(false)}
                />
                <PipelineViz utterance={vizUtterance} runKey={vizRunKey} />
              </div>
            </section>
          )}
        </div>

        {showContext && (
          <aside className="w-80 shrink-0 min-h-0 border-l border-white/10 bg-black/20 overflow-y-auto">
            <div className="px-5 pt-16 pb-5">
              <PanelHeader
                label="context"
                onHide={() => setShowContext(false)}
              />
              <ContextPanel snapshot={ctx} history={history.slice(1)} />
            </div>
          </aside>
        )}
      </div>

      <AccessibilityPanel
        settings={a11y.settings}
        onChange={a11y.setSettings}
      />
    </main>
  );
}

// ============================================================ empty state

function EmptyState({
  micState,
  level,
  errorMessage,
  onStart,
  onStop,
  connState,
  isLive,
}: {
  micState: string;
  level: number;
  errorMessage?: string;
  onStart: () => void | Promise<void>;
  onStop: () => void | Promise<void>;
  connState: string;
  isLive: boolean;
}) {
  return (
    <div className="text-center max-w-2xl mx-auto w-full">
      <div className="space-y-3 mb-12">
        <h1
          className="font-medium tracking-tight"
          style={{ fontSize: "calc(var(--phonemer-base-size) * 2.4)" }}
        >
          Phonemer is listening
        </h1>
        <p className="opacity-50 text-base">
          {isLive
            ? "Mic is on. The next thing said will appear here in large text."
            : "Click Start Listening below to begin."}
        </p>
      </div>

      <PrimaryControls
        micState={micState as any}
        level={level}
        errorMessage={errorMessage}
        onStart={onStart}
        onStop={onStop}
      />

      <div className="mt-10 flex items-center justify-center gap-3 text-[11px] opacity-50 font-mono">
        <ConnStateDot state={connState} />
        <span>ws {connState}</span>
      </div>
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

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center gap-3">
        <button
          disabled={isBusy || isBlocked}
          onClick={() => (isLive ? onStop() : onStart())}
          className={`px-6 py-3 rounded-full text-sm font-medium border transition-all duration-200 ${
            isLive
              ? "bg-rose-500/15 hover:bg-rose-500/25 border-rose-400/40 text-rose-100"
              : isBusy
                ? "bg-amber-500/10 border-amber-400/30 text-amber-200 cursor-wait"
                : isBlocked
                  ? "bg-white/[0.02] border-rose-500/30 text-rose-300 cursor-not-allowed"
                  : "bg-white/[0.04] hover:bg-white/[0.09] border-white/15 hover:border-white/30"
          }`}
        >
          <span className="flex items-center gap-2.5">
            <MicIndicator state={micState} level={level} />
            {label}
          </span>
        </button>

        <a
          href="/calibrate"
          className="px-6 py-3 rounded-full text-sm font-medium border border-white/10 hover:border-white/25 hover:bg-white/[0.04] transition-colors opacity-80"
        >
          Calibrate
        </a>
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
  primaryControl,
  connState,
}: {
  row: Translated;
  editing: { utteranceId: number; text: string } | null;
  onConfirm: (id: number) => void;
  onStartEdit: (row: Translated) => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  onChangeEdit: (t: string) => void;
  primaryControl: React.ReactNode;
  connState: string;
}) {
  const words = useMemo(
    () => row.primary.split(/\s+/).filter(Boolean),
    [row.primary],
  );

  return (
    <div className="max-w-5xl mx-auto w-full space-y-8">
      <div className="text-center space-y-4">
        <SpeakerBadge speaker={row.speaker} />
        {!editing ? (
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
        ) : (
          <div className="space-y-3">
            <textarea
              autoFocus
              value={editing.text}
              onChange={(e) => onChangeEdit(e.target.value)}
              className="w-full max-w-3xl mx-auto rounded-lg p-4 text-2xl bg-black/40 border border-white/15 focus:outline-none focus:border-teal-400/60"
              rows={3}
            />
            <div className="flex justify-center gap-2">
              <button
                onClick={onSaveEdit}
                className="px-4 py-2 rounded-full text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white"
              >
                Save correction
              </button>
              <button
                onClick={onCancelEdit}
                className="px-4 py-2 rounded-full text-sm border border-white/10 hover:bg-white/5"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        <ConfidenceLine row={row} />
      </div>

      {!editing && (
        <>
          {row.alternatives.length > 0 && (
            <div className="text-center space-y-1">
              <SectionLabel center>alternatives</SectionLabel>
              {row.alternatives.map((alt, i) => (
                <div key={i} className="opacity-60 text-base">
                  {alt}
                </div>
              ))}
            </div>
          )}

          {row.contextHints.length > 0 && (
            <div className="text-center text-xs opacity-50 italic">
              context: {row.contextHints.join(" · ")}
            </div>
          )}

          <div className="flex justify-center gap-3">
            {row.feedback ? (
              <span
                className={
                  row.feedback === "confirmed"
                    ? "text-emerald-300 text-sm"
                    : "text-amber-300 text-sm"
                }
              >
                {row.feedback === "confirmed" ? "Confirmed" : "Corrected"}
              </span>
            ) : (
              <>
                <button
                  onClick={() => onConfirm(row.utteranceId)}
                  className="px-5 py-2 rounded-full text-sm font-medium bg-emerald-600/80 hover:bg-emerald-500 text-white"
                >
                  Looks right
                </button>
                <button
                  onClick={() => onStartEdit(row)}
                  className="px-5 py-2 rounded-full text-sm font-medium bg-amber-600/80 hover:bg-amber-500 text-white"
                >
                  Correct
                </button>
              </>
            )}
          </div>

          <div className="pt-4 flex flex-col items-center gap-3">
            {primaryControl}
            <div className="flex items-center gap-2 text-[11px] opacity-50 font-mono">
              <ConnStateDot state={connState} />
              <span>ws {connState}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ShadedWord({ word, conf }: { word: string; conf: number }) {
  const color = conf >= 0.8 ? "#e5e7eb" : conf >= 0.5 ? "#fbbf24" : "#fb7185";
  const decoration = conf < 0.5 ? "underline" : "none";
  return (
    <span
      title={`${(conf * 100).toFixed(0)}% confidence`}
      style={{
        color,
        textDecoration: decoration,
        textUnderlineOffset: 6,
        textDecorationColor: "rgba(251, 113, 133, 0.5)",
      }}
    >
      {word}
    </span>
  );
}

function ConfidenceLine({ row }: { row: Translated }) {
  const c = row.overallConfidence;
  const color = c >= 0.8 ? "#10b981" : c >= 0.5 ? "#f59e0b" : "#ef4444";
  return (
    <div className="text-xs opacity-60 font-mono">
      overall confidence{" "}
      <span style={{ color }}>{(c * 100).toFixed(0)}%</span>
    </div>
  );
}

function SpeakerBadge({ speaker }: { speaker: string }) {
  const isPatient = speaker === "patient" || speaker === "PATIENT";
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
      {isPatient ? "PATIENT" : speaker.toUpperCase()}
    </span>
  );
}

// ============================================================ history

function HistoryRow({ row }: { row: Translated }) {
  const c = row.overallConfidence;
  const tone = c >= 0.8 ? "#34d399" : c >= 0.5 ? "#fbbf24" : "#fb7185";
  return (
    <li
      className="text-xs"
      style={{ borderLeft: `2px solid ${tone}55`, paddingLeft: "0.6rem" }}
    >
      <div className="flex items-baseline gap-2">
        <span className="opacity-50 font-mono shrink-0 tabular-nums">
          {(c * 100).toFixed(0)}%
        </span>
        {row.feedback && (
          <span
            className="text-[9px] uppercase tracking-wider ml-auto opacity-70 shrink-0"
            style={{ color: row.feedback === "confirmed" ? "#34d399" : "#fbbf24" }}
          >
            {row.feedback}
          </span>
        )}
      </div>
      <div className="opacity-90 leading-snug mt-0.5">
        {row.feedback === "corrected" && row.correctedText
          ? row.correctedText
          : row.primary}
      </div>
    </li>
  );
}

// ============================================================ context panel

function ContextPanel({
  snapshot,
  history,
}: {
  snapshot: ContextSnapshot | null;
  history: Translated[];
}) {
  return (
    <div className="space-y-5 text-xs">
      <div>
        <h3 className="text-[10px] tracking-[0.25em] font-mono opacity-50 mb-2">
          PATIENT UTTERANCES
        </h3>
        {history.length === 0 ? (
          <div className="opacity-50">no earlier translations yet</div>
        ) : (
          <ul className="space-y-2">
            {history.map((row) => (
              <HistoryRow key={row.utteranceId} row={row} />
            ))}
          </ul>
        )}
      </div>

      <div>
        <h3 className="text-[10px] tracking-[0.25em] font-mono opacity-50 mb-2">
          CONVERSATION
        </h3>
        {!snapshot || snapshot.recent.length === 0 ? (
          <div className="opacity-50">no recent utterances</div>
        ) : (
          <ul className="space-y-1.5">
            {snapshot.recent
              .slice(-12)
              .reverse()
              .map((u, i) => {
                const isPatient = u.speaker === "patient";
                return (
                  <li key={i} className="flex gap-2">
                    <span
                      className="shrink-0 w-16 truncate font-mono opacity-60"
                      style={{ color: isPatient ? "#fde68a" : "#bae6fd" }}
                    >
                      {isPatient ? "patient" : u.speaker}
                    </span>
                    <span className="opacity-90">{u.text}</span>
                  </li>
                );
              })}
          </ul>
        )}
      </div>

      <div>
        <h3 className="text-[10px] tracking-[0.25em] font-mono opacity-50 mb-2">
          ACTIVE TOPICS
        </h3>
        {!snapshot || snapshot.topics.length === 0 ? (
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
        )}
      </div>

      {snapshot && snapshot.pending_questions.length > 0 && (
        <div>
          <h3 className="text-[10px] tracking-[0.25em] font-mono opacity-50 mb-2">
            UNANSWERED
          </h3>
          <ul className="space-y-1">
            {snapshot.pending_questions.map((q, i) => (
              <li key={i} className="bg-white/5 rounded px-2 py-1">
                <span className="opacity-60 mr-1">[{q.type}]</span>
                {q.text}
              </li>
            ))}
          </ul>
        </div>
      )}

      {snapshot && (
        <div>
          <h3 className="text-[10px] tracking-[0.25em] font-mono opacity-50 mb-2">
            EMOTIONAL STATE
          </h3>
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
        </div>
      )}
    </div>
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

function SectionLabel({
  children,
  center,
}: {
  children: React.ReactNode;
  center?: boolean;
}) {
  return (
    <div
      className={`text-[10px] tracking-[0.25em] font-mono opacity-50 mb-3 uppercase ${
        center ? "text-center" : ""
      }`}
    >
      {children}
    </div>
  );
}

function PanelHeader({
  label,
  onHide,
}: {
  label: string;
  onHide: () => void;
}) {
  return (
    <div className="flex items-center justify-between mb-3">
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
        className="px-3 py-1 rounded bg-white/15 hover:bg-white/25 text-sm"
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
