"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AcousticFeatures,
  ContextSnapshot,
  PhonemeFrame,
  WsMessage,
  WsMessageType,
} from "../../lib/messages";
import { ConnectionState, createPhonemerWs } from "../../lib/phonemer-ws";
import { createAudioCapture, streamFileAsPcm } from "../../lib/audio-capture";
import { createRecorder, downloadRecording } from "../../lib/recorder";
import { createRecordingPlayer } from "../../lib/player";
import type { SessionRecording } from "../../lib/recording-types";

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";
const SERVER_WS = SERVER_HTTP.replace(/^http/, "ws") + "/ws/audio";

type LogEntry = { id: number; t_ms: number; payload: WsMessage };
type ActiveUtterance = {
  id: number;
  speaker: string;
  isPatient: boolean;
  startMs: number;
  endMs: number;
  similarity: number;
  frames: PhonemeFrame[];
  features?: AcousticFeatures;
  transcript?: string;
  translation?: {
    primary: string;
    alternatives: string[];
    per_word_confidence: number[];
    overall_confidence: number;
    emergency: boolean;
    context_hints: string[];
  };
};
type SpeakerInfo = { label: string; count: number; lastSimilarity: number };

const LAYER_TO_TYPE: Record<string, WsMessageType[]> = {
  VAD: ["utterance"],
  "Speaker ID": ["utterance"],
  Phonemes: ["phoneme_window"],
  Features: ["features"],
  STT: ["utterance_transcript"],
  LLM: ["translation"],
};
const LAYER_ORDER = Object.keys(LAYER_TO_TYPE);

export default function DebugPage() {
  const [conn, setConn] = useState<ConnectionState>("idle");
  const [mode, setMode] = useState<"live" | "replay">("live");
  const [log, setLog] = useState<LogEntry[]>([]);
  const [paused, setPaused] = useState(false);
  const [active, setActive] = useState<ActiveUtterance | null>(null);
  const [speakers, setSpeakers] = useState<Record<string, SpeakerInfo>>({});
  const [seenLayers, setSeenLayers] = useState<Set<string>>(new Set());
  const [ctx, setCtx] = useState<ContextSnapshot | null>(null);
  const [level, setLevel] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [recording, setRecording] = useState(false);
  const [replayProgress, setReplayProgress] = useState({
    currentMs: 0,
    totalMs: 0,
  });
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [includeAudio, setIncludeAudio] = useState(false);

  const [torgoClips, setTorgoClips] = useState<TorgoClipRow[] | null>(null);
  const [selectedClip, setSelectedClip] = useState<number | "">("");

  const lastIdRef = useRef(0);
  const startedAtRef = useRef(performance.now());
  const wsRef = useRef<ReturnType<typeof createPhonemerWs> | null>(null);
  const captureRef = useRef<ReturnType<typeof createAudioCapture> | null>(
    null,
  );
  const recorderRef = useRef(createRecorder());
  const playerRef = useRef<ReturnType<typeof createRecordingPlayer> | null>(
    null,
  );

  // ----- message ingest (used by both live WS and replay player) -----
  const handleMessage = useCallback(
    (msg: WsMessage) => {
      if (recorderRef.current.isRecording() && mode === "live") {
        recorderRef.current.observeMessage(msg);
      }
      const t_ms = performance.now() - startedAtRef.current;
      lastIdRef.current += 1;
      const entry = { id: lastIdRef.current, t_ms, payload: msg };
      if (!paused) {
        setLog((prev) => {
          const next = [entry, ...prev];
          return next.length > 500 ? next.slice(0, 500) : next;
        });
      }

      // Layer-status strip
      setSeenLayers((prev) => {
        const next = new Set(prev);
        for (const [layer, types] of Object.entries(LAYER_TO_TYPE)) {
          if ((types as readonly string[]).includes(msg.type)) next.add(layer);
        }
        return next;
      });

      switch (msg.type) {
        case "utterance":
          setActive({
            id: msg.utterance_id,
            speaker: msg.speaker,
            isPatient: msg.is_patient,
            startMs: msg.start_ms,
            endMs: msg.end_ms,
            similarity: msg.speaker_similarity,
            frames: [],
          });
          setSpeakers((prev) => {
            const cur = prev[msg.speaker];
            return {
              ...prev,
              [msg.speaker]: {
                label: msg.speaker,
                count: (cur?.count ?? 0) + 1,
                lastSimilarity: msg.speaker_similarity,
              },
            };
          });
          break;
        case "phoneme_window":
          setActive((prev) =>
            prev && prev.id === msg.utterance_id
              ? { ...prev, frames: [...prev.frames, ...msg.frames] }
              : prev,
          );
          break;
        case "features":
          setActive((prev) =>
            prev && prev.id === msg.utterance_id
              ? { ...prev, features: msg.features }
              : prev,
          );
          break;
        case "utterance_transcript":
          setActive((prev) =>
            prev && prev.id === msg.utterance_id
              ? { ...prev, transcript: msg.transcript }
              : prev,
          );
          break;
        case "translation":
          setActive((prev) =>
            prev && prev.id === msg.utterance_id
              ? {
                  ...prev,
                  translation: {
                    primary: msg.primary,
                    alternatives: msg.alternatives,
                    per_word_confidence: msg.per_word_confidence,
                    overall_confidence: msg.overall_confidence,
                    emergency: msg.emergency,
                    context_hints: msg.context_hints,
                  },
                }
              : prev,
          );
          break;
        case "context":
          setCtx(msg.context);
          break;
        default:
          break;
      }
    },
    [mode, paused],
  );

  // ----- live WebSocket -----
  useEffect(() => {
    const ws = createPhonemerWs(SERVER_WS, {
      onMessage: handleMessage,
      onState: setConn,
    });
    wsRef.current = ws;
    if (mode === "live") ws.connect();
    return () => ws.disconnect();
  }, [handleMessage, mode]);

  // ----- replay player -----
  useEffect(() => {
    playerRef.current = createRecordingPlayer({
      onMessage: handleMessage,
      onProgress: (currentMs, totalMs) =>
        setReplayProgress({ currentMs, totalMs }),
      onEnded: () => setReplayPlaying(false),
    });
  }, [handleMessage]);

  useEffect(() => {
    playerRef.current?.setSpeed(replaySpeed);
  }, [replaySpeed]);

  // ----- TORGO clips -----
  useEffect(() => {
    fetch(`${SERVER_HTTP}/torgo/clips?limit=500`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data) => setTorgoClips(data.clips ?? []))
      .catch(() => setTorgoClips([]));
  }, []);

  // ----- mic capture -----
  const startCapture = useCallback(async () => {
    if (captureRef.current?.isRunning()) return;
    const cap = createAudioCapture({
      onPcm: (chunk) => {
        wsRef.current?.sendPcm(chunk);
        if (recorderRef.current.isRecording()) {
          recorderRef.current.observePcm(chunk);
        }
      },
      onLevel: setLevel,
      onError: (err) => console.warn("capture error", err),
    });
    captureRef.current = cap;
    await cap.start();
    setCapturing(true);
  }, []);

  const stopCapture = useCallback(async () => {
    await captureRef.current?.stop();
    captureRef.current = null;
    setCapturing(false);
    wsRef.current?.sendControl({ type: "stop" });
  }, []);

  // ----- file / TORGO streaming -----
  const streamArrayBuffer = useCallback(
    async (buf: ArrayBuffer, source: "wav" | "torgo", label: string) => {
      if (!wsRef.current) return;
      // Reset active panel.
      setActive(null);
      // Re-open WS so the server starts a fresh session.
      wsRef.current.disconnect();
      wsRef.current.connect();
      // Wait briefly for the new socket to be open.
      await new Promise((r) => setTimeout(r, 250));
      await streamFileAsPcm(buf, (chunk) => {
        wsRef.current?.sendPcm(chunk);
        if (recorderRef.current.isRecording())
          recorderRef.current.observePcm(chunk);
      });
      // Tell server we're done so it flushes the trailing utterance.
      wsRef.current.sendControl({ type: "stop" });
    },
    [],
  );

  const onDropFile = useCallback(
    async (file: File) => {
      const buf = await file.arrayBuffer();
      await streamArrayBuffer(buf, "wav", file.name);
    },
    [streamArrayBuffer],
  );

  const onPlayClip = useCallback(async () => {
    if (!selectedClip) return;
    const res = await fetch(`${SERVER_HTTP}/torgo/audio/${selectedClip}`);
    const buf = await res.arrayBuffer();
    const meta = torgoClips?.find((c) => c.id === selectedClip);
    await streamArrayBuffer(buf, "torgo", meta?.intended_text || `clip ${selectedClip}`);
  }, [selectedClip, streamArrayBuffer, torgoClips]);

  // ----- enroll patient -----
  const onEnrollPatient = useCallback(async () => {
    if (!active) return;
    // We don't have raw audio in the client for the active utterance directly,
    // but the server already saved it. Easiest path: re-upload last recorded
    // PCM. For now, prompt the user with a hint.
    alert(
      "Enrollment needs a WAV upload. Use:\n" +
        "  curl -F audio=@calibration.wav http://localhost:8000/speaker/enroll/patient",
    );
  }, [active]);

  // ----- record/replay UI -----
  const onStartRecording = () => {
    recorderRef.current.start({ source: "mic", includeAudio });
    setRecording(true);
  };
  const onStopRecording = () => {
    setRecording(false);
  };
  const onDownloadRecording = () => {
    const snap = recorderRef.current.snapshot();
    if (snap) downloadRecording(snap);
    recorderRef.current.stop();
  };

  const onLoadRecording = useCallback(async (file: File) => {
    const text = await file.text();
    const rec = JSON.parse(text) as SessionRecording;
    wsRef.current?.disconnect();
    setMode("replay");
    setLog([]);
    setActive(null);
    setSpeakers({});
    setSeenLayers(new Set());
    playerRef.current?.load(rec);
  }, []);

  const onReplayPlay = () => {
    playerRef.current?.play();
    setReplayPlaying(true);
  };
  const onReplayPause = () => {
    playerRef.current?.pause();
    setReplayPlaying(false);
  };
  const onReplayBackToLive = () => {
    playerRef.current?.stop();
    setReplayPlaying(false);
    setMode("live");
    wsRef.current?.connect();
  };

  // ----- root drag handlers (file / recording auto-detect) -----
  const onRootDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (file.name.endsWith(".json")) await onLoadRecording(file);
      else await onDropFile(file);
    },
    [onDropFile, onLoadRecording],
  );

  const phonemeHeatmap = useMemo(
    () => buildHeatmap(active?.frames ?? []),
    [active?.frames],
  );

  return (
    <main
      onDragOver={(e) => e.preventDefault()}
      onDrop={onRootDrop}
      className="min-h-screen bg-neutral-950 text-neutral-100 font-mono text-xs"
    >
      <header className="px-4 py-3 border-b border-neutral-800 flex items-center gap-4 flex-wrap">
        <span className="text-base font-semibold tracking-wide">
          Phonemenal · Debug
        </span>
        <a
          href="/calibrate"
          className="text-xs px-2 py-0.5 rounded bg-amber-600/20 text-amber-300 hover:bg-amber-600/30"
        >
          Calibration →
        </a>
        <a
          href="/viz"
          className="text-xs px-2 py-0.5 rounded bg-teal-600/20 text-teal-300 hover:bg-teal-600/30"
        >
          Pipeline viz →
        </a>
        <a
          href="/audio"
          className="text-xs px-2 py-0.5 rounded bg-emerald-600/20 text-emerald-300 hover:bg-emerald-600/30"
        >
          Audio diag →
        </a>
        <a
          href="/"
          className="text-xs px-2 py-0.5 rounded bg-rose-600/20 text-rose-300 hover:bg-rose-600/30"
        >
          Home (live) →
        </a>
        <ModeBadge mode={mode} conn={conn} />
        <LayerStrip seen={seenLayers} />
        <div className="ml-auto flex items-center gap-2">
          <span className="opacity-70">level</span>
          <div className="w-24 h-2 bg-neutral-800 rounded">
            <div
              className="h-2 bg-emerald-500 rounded"
              style={{ width: `${Math.min(100, level * 600)}%` }}
            />
          </div>
        </div>
      </header>

      {/* Toolbar */}
      <div className="px-4 py-2 border-b border-neutral-800 flex items-center gap-3 flex-wrap">
        {mode === "live" ? (
          <>
            {capturing ? (
              <button onClick={stopCapture} className="btn-red">
                ⏹ Stop mic
              </button>
            ) : (
              <button onClick={startCapture} className="btn-green">
                🎙 Start mic
              </button>
            )}

            <select
              className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 disabled:opacity-50"
              value={selectedClip}
              disabled={!torgoClips || torgoClips.length === 0}
              onChange={(e) =>
                setSelectedClip(e.target.value ? Number(e.target.value) : "")
              }
            >
              <option value="">
                {torgoClips === null
                  ? "— loading TORGO… —"
                  : torgoClips.length === 0
                    ? "— no TORGO clips loaded —"
                    : "— TORGO clip —"}
              </option>
              {(torgoClips ?? []).map((c) => (
                <option key={c.id} value={c.id}>
                  {c.speaker_id} · {c.condition} · {(c.intended_text || "").slice(0, 40)}
                </option>
              ))}
            </select>
            <button
              disabled={!selectedClip}
              onClick={onPlayClip}
              className="btn"
            >
              ▶ Stream clip
            </button>
            {torgoClips && torgoClips.length === 0 && (
              <span className="text-amber-300/80 text-[11px]">
                no clips — see <code>make torgo-load</code> or drop any 16 kHz mono WAV onto the page
              </span>
            )}

            <span className="opacity-50 mx-2">|</span>

            {!recording ? (
              <button onClick={onStartRecording} className="btn">
                ⏺ Record
              </button>
            ) : (
              <button onClick={onStopRecording} className="btn-red">
                ⏹ Stop record
              </button>
            )}
            <button onClick={onDownloadRecording} className="btn">
              💾 Download
            </button>
            <label className="flex items-center gap-1 opacity-80">
              <input
                type="checkbox"
                checked={includeAudio}
                onChange={(e) => setIncludeAudio(e.target.checked)}
              />
              include audio
            </label>

            <span className="opacity-50 mx-2">|</span>
            <button onClick={onEnrollPatient} className="btn" title="See instructions">
              👤 Enroll patient…
            </button>
          </>
        ) : (
          <>
            {replayPlaying ? (
              <button onClick={onReplayPause} className="btn-red">
                ⏸ Pause
              </button>
            ) : (
              <button onClick={onReplayPlay} className="btn-green">
                ▶ Play
              </button>
            )}
            <select
              value={replaySpeed}
              onChange={(e) => setReplaySpeed(Number(e.target.value))}
              className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1"
            >
              {[0.5, 1, 2, 4, 16].map((s) => (
                <option key={s} value={s}>
                  {s}×
                </option>
              ))}
            </select>
            <input
              type="range"
              min={0}
              max={replayProgress.totalMs}
              value={replayProgress.currentMs}
              onChange={(e) => playerRef.current?.seek(Number(e.target.value))}
              className="flex-1"
            />
            <span className="opacity-70 tabular-nums">
              {fmt(replayProgress.currentMs)} / {fmt(replayProgress.totalMs)}
            </span>
            <button onClick={onReplayBackToLive} className="btn">
              🔌 Back to live
            </button>
          </>
        )}

        <span className="ml-auto opacity-50">
          drop a .wav (TORGO clip) or .json (recording) anywhere on the page
        </span>
      </div>

      {/* Three-column body */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-3">
        {/* Stream log */}
        <section className="border border-neutral-800 rounded">
          <Header label="Stream log">
            <button
              onClick={() => setPaused((p) => !p)}
              className="opacity-80 hover:opacity-100"
            >
              {paused ? "▶ resume" : "⏸ pause"}
            </button>
            <button
              onClick={() => setLog([])}
              className="opacity-80 hover:opacity-100"
            >
              🧹 clear
            </button>
          </Header>
          <div className="max-h-[70vh] overflow-auto p-2 space-y-1">
            {log.map((entry) => (
              <LogRow key={entry.id} entry={entry} />
            ))}
          </div>
        </section>

        {/* Active utterance */}
        <section className="border border-neutral-800 rounded">
          <Header label="Active utterance" />
          <div className="p-3 space-y-3">
            {active ? (
              <>
                <div className="flex items-center gap-2">
                  <span
                    className={
                      active.isPatient
                        ? "px-2 py-0.5 bg-amber-500/20 text-amber-300 rounded"
                        : "px-2 py-0.5 bg-sky-500/20 text-sky-300 rounded"
                    }
                  >
                    {active.isPatient ? "PATIENT" : active.speaker}
                  </span>
                  <span className="opacity-70">
                    sim {active.similarity.toFixed(2)}
                  </span>
                  <span className="opacity-70">
                    {active.startMs}–{active.endMs} ms
                  </span>
                </div>

                <Heatmap heatmap={phonemeHeatmap} />

                {active.translation && (
                  <TranslationPanel t={active.translation} />
                )}

                {active.features && <FeaturesPanel f={active.features} />}
                {active.transcript !== undefined && (
                  <div className="rounded bg-neutral-900 px-2 py-1">
                    <div className="opacity-60">transcript</div>
                    <div className="text-neutral-100">
                      {active.transcript || <span className="opacity-50">(empty)</span>}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="opacity-50">no active utterance yet</div>
            )}
          </div>
        </section>

        {/* Speakers + Context */}
        <section className="border border-neutral-800 rounded space-y-0">
          <Header label="Speaker registry" />
          <div className="p-3 space-y-1">
            {Object.values(speakers).length === 0 && (
              <div className="opacity-50">no speakers detected</div>
            )}
            {Object.values(speakers).map((s) => {
              const role = ctx?.speakers.find((x) => x.label === s.label);
              return (
                <div
                  key={s.label}
                  className="flex items-center justify-between bg-neutral-900 rounded px-2 py-1"
                >
                  <span>
                    {s.label === "patient" ? "👤 patient" : s.label}
                    {role?.role && (
                      <span className="ml-2 text-[10px] opacity-60">
                        · {role.role} ({role.role_confidence.toFixed(2)})
                      </span>
                    )}
                  </span>
                  <span className="opacity-60">
                    {s.count}× · sim {s.lastSimilarity.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="border-t border-neutral-800">
            <Header label="Context window" />
            <ContextPanel snapshot={ctx} />
          </div>
        </section>
      </div>

      <style>{`
        .btn { padding: 4px 10px; border: 1px solid #525252; border-radius: 4px; }
        .btn:hover { background: #1f1f1f; }
        .btn-green { padding: 4px 10px; border-radius: 4px; background: #047857; }
        .btn-green:hover { background: #059669; }
        .btn-red { padding: 4px 10px; border-radius: 4px; background: #b91c1c; }
        .btn-red:hover { background: #dc2626; }
      `}</style>
    </main>
  );
}

// ============================================================ helpers + UI

type TorgoClipRow = {
  id: number;
  speaker_id: string;
  condition: string;
  intelligibility?: string;
  prompt_set?: string;
  intended_text?: string;
  duration_ms: number;
};

function ModeBadge({
  mode,
  conn,
}: {
  mode: "live" | "replay";
  conn: ConnectionState;
}) {
  const live = mode === "live";
  const color =
    !live ? "bg-purple-500/20 text-purple-300"
    : conn === "open" ? "bg-emerald-500/20 text-emerald-300"
    : conn === "connecting" ? "bg-amber-500/20 text-amber-300"
    : "bg-rose-500/20 text-rose-300";
  return (
    <span className={`px-2 py-0.5 rounded ${color}`}>
      {live ? `🔌 LIVE · ${conn}` : "📼 REPLAY"}
    </span>
  );
}

function LayerStrip({ seen }: { seen: Set<string> }) {
  return (
    <div className="flex items-center gap-1">
      {LAYER_ORDER.map((name) => (
        <span
          key={name}
          className={`px-2 py-0.5 rounded text-[10px] ${
            seen.has(name)
              ? "bg-emerald-500/20 text-emerald-300"
              : "bg-neutral-800 text-neutral-500"
          }`}
        >
          {name}
        </span>
      ))}
    </div>
  );
}

function Header({
  label,
  children,
}: {
  label: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="px-3 py-2 border-b border-neutral-800 flex items-center justify-between">
      <span className="font-semibold tracking-wide">{label}</span>
      <span className="flex items-center gap-2">{children}</span>
    </div>
  );
}

function LogRow({ entry }: { entry: LogEntry }) {
  const colors: Record<string, string> = {
    utterance: "text-amber-300",
    phoneme_window: "text-sky-300",
    features: "text-purple-300",
    utterance_transcript: "text-emerald-300",
    translation: "text-pink-300",
    emergency: "text-rose-400",
    utterance_rejected: "text-neutral-500",
    context: "text-blue-300",
    translation_error: "text-amber-300",
  };
  const c = colors[entry.payload.type] ?? "text-neutral-300";
  const summary = summarize(entry.payload);
  return (
    <div className="flex gap-2">
      <span className="opacity-50 w-12 tabular-nums">
        {(entry.t_ms / 1000).toFixed(1)}s
      </span>
      <span className={`w-24 ${c}`}>{entry.payload.type}</span>
      <span className="opacity-80">{summary}</span>
    </div>
  );
}

function summarize(msg: WsMessage): string {
  switch (msg.type) {
    case "utterance":
      return `utt ${msg.utterance_id} · ${msg.speaker} · ${msg.start_ms}-${msg.end_ms}ms`;
    case "phoneme_window":
      return `${msg.frames.length} frames`;
    case "features":
      return `pitch ${msg.features.pitch_hz_mean.toFixed(0)}Hz · jitter ${msg.features.jitter.toFixed(3)}`;
    case "utterance_transcript":
      return msg.transcript || "(empty)";
    case "translation":
      return `${msg.primary} (${(msg.overall_confidence * 100).toFixed(0)}%)`;
    case "emergency":
      return `⚠ ${msg.phrase} (${msg.confidence.toFixed(2)})`;
    case "utterance_rejected":
      return `${msg.start_ms}-${msg.end_ms}ms · sim ${msg.speaker_similarity.toFixed(2)} · ${msg.reason}`;
    case "context":
      return `topics=${msg.context.topics.length} questions=${msg.context.pending_questions.length}`;
    case "translation_error":
      return `${msg.error}: ${msg.message}`;
    case "diag":
      return `diag ${msg.level}: ${msg.msg}`;
  }
}

function buildHeatmap(frames: PhonemeFrame[]) {
  // Take the top-5 phonemes per frame; render as rows × phonemes-cols.
  const cols = 5;
  const rows = frames.slice(-100); // last ~2s at 20ms stride
  return rows.map((f) => f.phonemes.slice(0, cols));
}

function Heatmap({
  heatmap,
}: {
  heatmap: { symbol: string; prob: number }[][];
}) {
  if (heatmap.length === 0) {
    return <div className="opacity-50">no phoneme frames yet</div>;
  }
  return (
    <div>
      <div className="opacity-60 mb-1">phoneme heatmap (last {heatmap.length} frames)</div>
      <div
        className="inline-grid gap-px"
        style={{
          gridTemplateColumns: `repeat(${heatmap[0].length}, 1.6rem)`,
        }}
      >
        {heatmap.map((row, ri) =>
          row.map((cell, ci) => (
            <div
              key={`${ri}-${ci}`}
              title={`${cell.symbol} · ${cell.prob.toFixed(3)}`}
              className="h-5 rounded-sm flex items-center justify-center text-[10px]"
              style={{
                background: `rgba(56, 189, 248, ${Math.max(0.05, cell.prob)})`,
              }}
            >
              {ci === 0 ? cell.symbol : ""}
            </div>
          )),
        )}
      </div>
    </div>
  );
}

function ContextPanel({ snapshot }: { snapshot: ContextSnapshot | null }) {
  if (!snapshot) {
    return <div className="p-3 opacity-50">no context yet</div>;
  }
  const { topics, pending_questions, emotion, session, vocabulary_boost } = snapshot;
  const boostKeys = Object.keys(vocabulary_boost).slice(0, 8);
  return (
    <div className="p-3 space-y-3 text-[11px]">
      <div>
        <div className="opacity-60 mb-1">topics (decayed weights)</div>
        {topics.length === 0 ? (
          <div className="opacity-40">none</div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {topics.slice(0, 8).map((t) => (
              <span
                key={t.keyword}
                className="px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-300"
                style={{ opacity: 0.4 + Math.min(0.6, t.weight / 2) }}
                title={`weight=${t.weight}`}
              >
                {t.keyword}·{t.weight.toFixed(1)}
              </span>
            ))}
          </div>
        )}
      </div>

      <div>
        <div className="opacity-60 mb-1">pending questions</div>
        {pending_questions.length === 0 ? (
          <div className="opacity-40">none</div>
        ) : (
          <div className="space-y-1">
            {pending_questions.map((q, i) => (
              <div key={i} className="flex justify-between gap-2 bg-neutral-900 rounded px-2 py-1">
                <span className="truncate">[{q.type}] {q.text}</span>
                <span className="opacity-60">u={q.urgency.toFixed(2)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <div className="opacity-60 mb-1">emotion (prosody)</div>
          <Bar label="frustration" value={emotion.frustration} color="#f59e0b" />
          <Bar label="distress" value={emotion.distress} color="#ef4444" />
        </div>
        <div>
          <div className="opacity-60 mb-1">session</div>
          <div className="bg-neutral-900 rounded px-2 py-1 space-y-0.5">
            <div className="flex justify-between"><span className="opacity-60">setting</span><span>{session.setting ?? "—"}</span></div>
            <div className="flex justify-between"><span className="opacity-60">listener</span><span>{session.listener_role_hint ?? "—"}</span></div>
            <div className="flex justify-between"><span className="opacity-60">time</span><span>{session.time_of_day ?? "—"}</span></div>
          </div>
        </div>
      </div>

      {boostKeys.length > 0 && (
        <div>
          <div className="opacity-60 mb-1">vocabulary boost (top {boostKeys.length})</div>
          <div className="flex flex-wrap gap-1">
            {boostKeys.map((k) => (
              <span
                key={k}
                className="px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-300"
                title={`boost=${vocabulary_boost[k].toFixed(2)}`}
              >
                {k}·{vocabulary_boost[k].toFixed(1)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 mb-0.5">
      <span className="opacity-60 w-20">{label}</span>
      <div className="flex-1 h-1.5 bg-neutral-800 rounded">
        <div
          className="h-1.5 rounded"
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }}
        />
      </div>
      <span className="opacity-60 w-10 text-right tabular-nums">{value.toFixed(2)}</span>
    </div>
  );
}


function TranslationPanel({
  t,
}: {
  t: NonNullable<ActiveUtterance["translation"]>;
}) {
  const words = t.primary.split(/\s+/).filter(Boolean);
  return (
    <div
      className={`rounded px-2 py-2 ${
        t.emergency
          ? "bg-rose-900/40 border border-rose-700"
          : "bg-pink-900/20"
      }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="opacity-60">translation</span>
        {t.emergency && (
          <span className="px-1.5 py-0.5 rounded bg-rose-600 text-white text-[10px]">
            ⚠ EMERGENCY
          </span>
        )}
        <span className="ml-auto opacity-60 text-[10px]">
          confidence {(t.overall_confidence * 100).toFixed(0)}%
        </span>
      </div>
      <div className="text-base flex flex-wrap gap-x-1.5 gap-y-1">
        {words.map((w, i) => {
          const c = t.per_word_confidence[i] ?? t.overall_confidence;
          const cls =
            c > 0.8
              ? "text-emerald-300"
              : c > 0.5
                ? "text-amber-300"
                : "text-rose-400 underline";
          return (
            <span key={i} className={cls} title={`${(c * 100).toFixed(0)}%`}>
              {w}
            </span>
          );
        })}
      </div>
      {t.alternatives.length > 0 && (
        <div className="mt-2 opacity-70 text-[11px] space-y-0.5">
          <div className="opacity-60">alternatives</div>
          {t.alternatives.map((a, i) => (
            <div key={i}>· {a}</div>
          ))}
        </div>
      )}
      {t.context_hints.length > 0 && (
        <div className="mt-2 opacity-50 text-[11px] italic">
          context: {t.context_hints.join(" · ")}
        </div>
      )}
    </div>
  );
}


function FeaturesPanel({ f }: { f: AcousticFeatures }) {
  const items: [string, string][] = [
    ["pitch", `${f.pitch_hz_mean.toFixed(0)} ± ${f.pitch_hz_std.toFixed(0)} Hz`],
    ["jitter", f.jitter.toFixed(3)],
    ["shimmer", f.shimmer.toFixed(3)],
    [
      "formants",
      `F1 ${f.formants_hz[0].toFixed(0)} · F2 ${f.formants_hz[1].toFixed(0)} · F3 ${f.formants_hz[2].toFixed(0)}`,
    ],
    ["VOT proxy", `${f.vot_proxy_ms.toFixed(0)} ms`],
    ["pauses", `${f.pause_count} (${f.pause_total_ms} ms)`],
    [
      "spectral flux",
      `${f.spectral_flux_mean.toFixed(2)} ± ${f.spectral_flux_std.toFixed(2)}`,
    ],
  ];
  return (
    <div className="rounded bg-neutral-900 px-2 py-1">
      <div className="opacity-60 mb-1">features</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        {items.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2">
            <span className="opacity-60">{k}</span>
            <span>{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function fmt(ms: number) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}
