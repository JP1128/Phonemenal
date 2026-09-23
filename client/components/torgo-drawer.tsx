"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SeverityBadge } from "./severity-badge";
import { streamFileAsPcm } from "../lib/audio-capture";
import {
  buildRankingIndex,
  loadDemoRanking,
  type DemoRanking,
  type RankingEntry,
} from "../lib/demo-ranking";
import type { WsMessage } from "../lib/messages";
import { createPhonemerWs } from "../lib/phonemer-ws";
import type { Profile } from "../lib/profile";
import { SPEAKER_SEVERITY } from "../lib/torgo-speakers";

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";

type TorgoClip = {
  id: number;
  speaker_id: string;
  condition: string;
  intelligibility: string | null;
  prompt_set: string | null;
  intended_text: string | null;
  duration_ms: number;
  // Deterministic train/eval bucket. Server hashes (seed | speaker | id)
  // and assigns 20% to "eval" — the same split used by the per-speaker
  // adapter trainer, so this lets the panel show "did the adapter ever
  // see this clip during training?".
  split?: "cal" | "eval" | null;
};

// The old TORGO panel used to expose Calibrated / Not calibrated / Control
// grouping inline. That role moved to the header-level ProfileSelector:
// the active Profile already locks the audience the drawer is showing
// clips for, so the drawer no longer needs its own group selector.

// Per-speaker dysarthria severity now lives in
// `client/lib/torgo-speakers.ts` so the profile selector can reuse it.

// prompt_set values written by the loader. "non-word" is the DDK-stimuli
// bucket — single repeated CV syllables ("papa", "kakaka") that aren't
// useful for translation testing; hidden by default.
type PromptSet = "timit" | "intelligibility" | "image" | "non-word" | "unknown";

const PROMPT_SET_LABELS: Record<PromptSet, string> = {
  timit: "sentence",
  intelligibility: "word",
  image: "image",
  "non-word": "non-word",
  unknown: "?",
};

export type TorgoDrawerProps = {
  open: boolean;
  onClose: () => void;
  /**
   * Receives every WS message produced by the streamed clip. The home page
   * pipes this into its primary handleMessage so utterances + translations
   * land in the same history shown to the user.
   */
  onMessage: (msg: WsMessage) => void;
  /** Called when a clip starts streaming through the dedicated WS. */
  onStreamStart?: (clip: TorgoClip) => void;
  /** Called when the WS closes (clip finished or errored). */
  onStreamEnd?: () => void;
  /** "overlay" = fixed sidebar (default). "inline" = panel that flows in parent layout. */
  variant?: "overlay" | "inline";
  /** Active patient profile; scopes the clip list to one speaker / a bucket. */
  profile: Profile;
  /**
   * Uppercase TORGO speaker ids with a per-speaker adapter on disk.
   * Supplied by the parent (already fetched for the ProfileSelector) so
   * the drawer doesn't double-fetch `/adapters`.
   */
  calibratedSpeakers: Set<string>;
};

// One in-app diagnostic log per clip stream. Captures the server's
// own ``diag`` events plus derived entries from the standard pipeline
// messages (utterance / features / translation / rejected) so the user
// can copy a complete, deterministic timeline without sifting through
// the server's stdout. Each stream is delimited so consecutive runs
// of different clips stay distinct in the copy buffer.
type StreamLogEntry = {
  ts_ms: number;
  level: "info" | "warn" | "error";
  text: string;
};
type StreamLog = {
  // Stable client-side id so React keys stay valid before the server's
  // session_id has arrived (it lands with the first diag/utterance).
  key: string;
  clipLabel: string;
  startedAt: number; // Date.now() at click
  sessionId: number | null;
  entries: StreamLogEntry[];
};

// Cap how many past streams we keep in memory. Twenty is plenty for
// a single demo session and prevents unbounded growth if the user
// rapidly clicks through dozens of clips.
const STREAM_LOG_CAP = 20;

export function TorgoDrawer({
  open,
  onClose,
  onMessage,
  onStreamStart,
  onStreamEnd,
  variant = "overlay",
  profile,
  calibratedSpeakers,
}: TorgoDrawerProps) {
  const [clips, setClips] = useState<TorgoClip[] | null>(null);
  // Pre-computed per-clip adapted-WER + gate-flag, keyed by speaker id
  // (uppercase). Used as the primary sort key for TIMIT clips so the
  // best-performing clips float to the top for the demo. Missing → null.
  const [ranking, setRanking] = useState<DemoRanking | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Per-stream diagnostic log buffer (newest first). Updated as
  // messages flow in for the currently-streaming WS.
  const [streamLogs, setStreamLogs] = useState<StreamLog[]>([]);
  // The active stream's key — log entries are appended to the head of
  // streamLogs only when this matches, so a stale WS that's still
  // closing can't pollute the next clip's log.
  const activeLogKey = useRef<string | null>(null);
  const [showLogs, setShowLogs] = useState(true);
  // For the Demo profile we let the user sub-pick which uncalibrated
  // dysarthric speaker to view. For the Calibrated profile the speaker
  // is locked by `profile.speakerId`, so this state is only used in the
  // demo branch.
  const [demoSpeaker, setDemoSpeaker] = useState<string | null>(null);
  const [streamingId, setStreamingId] = useState<number | null>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const [showNonWord, setShowNonWord] = useState(false);
  const [setFilter, setSetFilter] = useState<"all" | "timit" | "intelligibility">(
    "all",
  );
  // Train / test split filter. Maps to the server's cal / eval buckets
  // — "train" = clips the calibrated adapter was trained on, "test" = the
  // 20% held-out evaluation set.
  type SplitFilter = "all" | "train" | "test";
  const [splitFilter, setSplitFilter] = useState<SplitFilter>("all");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Track the most recent blob URL handed to the audio element so we can
  // revoke it when assigning a new one (and on unmount). Avoids leaking
  // ObjectURLs as the user clicks through clips.
  const blobUrlRef = useRef<string | null>(null);
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    if (clips !== null) return;
    // Limit needs to comfortably exceed the full HF-loaded manifest
    // (~8,700 clips). 2,000 was truncating the alphabetical fetch
    // before reaching M speakers, hiding them from the panel entirely.
    fetch(`${SERVER_HTTP}/torgo/clips?limit=20000`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)))
      .then((data) => setClips(data.clips ?? []))
      .catch((err) => setError(`could not load TORGO clips: ${err}`));
  }, [open, clips]);

  // Fetch the demo ranking once per drawer open. Cached in-module via
  // loadDemoRanking, so re-opens are free. A missing file is non-fatal —
  // the clip list just falls back to the existing prompt-set / id order.
  useEffect(() => {
    if (!open) return;
    if (ranking !== null) return;
    let cancelled = false;
    loadDemoRanking().then((r) => {
      if (!cancelled && r) setRanking(r);
    });
    return () => {
      cancelled = true;
    };
  }, [open, ranking]);

  // For the Demo profile the speaker picker shows the uncalibrated
  // dysarthric set (F01 / F04 / M02 / M03 / M04 / M05 in practice).
  // Controls are intentionally excluded — the demo bucket is for
  // exercising the patient pipeline against uncalibrated speakers, not
  // sanity-checking on non-dysarthric audio.
  const demoSpeakers = useMemo(() => {
    if (!clips) return [];
    const set = new Set<string>();
    for (const c of clips) {
      const up = c.speaker_id.toUpperCase();
      if (calibratedSpeakers.has(up)) continue;
      if (c.condition === "control") continue;
      set.add(up);
    }
    return Array.from(set).sort();
  }, [clips, calibratedSpeakers]);

  // The single speaker the clip list is currently scoped to. For
  // Calibrated profiles that's pinned to profile.speakerId. For Demo
  // profiles it follows the user's sub-pick, defaulting to the first
  // available speaker.
  const activeSpeaker = useMemo(() => {
    if (profile.kind === "calibrated") return profile.speakerId.toUpperCase();
    if (demoSpeaker && demoSpeakers.includes(demoSpeaker)) return demoSpeaker;
    return demoSpeakers[0] ?? null;
  }, [profile, demoSpeaker, demoSpeakers]);

  // Auto-seed demoSpeaker when entering Demo mode without a prior pick
  // (or after the available demo set changed).
  useEffect(() => {
    if (profile.kind !== "demo") return;
    if (demoSpeakers.length === 0) return;
    if (!demoSpeaker || !demoSpeakers.includes(demoSpeaker)) {
      setDemoSpeaker(demoSpeakers[0]);
    }
  }, [profile.kind, demoSpeakers, demoSpeaker]);

  // Per-speaker clip → ranking entry lookup. Built once per
  // (ranking, activeSpeaker) pair; the sort below uses it to pull
  // adapted_wer / adapted_flagged into the comparator without
  // re-scanning the ranking array per comparison.
  const rankingIndex = useMemo<Map<number, RankingEntry>>(() => {
    if (!activeSpeaker) return new Map();
    return buildRankingIndex(ranking, activeSpeaker);
  }, [ranking, activeSpeaker]);

  const speakerClips = useMemo(() => {
    if (!clips || !activeSpeaker) return [];
    // Order: sentences first (timit), then single words (intelligibility),
    // then image-based prompts, then non-word DDK stimuli, then unknowns.
    // Within each bucket preserve clip id order so the per-row ordinal
    // ("Clip 1, 2, …") stays stable on subsequent renders.
    const PROMPT_ORDER: Record<string, number> = {
      timit: 0,
      intelligibility: 1,
      image: 2,
      "non-word": 3,
      unknown: 4,
    };
    const targetSpeaker = activeSpeaker.toUpperCase();
    // The split filter only makes sense when an adapter actually exists
    // for this speaker. For demo (uncalibrated) speakers, no adapter has
    // been trained, so the per-clip cal/eval bucket is just a pre-computed
    // label that doesn't correspond to anything real — ignore the filter.
    const applySplit = profile.kind === "calibrated";
    // Demo-ranking sort key (lower = plays earlier). Ranked + unflagged
    // clips sort first by adapted_wer ascending; flagged ranked clips
    // sink to the bottom of the ranked block; unranked clips keep their
    // historical (prompt-set, id) order behind everything ranked.
    const rankKey = (c: TorgoClip): [number, number, number] => {
      const r = rankingIndex.get(c.id);
      if (!r) {
        return [
          1, // unranked → after every ranked clip
          PROMPT_ORDER[c.prompt_set ?? "unknown"] ?? 99,
          c.id,
        ];
      }
      // Ranked block split into two strata: unflagged (0) before flagged (1),
      // each internally sorted by adapted_wer ascending.
      return [0, r.adapted_flagged ? 1 : 0, r.adapted_wer];
    };
    return clips
      .filter((c) => {
        if (c.speaker_id.toUpperCase() !== targetSpeaker) return false;
        if (!showNonWord && c.prompt_set === "non-word") return false;
        if (setFilter !== "all" && c.prompt_set !== setFilter) return false;
        if (applySplit) {
          if (splitFilter === "train" && c.split !== "cal") return false;
          if (splitFilter === "test" && c.split !== "eval") return false;
        }
        return true;
      })
      .slice()
      .sort((a, b) => {
        const ka = rankKey(a);
        const kb = rankKey(b);
        for (let i = 0; i < ka.length; i++) {
          if (ka[i] !== kb[i]) return ka[i] - kb[i];
        }
        return 0;
      });
  }, [
    clips,
    activeSpeaker,
    showNonWord,
    setFilter,
    splitFilter,
    profile.kind,
    rankingIndex,
  ]);

  const playClipAudio = useCallback((clip: TorgoClip) => {
    const el = audioRef.current;
    if (!el) return;
    el.src = `${SERVER_HTTP}/torgo/audio/${clip.id}`;
    el.currentTime = 0;
    setPlayingId(clip.id);
    el.play().catch((err) => {
      console.warn("audio play failed", err);
      setPlayingId(null);
    });
  }, []);

  const stopAudio = useCallback(() => {
    const el = audioRef.current;
    if (!el) return;
    el.pause();
    el.currentTime = 0;
    setPlayingId(null);
  }, []);

  // Format a single incoming WS message into a one-line log entry.
  // Returns null for messages we don't want in the timeline (chatty
  // per-frame phoneme windows etc.).
  const formatLogEntry = useCallback(
    (msg: WsMessage, streamStartedAt: number): StreamLogEntry | null => {
      // Server-emitted diag events carry their own ts_ms; everything
      // else gets timestamped by client arrival relative to stream start.
      const elapsed = (msg as { ts_ms?: number }).ts_ms ?? Date.now() - streamStartedAt;
      switch (msg.type) {
        case "diag": {
          const data = msg.data ? " " + JSON.stringify(msg.data) : "";
          return { ts_ms: elapsed, level: msg.level, text: msg.msg + data };
        }
        case "utterance":
          return {
            ts_ms: elapsed,
            level: "info",
            text:
              `utterance opened #${msg.utterance_id} ` +
              `(speaker=${msg.speaker}, role=${msg.role ?? "—"}, ` +
              `is_patient=${msg.is_patient}, sim=${msg.speaker_similarity.toFixed(2)}, ` +
              `${msg.start_ms}–${msg.end_ms} ms)`,
          };
        case "utterance_rejected":
          return {
            ts_ms: elapsed,
            level: "warn",
            text:
              `utterance REJECTED ${msg.start_ms}–${msg.end_ms} ms ` +
              `(${msg.reason}, sim=${msg.speaker_similarity.toFixed(2)})`,
          };
        case "features": {
          const f = msg.features;
          return {
            ts_ms: elapsed,
            level: "info",
            text:
              `features: duration=${f.duration_ms}ms ` +
              `pause_count=${f.pause_count} pause_total=${f.pause_total_ms}ms ` +
              `jitter=${f.jitter.toFixed(3)} shimmer=${f.shimmer.toFixed(3)}`,
          };
        }
        case "translation": {
          const flag = msg.flagged
            ? `FLAGGED(${msg.reason ?? "unknown"})`
            : "ok";
          const align =
            msg.align_score !== undefined
              ? msg.align_score.toFixed(3)
              : "—";
          const density =
            msg.phoneme_density !== undefined
              ? msg.phoneme_density.toFixed(2)
              : "—";
          return {
            ts_ms: elapsed,
            level: msg.flagged ? "warn" : "info",
            text:
              `translation: "${msg.primary}" ` +
              `align=${align} density=${density} ` +
              `conf=${(msg.overall_confidence * 100).toFixed(0)}% ${flag}`,
          };
        }
        case "translation_error":
          return {
            ts_ms: elapsed,
            level: "error",
            text: `translation_error: ${msg.error} — ${msg.message}`,
          };
        case "emergency":
          return {
            ts_ms: elapsed,
            level: "warn",
            text:
              `emergency: "${msg.phrase}" ` +
              `(${msg.category}, conf=${(msg.confidence * 100).toFixed(0)}%)`,
          };
        case "context":
        case "utterance_transcript":
        case "phoneme_window":
          // Skip — too chatty for the diagnostic copy buffer.
          return null;
        default:
          return null;
      }
    },
    [],
  );

  // Append a log entry to whichever stream is currently active. Also
  // pick up the server's session_id from the first message that carries
  // one, so the copy buffer can label streams correctly.
  const appendToActiveLog = useCallback(
    (entry: StreamLogEntry, sessionId?: number) => {
      setStreamLogs((prev) => {
        if (prev.length === 0) return prev;
        const [head, ...rest] = prev;
        if (head.key !== activeLogKey.current) return prev;
        return [
          {
            ...head,
            sessionId: head.sessionId ?? sessionId ?? null,
            entries: [...head.entries, entry],
          },
          ...rest,
        ];
      });
    },
    [],
  );

  const streamClip = useCallback(
    async (clip: TorgoClip) => {
      // Clear any previous error so a retry doesn't keep the old message
      // pinned after success.
      setError(null);
      setStreamingId(clip.id);
      onStreamStart?.(clip);
      // Seed a new stream log entry. Subsequent messages from this WS
      // append to its head; older streams stay visible (capped) below.
      const streamKey =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `stream-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const startedAt = Date.now();
      const clipLabel =
        `${clip.speaker_id.toUpperCase()} clip #${clip.id}` +
        (clip.intended_text ? ` — “${clip.intended_text}”` : "");
      activeLogKey.current = streamKey;
      setStreamLogs((prev) =>
        [
          {
            key: streamKey,
            clipLabel,
            startedAt,
            sessionId: null,
            entries: [],
          },
          ...prev,
        ].slice(0, STREAM_LOG_CAP),
      );
      try {
        // Single fetch: previously playClipAudio() also kicked off its own
        // request via <audio>.src, racing this one — that double-request was
        // the source of "TypeError: failed to fetch" on first click. Now we
        // fetch once and feed both the audio element (via Blob URL) and the
        // streaming WS from the same buffer.
        //
        // cache: "no-store" — a no-CORS audio-element response from an
        // earlier run can sit in HTTP cache without usable CORS headers and
        // get served back to this CORS fetch, producing a spurious
        // "blocked by CORS policy" failure even though the live server
        // sends Access-Control-Allow-Origin correctly.
        const audioRes = await fetch(
          `${SERVER_HTTP}/torgo/audio/${clip.id}`,
          { cache: "no-store" },
        );
        if (!audioRes.ok) {
          const body = await audioRes.text().catch(() => "");
          throw new Error(
            `audio fetch failed (${audioRes.status} ${audioRes.statusText})${
              body ? `: ${body}` : ""
            }`,
          );
        }
        const buf = await audioRes.arrayBuffer();

        const el = audioRef.current;
        if (el) {
          const url = URL.createObjectURL(
            new Blob([buf], { type: "audio/wav" }),
          );
          if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
          blobUrlRef.current = url;
          el.src = url;
          el.currentTime = 0;
          setPlayingId(clip.id);
          el.play().catch((err) => {
            console.warn("audio play failed", err);
            setPlayingId(null);
          });
        }

        // Open a fresh dedicated WS with:
        //   force_patient=1   — route through patient pipeline regardless
        //                       of who's enrolled
        //   adapter_speaker=X — pin matching per-speaker LoRA for clips
        //                       belonging to a calibrated TORGO speaker
        //   bypass_vad=1      — skip silero VAD; treat the entire streamed
        //                       PCM as one utterance. Pre-recorded TORGO
        //                       clips are *by definition* a single intended
        //                       utterance, and silero VAD was producing
        //                       non-deterministic splits on dysarthric
        //                       audio (e.g. M01 "The islands are sparsely
        //                       populated" → "silent sun" when VAD
        //                       happened to capture only 1.7 s of the
        //                       10 s clip). Bypassing eliminates that
        //                       coin-flip entirely.
        const speakerForAdapter = clip.speaker_id.toUpperCase();
        const adapterParam = calibratedSpeakers.has(speakerForAdapter)
          ? `&adapter_speaker=${encodeURIComponent(speakerForAdapter)}`
          : "";
        const wsUrl =
          SERVER_HTTP.replace(/^http/, "ws") +
          `/ws/audio?force_patient=1&bypass_vad=1${adapterParam}`;
        // Single-shot: stream once, hard-close, never auto-reconnect.
        let didStream = false;
        let fallbackDisconnect: ReturnType<typeof setTimeout> | null = null;
        const clearFallback = () => {
          if (fallbackDisconnect) {
            clearTimeout(fallbackDisconnect);
            fallbackDisconnect = null;
          }
        };
        const ws = createPhonemerWs(wsUrl, {
          // Do NOT auto-disconnect on the first translation. A single
          // TORGO clip can produce *multiple* VAD-segmented utterances
          // (especially M01 with its >3 s intra-sentence pauses), and
          // tearing down after the first translation would silently
          // drop every subsequent fragment — making the same clip
          // produce wildly different results across runs depending
          // on where VAD happened to split it.
          //
          // Instead let the *server* drive the close: it emits its
          // final translation(s), then ``diag("stream closing", …)``,
          // then ``websocket.close()``. The client just rides along.
          // The 30 s fallback timer below is the safety net for cases
          // where the server hangs.
          onMessage: (msg) => {
            onMessage(msg);
            const entry = formatLogEntry(msg, startedAt);
            if (entry) {
              const sid = (msg as { session_id?: number }).session_id;
              appendToActiveLog(entry, sid);
            }
          },
          onState: (state) => {
            if (state === "open" && !didStream) {
              didStream = true;
              // realtime: true so the WS pace matches the audio playback —
              // user hears the clip and then sees the pipeline animation.
              streamFileAsPcm(buf, (chunk) => ws.sendPcm(chunk), {
                realtime: true,
              })
                .then(() => {
                  ws.sendControl({ type: "stop" });
                  // Fallback disconnect: 30 s after sending stop.
                  // Generous because the first utterance of a session
                  // pays a cold cost (adapter load 3 s + first decode
                  // 5 s + alignment ~2 s) and we'd rather wait too long
                  // than tear down before the translation lands.
                  fallbackDisconnect = setTimeout(
                    () => ws.disconnect(),
                    30_000,
                  );
                })
                .catch((err) => {
                  setError(String(err));
                  ws.disconnect();
                });
            }
            if (state === "closed") {
              clearFallback();
              setStreamingId(null);
              onStreamEnd?.();
              // Stop appending to this stream's log buffer. Any
              // messages that arrive after this point (rare — the
              // grace timeout above covers ordinary deliveries) won't
              // accidentally polute the next clip's log.
              if (activeLogKey.current === streamKey) {
                activeLogKey.current = null;
              }
            }
          },
          onError: (err) => {
            setError(err instanceof Error ? err.message : String(err));
          },
        });
        ws.connect();
      } catch (err) {
        // Distinguish a transport-layer fetch failure (server unreachable,
        // CORS, abort) from server-emitted errors so the message is useful.
        const isNetwork =
          err instanceof TypeError && /fetch/i.test(err.message);
        setError(
          isNetwork
            ? "could not reach the server — check that the FastAPI backend is running, then retry"
            : err instanceof Error
              ? err.message
              : String(err),
        );
        setStreamingId(null);
        onStreamEnd?.();
      }
    },
    [onStreamStart, onStreamEnd, onMessage, formatLogEntry, appendToActiveLog,
     calibratedSpeakers],
  );

  // Format the in-memory streamLogs as a single copy-able text block.
  // The text format intentionally uses '===' fences between streams so
  // a clinician (or future-me) can scan multiple runs without
  // ambiguity about where one stream ends and the next begins.
  const buildLogText = useCallback(() => {
    if (streamLogs.length === 0) return "";
    const blocks = streamLogs.map((log) => {
      const header =
        `=== Stream ${log.sessionId != null ? `session=${log.sessionId} ` : ""}` +
        `${log.clipLabel}\n` +
        `=== Started ${new Date(log.startedAt).toLocaleTimeString()} ` +
        `(${new Date(log.startedAt).toISOString()})\n`;
      const lines = log.entries
        .map((e) => {
          const ts = `+${e.ts_ms.toString().padStart(5)}ms`;
          const lv = e.level.toUpperCase().padEnd(5);
          return `${ts}  [${lv}] ${e.text}`;
        })
        .join("\n");
      return header + lines;
    });
    return blocks.join("\n\n");
  }, [streamLogs]);

  const [copied, setCopied] = useState(false);
  const onCopyLog = useCallback(async () => {
    const text = buildLogText();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignored — clipboard might be unavailable; user can still drag-select
    }
  }, [buildLogText]);

  if (!open) return null;

  const containerClass =
    variant === "inline"
      ? "h-full w-full flex flex-col bg-neutral-950/40 border-r border-white/10"
      : "fixed inset-y-0 right-0 w-full max-w-md bg-neutral-950/95 backdrop-blur-md border-l border-white/10 z-40 flex flex-col";

  const profileSubtitle =
    profile.kind === "calibrated"
      ? `Demo clips for ${profile.speakerId} (calibrated adapter active)`
      : "Demo clips · pick an uncalibrated TORGO speaker below";

  return (
    <div className={containerClass}>
      <header className="px-4 py-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold tracking-wide">Demo clips</h2>
          <button
            onClick={onClose}
            aria-label="Hide clips panel"
            title="hide"
            className="ml-auto w-6 h-6 flex items-center justify-center rounded text-xs opacity-60 hover:opacity-100 hover:bg-white/10 transition-colors"
          >
            ✕
          </button>
        </div>
        <div className="mt-0.5 text-[10px] opacity-50 leading-snug">
          {profileSubtitle}
          {clips && (
            <span className="ml-1 font-mono">
              · {speakerClips.length} clips
            </span>
          )}
        </div>
      </header>

      {error && (
        <div className="mx-3 mt-3 rounded-md bg-rose-900/40 px-3 py-2 text-rose-200 text-xs">
          {error}
        </div>
      )}

      {profile.kind === "calibrated" && (
        <div className="px-4 py-3 border-b border-white/10">
          <div className="flex items-center gap-2 text-[11px]">
            <span className="text-[10px] tracking-[0.25em] font-mono opacity-50 uppercase">
              speaker
            </span>
            <span className="font-mono text-emerald-200">
              {profile.speakerId}
            </span>
            {(() => {
              const sev =
                SPEAKER_SEVERITY[profile.speakerId.toUpperCase()];
              return sev ? <SeverityBadge severity={sev} /> : null;
            })()}
            <span className="ml-auto text-[10px] opacity-50 font-mono uppercase tracking-wider">
              adapter active
            </span>
          </div>
        </div>
      )}

      {profile.kind === "demo" && (
        <div className="px-4 py-3 border-b border-white/10">
          <div className="text-[10px] tracking-[0.25em] font-mono opacity-50 mb-1.5">
            UNCALIBRATED SPEAKER
          </div>
          <div className="flex flex-wrap gap-1">
            {demoSpeakers.length === 0 ? (
              <span className="opacity-50 text-xs">no uncalibrated speakers</span>
            ) : (
              demoSpeakers.map((s, idx) => {
                const severity = SPEAKER_SEVERITY[s];
                return (
                  <button
                    key={s}
                    onClick={() => setDemoSpeaker(s)}
                    title={`TORGO speaker ${s}${severity ? ` · ${severity} dysarthria` : ""}`}
                    className={`inline-flex items-center gap-1.5 text-[11px] px-2 py-1 rounded ${
                      s === activeSpeaker
                        ? "bg-amber-500/20 text-amber-100 border border-amber-400/40"
                        : "border border-white/10 hover:bg-white/5"
                    }`}
                  >
                    <span>Speaker {idx + 1}</span>
                    {severity && <SeverityBadge severity={severity} />}
                    <span className="font-mono text-[9px] opacity-50">{s}</span>
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}

      <div className="px-4 py-3 border-b border-white/10 space-y-2">
        <div className="text-[10px] tracking-[0.25em] font-mono opacity-50">
          PROMPT TYPE
        </div>
        <div className="flex gap-1">
          {(
            [
              ["all", "all"],
              ["timit", "sentences"],
              ["intelligibility", "single words"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setSetFilter(key)}
              className={`text-[11px] px-2 py-1 rounded ${
                setFilter === key
                  ? "border border-teal-400/60 bg-teal-400/10 text-teal-200"
                  : "border border-white/10 hover:bg-white/5"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-[10px] opacity-70">
          <input
            type="checkbox"
            checked={showNonWord}
            onChange={(e) => setShowNonWord(e.target.checked)}
            className="accent-teal-400"
          />
          show non-word DDK clips (papa, kakaka — not useful for translation)
        </label>

        {profile.kind === "calibrated" && (
          <>
            <div className="text-[10px] tracking-[0.25em] font-mono opacity-50 pt-1">
              ADAPTER SPLIT
            </div>
            <div className="flex gap-1">
              {(
                [
                  ["all", "all"],
                  ["train", "train (80%)"],
                  ["test", "test (20%)"],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setSplitFilter(key)}
                  title={
                    key === "train"
                      ? "Clips this speaker's adapter trained on (cal split)"
                      : key === "test"
                        ? "Held-out clips the adapter never saw during training (eval split)"
                        : "Both splits"
                  }
                  className={`text-[11px] px-2 py-1 rounded ${
                    splitFilter === key
                      ? "border border-teal-400/60 bg-teal-400/10 text-teal-200"
                      : "border border-white/10 hover:bg-white/5"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2">
        <div className="text-[10px] tracking-[0.25em] font-mono opacity-50 px-2 mb-1">
          CLIPS ({speakerClips.length})
        </div>
        {speakerClips.length === 0 && clips && (
          <div className="px-2 py-6 text-center text-xs opacity-50 leading-snug">
            no clips match the current filters
            <div className="mt-1 text-[10px] opacity-70">
              try toggling the prompt-type or split filters above
            </div>
          </div>
        )}
        {speakerClips.slice(0, 200).map((c, idx) => {
          const isStreaming = c.id === streamingId;
          const isPlaying = c.id === playingId;
          const promptLabel =
            c.prompt_set && c.prompt_set !== "unknown"
              ? (PROMPT_SET_LABELS[c.prompt_set as PromptSet] ?? c.prompt_set)
              : null;
          const promptColor =
            c.prompt_set === "timit"
              ? "text-emerald-300/80"
              : c.prompt_set === "non-word"
                ? "text-rose-300/80"
                : "opacity-70";
          return (
            <div
              key={c.id}
              className={`flex items-stretch rounded-md transition-colors ${
                isStreaming ? "bg-teal-500/20" : "hover:bg-white/5"
              }`}
            >
              <button
                aria-label={isPlaying ? "Stop audio" : "Play audio"}
                onClick={(e) => {
                  e.stopPropagation();
                  if (isPlaying) stopAudio();
                  else playClipAudio(c);
                }}
                disabled={streamingId !== null && !isStreaming}
                className={`flex items-center justify-center w-7 shrink-0 rounded-l text-[11px] transition-colors ${
                  isPlaying
                    ? "text-rose-300 bg-rose-500/10"
                    : "opacity-60 hover:opacity-100 hover:bg-white/5 disabled:opacity-30"
                }`}
                title={isPlaying ? "stop audio" : "play audio"}
              >
                {isPlaying ? "■" : "▶"}
              </button>
              <button
                onClick={() => streamClip(c)}
                disabled={streamingId !== null}
                title={`TORGO clip #${c.id}`}
                className={`flex-1 text-left px-2 py-1.5 rounded-r transition-colors min-w-0 ${
                  isStreaming ? "text-teal-100" : "disabled:opacity-50"
                }`}
              >
                <div className="text-[13px] leading-snug truncate opacity-95 mb-0.5">
                  {c.intended_text || "—"}
                </div>
                <div className="flex items-center gap-1.5 text-[10px] font-mono opacity-50">
                  <span>Clip {idx + 1}</span>
                  <span aria-hidden>·</span>
                  <span>{(c.duration_ms / 1000).toFixed(1)}s</span>
                  {promptLabel && (
                    <>
                      <span aria-hidden>·</span>
                      <span className={`uppercase tracking-wide ${promptColor}`}>
                        {promptLabel}
                      </span>
                    </>
                  )}
                  {c.split && profile.kind === "calibrated" && (
                    <>
                      <span aria-hidden>·</span>
                      <span
                        className={`uppercase tracking-wide ${
                          c.split === "cal"
                            ? "text-sky-300/80"
                            : "text-violet-300/80"
                        }`}
                        title={
                          c.split === "cal"
                            ? "This speaker's adapter trained on this clip"
                            : "Held out — this speaker's adapter never saw this clip"
                        }
                      >
                        {c.split === "cal" ? "train" : "test"}
                      </span>
                    </>
                  )}
                  {isStreaming && (
                    <>
                      <span aria-hidden>·</span>
                      <span className="text-teal-300">streaming…</span>
                    </>
                  )}
                </div>
              </button>
            </div>
          );
        })}
        {speakerClips.length > 200 && (
          <div className="text-[10px] opacity-50 px-2 pt-1">
            …and {speakerClips.length - 200} more
          </div>
        )}
      </div>

      <audio
        ref={audioRef}
        preload="none"
        onEnded={() => setPlayingId(null)}
        onPause={() => {
          if (audioRef.current?.ended) setPlayingId(null);
        }}
        className="hidden"
      />

      {/* Per-stream diagnostic log. Collects the server's ``diag`` events
          plus formatted entries from the standard pipeline messages
          (utterance / features / translation / rejected). Each clip
          stream is delimited so the copy buffer keeps multiple runs
          distinct. */}
      <div className="border-t border-white/10 bg-neutral-950/50">
        <div className="flex items-center justify-between px-3 py-2">
          <button
            onClick={() => setShowLogs((v) => !v)}
            className="text-[10px] uppercase tracking-[0.2em] font-mono text-neutral-400 hover:text-neutral-200 transition-colors"
          >
            {showLogs ? "▾" : "▸"} stream log
            {streamLogs.length > 0 && (
              <span className="ml-2 opacity-60">({streamLogs.length})</span>
            )}
          </button>
          <div className="flex items-center gap-2">
            {streamLogs.length > 0 && (
              <>
                <button
                  onClick={() => {
                    setStreamLogs([]);
                    activeLogKey.current = null;
                  }}
                  className="text-[10px] font-mono text-neutral-500 hover:text-neutral-300 transition-colors"
                >
                  clear
                </button>
                <button
                  onClick={onCopyLog}
                  className={`text-[10px] font-mono px-2 py-1 rounded-md transition-colors ${
                    copied
                      ? "bg-emerald-700/50 text-emerald-200"
                      : "bg-neutral-800 text-neutral-200 hover:bg-neutral-700"
                  }`}
                >
                  {copied ? "copied ✓" : "copy"}
                </button>
              </>
            )}
          </div>
        </div>
        {showLogs && (
          <div className="px-3 pb-3 max-h-64 overflow-y-auto font-mono text-[10px] leading-snug">
            {streamLogs.length === 0 ? (
              <div className="text-neutral-600 italic">
                No streams yet — click a clip to start one. Each run will be
                recorded here with a separator so consecutive runs stay
                distinct in the copy buffer.
              </div>
            ) : (
              streamLogs.map((log, i) => (
                <div
                  key={log.key}
                  className={i === 0 ? "" : "mt-3 pt-2 border-t border-white/5"}
                >
                  <div className="text-neutral-300 truncate mb-1">
                    <span className="opacity-50">
                      {log.sessionId != null
                        ? `session ${log.sessionId} · `
                        : ""}
                    </span>
                    {log.clipLabel}
                  </div>
                  {log.entries.length === 0 ? (
                    <div className="text-neutral-600 italic">
                      (waiting for first event…)
                    </div>
                  ) : (
                    log.entries.map((e, j) => (
                      <div
                        key={j}
                        className={
                          e.level === "error"
                            ? "text-red-300"
                            : e.level === "warn"
                              ? "text-amber-300"
                              : "text-neutral-400"
                        }
                      >
                        <span className="opacity-60">
                          +{e.ts_ms.toString().padStart(5)}ms
                        </span>{" "}
                        {e.text}
                      </div>
                    ))
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <div className="px-4 py-2 border-t border-white/10 text-[10px] opacity-60 leading-relaxed">
        <span className="opacity-90">▶</span> plays the clip locally · clicking
        the row plays + streams it through the pipeline (
        <code className="opacity-90">force_patient=1</code>) so the translation
        and animation appear in the main view.
      </div>
    </div>
  );
}

