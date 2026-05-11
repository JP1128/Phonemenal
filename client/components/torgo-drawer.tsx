"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { streamFileAsPcm } from "../lib/audio-capture";
import type { WsMessage } from "../lib/messages";
import { createPhonemerWs } from "../lib/phonemer-ws";

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
};

const GROUPS = [
  { key: "F_Con", label: "F · Control", gender: "F", condition: "control" },
  { key: "F_Dys", label: "F · Dys", gender: "F", condition: "dysarthria" },
  { key: "M_Con", label: "M · Control", gender: "M", condition: "control" },
  { key: "M_Dys", label: "M · Dys", gender: "M", condition: "dysarthria" },
] as const;

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
};

export function TorgoDrawer({
  open,
  onClose,
  onMessage,
  onStreamStart,
  onStreamEnd,
  variant = "overlay",
}: TorgoDrawerProps) {
  const [clips, setClips] = useState<TorgoClip[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [groupKey, setGroupKey] = useState<string>("F_Dys");
  const [speaker, setSpeaker] = useState<string | null>(null);
  const [streamingId, setStreamingId] = useState<number | null>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);
  const [showNonWord, setShowNonWord] = useState(false);
  const [setFilter, setSetFilter] = useState<"all" | "timit" | "intelligibility">(
    "all",
  );
  const audioRef = useRef<HTMLAudioElement | null>(null);

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

  const groupClips = useMemo(() => {
    if (!clips) return [];
    const grp = GROUPS.find((g) => g.key === groupKey);
    if (!grp) return [];
    return clips.filter(
      (c) =>
        c.condition === grp.condition &&
        c.speaker_id.toUpperCase().startsWith(grp.gender) &&
        (showNonWord || c.prompt_set !== "non-word") &&
        (setFilter === "all" || c.prompt_set === setFilter),
    );
  }, [clips, groupKey, showNonWord, setFilter]);

  const speakers = useMemo(() => {
    const set = new Set<string>();
    groupClips.forEach((c) => set.add(c.speaker_id));
    return Array.from(set).sort();
  }, [groupClips]);

  useEffect(() => {
    if (speakers.length > 0 && (!speaker || !speakers.includes(speaker))) {
      setSpeaker(speakers[0]);
    }
  }, [speakers, speaker]);

  const speakerClips = useMemo(() => {
    if (!speaker) return [];
    return groupClips.filter((c) => c.speaker_id === speaker);
  }, [groupClips, speaker]);

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

  const streamClip = useCallback(
    async (clip: TorgoClip) => {
      setStreamingId(clip.id);
      onStreamStart?.(clip);
      playClipAudio(clip);
      try {
        const audioRes = await fetch(`${SERVER_HTTP}/torgo/audio/${clip.id}`);
        if (!audioRes.ok) throw new Error(await audioRes.text());
        const buf = await audioRes.arrayBuffer();

        // Open a fresh dedicated WS with force_patient=1 so results route
        // through the LLM pipeline regardless of who's enrolled.
        const wsUrl =
          SERVER_HTTP.replace(/^http/, "ws") + "/ws/audio?force_patient=1";
        // Single-shot: stream once, hard-close, never auto-reconnect.
        let didStream = false;
        const ws = createPhonemerWs(wsUrl, {
          onMessage,
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
                  setTimeout(() => ws.disconnect(), 6000);
                })
                .catch((err) => {
                  setError(String(err));
                  ws.disconnect();
                });
            }
            if (state === "closed") {
              setStreamingId(null);
              onStreamEnd?.();
            }
          },
          onError: (err) => {
            setError(err instanceof Error ? err.message : String(err));
          },
        });
        ws.connect();
      } catch (err) {
        setError(String(err));
        setStreamingId(null);
        onStreamEnd?.();
      }
    },
    [onStreamStart, onStreamEnd, onMessage, playClipAudio],
  );

  if (!open) return null;

  const containerClass =
    variant === "inline"
      ? "h-full w-full flex flex-col bg-neutral-950/40 border-r border-white/10"
      : "fixed inset-y-0 right-0 w-full max-w-md bg-neutral-950/95 backdrop-blur-md border-l border-white/10 z-40 flex flex-col";

  return (
    <div className={containerClass}>
      <header className="px-4 py-3 border-b border-white/10 flex items-center gap-2">
        <h2 className="text-sm font-semibold tracking-wide">TORGO</h2>
        <span className="text-[10px] opacity-50 font-mono">
          {clips
            ? `${groupClips.length} / ${clips.length} clips`
            : "… clips"}
        </span>
        <button
          onClick={onClose}
          aria-label="Hide TORGO panel"
          title="hide"
          className="ml-auto w-6 h-6 flex items-center justify-center rounded text-xs opacity-60 hover:opacity-100 hover:bg-white/10 transition-colors"
        >
          ✕
        </button>
      </header>

      {error && (
        <div className="mx-3 mt-3 rounded bg-rose-900/40 px-3 py-2 text-rose-200 text-xs">
          {error}
        </div>
      )}

      <div className="px-4 py-3 border-b border-white/10">
        <div className="grid grid-cols-2 gap-1">
          {GROUPS.map((g) => {
            const active = g.key === groupKey;
            return (
              <button
                key={g.key}
                onClick={() => {
                  setGroupKey(g.key);
                  setSpeaker(null);
                }}
                className={`text-[11px] px-2 py-1.5 rounded ${
                  active
                    ? "border border-teal-400/60 bg-teal-400/10 text-teal-200"
                    : "border border-white/10 hover:bg-white/5"
                }`}
              >
                {g.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="px-4 py-3 border-b border-white/10">
        <div className="text-[9px] tracking-[0.25em] font-mono opacity-50 mb-1.5">
          SPEAKER
        </div>
        <div className="flex flex-wrap gap-1">
          {speakers.length === 0 ? (
            <span className="opacity-50 text-xs">no clips</span>
          ) : (
            speakers.map((s) => (
              <button
                key={s}
                onClick={() => setSpeaker(s)}
                className={`text-[11px] px-2 py-1 rounded font-mono ${
                  s === speaker
                    ? "bg-teal-500/20 text-teal-200 border border-teal-400/40"
                    : "border border-white/10 hover:bg-white/5"
                }`}
              >
                {s}
              </button>
            ))
          )}
        </div>
      </div>

      <div className="px-4 py-3 border-b border-white/10 space-y-2">
        <div className="text-[9px] tracking-[0.25em] font-mono opacity-50">
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
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2">
        <div className="text-[9px] tracking-[0.25em] font-mono opacity-50 px-2 mb-1">
          CLIPS ({speakerClips.length})
        </div>
        {speakerClips.slice(0, 200).map((c) => {
          const isStreaming = c.id === streamingId;
          const isPlaying = c.id === playingId;
          return (
            <div
              key={c.id}
              className={`flex items-stretch rounded font-mono transition-colors ${
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
                className={`flex-1 text-left text-xs px-2 py-1.5 rounded-r transition-colors min-w-0 ${
                  isStreaming
                    ? "text-teal-100"
                    : "disabled:opacity-50"
                }`}
              >
                <div className="flex items-center gap-2 text-[10px] opacity-60 mb-0.5">
                  <span>#{c.id}</span>
                  <span>{(c.duration_ms / 1000).toFixed(1)}s</span>
                  {c.prompt_set && c.prompt_set !== "unknown" && (
                    <span
                      className={`uppercase tracking-wide ${
                        c.prompt_set === "timit"
                          ? "text-emerald-300/80"
                          : c.prompt_set === "non-word"
                            ? "text-rose-300/80"
                            : "opacity-70"
                      }`}
                    >
                      {PROMPT_SET_LABELS[c.prompt_set as PromptSet] ??
                        c.prompt_set}
                    </span>
                  )}
                  {isStreaming && (
                    <span className="text-teal-300">streaming…</span>
                  )}
                </div>
                <div className="text-[12px] leading-snug truncate font-sans opacity-90">
                  {c.intended_text || "—"}
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

      <div className="px-4 py-2 border-t border-white/10 text-[10px] opacity-60 leading-relaxed">
        <span className="opacity-90">▶</span> plays the clip locally · clicking
        the row plays + streams it through the pipeline (
        <code className="opacity-90">force_patient=1</code>) so the translation
        and animation appear in the main view.
      </div>
    </div>
  );
}
