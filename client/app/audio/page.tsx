"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MicButton } from "../../components/mic-button";
import {
  listInputDevices,
  queryMicPermission,
} from "../../lib/audio-devices";
import { usePhonemerSession } from "../../lib/use-phonemer-session";
import type { WsMessage } from "../../lib/messages";

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";
const WS_URL = SERVER_HTTP.replace(/^http/, "ws") + "/ws/audio";

export default function AudioDiagnosticsPage() {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [perm, setPerm] = useState<string>("unknown");
  const [history, setHistory] = useState<number[]>([]);
  const [recentMessages, setRecentMessages] = useState<WsMessage[]>([]);
  const historyRef = useRef<number[]>([]);

  const session = usePhonemerSession({
    url: WS_URL,
    onMessage: (msg) => {
      setRecentMessages((prev) => [msg, ...prev].slice(0, 12));
    },
  });

  // Pump the session's level value into a rolling history for the graph.
  useEffect(() => {
    historyRef.current.push(session.level);
    if (historyRef.current.length > 240) historyRef.current.shift();
    setHistory([...historyRef.current]);
  }, [session.level]);

  useEffect(() => {
    queryMicPermission().then((s) => setPerm(String(s)));
    listInputDevices().then(setDevices);
  }, []);

  const refreshDevices = useCallback(async () => {
    const [p, d] = await Promise.all([queryMicPermission(), listInputDevices()]);
    setPerm(String(p));
    setDevices(d);
  }, []);

  const constraints = {
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  };

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 font-mono p-6 space-y-5">
      <header className="flex items-center gap-3 flex-wrap">
        <h1 className="text-xl font-semibold tracking-wide">
          Phonemer · Audio diagnostics
        </h1>
        <a
          href="/debug"
          className="text-xs px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700"
        >
          ← Debug
        </a>
        <span className="ml-auto text-xs opacity-60">
          permission: <span className="text-neutral-100">{perm}</span> ·{" "}
          devices: {devices.length}
        </span>
      </header>

      <section className="border border-neutral-800 rounded p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Capture</h2>
          <span className="text-[11px] opacity-60">
            ws state: <span className="text-neutral-100">{session.health.state}</span>
            {" · "}queued: {session.health.queuedChunks}
            {" · "}dropped: {session.health.droppedChunks}
            {session.health.lastMessageAgeMs !== null && (
              <> {" · "}last msg: {session.health.lastMessageAgeMs} ms ago</>
            )}
          </span>
        </div>
        <MicButton
          state={session.micState}
          level={session.level}
          errorMessage={session.errorMessage}
          onStart={session.start}
          onStop={session.stop}
        />
        <LevelHistory history={history} />
      </section>

      <section className="border border-neutral-800 rounded p-4 space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Input devices</h2>
          <button
            onClick={refreshDevices}
            className="text-xs px-2 py-0.5 rounded bg-neutral-800 hover:bg-neutral-700"
          >
            refresh
          </button>
        </div>
        {devices.length === 0 ? (
          <div className="opacity-60 text-xs">
            (No devices listed. Browsers hide labels until permission is granted —
            click "Start mic" once, then refresh.)
          </div>
        ) : (
          <ul className="space-y-1 text-xs">
            {devices.map((d) => (
              <li
                key={d.deviceId}
                className="flex items-center gap-2 bg-neutral-900 rounded px-2 py-1"
              >
                <span className="opacity-50">[{d.deviceId.slice(0, 6)}]</span>
                <span>{d.label || "(unlabelled — grant permission for names)"}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="border border-neutral-800 rounded p-4 space-y-2">
        <h2 className="text-sm font-semibold">getUserMedia constraints</h2>
        <pre className="bg-neutral-900 rounded p-3 text-[11px] leading-tight overflow-auto">
{JSON.stringify(constraints, null, 2)}
        </pre>
        <div className="text-[11px] opacity-60 leading-relaxed">
          Browser DSP is intentionally disabled. Noise suppression and AGC
          remove the distortion signals the model is trained to detect, so we
          ask for raw PCM. The AudioWorklet at <code>/pcm-worklet.js</code>{" "}
          downsamples to 16 kHz Int16 with a linear resampler and posts ~100 ms
          chunks to the main thread.
        </div>
      </section>

      <section className="border border-neutral-800 rounded p-4 space-y-2">
        <h2 className="text-sm font-semibold">Recent server messages</h2>
        {recentMessages.length === 0 ? (
          <div className="opacity-60 text-xs">no messages yet</div>
        ) : (
          <ul className="text-[11px] space-y-1 max-h-[40vh] overflow-auto">
            {recentMessages.map((m, i) => (
              <li key={i} className="bg-neutral-900 rounded px-2 py-1">
                <span className="opacity-60">{m.type}</span>{" "}
                <span className="opacity-80">{summarizeForDiag(m)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

function summarizeForDiag(m: WsMessage): string {
  switch (m.type) {
    case "utterance":
      return `${m.speaker} ${m.start_ms}-${m.end_ms}ms`;
    case "phoneme_window":
      return `${m.frames.length} frames`;
    case "features":
      return `pitch=${m.features.pitch_hz_mean.toFixed(0)}Hz`;
    case "translation":
      return m.primary;
    case "utterance_transcript":
      return m.transcript || "(empty)";
    case "utterance_rejected":
      return `sim=${m.speaker_similarity.toFixed(2)} ${m.reason}`;
    case "context":
      return `topics=${m.context.topics.length}`;
    case "emergency":
      return `⚠ ${m.phrase ?? m.category ?? "alert"}`;
    case "translation_error":
      return `LLM error: ${m.message}`;
  }
}

function LevelHistory({ history }: { history: number[] }) {
  // Sparkline. Most recent value on the right.
  const max = 0.5;
  const points = history.length || 1;
  const w = 600;
  const h = 60;
  const path = history
    .map((v, i) => {
      const x = (i / Math.max(1, points - 1)) * w;
      const y = h - Math.min(1, v / max) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full h-14 bg-neutral-900 rounded"
      preserveAspectRatio="none"
    >
      <path
        d={path || `M0 ${h} L${w} ${h}`}
        stroke="#10b981"
        strokeWidth="1.4"
        fill="none"
      />
    </svg>
  );
}
