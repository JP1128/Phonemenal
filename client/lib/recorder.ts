"use client";

import type { WsMessage } from "./messages";
import type {
  PcmChunk,
  RecordedEvent,
  SessionRecording,
} from "./recording-types";

export type RecorderHandle = {
  start: (opts: { source: SessionRecording["source"]; label?: string; includeAudio?: boolean }) => void;
  stop: () => void;
  observeMessage: (msg: WsMessage) => void;
  observePcm: (chunk: Int16Array) => void;
  isRecording: () => boolean;
  snapshot: () => SessionRecording | null;
};

export function createRecorder(): RecorderHandle {
  let state: {
    started: number;
    source: SessionRecording["source"];
    label?: string;
    sessionId: number | null;
    includeAudio: boolean;
    events: RecordedEvent[];
    pcm: PcmChunk[];
  } | null = null;

  return {
    start: (opts) => {
      state = {
        started: performance.now(),
        source: opts.source,
        label: opts.label,
        sessionId: null,
        includeAudio: !!opts.includeAudio,
        events: [],
        pcm: [],
      };
    },

    stop: () => {
      state = null;
    },

    observeMessage: (msg) => {
      if (!state) return;
      if (msg.type === "utterance" && state.sessionId === null) {
        state.sessionId = msg.session_id;
      }
      state.events.push({
        t_ms: performance.now() - state.started,
        payload: msg,
      });
    },

    observePcm: (chunk) => {
      if (!state || !state.includeAudio) return;
      state.pcm.push({
        t_ms: performance.now() - state.started,
        data_b64: int16ToBase64(chunk),
      });
    },

    isRecording: () => state !== null,

    snapshot: () => {
      if (!state) return null;
      return {
        version: 1,
        started_at: new Date(
          Date.now() - (performance.now() - state.started),
        ).toISOString(),
        session_id: state.sessionId,
        source: state.source,
        source_label: state.label,
        duration_ms: Math.round(performance.now() - state.started),
        events: [...state.events],
        pcm: state.includeAudio ? [...state.pcm] : undefined,
      };
    },
  };
}

export function downloadRecording(rec: SessionRecording): void {
  const ts = rec.started_at.replace(/[:.]/g, "-");
  const sess = rec.session_id ?? "unknown";
  const blob = new Blob([JSON.stringify(rec)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `phonemer-session-${sess}-${ts}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function int16ToBase64(chunk: Int16Array): string {
  const bytes = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(
      null,
      Array.from(bytes.subarray(i, i + CHUNK)),
    );
  }
  return btoa(binary);
}
