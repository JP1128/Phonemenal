"use client";

import type { WsMessage } from "./messages";
import type { SessionRecording } from "./recording-types";

export type PlayerHandlers = {
  onMessage: (msg: WsMessage) => void;
  onProgress?: (currentMs: number, totalMs: number) => void;
  onEnded?: () => void;
};

export type RecordingPlayer = {
  load: (rec: SessionRecording) => void;
  play: () => void;
  pause: () => void;
  stop: () => void;
  setSpeed: (speed: number) => void;
  seek: (toMs: number) => void;
  isPlaying: () => boolean;
  current: () => { currentMs: number; totalMs: number };
};

export function createRecordingPlayer(handlers: PlayerHandlers): RecordingPlayer {
  let recording: SessionRecording | null = null;
  let nextIndex = 0;
  let speed = 1;
  let playing = false;
  let virtualMs = 0; // current playhead in recording time
  let lastWallTickMs = 0;
  let raf: number | null = null;

  const tick = () => {
    if (!playing || !recording) return;
    const now = performance.now();
    const elapsed = (now - lastWallTickMs) * speed;
    lastWallTickMs = now;
    virtualMs += elapsed;

    while (
      nextIndex < recording.events.length &&
      recording.events[nextIndex].t_ms <= virtualMs
    ) {
      handlers.onMessage(recording.events[nextIndex].payload);
      nextIndex += 1;
    }

    handlers.onProgress?.(virtualMs, recording.duration_ms);

    if (
      virtualMs >= recording.duration_ms &&
      nextIndex >= recording.events.length
    ) {
      playing = false;
      handlers.onEnded?.();
      return;
    }
    raf = requestAnimationFrame(tick);
  };

  return {
    load: (rec) => {
      recording = rec;
      nextIndex = 0;
      virtualMs = 0;
      playing = false;
      handlers.onProgress?.(0, rec.duration_ms);
    },
    play: () => {
      if (!recording || playing) return;
      playing = true;
      lastWallTickMs = performance.now();
      raf = requestAnimationFrame(tick);
    },
    pause: () => {
      playing = false;
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
    },
    stop: () => {
      playing = false;
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null;
      nextIndex = 0;
      virtualMs = 0;
      if (recording) handlers.onProgress?.(0, recording.duration_ms);
    },
    setSpeed: (s) => {
      speed = Math.max(0.1, s);
    },
    seek: (toMs) => {
      if (!recording) return;
      virtualMs = Math.max(0, Math.min(recording.duration_ms, toMs));
      // Move event cursor to the first event >= virtualMs without firing
      // skipped events (so scrubbing forward doesn't dump 1k messages).
      nextIndex = 0;
      while (
        nextIndex < recording.events.length &&
        recording.events[nextIndex].t_ms < virtualMs
      ) {
        nextIndex += 1;
      }
      handlers.onProgress?.(virtualMs, recording.duration_ms);
    },
    isPlaying: () => playing,
    current: () => ({
      currentMs: virtualMs,
      totalMs: recording?.duration_ms ?? 0,
    }),
  };
}
