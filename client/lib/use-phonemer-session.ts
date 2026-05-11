"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createAudioCapture, streamFileAsPcm } from "./audio-capture";
import {
  ConnectionHealth,
  createConnectionManager,
} from "./connection-manager";
import type { ConnectionState } from "./phonemer-ws";
import type { MicState } from "../components/mic-button";
import type { WsMessage } from "./messages";

export type SessionOptions = {
  url: string;
  onMessage: (msg: WsMessage) => void;
  onError?: (err: Error | Event) => void;
};

export type SessionApi = {
  micState: MicState;
  connState: ConnectionState;
  health: ConnectionHealth;
  level: number;
  errorMessage?: string;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  sendStop: () => void;
  // Stream a WAV buffer through the same connection (TORGO clips, dropped files).
  streamBuffer: (buf: ArrayBuffer) => Promise<void>;
};

const INITIAL_HEALTH: ConnectionHealth = {
  state: "idle",
  reconnectAttempt: 0,
  queuedChunks: 0,
  droppedChunks: 0,
  lastMessageAgeMs: null,
};

export function usePhonemerSession(opts: SessionOptions): SessionApi {
  const { url, onMessage, onError } = opts;
  const [micState, setMicState] = useState<MicState>("idle");
  const [level, setLevel] = useState(0);
  const [health, setHealth] = useState<ConnectionHealth>(INITIAL_HEALTH);
  const [errorMessage, setErrorMessage] = useState<string | undefined>();

  const captureRef = useRef<ReturnType<typeof createAudioCapture> | null>(null);
  const connRef = useRef<ReturnType<typeof createConnectionManager> | null>(
    null,
  );
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  // Lazily build the connection manager on first interaction; tear down on unmount.
  const ensureConn = useCallback(() => {
    if (connRef.current) return connRef.current;
    connRef.current = createConnectionManager(url, {
      onMessage: (m) => onMessageRef.current(m),
      onHealth: setHealth,
      onError: (err) => {
        onError?.(err);
        setErrorMessage(
          err instanceof Error ? err.message : "WebSocket transport error",
        );
      },
    });
    connRef.current.connect();
    return connRef.current;
  }, [url, onError]);

  useEffect(() => {
    return () => {
      captureRef.current?.stop();
      connRef.current?.disconnect();
      captureRef.current = null;
      connRef.current = null;
    };
  }, []);

  const start = useCallback(async () => {
    setErrorMessage(undefined);
    setMicState("connecting");
    const conn = ensureConn();
    if (captureRef.current?.isRunning()) {
      setMicState("recording");
      return;
    }
    const cap = createAudioCapture({
      onPcm: (chunk) => conn.sendPcm(chunk),
      onLevel: setLevel,
      onError: (err) => {
        setErrorMessage(err.message || String(err));
        setMicState(
          err.name === "NotAllowedError" || err.name === "SecurityError"
            ? "blocked"
            : "error",
        );
      },
    });
    captureRef.current = cap;
    try {
      await cap.start();
      setMicState("recording");
    } catch (err) {
      // onError handler above already updated state; fall back if not.
      setMicState((s) => (s === "recording" ? "error" : s));
    }
  }, [ensureConn]);

  const stop = useCallback(async () => {
    await captureRef.current?.stop();
    captureRef.current = null;
    setLevel(0);
    setMicState("idle");
    connRef.current?.sendControl({ type: "stop" });
  }, []);

  const streamBuffer = useCallback(
    async (buf: ArrayBuffer) => {
      const conn = ensureConn();
      // Reconnect to start a fresh session, then wait briefly for the WS to open.
      conn.disconnect();
      conn.connect();
      const start = performance.now();
      while (
        conn.health().state !== "open" &&
        performance.now() - start < 2000
      ) {
        await new Promise((r) => setTimeout(r, 50));
      }
      await streamFileAsPcm(buf, (chunk) => conn.sendPcm(chunk));
      conn.sendControl({ type: "stop" });
    },
    [ensureConn],
  );

  const sendStop = useCallback(() => {
    connRef.current?.sendControl({ type: "stop" });
  }, []);

  return {
    micState,
    connState: health.state,
    health,
    level,
    errorMessage,
    start,
    stop,
    sendStop,
    streamBuffer,
  };
}
