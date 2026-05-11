"use client";

import type { ControlMessage, WsMessage } from "./messages";
import { ConnectionState, createPhonemerWs } from "./phonemer-ws";

export type ConnectionHealth = {
  state: ConnectionState;
  reconnectAttempt: number;
  queuedChunks: number;
  droppedChunks: number;        // queue overflow drops since last open
  lastMessageAgeMs: number | null;
};

export type ConnectionHandlers = {
  onMessage: (msg: WsMessage) => void;
  onHealth?: (health: ConnectionHealth) => void;
  onError?: (err: Error | Event) => void;
};

export type ConnectionManager = {
  connect: () => void;
  disconnect: () => void;
  sendPcm: (chunk: Int16Array) => void;
  sendControl: (msg: ControlMessage) => void;
  health: () => ConnectionHealth;
};

const QUEUE_CAPACITY = 256; // ~25 s of 100 ms chunks
const HEALTH_TICK_MS = 500;

export function createConnectionManager(
  url: string,
  handlers: ConnectionHandlers,
  opts: { queueCapacity?: number } = {},
): ConnectionManager {
  const capacity = opts.queueCapacity ?? QUEUE_CAPACITY;

  // Bounded queue: oldest dropped on overflow so live audio takes priority.
  const queue: Int16Array[] = [];
  let droppedChunks = 0;
  let reconnectAttempt = 0;
  let lastMessageAt: number | null = null;

  const ws = createPhonemerWs(url, {
    onMessage: (msg) => {
      lastMessageAt = performance.now();
      handlers.onMessage(msg);
    },
    onState: (state) => {
      if (state === "open") {
        // Drain whatever's queued on reconnect.
        while (queue.length) {
          const chunk = queue.shift()!;
          ws.sendPcm(chunk);
        }
        reconnectAttempt = 0;
      } else if (state === "connecting" || state === "error") {
        reconnectAttempt += 1;
      }
      pushHealth();
    },
    onError: (err) => handlers.onError?.(err),
  });

  const pushHealth = () => {
    const now = performance.now();
    handlers.onHealth?.({
      state: ws.state(),
      reconnectAttempt,
      queuedChunks: queue.length,
      droppedChunks,
      lastMessageAgeMs: lastMessageAt === null ? null : Math.round(now - lastMessageAt),
    });
  };

  const tick = setInterval(pushHealth, HEALTH_TICK_MS);
  // Best-effort cleanup if the page is torn down without explicit disconnect.
  if (typeof window !== "undefined") {
    window.addEventListener("beforeunload", () => clearInterval(tick));
  }

  return {
    connect: () => {
      droppedChunks = 0;
      ws.connect();
    },
    disconnect: () => {
      ws.disconnect();
      queue.length = 0;
      lastMessageAt = null;
      reconnectAttempt = 0;
      pushHealth();
    },
    sendPcm: (chunk: Int16Array) => {
      if (ws.state() === "open") {
        ws.sendPcm(chunk);
        return;
      }
      // Buffer until we reconnect; drop oldest on overflow.
      if (queue.length >= capacity) {
        queue.shift();
        droppedChunks += 1;
      }
      queue.push(chunk);
    },
    sendControl: (msg: ControlMessage) => ws.sendControl(msg),
    health: () => ({
      state: ws.state(),
      reconnectAttempt,
      queuedChunks: queue.length,
      droppedChunks,
      lastMessageAgeMs:
        lastMessageAt === null
          ? null
          : Math.round(performance.now() - lastMessageAt),
    }),
  };
}
