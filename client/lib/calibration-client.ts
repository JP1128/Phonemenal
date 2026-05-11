"use client";

import type {
  CalibrationServerMessage,
  CalibrationBeginMessage,
} from "./messages";

export type CalibrationClient = {
  connect: () => void;
  disconnect: () => void;
  begin: (msg: Omit<CalibrationBeginMessage, "type">) => void;
  end: () => void;
  discard: () => void;
  sendPcm: (chunk: Int16Array) => void;
};

export type CalibrationHandlers = {
  onMessage: (msg: CalibrationServerMessage) => void;
  onState?: (state: "open" | "closed" | "error") => void;
};

// PCM payloads can pile up if the user starts recording before the socket
// opens. We buffer each kind separately so we can drain in the right order:
// every queued JSON command (begin / end / discard) first, then the queued
// audio. ``MAX_QUEUED_PCM`` is generous — ~30 s of 100 ms chunks — so even
// a slow handshake doesn't drop audio.
const MAX_QUEUED_PCM = 320;

export function createCalibrationClient(
  url: string,
  handlers: CalibrationHandlers,
): CalibrationClient {
  let socket: WebSocket | null = null;
  const pendingText: string[] = [];
  const pendingPcm: ArrayBuffer[] = [];

  const flush = () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    while (pendingText.length) {
      socket.send(pendingText.shift()!);
    }
    while (pendingPcm.length) {
      socket.send(pendingPcm.shift()!);
    }
  };

  const open = () => {
    socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      handlers.onState?.("open");
      flush();
    };
    socket.onclose = () => {
      socket = null;
      handlers.onState?.("closed");
    };
    socket.onerror = () => handlers.onState?.("error");
    socket.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      try {
        handlers.onMessage(JSON.parse(ev.data) as CalibrationServerMessage);
      } catch {
        /* ignore */
      }
    };
  };

  const sendText = (obj: object) => {
    const json = JSON.stringify(obj);
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(json);
    } else {
      pendingText.push(json);
    }
  };

  return {
    connect: () => {
      if (socket && socket.readyState !== WebSocket.CLOSED) return;
      open();
    },
    disconnect: () => {
      socket?.close();
      socket = null;
      pendingText.length = 0;
      pendingPcm.length = 0;
    },
    begin: (msg) => sendText({ type: "begin", ...msg }),
    end: () => sendText({ type: "end" }),
    discard: () => sendText({ type: "discard" }),
    sendPcm: (chunk) => {
      // Always copy into a fresh ArrayBuffer so we're independent of whatever
      // backing store the worklet handed us (and TS narrows away SharedArrayBuffer).
      const copy = new ArrayBuffer(chunk.byteLength);
      new Uint8Array(copy).set(
        new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength),
      );
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(copy);
      } else {
        if (pendingPcm.length >= MAX_QUEUED_PCM) {
          pendingPcm.shift();
        }
        pendingPcm.push(copy);
      }
    },
  };
}
