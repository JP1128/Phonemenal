"use client";

import type { ControlMessage, WsMessage } from "./messages";

export type ConnectionState = "idle" | "connecting" | "open" | "closed" | "error";

export type WsHandlers = {
  onMessage: (msg: WsMessage) => void;
  onState?: (state: ConnectionState) => void;
  onError?: (err: Event | Error) => void;
};

export type PhonemerWs = {
  connect: () => void;
  disconnect: () => void;
  sendPcm: (chunk: Int16Array) => void;
  sendControl: (msg: ControlMessage) => void;
  state: () => ConnectionState;
};

export function createPhonemerWs(
  url: string,
  handlers: WsHandlers,
): PhonemerWs {
  let socket: WebSocket | null = null;
  let state: ConnectionState = "idle";
  let manualClose = false;
  let reconnectAttempt = 0;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const setState = (next: ConnectionState) => {
    state = next;
    handlers.onState?.(next);
  };

  const scheduleReconnect = () => {
    if (manualClose) return;
    const delay = Math.min(30_000, 500 * 2 ** reconnectAttempt);
    reconnectAttempt += 1;
    reconnectTimer = setTimeout(open, delay);
  };

  const open = () => {
    setState("connecting");
    try {
      socket = new WebSocket(url);
    } catch (err) {
      setState("error");
      handlers.onError?.(err as Error);
      scheduleReconnect();
      return;
    }
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
      reconnectAttempt = 0;
      setState("open");
    };

    socket.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      try {
        const payload = JSON.parse(ev.data) as WsMessage;
        handlers.onMessage(payload);
      } catch (err) {
        handlers.onError?.(err as Error);
      }
    };

    socket.onerror = (ev) => {
      setState("error");
      handlers.onError?.(ev);
    };

    socket.onclose = () => {
      setState("closed");
      socket = null;
      scheduleReconnect();
    };
  };

  return {
    connect: () => {
      manualClose = false;
      if (socket && socket.readyState !== WebSocket.CLOSED) return;
      open();
    },
    disconnect: () => {
      manualClose = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      socket?.close();
      socket = null;
      setState("idle");
    },
    sendPcm: (chunk: Int16Array) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      // Send the underlying buffer slice (handles offset/length correctly).
      socket.send(
        chunk.buffer.slice(
          chunk.byteOffset,
          chunk.byteOffset + chunk.byteLength,
        ),
      );
    },
    sendControl: (msg: ControlMessage) => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify(msg));
    },
    state: () => state,
  };
}
