import type { WsMessage } from "./messages";

export type RecordedEvent = {
  t_ms: number;          // ms since recording start
  payload: WsMessage;
};

// Optional captured PCM (for round-trip backend replay).
export type PcmChunk = {
  t_ms: number;
  // Base64-encoded little-endian Int16 PCM.
  data_b64: string;
};

export type SessionRecording = {
  version: 1;
  started_at: string;        // ISO 8601
  session_id: number | null;
  source: "mic" | "wav" | "torgo";
  source_label?: string;     // filename / TORGO speaker id / clip id
  duration_ms: number;
  events: RecordedEvent[];
  pcm?: PcmChunk[];          // present when "include audio" was on
};
