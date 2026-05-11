// Shared message types between server and client.
// Keep aligned with server/app/main.py response payloads.

export type PhonemeProb = { symbol: string; prob: number };

export type PhonemeFrame = {
  timestamp_ms: number;
  phonemes: PhonemeProb[];
};

export type AcousticFeatures = {
  duration_ms: number;
  mfcc_mean: number[];
  mfcc_std: number[];
  formants_hz: number[];
  pitch_hz_mean: number;
  pitch_hz_std: number;
  jitter: number;
  shimmer: number;
  spectral_flux_mean: number;
  spectral_flux_std: number;
  vot_proxy_ms: number;
  pause_count: number;
  pause_total_ms: number;
};

export type UtteranceMessage = {
  type: "utterance";
  utterance_id: number;
  session_id: number;
  speaker: string;
  role: string | null;
  is_patient: boolean;
  speaker_similarity: number;
  start_ms: number;
  end_ms: number;
};

export type PhonemeWindowMessage = {
  type: "phoneme_window";
  utterance_id: number;
  frames: PhonemeFrame[];
};

export type FeaturesMessage = {
  type: "features";
  utterance_id: number;
  features: AcousticFeatures;
};

export type UtteranceTranscriptMessage = {
  type: "utterance_transcript";
  utterance_id: number;
  transcript: string;
};

export type TranslationMessage = {
  type: "translation";
  utterance_id: number;
  primary: string;
  alternatives: string[];
  per_word_confidence: number[];
  overall_confidence: number;
  emergency: boolean;
  context_hints: string[];
  // Phoneme-alignment safety gate fields. The patient pipeline now uses
  // Whisper + a wav2vec2-alignment check; `flagged` is true when the audio
  // doesn't support the proposed transcription (Whisper-style hallucination).
  // The UI renders flagged outputs with a red border + "uncertain" hint
  // — text stays visible, but it isn't presented as confidently correct.
  flagged?: boolean;
  reason?: "low_alignment" | "phoneme_density" | null;
  align_score?: number;
  phoneme_density?: number;
};

export type TranslationErrorMessage = {
  type: "translation_error";
  utterance_id: number;
  error: string;
  message: string;
};

export type EmergencyMessage = {
  type: "emergency";
  utterance_id: number;
  phrase: string | null;
  confidence: number;
  category?: string;
  sources?: string[];
};

export type ContextSnapshot = {
  session_id: number;
  recent: {
    speaker: string;
    role?: string | null;
    text: string;
    age_s: number;
    relevance: number;
  }[];
  topics: { keyword: string; weight: number }[];
  vocabulary_boost: Record<string, number>;
  pending_questions: { text: string; type: string; age_s: number; urgency: number }[];
  speakers: { label: string; role: string | null; role_confidence: number }[];
  emotion: { frustration: number; distress: number; last_updated_s_ago: number };
  session: {
    setting: string | null;
    listener_role_hint: string | null;
    time_of_day: string | null;
    activity_context: string | null;
    notes: string | null;
  };
};

export type ContextMessage = {
  type: "context";
  session_id: number;
  context: ContextSnapshot;
};

export type UtteranceRejectedMessage = {
  type: "utterance_rejected";
  session_id: number;
  start_ms: number;
  end_ms: number;
  speaker_similarity: number;
  reason: "non_speech" | string;
};

export type WsMessage =
  | UtteranceMessage
  | PhonemeWindowMessage
  | FeaturesMessage
  | UtteranceTranscriptMessage
  | TranslationMessage
  | TranslationErrorMessage
  | EmergencyMessage
  | UtteranceRejectedMessage
  | ContextMessage;

export type WsMessageType = WsMessage["type"];

export type ControlMessage = { type: "stop" };

// Calibration WebSocket protocol (Phase 2.1)
export type CalibrationBeginMessage = {
  type: "begin";
  set_id: string;
  sentence_index: number;
  intended_text: string;
};
export type CalibrationEndMessage = { type: "end" };
export type CalibrationDiscardMessage = { type: "discard" };

export type CalibrationReadyMessage = { type: "ready" };
export type CalibrationUtteranceCommittedMessage = {
  type: "calibration_utterance";
  utterance_id: number;
  duration_ms: number;
};

export type CalibrationErrorMessage = {
  type: "calibration_error";
  error: string;
  message: string;
  received_bytes?: number;
};

export type CalibrationServerMessage =
  | CalibrationReadyMessage
  | CalibrationUtteranceCommittedMessage
  | CalibrationErrorMessage;
