// Data structure consumed by /viz. Hardcoded samples conform to this; the
// live feed adapter (lib/viz-from-ws.ts) maps WS messages onto it.

export type VizSpeakerContext = {
  speaker: string;
  text: string;
};

export type VizPhoneme = {
  produced: string;
  intended: string;
  conf: number;
  match: boolean;
  time: number;            // ms from utterance start
  distortionType?: string; // shown under mismatched phonemes
};

export type VizWord = {
  raw: string;
  translated: string;
  conf: number;
  corrected?: boolean;
  contextUsed?: boolean;
};

export type VizProbCandidate = {
  phoneme: string;
  prob: number;
};

export type VizProbDistribution = {
  label: string;
  candidates: VizProbCandidate[];
  correctedTo: string;
  reason: string;
};

export type VizUtterance = {
  id: string;
  speakerContext?: VizSpeakerContext;
  raw: string;
  translated: string;
  phonemes: VizPhoneme[];
  words: VizWord[];
  probDistributions: VizProbDistribution[];
  durationMs: number;
};
