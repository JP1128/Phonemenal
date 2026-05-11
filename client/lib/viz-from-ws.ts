// Adapter from the live WebSocket protocol → VizUtterance shape used by the
// PipelineViz component. Builds a richer view than viz-samples.ts: phonemes
// come from collapsing frames, probability cards from the lowest-confidence
// frames, words from the LLM's translation primary + per-word confidence.

import type {
  PhonemeFrame,
  TranslationMessage,
  UtteranceMessage,
} from "./messages";
import type {
  VizPhoneme,
  VizProbDistribution,
  VizUtterance,
  VizWord,
} from "./viz-types";

export type LiveAssemblyInput = {
  utterance: UtteranceMessage;
  frames: PhonemeFrame[];
  translation: TranslationMessage;
  rawText?: string;
  speakerContext?: { speaker: string; text: string };
};

export function buildVizFromLive(input: LiveAssemblyInput): VizUtterance {
  const { utterance, frames, translation } = input;

  const phonemes = collapseFramesToPhonemes(frames, utterance.start_ms);
  const words = pairWords(input.rawText, translation);
  const probDistributions = takeProbDistributions(frames, 3);

  // The LLM's per-word confidence already encodes which words are corrected.
  // Mark anything below 0.85 as corrected; if context_hints non-empty mark
  // low-confidence words as context-assisted.
  const usedContext = (translation.context_hints?.length ?? 0) > 0;
  for (const w of words) {
    if (w.conf < 0.85) w.corrected = true;
    if (usedContext && w.conf < 0.75) w.contextUsed = true;
  }

  return {
    id: `live-${utterance.utterance_id}`,
    speakerContext: input.speakerContext,
    raw: input.rawText ?? translation.primary,
    translated: translation.primary,
    phonemes,
    words,
    probDistributions,
    durationMs: Math.max(1, utterance.end_ms - utterance.start_ms),
  };
}

function collapseFramesToPhonemes(
  frames: PhonemeFrame[],
  startMs: number,
): VizPhoneme[] {
  // Greedy collapse: merge consecutive frames whose top phoneme matches.
  // Treat <pad> / silence specially so we don't render dozens of blank chips.
  const out: VizPhoneme[] = [];
  let runStart = 0;
  let runSym = "";
  let runProb = 0;
  let runCount = 0;

  const flush = () => {
    if (runCount === 0) return;
    if (runSym === "<pad>" || runSym === "" || runSym === "<unk>") return;
    out.push({
      produced: runSym,
      intended: runSym,
      conf: runProb / runCount,
      match: runProb / runCount > 0.7,
      time: frames[runStart].timestamp_ms - startMs,
      distortionType:
        runProb / runCount <= 0.7 ? "low confidence" : undefined,
    });
  };

  for (let i = 0; i < frames.length; i++) {
    const top = frames[i].phonemes[0];
    if (!top) continue;
    if (top.symbol !== runSym) {
      flush();
      runSym = top.symbol;
      runProb = top.prob;
      runCount = 1;
      runStart = i;
    } else {
      runProb += top.prob;
      runCount += 1;
    }
  }
  flush();
  return out.slice(0, 60);  // viz cap
}

function pairWords(
  raw: string | undefined,
  translation: TranslationMessage,
): VizWord[] {
  const transWords = translation.primary.split(/\s+/).filter(Boolean);
  const rawWords = raw ? raw.split(/\s+/).filter(Boolean) : [];
  const len = transWords.length;
  const out: VizWord[] = [];
  for (let i = 0; i < len; i++) {
    const r = rawWords[i] ?? transWords[i];
    const t = transWords[i];
    const conf =
      translation.per_word_confidence[i] ?? translation.overall_confidence ?? 0;
    out.push({
      raw: r,
      translated: t,
      conf,
      corrected: r !== t,
      contextUsed: false,
    });
  }
  return out;
}

function takeProbDistributions(
  frames: PhonemeFrame[],
  k: number,
): VizProbDistribution[] {
  // Surface the k most-ambiguous (lowest top-1) frames as probability cards.
  // These are the moments where the system did real work — perfect for the
  // viz to highlight.
  const ranked = frames
    .map((f, i) => ({ f, i, top: f.phonemes[0]?.prob ?? 1 }))
    .filter((r) => r.f.phonemes.length >= 2)
    .sort((a, b) => a.top - b.top)
    .slice(0, k);

  return ranked.map(({ f, i }) => ({
    label: `frame ${i} · ${f.timestamp_ms} ms`,
    candidates: f.phonemes.slice(0, 4).map((p) => ({
      phoneme: `/${p.symbol}/`,
      prob: p.prob,
    })),
    correctedTo: `/${f.phonemes[0]?.symbol ?? ""}/`,
    reason:
      f.phonemes[0]?.prob && f.phonemes[0].prob < 0.4
        ? "Low-confidence frame — LLM resolved using profile + context"
        : "Frame resolved by patient distortion model",
  }));
}
