/**
 * Loads a pre-computed per-clip performance ranking baked from
 * ``server/scripts/rank_demo_clips.py`` runs (one per calibrated
 * speaker). The TORGO clips drawer uses this ranking to put the
 * known-good clips at the top of the list during a live demo, so the
 * presenter can play them top-to-bottom and consistently get a strong
 * transcription result.
 *
 * Ranking entries carry per-clip WER (adapted Whisper output vs the
 * intended_text) and the safety-gate verdict (flagged=true means the
 * gate would mark this clip "uncertain" in the UI). The drawer's sort
 * key is ``(flagged ASC, adapted_wer ASC)`` — unflagged + low-WER first,
 * flagged clips sink to the bottom.
 *
 * The JSON lives at ``/demo_ranking.json`` (i.e. client/public/), keyed
 * by speaker id. The drawer fetches it once on first open and caches in
 * memory; missing entries (no ranking for that speaker, or that clip)
 * simply fall back to the existing prompt-set / id ordering.
 */

export type RankingEntry = {
  id: number;
  intended?: string;
  adapted_text?: string | null;
  adapted_wer: number;
  adapted_flagged: boolean;
  vanilla_wer?: number;
};

export type DemoRanking = Record<string, RankingEntry[]>;

let cached: DemoRanking | null = null;
let inflight: Promise<DemoRanking | null> | null = null;

/**
 * Resolve the demo ranking, fetching the static JSON if needed. Returns
 * ``null`` if the file is absent — the drawer treats that the same as
 * "no ranking" and skips the reorder.
 */
export async function loadDemoRanking(): Promise<DemoRanking | null> {
  if (cached) return cached;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const res = await fetch("/demo_ranking.json", { cache: "no-store" });
      if (!res.ok) return null;
      const raw = (await res.json()) as unknown;
      if (!raw || typeof raw !== "object") return null;
      const out: DemoRanking = {};
      for (const [speaker, entries] of Object.entries(
        raw as Record<string, unknown>,
      )) {
        if (!Array.isArray(entries)) continue;
        const cleaned: RankingEntry[] = [];
        for (const e of entries) {
          if (!e || typeof e !== "object") continue;
          const obj = e as Record<string, unknown>;
          const id = typeof obj.id === "number" ? obj.id : null;
          if (id == null) continue;
          const adapted_wer =
            typeof obj.adapted_wer === "number" ? obj.adapted_wer : 1;
          const adapted_flagged = obj.adapted_flagged === true;
          cleaned.push({
            id,
            intended: typeof obj.intended === "string" ? obj.intended : undefined,
            adapted_text:
              typeof obj.adapted_text === "string" ? obj.adapted_text : null,
            adapted_wer,
            adapted_flagged,
            vanilla_wer:
              typeof obj.vanilla_wer === "number" ? obj.vanilla_wer : undefined,
          });
        }
        out[speaker.toUpperCase()] = cleaned;
      }
      cached = out;
      return out;
    } catch {
      return null;
    }
  })();
  return inflight;
}

/**
 * Build a per-clip lookup ``id → RankingEntry`` for one speaker. Returns
 * an empty Map when the speaker has no ranking entries.
 */
export function buildRankingIndex(
  ranking: DemoRanking | null,
  speakerId: string,
): Map<number, RankingEntry> {
  const out = new Map<number, RankingEntry>();
  if (!ranking) return out;
  const entries = ranking[speakerId.toUpperCase()];
  if (!entries) return out;
  for (const e of entries) out.set(e.id, e);
  return out;
}
