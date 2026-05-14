/**
 * Per-speaker metadata for the TORGO corpus, shared across the UI so
 * the TORGO panel and the profile selector agree on severity ratings.
 *
 * Severities come from the Frenchay Dysarthria Assessment scores
 * published in Rudzicz et al. 2008 (out of 40 — higher = more
 * intelligible). Mapping each score to a three-bucket label:
 *
 *   F01: 19/40  → severe
 *   F03: 28/40  → moderate   (much more intelligible than F01)
 *   F04: 36/40  → mild        (close to control)
 *   M01: 10/40  → severe      (most severe in the corpus; common worst-case)
 *   M02:  8/40  → severe      (very limited data, often excluded in papers)
 *   M03: 23/40  → severe
 *   M04: 24/40  → severe
 *   M05: 28/40  → moderate    (ALS, less severe than the CP-based speakers)
 *
 * Buckets are coarse on purpose — a clinician glancing at a chip in
 * the panel cares about "how impaired is this speaker" at a glance,
 * not the exact FDA score. The full numbers are in the comment above
 * so the bucketing is traceable.
 */

export type Severity = "mild" | "moderate" | "severe";

export const SPEAKER_SEVERITY: Record<string, Severity> = {
  F01: "severe",
  F03: "moderate",
  F04: "mild",
  M01: "severe",
  M02: "severe",
  M03: "severe",
  M04: "severe",
  M05: "moderate",
};

/**
 * TORGO control speakers are non-dysarthric — they're labelled with a
 * "C" infix (FC01–FC03, MC01–MC04). The corpus has no other prefix
 * conventions, so a regex on the id is reliable.
 */
export function isControlSpeaker(speakerId: string): boolean {
  return /^[FM]C\d+$/i.test(speakerId.trim());
}

export function severityFor(speakerId: string): Severity | null {
  return SPEAKER_SEVERITY[speakerId.toUpperCase()] ?? null;
}
