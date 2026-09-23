"use client";

import { displayNameFor } from "./torgo-speakers";

/**
 * The active "patient" the UI is configured for.
 *
 * Phonemer is a single-calibrated-patient app — the whole product makes
 * the most sense when there's a specific speaker whose voice / adapter /
 * prosody baseline the runtime is pinned to. A Profile expresses that
 * identity at the UI level:
 *
 *   - `calibrated`: a speaker we have a per-speaker Whisper adapter for
 *     on disk (F03 / FC02 / M01 / etc). Streamed TORGO clips for this
 *     speaker route through the matching LoRA; live mic uses the default
 *     adapter today, but will route via this profile in a future change.
 *
 *   - `demo`: a bundle of uncalibrated dysarthric TORGO speakers (F01,
 *     F04, M02, M03, M04, M05). Not a real patient identity — it's a
 *     test bench against the baseline (non-adapted) Whisper path.
 *     Inside this profile the user picks which uncalibrated speaker to
 *     view.
 *
 * "+ New profile" is a menu item that navigates to /calibrate; it isn't
 * a profile state itself.
 */
export type Profile =
  | { kind: "calibrated"; speakerId: string }
  | { kind: "demo" };

/**
 * Enriched calibrated-speaker entry surfaced by ``GET /adapters``.
 *
 *   id    — routing key (matches `Profile.speakerId` and the WS
 *           `adapter_speaker` query param). For TORGO speakers this is
 *           the canonical id (F03, FC02, …). For patient-trained
 *           adapters it's the literal directory name (Jane_Doe).
 *   label — human-readable display string. Identical to `id` for TORGO,
 *           humanized (underscores → spaces) for patients.
 *   kind  — "torgo" or "patient". The UI uses it to decide between
 *           severity / control badges (TORGO) and a generic label
 *           (patient), and to filter the demo clips drawer's set of
 *           "is this speaker calibrated?" matches to TORGO only.
 */
export type CalibratedSpeaker = {
  id: string;
  label: string;
  kind: "torgo" | "patient";
};

/**
 * Best-effort parser for the ``GET /adapters`` `speakers` payload. The
 * server returns enriched objects; we accept legacy `string[]` too in
 * case the server is older than the client, treating them as TORGO ids.
 */
export function parseCalibratedSpeakers(raw: unknown): CalibratedSpeaker[] {
  if (!Array.isArray(raw)) return [];
  const out: CalibratedSpeaker[] = [];
  for (const item of raw) {
    if (typeof item === "string") {
      const upper = item.toUpperCase();
      out.push({ id: upper, label: upper, kind: "torgo" });
      continue;
    }
    if (item && typeof item === "object") {
      const obj = item as Record<string, unknown>;
      const id = typeof obj.id === "string" ? obj.id : null;
      if (!id) continue;
      const label =
        typeof obj.label === "string" && obj.label.length > 0
          ? obj.label
          : id;
      const kind = obj.kind === "patient" ? "patient" : "torgo";
      out.push({ id, label, kind });
    }
  }
  return out;
}

const STORAGE_KEY = "phonemer.profile";

export function defaultProfile(calibratedSpeakers: string[]): Profile {
  if (calibratedSpeakers.length > 0) {
    return { kind: "calibrated", speakerId: calibratedSpeakers[0] };
  }
  return { kind: "demo" };
}

export function loadProfile(): Profile | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (
      parsed &&
      typeof parsed === "object" &&
      "kind" in parsed
    ) {
      const p = parsed as Record<string, unknown>;
      if (
        p.kind === "calibrated" &&
        typeof p.speakerId === "string" &&
        p.speakerId.length > 0
      ) {
        return { kind: "calibrated", speakerId: p.speakerId };
      }
      if (p.kind === "demo") {
        return { kind: "demo" };
      }
    }
  } catch {
    // Ignore — corrupt storage shouldn't crash the app.
  }
  return null;
}

export function saveProfile(p: Profile): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch {
    // localStorage may be unavailable (private mode, quota); profile
    // just won't persist across reloads.
  }
}

export function profileLabel(
  p: Profile,
  speakers?: CalibratedSpeaker[] | null,
): string {
  if (p.kind === "calibrated") {
    const demoName = displayNameFor(p.speakerId);
    if (demoName) return demoName;
    if (speakers) {
      const hit = speakers.find((s) => s.id === p.speakerId);
      if (hit) return hit.label;
    }
    return p.speakerId;
  }
  return "Demo TORGO";
}

/**
 * If the saved profile references a speaker that no longer has an
 * adapter on disk (e.g., it was deleted), reconcile to a valid default.
 *
 * Accepts either the enriched ``CalibratedSpeaker[]`` (preferred) or a
 * legacy ``string[]`` of speaker ids so this stays usable from older
 * call sites.
 */
export function reconcileProfile(
  saved: Profile | null,
  available: CalibratedSpeaker[] | string[],
): Profile {
  const ids: string[] =
    available.length > 0 && typeof (available as unknown[])[0] === "string"
      ? (available as string[])
      : (available as CalibratedSpeaker[]).map((s) => s.id);
  if (!saved) return defaultProfile(ids);
  if (saved.kind === "calibrated") {
    if (!ids.includes(saved.speakerId)) {
      return defaultProfile(ids);
    }
  }
  return saved;
}
