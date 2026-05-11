"use client";

// A short two-tone chime synthesized via WebAudio so we don't ship an audio
// file. Used by the emergency banner; deliberately attention-grabbing but
// not so loud as to startle a patient nearby.

let ctx: AudioContext | null = null;

function ensureCtx(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (ctx && ctx.state !== "closed") return ctx;
  try {
    ctx = new AudioContext();
  } catch {
    ctx = null;
  }
  return ctx;
}

export function playEmergencyChime(): void {
  const c = ensureCtx();
  if (!c) return;
  const now = c.currentTime;

  const make = (freq: number, start: number, dur: number, gain: number) => {
    const osc = c.createOscillator();
    const env = c.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(freq, now + start);
    env.gain.setValueAtTime(0, now + start);
    env.gain.linearRampToValueAtTime(gain, now + start + 0.04);
    env.gain.linearRampToValueAtTime(0, now + start + dur);
    osc.connect(env).connect(c.destination);
    osc.start(now + start);
    osc.stop(now + start + dur + 0.05);
  };

  // Two ascending pulses → urgent but not panic-inducing.
  make(880, 0.0, 0.18, 0.18);
  make(1320, 0.22, 0.22, 0.22);
}
