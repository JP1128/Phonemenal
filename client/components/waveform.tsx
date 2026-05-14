"use client";

import { useEffect, useRef, useState } from "react";

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";

const PEAK_COUNT = 32;
// Peaks per utterance, shared across the page so navigating away and back
// doesn't re-decode. Bounded by the thread cap upstream (MAX_HISTORY).
const peakCache = new Map<number, number[]>();
const inflight = new Map<number, Promise<number[]>>();
let _ctx: AudioContext | null = null;

function getCtx(): AudioContext {
  // Lazy: AudioContext creation requires the page to be interactive in some
  // browsers. Holding a single instance also avoids the overhead of spinning
  // one up per row.
  if (!_ctx) {
    const Ctor: typeof AudioContext =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    _ctx = new Ctor();
  }
  return _ctx;
}

async function loadPeaks(utteranceId: number): Promise<number[]> {
  const cached = peakCache.get(utteranceId);
  if (cached) return cached;
  const pending = inflight.get(utteranceId);
  if (pending) return pending;
  const p = (async () => {
    const res = await fetch(
      `${SERVER_HTTP}/utterance/${utteranceId}/audio`,
    );
    if (!res.ok) throw new Error(`audio fetch failed (${res.status})`);
    const buf = await res.arrayBuffer();
    const audio = await getCtx().decodeAudioData(buf);
    const ch = audio.getChannelData(0);
    const block = Math.max(1, Math.floor(ch.length / PEAK_COUNT));
    const peaks: number[] = [];
    for (let i = 0; i < PEAK_COUNT; i++) {
      let max = 0;
      const start = i * block;
      const end = Math.min(start + block, ch.length);
      for (let j = start; j < end; j++) {
        const a = Math.abs(ch[j]);
        if (a > max) max = a;
      }
      peaks.push(max);
    }
    peakCache.set(utteranceId, peaks);
    inflight.delete(utteranceId);
    return peaks;
  })();
  inflight.set(utteranceId, p);
  return p;
}

export function Waveform({
  utteranceId,
  className = "",
}: {
  utteranceId: number;
  className?: string;
}) {
  const [peaks, setPeaks] = useState<number[] | null>(
    peakCache.get(utteranceId) ?? null,
  );
  const [errored, setErrored] = useState(false);
  const ref = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (peaks || errored) return;
    const el = ref.current;
    if (!el) return;
    let cancelled = false;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          obs.disconnect();
          loadPeaks(utteranceId)
            .then((p) => {
              if (!cancelled) setPeaks(p);
            })
            .catch(() => {
              if (!cancelled) setErrored(true);
            });
        }
      },
      { threshold: 0, rootMargin: "100px" },
    );
    obs.observe(el);
    return () => {
      cancelled = true;
      obs.disconnect();
    };
  }, [utteranceId, peaks, errored]);

  return (
    <span
      ref={ref}
      aria-hidden
      className={`inline-block align-middle h-3 w-10 ${className}`}
    >
      {errored ? null : peaks ? (
        <svg
          viewBox={`0 0 ${PEAK_COUNT} 100`}
          preserveAspectRatio="none"
          className="block w-full h-full opacity-50"
        >
          {peaks.map((p, i) => {
            // Scale so even quiet bars are perceptible; clamp to leave a
            // 1px floor so silent regions still have a visible spine.
            const h = Math.max(8, Math.min(100, p * 140));
            return (
              <rect
                key={i}
                x={i + 0.15}
                y={(100 - h) / 2}
                width={0.7}
                height={h}
                fill="currentColor"
              />
            );
          })}
        </svg>
      ) : (
        <span className="block w-full h-px bg-white/10 rounded mt-[5px]" />
      )}
    </span>
  );
}
