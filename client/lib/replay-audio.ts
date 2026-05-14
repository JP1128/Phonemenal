"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const SERVER_HTTP =
  (typeof window !== "undefined" &&
    (window as unknown as { __PHONEMER_HTTP?: string }).__PHONEMER_HTTP) ||
  "http://localhost:8000";

/**
 * Single-track replay manager: only one utterance plays at a time. Clicking
 * a different utterance stops the current one and starts the new one.
 */
export function useReplayAudio() {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<number | null>(null);

  useEffect(() => {
    const a = new Audio();
    audioRef.current = a;
    const onEnded = () => setPlayingId(null);
    a.addEventListener("ended", onEnded);
    return () => {
      a.pause();
      a.removeEventListener("ended", onEnded);
      audioRef.current = null;
    };
  }, []);

  const toggle = useCallback(
    (utteranceId: number) => {
      const a = audioRef.current;
      if (!a) return;
      if (playingId === utteranceId) {
        a.pause();
        setPlayingId(null);
        return;
      }
      a.src = `${SERVER_HTTP}/utterance/${utteranceId}/audio`;
      a.currentTime = 0;
      a.play()
        .then(() => setPlayingId(utteranceId))
        .catch(() => setPlayingId(null));
    },
    [playingId],
  );

  // Stable across renders so callers can safely add it to useEffect deps.
  const stop = useCallback(() => {
    const a = audioRef.current;
    if (!a) return;
    a.pause();
    setPlayingId(null);
  }, []);

  return { playingId, toggle, stop };
}
