"use client";

export type CaptureHandlers = {
  onPcm: (chunk: Int16Array) => void;
  onLevel?: (rms: number) => void;
  onError?: (err: Error) => void;
};

export type AudioCapture = {
  start: () => Promise<void>;
  stop: () => Promise<void>;
  isRunning: () => boolean;
};

export function createAudioCapture(handlers: CaptureHandlers): AudioCapture {
  let ctx: AudioContext | null = null;
  let stream: MediaStream | null = null;
  let workletNode: AudioWorkletNode | null = null;
  let analyser: AnalyserNode | null = null;
  let levelTimer: ReturnType<typeof setInterval> | null = null;

  const start = async () => {
    if (ctx) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
    } catch (err) {
      handlers.onError?.(err as Error);
      throw err;
    }

    ctx = new AudioContext();
    await ctx.audioWorklet.addModule("/pcm-worklet.js");

    const source = ctx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(ctx, "phonemer-pcm", {
      processorOptions: { targetRate: 16000, chunkMs: 100 },
    });
    workletNode.port.onmessage = (ev) => {
      handlers.onPcm(ev.data as Int16Array);
    };
    source.connect(workletNode);
    // Worklet outputs nothing audible, but it must be connected to the
    // graph or the browser may suspend it.
    workletNode.connect(ctx.destination);

    if (handlers.onLevel) {
      analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      levelTimer = setInterval(() => {
        analyser!.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        handlers.onLevel!(Math.sqrt(sum / buf.length));
      }, 100);
    }
  };

  const stop = async () => {
    if (levelTimer) {
      clearInterval(levelTimer);
      levelTimer = null;
    }
    workletNode?.disconnect();
    workletNode = null;
    analyser?.disconnect();
    analyser = null;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    if (ctx) {
      await ctx.close();
      ctx = null;
    }
  };

  return {
    start,
    stop,
    isRunning: () => ctx !== null,
  };
}

// Helper for the file/TORGO drag-and-drop path: decodes a WAV/audio file
// into a 16 kHz Int16Array, then yields it in 100 ms chunks paced at
// realtime so the server's VAD sees the same arrival cadence as a mic.
export async function streamFileAsPcm(
  file: ArrayBuffer | Blob,
  send: (chunk: Int16Array) => void,
  opts: { realtime?: boolean } = {},
): Promise<void> {
  const buf = file instanceof Blob ? await file.arrayBuffer() : file;
  const tmpCtx = new AudioContext();
  const decoded = await tmpCtx.decodeAudioData(buf.slice(0));
  await tmpCtx.close();

  // Mono mix-down.
  const channelCount = decoded.numberOfChannels;
  const length = decoded.length;
  const mono = new Float32Array(length);
  for (let c = 0; c < channelCount; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < length; i++) mono[i] += data[i] / channelCount;
  }

  // Resample to 16 kHz with a linear interpolator.
  const targetRate = 16000;
  const ratio = decoded.sampleRate / targetRate;
  const targetLen = Math.floor(length / ratio);
  const out = new Int16Array(targetLen);
  for (let i = 0; i < targetLen; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const frac = x - i0;
    const a = mono[i0] || 0;
    const b = mono[Math.min(length - 1, i0 + 1)] || 0;
    const s = Math.max(-1, Math.min(1, a + (b - a) * frac));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  // Chunk + send.
  const chunkSamples = (targetRate * 100) / 1000; // 100 ms
  for (let i = 0; i < out.length; i += chunkSamples) {
    const chunk = out.slice(i, Math.min(out.length, i + chunkSamples));
    send(chunk);
    if (opts.realtime !== false) {
      await new Promise((r) => setTimeout(r, 100));
    }
  }
}
