// AudioWorklet processor: takes the input mic stream (browser native rate,
// usually 44.1k or 48k Float32) and emits 16 kHz Int16 chunks to the main
// thread every ~100 ms.
//
// Browser-side WebAudio constraints intentionally disable noiseSuppression,
// echoCancellation and autoGainControl, since the distortion signal we need
// to learn would otherwise be normalized away.

class PcmWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 16000;
    this.chunkMs = opts.chunkMs || 100;
    this.targetSamplesPerChunk = Math.round(
      (this.targetRate * this.chunkMs) / 1000,
    );

    // Linear-resampler state.
    this.resamplePos = 0; // fractional position in input samples
    this.lastSample = 0;

    // Output buffer (Int16) we fill and post when full.
    this.out = new Int16Array(this.targetSamplesPerChunk);
    this.outIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    const inputRate = sampleRate; // global from AudioWorkletGlobalScope
    const ratio = inputRate / this.targetRate;
    let pos = this.resamplePos;

    while (pos < channel.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = i === 0 ? this.lastSample : channel[i - 1];
      const b = channel[i];
      const sample = a + (b - a) * frac;

      // Float32 [-1, 1] -> Int16
      const s = Math.max(-1, Math.min(1, sample));
      this.out[this.outIndex++] = s < 0 ? s * 0x8000 : s * 0x7fff;

      if (this.outIndex >= this.targetSamplesPerChunk) {
        // Copy because we'll keep filling `this.out`.
        const copy = new Int16Array(this.out);
        this.port.postMessage(copy, [copy.buffer]);
        this.out = new Int16Array(this.targetSamplesPerChunk);
        this.outIndex = 0;
      }

      pos += ratio;
    }

    this.lastSample = channel[channel.length - 1] || 0;
    this.resamplePos = pos - channel.length;
    return true;
  }
}

registerProcessor("phonemer-pcm", PcmWorklet);
