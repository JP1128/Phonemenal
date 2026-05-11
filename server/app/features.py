"""Acoustic feature extraction (Layer 2).

Captures motor-pattern signals required by the distortion model: vocal-tract
shape (MFCC), vowel positioning (formants), consonant timing (VOT proxy via
spectral flux), prosody breakdown (pitch, jitter, shimmer), and motor-planning
delays (pause structure).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import librosa
import numpy as np
from scipy.signal import lfilter


@dataclass
class AcousticFeatures:
    duration_ms: int
    mfcc_mean: list[float]                    # 13 coeffs
    mfcc_std: list[float]
    formants_hz: list[float]                  # F1, F2, F3
    pitch_hz_mean: float
    pitch_hz_std: float
    jitter: float                             # period-to-period variation
    shimmer: float                            # amplitude-to-amplitude variation
    spectral_flux_mean: float                 # consonant articulation proxy
    spectral_flux_std: float
    vot_proxy_ms: float                       # onset gap before voicing
    pause_count: int
    pause_total_ms: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def extract(audio: np.ndarray, sample_rate: int = 16000) -> AcousticFeatures:
    """Compute the full acoustic feature bundle for one utterance."""
    if audio.size == 0:
        return _empty()

    audio = audio.astype(np.float32, copy=False)
    duration_ms = int(1000 * len(audio) / sample_rate)

    # MFCCs (13 coeffs is standard for speech).
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1).tolist()
    mfcc_std = mfcc.std(axis=1).tolist()

    formants = _formants(audio, sample_rate)

    pitch_mean, pitch_std, jitter = _pitch_stats(audio, sample_rate)
    shimmer = _shimmer(audio)

    flux_mean, flux_std = _spectral_flux_stats(audio, sample_rate)
    vot_ms = _vot_proxy_ms(audio, sample_rate)
    pause_count, pause_total_ms = _pause_stats(audio, sample_rate)

    return AcousticFeatures(
        duration_ms=duration_ms,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        formants_hz=formants,
        pitch_hz_mean=pitch_mean,
        pitch_hz_std=pitch_std,
        jitter=jitter,
        shimmer=shimmer,
        spectral_flux_mean=flux_mean,
        spectral_flux_std=flux_std,
        vot_proxy_ms=vot_ms,
        pause_count=pause_count,
        pause_total_ms=pause_total_ms,
    )


# --------------------------------------------------------------------- helpers


def _empty() -> AcousticFeatures:
    return AcousticFeatures(
        duration_ms=0,
        mfcc_mean=[0.0] * 13,
        mfcc_std=[0.0] * 13,
        formants_hz=[0.0, 0.0, 0.0],
        pitch_hz_mean=0.0,
        pitch_hz_std=0.0,
        jitter=0.0,
        shimmer=0.0,
        spectral_flux_mean=0.0,
        spectral_flux_std=0.0,
        vot_proxy_ms=0.0,
        pause_count=0,
        pause_total_ms=0,
    )


def _formants(audio: np.ndarray, sr: int) -> list[float]:
    """Estimate first three formants via LPC root analysis on a center window."""
    if len(audio) < sr // 4:
        return [0.0, 0.0, 0.0]
    center = len(audio) // 2
    half = min(len(audio) // 2, sr // 4)  # up to 250 ms window
    segment = audio[center - half : center + half]
    if segment.size == 0:
        return [0.0, 0.0, 0.0]

    pre_emphasis = lfilter([1.0, -0.97], [1.0], segment)
    windowed = pre_emphasis * np.hamming(len(pre_emphasis))
    order = 2 + sr // 1000  # rule-of-thumb LPC order
    try:
        a = librosa.lpc(windowed.astype(np.float64), order=order)
    except Exception:
        return [0.0, 0.0, 0.0]

    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = sorted(angles * (sr / (2 * np.pi)))
    formants = [f for f in freqs if 90.0 < f < 5000.0][:3]
    while len(formants) < 3:
        formants.append(0.0)
    return [float(f) for f in formants]


def _pitch_stats(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
    except Exception:
        return 0.0, 0.0, 0.0

    voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    voiced = voiced[~np.isnan(voiced)]
    if voiced.size < 3:
        return 0.0, 0.0, 0.0

    periods = 1.0 / voiced
    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
    return float(np.mean(voiced)), float(np.std(voiced)), jitter


def _shimmer(audio: np.ndarray) -> float:
    peaks = np.abs(audio)
    if peaks.size < 1024:
        return 0.0
    # RMS over short hops as an amplitude envelope.
    hop = 256
    n = peaks.size // hop
    if n < 3:
        return 0.0
    env = np.array([peaks[i * hop : (i + 1) * hop].mean() for i in range(n)])
    env = env[env > 1e-6]
    if env.size < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(env))) / np.mean(env))


def _spectral_flux_stats(audio: np.ndarray, sr: int) -> tuple[float, float]:
    if len(audio) < 1024:
        return 0.0, 0.0
    S = np.abs(librosa.stft(audio, n_fft=512, hop_length=256))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    return float(flux.mean()), float(flux.std())


def _vot_proxy_ms(audio: np.ndarray, sr: int) -> float:
    """Approximate voice-onset-time as ms from utterance start to first voiced
    frame (above an energy threshold). Useful for consonant-precision tracking.
    """
    if len(audio) < sr // 50:
        return 0.0
    frame = sr // 100  # 10 ms
    energy = np.array(
        [np.sqrt(np.mean(audio[i : i + frame] ** 2)) for i in range(0, len(audio), frame)]
    )
    if energy.size == 0:
        return 0.0
    threshold = max(1e-3, 0.2 * energy.max())
    onset_idx = int(np.argmax(energy > threshold))
    return float(onset_idx * 10)


def _pause_stats(audio: np.ndarray, sr: int) -> tuple[int, int]:
    frame = sr // 100  # 10 ms
    if len(audio) < frame * 5:
        return 0, 0
    rms = np.array(
        [np.sqrt(np.mean(audio[i : i + frame] ** 2)) for i in range(0, len(audio), frame)]
    )
    if rms.size == 0:
        return 0, 0
    threshold = max(1e-3, 0.1 * rms.max())
    silent = rms < threshold

    pauses = 0
    pause_frames = 0
    run = 0
    for s in silent:
        if s:
            run += 1
        else:
            if run >= 8:  # >= 80 ms counts as a pause
                pauses += 1
                pause_frames += run
            run = 0
    if run >= 8:
        pauses += 1
        pause_frames += run
    return pauses, pause_frames * 10
