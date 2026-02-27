from __future__ import annotations

import numpy as np


def snr_db(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(min(x.size, y.size))
    if n == 0:
        return float("nan")
    x = x[:n]
    y = y[:n]
    num = float(np.sum(x * x))
    den = float(np.sum((x - y) * (x - y)) + eps)
    return 10.0 * float(np.log10((num + eps) / den))


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(
    fs: int,
    n_fft: int,
    n_mels: int = 40,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Create a triangular mel filterbank matrix with shape (n_mels, n_freq_bins).

    This is a lightweight implementation (no external deps), intended for
    research/inspection metrics rather than strict compatibility with any library.
    """
    fs = int(fs)
    n_fft = int(n_fft)
    n_mels = int(n_mels)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if n_mels <= 0:
        raise ValueError("n_mels must be > 0")

    nyq = 0.5 * float(fs)
    fmin = float(max(0.0, fmin))
    fmax = float(nyq if fmax is None else fmax)
    fmax = float(min(max(fmax, fmin + 1e-6), nyq))

    n_freq = n_fft // 2 + 1
    freqs = np.linspace(0.0, nyq, n_freq, dtype=np.float64)

    mel_min = float(_hz_to_mel(np.array([fmin], dtype=np.float64))[0])
    mel_max = float(_hz_to_mel(np.array([fmax], dtype=np.float64))[0])
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_pts = _mel_to_hz(mel_pts)

    # FFT bin indices for mel points
    bins = np.floor((n_fft + 1) * hz_pts / float(fs)).astype(np.int32)
    bins = np.clip(bins, 0, n_freq - 1)

    fb = np.zeros((n_mels, n_freq), dtype=np.float64)
    for m in range(n_mels):
        left = int(bins[m])
        center = int(bins[m + 1])
        right = int(bins[m + 2])
        if center <= left or right <= center:
            continue

        # rising edge
        fb[m, left:center] = (freqs[left:center] - freqs[left]) / max(freqs[center] - freqs[left], 1e-12)
        # falling edge
        fb[m, center:right] = (freqs[right] - freqs[center:right]) / max(freqs[right] - freqs[center], 1e-12)

    # Normalize each filter to have roughly unit area (helps comparability).
    denom = np.sum(fb, axis=1, keepdims=True)
    fb = fb / np.maximum(denom, 1e-12)
    return fb


def _stft_power(
    x: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    n_fft = int(n_fft)
    win_length = int(win_length)
    hop_length = int(hop_length)
    if n_fft <= 0:
        raise ValueError("n_fft must be > 0")
    if win_length <= 0:
        raise ValueError("win_length must be > 0")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0")

    if x.size < win_length:
        x = np.pad(x, (0, win_length - x.size), mode="constant")

    n_frames = 1 + (x.size - win_length) // hop_length
    if n_frames <= 0:
        n_frames = 1

    win = np.hanning(win_length).astype(np.float64, copy=False)
    frames = np.empty((n_frames, n_fft), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        seg = x[start : start + win_length]
        if seg.size < win_length:
            seg = np.pad(seg, (0, win_length - seg.size), mode="constant")
        seg = seg * win
        if win_length < n_fft:
            frames[i, :win_length] = seg
            frames[i, win_length:] = 0.0
        else:
            frames[i, :] = seg[:n_fft]

    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float64, copy=False)
    return power  # (n_frames, n_freq)


def mel_snr_db(
    x: np.ndarray,
    y: np.ndarray,
    fs: int,
    eps: float = 1e-12,
    n_mels: int = 40,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> float:
    """
    A simple "mel-domain SNR" computed on log-mel energies:

      melSNR = 10*log10( sum(Mx^2) / sum((Mx-My)^2) )

    where M is the log-mel energy matrix (frames x mels).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(min(x.size, y.size))
    if n == 0:
        return float("nan")
    x = x[:n]
    y = y[:n]

    fs = int(fs)
    if fs <= 0:
        raise ValueError("fs must be > 0")

    win_length = int(round(float(fs) * (float(win_ms) / 1000.0)))
    hop_length = int(round(float(fs) * (float(hop_ms) / 1000.0)))
    win_length = int(max(16, win_length))
    hop_length = int(max(1, hop_length))
    n_fft = 1
    while n_fft < win_length:
        n_fft <<= 1
    n_fft = int(max(256, n_fft))

    px = _stft_power(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    py = _stft_power(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    fb = mel_filterbank(fs=fs, n_fft=n_fft, n_mels=int(n_mels), fmin=float(fmin), fmax=fmax)

    mx = px @ fb.T
    my = py @ fb.T
    mx = np.log(mx + eps)
    my = np.log(my + eps)

    num = float(np.sum(mx * mx))
    den = float(np.sum((mx - my) * (mx - my)) + eps)
    return 10.0 * float(np.log10((num + eps) / den))


def seg_snr_db(
    x: np.ndarray,
    y: np.ndarray,
    frame_len: int,
    eps: float = 1e-12,
    clip_low: float = -10.0,
    clip_high: float = 35.0,
) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(min(x.size, y.size))
    if n == 0:
        return float("nan")
    x = x[:n]
    y = y[:n]

    L = int(frame_len)
    if L <= 0:
        raise ValueError("frame_len must be > 0")

    n_frames = n // L
    if n_frames == 0:
        return snr_db(x, y, eps=eps)

    vals = []
    for i in range(n_frames):
        xs = x[i * L : (i + 1) * L]
        ys = y[i * L : (i + 1) * L]
        e = xs - ys
        sx = float(np.sum(xs * xs))
        se = float(np.sum(e * e))
        if sx <= eps:
            continue
        v = 10.0 * float(np.log10((sx + eps) / (se + eps)))
        v = float(min(max(v, clip_low), clip_high))
        vals.append(v)
    if not vals:
        return snr_db(x, y, eps=eps)
    return float(np.mean(vals))
