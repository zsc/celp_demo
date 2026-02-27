from __future__ import annotations

import contextlib
import wave
from pathlib import Path

import numpy as np


def _pcm24_to_int32(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8)
    if b.size % 3 != 0:
        raise ValueError("Invalid 24-bit PCM byte length.")
    b = b.reshape(-1, 3)
    x = (
        b[:, 0].astype(np.int32)
        | (b[:, 1].astype(np.int32) << 8)
        | (b[:, 2].astype(np.int32) << 16)
    )
    sign = (b[:, 2] & 0x80) != 0
    x[sign] |= -1 << 24
    return x


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Read WAV via soundfile if available, else stdlib wave.

    Returns mono float64 in [-1, 1] and sample rate.
    """
    p = Path(path)

    try:
        import soundfile as sf  # type: ignore

        data, fs = sf.read(str(p), always_2d=True)
        data = np.asarray(data, dtype=np.float64)
        if data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True)
        x = data[:, 0]
        x = np.clip(x, -1.0, 1.0)
        return x, int(fs)
    except Exception:
        pass

    with contextlib.closing(wave.open(str(p), "rb")) as w:
        ch = w.getnchannels()
        fs = int(w.getframerate())
        sampwidth = int(w.getsampwidth())
        nframes = int(w.getnframes())
        raw = w.readframes(nframes)

    if sampwidth == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 3:
        x = _pcm24_to_int32(raw).astype(np.float64) / float(1 << 23)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sampwidth: {sampwidth} bytes")

    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x = np.clip(x, -1.0, 1.0)
    return x, fs


def write_wav(path: str | Path, x: np.ndarray, fs: int, clip: bool = True) -> None:
    p = Path(path)
    x = np.asarray(x, dtype=np.float64).ravel()
    if clip:
        x = np.clip(x, -1.0, 1.0)
    y = (x * 32767.0).round().astype(np.int16)

    try:
        import soundfile as sf  # type: ignore

        sf.write(str(p), y, int(fs), subtype="PCM_16")
        return
    except Exception:
        pass

    with contextlib.closing(wave.open(str(p), "wb")) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(y.tobytes())


def resample_to_fs(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    fs_in = int(fs_in)
    fs_out = int(fs_out)
    if fs_in == fs_out or x.size == 0:
        return x.copy()

    try:
        from scipy.signal import resample_poly  # type: ignore

        # rational approximation
        import math

        g = math.gcd(fs_in, fs_out)
        up = fs_out // g
        down = fs_in // g
        y = resample_poly(x, up, down).astype(np.float64, copy=False)
        return y
    except Exception:
        pass

    n_out = int(round(x.size * (fs_out / float(fs_in))))
    t_in = np.arange(x.size, dtype=np.float64) / float(fs_in)
    t_out = np.arange(n_out, dtype=np.float64) / float(fs_out)
    y = np.interp(t_out, t_in, x).astype(np.float64, copy=False)
    return y
