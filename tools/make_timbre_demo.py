from __future__ import annotations

import argparse
import html
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from celp_codec import codec, gains, lpc, metrics, pitch, timbre, wav_io  # noqa: E402
from celp_codec.bitstream import BitReader, BitstreamHeaderV1, BitstreamHeaderV2, read_header  # noqa: E402


@dataclass(frozen=True)
class Item:
    key: str
    wav_path: Path
    name: str


@dataclass(frozen=True)
class StyleStats:
    lag_median: float
    gp_mean: float
    gc_mean: float


@dataclass(frozen=True)
class AugSpec:
    tag: str
    title: str
    note: str
    params: timbre.TimbreParams


@dataclass(frozen=True)
class FormantViz:
    mel_png_rel: str
    track_png_rel: str
    spec_png_rel: str
    fs: int
    order: int
    spec_max_hz: float
    f1_hz: float | None
    f2_hz: float | None
    f3_hz: float | None
    stats: "FormantTrackStats"


@dataclass(frozen=True)
class FormantTrackStats:
    frame_ms: float
    hop_ms: float
    n_total: int
    n_valid_f1: int
    n_valid_f2: int
    n_valid_f3: int
    f1_med_hz: float | None
    f1_p10_hz: float | None
    f1_p90_hz: float | None
    f2_med_hz: float | None
    f2_p10_hz: float | None
    f2_p90_hz: float | None
    f3_med_hz: float | None
    f3_p10_hz: float | None
    f3_p90_hz: float | None


@dataclass(frozen=True)
class FormantNotch:
    f1_wav: str | None
    f2_wav: str | None
    f3_wav: str | None
    f1_ultra_wav: str | None
    f2_ultra_wav: str | None
    f3_ultra_wav: str | None


def _pick_three(pattern: str) -> list[Path]:
    pat = os.path.expanduser(str(pattern))
    if not os.path.isabs(pat):
        pat = str((REPO_ROOT / pat).resolve())
    paths = [Path(p) for p in sorted(glob(pat)) if p.lower().endswith(".wav")]
    if not paths:
        raise FileNotFoundError(f"No wavs matched pattern: {pattern!r}")

    prefer = ["en_happy_prompt.wav", "fear_zh_female_prompt.wav", "whisper_prompt.wav"]
    by_name = {p.name: p for p in paths}
    preferred = [by_name[n] for n in prefer if n in by_name]
    if len(preferred) >= 3:
        return preferred[:3]

    uniq: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        uniq.append(path)
        if len(uniq) >= 3:
            break
    if len(uniq) < 3:
        raise FileNotFoundError(f"Need at least 3 wavs, got {len(uniq)} from {pattern!r}")
    return uniq


def _encode_source(wav_path: Path, out_dir: Path, key: str, fs: int) -> bytes:
    x, fs_in = wav_io.read_wav(wav_path)
    x = wav_io.resample_to_fs(x, int(fs_in), int(fs))

    cfg = codec.CodecConfig(
        mode="acelp",
        fs=int(fs),
        frame_ms=20,
        subframe_ms=5,
        dp_pitch=True,
        dp_topk=10,
        dp_lambda=0.05,
        lpc_interp=True,
        rc_bits=10,
        gain_bits_p=7,
        gain_bits_c=7,
        seed=1234,
        acelp_solver="ista",
        ista_iters=25,
        ista_lambda=0.02,
        acelp_K=10,
        acelp_weight_bits=5,
    )
    bitstream_bytes, recon, _, _ = codec.encode_samples(x, cfg)

    wav_io.write_wav(out_dir / f"{key}_orig.wav", x, int(fs), clip=True)
    wav_io.write_wav(out_dir / f"{key}_recon.wav", recon, int(fs), clip=True)
    (out_dir / f"{key}.celpbin").write_bytes(bitstream_bytes)
    return bitstream_bytes


def _decode_to_wav(out_bytes: bytes, out_wav: Path) -> None:
    y, header, _ = codec.decode_bitstream(out_bytes, clip=True)
    wav_io.write_wav(out_wav, y, int(getattr(header, "fs")), clip=True)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    if len(tag) != 4:
        raise ValueError("PNG chunk tag must be 4 bytes")
    crc = zlib.crc32(tag)
    crc = zlib.crc32(data, crc)
    crc &= 0xFFFFFFFF
    return len(data).to_bytes(4, "big") + tag + data + crc.to_bytes(4, "big")


def _png_encode_rgb8(rgb: np.ndarray) -> bytes:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    comp = zlib.compress(raw, level=6)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")


def _mel_spectrogram_db(
    x: np.ndarray,
    fs: int,
    n_mels: int = 64,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    fs = int(fs)
    win_length = int(round(float(fs) * (float(win_ms) / 1000.0)))
    hop_length = int(round(float(fs) * (float(hop_ms) / 1000.0)))
    win_length = int(max(16, win_length))
    hop_length = int(max(1, hop_length))
    n_fft = 1
    while n_fft < win_length:
        n_fft <<= 1
    n_fft = int(max(256, n_fft))

    power = metrics._stft_power(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    fb = metrics.mel_filterbank(fs=fs, n_fft=n_fft, n_mels=int(n_mels), fmin=float(fmin), fmax=fmax)
    mel = power @ fb.T
    mel_db = 10.0 * np.log10(mel + 1e-12)
    return mel_db


def _resample_time(m: np.ndarray, width: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    width = int(width)
    n_frames, n_mels = int(m.shape[0]), int(m.shape[1])
    if n_frames == width:
        return m
    if n_frames == 1:
        return np.repeat(m, width, axis=0)
    t0 = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    t1 = np.linspace(0.0, 1.0, width, dtype=np.float64)
    out = np.empty((width, n_mels), dtype=np.float64)
    for i in range(n_mels):
        out[:, i] = np.interp(t1, t0, m[:, i])
    return out


def _write_mel_png(wav_path: Path, png_path: Path, width: int = 220, n_mels: int = 64) -> bool:
    try:
        x, fs = wav_io.read_wav(wav_path)
        mel_db = _mel_spectrogram_db(x, fs=fs, n_mels=int(n_mels))
        mel_db = _resample_time(mel_db, width=int(width))
        img = mel_db.T[::-1, :]
        lo = float(np.percentile(img, 5.0))
        hi = float(np.percentile(img, 95.0))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo + 1e-9:
            lo = float(np.min(img))
            hi = float(np.max(img))
        if hi <= lo + 1e-9:
            hi = lo + 1.0
        norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        u8 = np.round(norm * 255.0).astype(np.uint8)
        rgb = np.repeat(u8[:, :, None], 3, axis=2)
        png_path.write_bytes(_png_encode_rgb8(rgb))
        return True
    except Exception:
        return False


def _clip_int(v: int, lo: int, hi: int) -> int:
    return int(min(max(int(v), int(lo)), int(hi)))


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w = int(img.shape[0]), int(img.shape[1])
    x0 = _clip_int(x0, 0, w - 1)
    x1 = _clip_int(x1, 0, w - 1)
    y0 = _clip_int(y0, 0, h - 1)
    y1 = _clip_int(y1, 0, h - 1)

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        img[y, x, 0] = color[0]
        img[y, x, 1] = color[1]
        img[y, x, 2] = color[2]
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _draw_polyline_xy(
    img: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
) -> None:
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < 2 or ys.size != xs.size:
        return
    thickness = int(max(1, thickness))
    offsets = [0]
    if thickness >= 2:
        offsets = [0, 1, -1]
    if thickness >= 3:
        offsets = [0, 1, -1, 2, -2]
    for i in range(xs.size - 1):
        x0 = float(xs[i])
        y0 = float(ys[i])
        x1 = float(xs[i + 1])
        y1 = float(ys[i + 1])
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
            continue
        for dy in offsets:
            _draw_line(
                img,
                int(round(x0)),
                int(round(y0 + float(dy))),
                int(round(x1)),
                int(round(y1 + float(dy))),
                color,
            )


def _pick_analysis_frame(x: np.ndarray, fs: int, frame_ms: float = 30.0, hop_ms: float = 10.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    fs = int(fs)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    frame_len = int(max(32, round(float(fs) * float(frame_ms) / 1000.0)))
    hop_len = int(max(1, round(float(fs) * float(hop_ms) / 1000.0)))
    if x.size <= frame_len:
        out = np.zeros((frame_len,), dtype=np.float64)
        out[: x.size] = x
        return out

    best_i = 0
    best_e = -1.0
    for i in range(0, x.size - frame_len + 1, hop_len):
        fr = x[i : i + frame_len]
        e = float(np.dot(fr, fr))
        if e > best_e:
            best_e = e
            best_i = i
    return x[best_i : best_i + frame_len].copy()


def _estimate_formants_from_lpc_roots(a: np.ndarray, fs: int, max_f_hz: float) -> list[float]:
    fs = int(fs)
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size < 3:
        return []
    roots = np.roots(a)
    cand: list[tuple[float, float]] = []
    for r in roots:
        if np.imag(r) <= 1e-6:
            continue
        mag = float(np.abs(r))
        if mag <= 0.0:
            continue
        freq = float(np.angle(r) * float(fs) / (2.0 * float(np.pi)))
        if freq < 90.0 or freq > float(max_f_hz):
            continue
        bw = float(-float(fs) * np.log(mag + 1e-12) / float(np.pi))
        if bw < 20.0 or bw > 900.0:
            continue
        cand.append((freq, bw))
    cand.sort(key=lambda t: t[0])

    out: list[float] = []
    min_sep = 120.0
    for freq, _bw in cand:
        if not out or freq - out[-1] >= min_sep:
            out.append(freq)
        if len(out) >= 3:
            break
    return out


def _estimate_formants_from_envelope(freq_hz: np.ndarray, env_db: np.ndarray, max_f_hz: float) -> list[float]:
    f = np.asarray(freq_hz, dtype=np.float64).ravel()
    y = np.asarray(env_db, dtype=np.float64).ravel()
    if f.size < 5 or y.size != f.size:
        return []

    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if idx.size == 0:
        return []

    # Keep peaks in speech formant band and with moderate prominence.
    lo = float(np.percentile(y, 50.0))
    peak_freqs = [float(f[i]) for i in idx if 90.0 <= float(f[i]) <= float(max_f_hz) and float(y[i]) >= lo]
    peak_freqs.sort()

    out: list[float] = []
    min_sep = 120.0
    for freq in peak_freqs:
        if not out or freq - out[-1] >= min_sep:
            out.append(freq)
        if len(out) >= 3:
            break
    return out


def _analyze_lpc_envelope_and_formants(
    x: np.ndarray,
    fs: int,
    order: int | None = None,
    preemph: float = 0.97,
) -> tuple[np.ndarray, np.ndarray, tuple[float | None, float | None, float | None], int]:
    fs = int(fs)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or fs <= 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            (None, None, None),
            0,
        )

    if order is None:
        order = 16 if fs >= 16000 else 10
    p = int(max(4, order))

    fr = _pick_analysis_frame(x, fs=fs, frame_ms=30.0, hop_ms=10.0)
    if fr.size <= p + 2:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            (None, None, None),
            p,
        )
    if preemph != 0.0:
        fr[1:] = fr[1:] - float(preemph) * fr[:-1]
    fr *= np.hamming(fr.size).astype(np.float64)

    r = lpc.autocorrelation(fr, p)
    a, _k = lpc.levinson_durbin(r, p)
    if not np.all(np.isfinite(a)):
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            (None, None, None),
            p,
        )
    a = np.asarray(a, dtype=np.float64).ravel()
    if a[0] == 0.0:
        a[0] = 1.0
    if a[0] != 1.0:
        a /= float(a[0])

    max_f = min(float(fs) * 0.5 - 50.0, 5000.0 if fs >= 12000 else 3800.0)
    if max_f <= 200.0:
        max_f = float(fs) * 0.5 - 20.0

    n = 512
    freq = np.linspace(0.0, max_f, n, dtype=np.float64)
    w = 2.0 * np.pi * freq / float(fs)
    k = np.arange(a.size, dtype=np.float64)
    A = np.exp(-1j * np.outer(w, k)) @ a
    env_db = 20.0 * np.log10(1.0 / (np.abs(A) + 1e-9))

    f_roots = _estimate_formants_from_lpc_roots(a, fs=fs, max_f_hz=max_f)
    if len(f_roots) < 3:
        f_env = _estimate_formants_from_envelope(freq, env_db, max_f_hz=max_f)
    else:
        f_env = []
    combined = (f_roots + [v for v in f_env if v not in f_roots])[:3]
    while len(combined) < 3:
        combined.append(float("nan"))

    f1 = None if not np.isfinite(combined[0]) else float(combined[0])
    f2 = None if not np.isfinite(combined[1]) else float(combined[1])
    f3 = None if not np.isfinite(combined[2]) else float(combined[2])
    return freq, env_db, (f1, f2, f3), p


def _estimate_formants_from_envelope_banded(
    freq_hz: np.ndarray,
    env_db: np.ndarray,
    max_f_hz: float,
) -> tuple[float | None, float | None, float | None]:
    f = np.asarray(freq_hz, dtype=np.float64).ravel()
    y = np.asarray(env_db, dtype=np.float64).ravel()
    if f.size < 5 or y.size != f.size:
        return (None, None, None)

    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    lo_db = float(np.percentile(y[np.isfinite(y)], 50.0)) if np.any(np.isfinite(y)) else float("nan")
    peaks: list[tuple[float, float]] = []
    for i in idx.tolist():
        fhz = float(f[i])
        if fhz < 90.0 or fhz > float(max_f_hz):
            continue
        amp = float(y[i])
        if not np.isfinite(amp):
            continue
        if np.isfinite(lo_db) and amp < lo_db:
            continue
        peaks.append((fhz, amp))

    def best_in_band(lo_hz: float, hi_hz: float) -> float | None:
        lo_hz = float(lo_hz)
        hi_hz = float(hi_hz)
        cand = [p for p in peaks if lo_hz <= p[0] <= hi_hz]
        if cand:
            cand.sort(key=lambda t: t[1], reverse=True)
            return float(cand[0][0])
        mask = (f >= lo_hz) & (f <= hi_hz) & np.isfinite(y)
        if not bool(np.any(mask)):
            return None
        j = int(np.argmax(y[mask]))
        fhz = float(f[mask][j])
        if fhz < 90.0 or fhz > float(max_f_hz):
            return None
        return fhz

    f1_lo, f1_hi = 150.0, min(1200.0, float(max_f_hz))
    f2_lo, f2_hi = 700.0, min(3500.0 if float(max_f_hz) >= 4500.0 else 2600.0, float(max_f_hz))
    f3_lo, f3_hi = 1500.0, float(max_f_hz)

    f1 = best_in_band(f1_lo, f1_hi)
    if f1 is not None:
        f2_lo = max(f2_lo, float(f1) + 200.0)
    f2 = best_in_band(f2_lo, f2_hi)
    if f2 is not None:
        f3_lo = max(f3_lo, float(f2) + 200.0)
    f3 = best_in_band(f3_lo, f3_hi)
    return (f1, f2, f3)


def _median_filter_nan(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    win = int(max(1, win))
    if win % 2 == 0:
        win += 1
    if x.size == 0 or win <= 1:
        return x.copy()
    half = win // 2
    out = np.full_like(x, np.nan)
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        w = x[lo:hi]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = float(np.median(w))
    return out


def _track_stats_1d(x: np.ndarray) -> tuple[int, float | None, float | None, float | None]:
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0, None, None, None
    return int(arr.size), float(np.median(arr)), float(np.percentile(arr, 10.0)), float(np.percentile(arr, 90.0))


def _analyze_lpc_formant_tracks(
    x: np.ndarray,
    fs: int,
    order: int | None = None,
    preemph: float = 0.97,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    smooth_win: int = 5,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    FormantTrackStats,
    int,
    float,
]:
    fs = int(fs)
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or fs <= 0:
        empty = np.zeros((0,), dtype=np.float64)
        stats = FormantTrackStats(
            frame_ms=float(frame_ms),
            hop_ms=float(hop_ms),
            n_total=0,
            n_valid_f1=0,
            n_valid_f2=0,
            n_valid_f3=0,
            f1_med_hz=None,
            f1_p10_hz=None,
            f1_p90_hz=None,
            f2_med_hz=None,
            f2_p10_hz=None,
            f2_p90_hz=None,
            f3_med_hz=None,
            f3_p10_hz=None,
            f3_p90_hz=None,
        )
        return empty, empty, empty, empty, empty, empty, stats, 0, 0.0

    if order is None:
        order = 16 if fs >= 16000 else 10
    p = int(max(4, int(order)))

    frame_len = int(max(p + 8, round(float(fs) * float(frame_ms) / 1000.0)))
    hop_len = int(max(1, round(float(fs) * float(hop_ms) / 1000.0)))
    if x.size < frame_len:
        tmp = np.zeros((frame_len,), dtype=np.float64)
        tmp[: x.size] = x
        x = tmp

    starts = list(range(0, int(x.size) - frame_len + 1, hop_len))
    n_frames = int(len(starts))
    if n_frames <= 0:
        empty = np.zeros((0,), dtype=np.float64)
        stats = FormantTrackStats(
            frame_ms=float(frame_ms),
            hop_ms=float(hop_ms),
            n_total=0,
            n_valid_f1=0,
            n_valid_f2=0,
            n_valid_f3=0,
            f1_med_hz=None,
            f1_p10_hz=None,
            f1_p90_hz=None,
            f2_med_hz=None,
            f2_p10_hz=None,
            f2_p90_hz=None,
            f3_med_hz=None,
            f3_p10_hz=None,
            f3_p90_hz=None,
        )
        return empty, empty, empty, empty, empty, empty, stats, p, 0.0

    energies = np.asarray([float(np.dot(x[s : s + frame_len], x[s : s + frame_len])) for s in starts], dtype=np.float64)
    max_e = float(np.max(energies)) if energies.size else 0.0
    thr = 0.0 if max_e <= 0.0 else max_e * 1e-2
    keep = energies >= thr
    if int(np.count_nonzero(keep)) < max(5, n_frames // 25):
        thr = 0.0 if max_e <= 0.0 else max_e * 1e-3
        keep = energies >= thr
    if int(np.count_nonzero(keep)) < 3:
        keep[:] = True

    max_f = min(float(fs) * 0.5 - 50.0, 5000.0 if fs >= 12000 else 3800.0)
    if max_f <= 200.0:
        max_f = float(fs) * 0.5 - 20.0
    n = 512
    freq = np.linspace(0.0, float(max_f), n, dtype=np.float64)
    w = 2.0 * np.pi * freq / float(fs)
    k_idx = np.arange(p + 1, dtype=np.float64)
    E = np.exp(-1j * np.outer(w, k_idx))

    times_s = np.empty((n_frames,), dtype=np.float64)
    f1 = np.full((n_frames,), np.nan, dtype=np.float64)
    f2 = np.full((n_frames,), np.nan, dtype=np.float64)
    f3 = np.full((n_frames,), np.nan, dtype=np.float64)

    sum_mag = np.zeros((n,), dtype=np.float64)
    count_mag = 0

    window = np.hamming(frame_len).astype(np.float64)
    for i, start in enumerate(starts):
        times_s[i] = (float(start) + 0.5 * float(frame_len)) / float(fs)
        if not bool(keep[i]):
            continue
        fr = x[start : start + frame_len].astype(np.float64, copy=True)
        if preemph != 0.0:
            fr[1:] = fr[1:] - float(preemph) * fr[:-1]
        fr *= window

        r = lpc.autocorrelation(fr, p)
        a, _k = lpc.levinson_durbin(r, p)
        if not np.all(np.isfinite(a)):
            continue
        a = np.asarray(a, dtype=np.float64).ravel()
        if a.size != p + 1:
            continue
        if a[0] == 0.0:
            a[0] = 1.0
        if a[0] != 1.0:
            a /= float(a[0])

        A = E @ a
        env_mag = 1.0 / (np.abs(A) + 1e-9)
        env_db = 20.0 * np.log10(env_mag + 1e-12)

        b1, b2, b3 = _estimate_formants_from_envelope_banded(freq, env_db, max_f_hz=float(max_f))
        if b1 is None or b2 is None or b3 is None:
            f_roots = _estimate_formants_from_lpc_roots(a, fs=fs, max_f_hz=float(max_f))
            f1_hi = min(1200.0, float(max_f))
            f2_hi = min(3500.0 if float(max_f) >= 4500.0 else 2600.0, float(max_f))
            for v in f_roots:
                vv = float(v)
                if b1 is None and 150.0 <= vv <= float(f1_hi):
                    b1 = vv
                    continue
                if b2 is None:
                    lo = 700.0 if b1 is None else max(700.0, float(b1) + 200.0)
                    if lo <= vv <= float(f2_hi):
                        b2 = vv
                        continue
                if b3 is None:
                    lo = 1500.0 if b2 is None else max(1500.0, float(b2) + 200.0)
                    if lo <= vv <= float(max_f):
                        b3 = vv
                        continue
        if b1 is not None and np.isfinite(b1):
            f1[i] = float(b1)
        if b2 is not None and np.isfinite(b2):
            f2[i] = float(b2)
        if b3 is not None and np.isfinite(b3):
            f3[i] = float(b3)

        sum_mag += env_mag
        count_mag += 1

    if count_mag > 0:
        env_mag_mean = sum_mag / float(count_mag)
        env_db_mean = 20.0 * np.log10(env_mag_mean + 1e-12)
    else:
        _freq, _env_db, _f123, _p2 = _analyze_lpc_envelope_and_formants(x, fs=fs, order=p, preemph=preemph)
        if _freq.size:
            freq = _freq
            env_db_mean = _env_db
            max_f = float(_freq[-1])
        else:
            env_db_mean = np.zeros_like(freq)

    f1_s = _median_filter_nan(f1, smooth_win)
    f2_s = _median_filter_nan(f2, smooth_win)
    f3_s = _median_filter_nan(f3, smooth_win)

    n1, med1, p10_1, p90_1 = _track_stats_1d(f1_s)
    n2, med2, p10_2, p90_2 = _track_stats_1d(f2_s)
    n3, med3, p10_3, p90_3 = _track_stats_1d(f3_s)
    stats = FormantTrackStats(
        frame_ms=float(frame_ms),
        hop_ms=float(hop_ms),
        n_total=int(n_frames),
        n_valid_f1=int(n1),
        n_valid_f2=int(n2),
        n_valid_f3=int(n3),
        f1_med_hz=med1,
        f1_p10_hz=p10_1,
        f1_p90_hz=p90_1,
        f2_med_hz=med2,
        f2_p10_hz=p10_2,
        f2_p90_hz=p90_2,
        f3_med_hz=med3,
        f3_p10_hz=p10_3,
        f3_p90_hz=p90_3,
    )
    return freq, env_db_mean, times_s, f1_s, f2_s, f3_s, stats, p, float(max_f)


def _write_formant_spectrum_png_from_envelope(
    png_path: Path,
    freq_hz: np.ndarray,
    env_db: np.ndarray,
    f123: tuple[float | None, float | None, float | None],
    order: int,
    width: int = 280,
    height: int = 140,
) -> bool:
    freq = np.asarray(freq_hz, dtype=np.float64).ravel()
    env = np.asarray(env_db, dtype=np.float64).ravel()
    if freq.size < 2 or env.size != freq.size:
        return False
    try:
        img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        img[:, :, 0] = 14
        img[:, :, 1] = 18
        img[:, :, 2] = 24

        lo = float(np.percentile(env, 5.0))
        hi = float(np.percentile(env, 95.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            lo = float(np.min(env))
            hi = float(np.max(env))
        if hi <= lo + 1e-6:
            hi = lo + 1.0

        # Horizontal reference lines.
        for frac in (0.2, 0.4, 0.6, 0.8):
            y = int(round((1.0 - frac) * (height - 1)))
            _draw_line(img, 0, y, width - 1, y, (35, 45, 58))

        # Vertical Hz grid lines.
        fmax = float(freq[-1])
        for hz in (500.0, 1000.0, 2000.0, 3000.0, 4000.0):
            if hz >= fmax:
                continue
            xf = int(round((hz / max(fmax, 1e-6)) * float(width - 1)))
            _draw_line(img, xf, 0, xf, height - 1, (35, 45, 58))

        # Envelope polyline.
        xpix = np.linspace(0, width - 1, freq.size, dtype=np.float64)
        ynorm = (env - lo) / (hi - lo)
        ypix = (height - 1) - np.clip(ynorm, 0.0, 1.0) * float(height - 1)
        _draw_polyline_xy(img, xpix, ypix, (77, 215, 255), thickness=2)

        # Mark F1/F2/F3 with colored vertical lines.
        marker_colors = [(255, 120, 120), (140, 255, 150), (255, 220, 120)]
        for idx, f_hz in enumerate(f123):
            if f_hz is None or not np.isfinite(f_hz):
                continue
            xf = int(round((float(f_hz) / max(fmax, 1e-6)) * float(width - 1)))
            _draw_line(img, xf, 0, xf, height - 1, marker_colors[idx])

        # Small order hint stripe (visual only).
        stripe_w = int(min(max(10, order), width // 3))
        img[0:2, 0:stripe_w, :] = np.asarray([48, 65, 84], dtype=np.uint8)[None, None, :]

        png_path.write_bytes(_png_encode_rgb8(img))
        return True
    except Exception:
        return False


def _write_formant_track_png(
    png_path: Path,
    times_s: np.ndarray,
    f1_hz: np.ndarray,
    f2_hz: np.ndarray,
    f3_hz: np.ndarray,
    max_hz: float,
    width: int = 280,
    height: int = 140,
) -> bool:
    t = np.asarray(times_s, dtype=np.float64).ravel()
    if t.size < 2:
        return False
    try:
        img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        img[:, :, 0] = 14
        img[:, :, 1] = 18
        img[:, :, 2] = 24

        t0 = float(t[0])
        t1 = float(t[-1])
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0 + 1e-9:
            t0 = 0.0
            t1 = float(max(t.size - 1, 1))

        fmax = float(max_hz)
        if not np.isfinite(fmax) or fmax <= 200.0:
            fmax = 4000.0

        # Grid: time.
        for frac in (0.0, 0.25, 0.50, 0.75, 1.0):
            x = int(round(float(frac) * float(width - 1)))
            _draw_line(img, x, 0, x, height - 1, (30, 40, 52))

        # Grid: Hz.
        for hz in (500.0, 1000.0, 2000.0, 3000.0, 4000.0):
            if hz >= fmax:
                continue
            y = int(round((1.0 - hz / max(fmax, 1e-6)) * float(height - 1)))
            _draw_line(img, 0, y, width - 1, y, (35, 45, 58))

        def to_xy(ff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            ff = np.asarray(ff, dtype=np.float64).ravel()
            if ff.size != t.size:
                ff = np.resize(ff, t.size)
            xpix = (t - t0) / max(t1 - t0, 1e-9) * float(width - 1)
            ypix = (1.0 - ff / max(fmax, 1e-6)) * float(height - 1)
            return xpix, ypix

        x1, y1 = to_xy(f1_hz)
        x2, y2 = to_xy(f2_hz)
        x3, y3 = to_xy(f3_hz)
        _draw_polyline_xy(img, x1, y1, (255, 120, 120), thickness=2)
        _draw_polyline_xy(img, x2, y2, (140, 255, 150), thickness=2)
        _draw_polyline_xy(img, x3, y3, (255, 220, 120), thickness=2)

        png_path.write_bytes(_png_encode_rgb8(img))
        return True
    except Exception:
        return False


def _write_formant_mel_png_with_tracks(
    wav_path: Path,
    png_path: Path,
    times_s: np.ndarray,
    f1_hz: np.ndarray,
    f2_hz: np.ndarray,
    f3_hz: np.ndarray,
    width: int = 280,
    height: int = 140,
    n_mels: int = 80,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> tuple[bool, int]:
    try:
        x, fs = wav_io.read_wav(wav_path)
        fs = int(fs)
        mel_db = _mel_spectrogram_db(x, fs=fs, n_mels=int(n_mels), win_ms=float(win_ms), hop_ms=float(hop_ms))

        # Approximate time stamps for mel frames (center of window).
        win_length = int(round(float(fs) * (float(win_ms) / 1000.0)))
        hop_length = int(round(float(fs) * (float(hop_ms) / 1000.0)))
        win_length = int(max(16, win_length))
        hop_length = int(max(1, hop_length))
        mel_frames = int(mel_db.shape[0])
        if mel_frames <= 0:
            return False, fs
        mel_times = (np.arange(mel_frames, dtype=np.float64) * float(hop_length) + 0.5 * float(win_length)) / float(fs)
        mel_t0 = float(mel_times[0])
        mel_t1 = float(mel_times[-1]) if mel_times.size >= 2 else float(mel_times[0] + 1e-3)

        mel_db_rs = _resample_time(mel_db, width=int(width))
        img = mel_db_rs.T[::-1, :]

        lo = float(np.percentile(img, 5.0))
        hi = float(np.percentile(img, 95.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-9:
            lo = float(np.min(img))
            hi = float(np.max(img))
        if hi <= lo + 1e-9:
            hi = lo + 1.0

        norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        u8 = np.round(norm * 255.0).astype(np.uint8)
        rgb = np.repeat(u8[:, :, None], 3, axis=2)

        # Hz grid lines on mel (horizontal lines mapped in mel scale).
        nyq = 0.5 * float(fs)
        mel_max = 2595.0 * np.log10(1.0 + nyq / 700.0)
        for hz in (500.0, 1000.0, 2000.0, 3000.0, 4000.0):
            if hz >= nyq:
                continue
            mel = 2595.0 * np.log10(1.0 + hz / 700.0)
            y = int(round((1.0 - mel / max(mel_max, 1e-9)) * float(height - 1)))
            _draw_line(rgb, 0, y, width - 1, y, (70, 80, 95))

        t = np.asarray(times_s, dtype=np.float64).ravel()
        if t.size >= 2 and mel_t1 > mel_t0 + 1e-9:
            xpix = (t - mel_t0) / max(mel_t1 - mel_t0, 1e-9) * float(width - 1)

            def to_ypix(ff: np.ndarray) -> np.ndarray:
                ff = np.asarray(ff, dtype=np.float64).ravel()
                if ff.size != t.size:
                    ff = np.resize(ff, t.size)
                mel = 2595.0 * np.log10(1.0 + np.clip(ff, 0.0, nyq) / 700.0)
                ypix = (1.0 - mel / max(mel_max, 1e-9)) * float(height - 1)
                return ypix

            y1 = to_ypix(f1_hz)
            y2 = to_ypix(f2_hz)
            y3 = to_ypix(f3_hz)
            _draw_polyline_xy(rgb, xpix, y1, (255, 120, 120), thickness=2)
            _draw_polyline_xy(rgb, xpix, y2, (140, 255, 150), thickness=2)
            _draw_polyline_xy(rgb, xpix, y3, (255, 220, 120), thickness=2)

        png_path.write_bytes(_png_encode_rgb8(rgb))
        return True, fs
    except Exception:
        return False, 0


def _write_formant_viz_for_wav(
    wav_path: Path,
    formant_dir: Path,
    width: int = 280,
    height: int = 140,
) -> FormantViz | None:
    try:
        x, fs_in = wav_io.read_wav(wav_path)
        fs = int(fs_in)
        freq, env_db, times_s, f1, f2, f3, stats, order, spec_max_hz = _analyze_lpc_formant_tracks(
            x,
            fs=fs,
            order=None,
            preemph=0.97,
            frame_ms=30.0,
            hop_ms=10.0,
            smooth_win=5,
        )
        if freq.size == 0 or times_s.size == 0:
            return None

        f123 = (stats.f1_med_hz, stats.f2_med_hz, stats.f3_med_hz)
        if any(v is None or not np.isfinite(v) for v in f123):
            b1, b2, b3 = _estimate_formants_from_envelope_banded(freq, env_db, max_f_hz=float(spec_max_hz))
            f123 = (
                stats.f1_med_hz if stats.f1_med_hz is not None else b1,
                stats.f2_med_hz if stats.f2_med_hz is not None else b2,
                stats.f3_med_hz if stats.f3_med_hz is not None else b3,
            )

        stem = wav_path.stem
        spec_name = f"{stem}_spec.png"
        mel_name = f"{stem}_mel.png"
        track_name = f"{stem}_track.png"
        spec_path = formant_dir / spec_name
        mel_path = formant_dir / mel_name
        track_path = formant_dir / track_name

        ok_spec = _write_formant_spectrum_png_from_envelope(
            spec_path,
            freq_hz=freq,
            env_db=env_db,
            f123=f123,
            order=int(order),
            width=int(width),
            height=int(height),
        )
        ok_mel, fs_ref = _write_formant_mel_png_with_tracks(
            wav_path,
            mel_path,
            times_s=times_s,
            f1_hz=f1,
            f2_hz=f2,
            f3_hz=f3,
            width=int(width),
            height=int(height),
            n_mels=80,
        )
        ok_track = _write_formant_track_png(
            track_path,
            times_s=times_s,
            f1_hz=f1,
            f2_hz=f2,
            f3_hz=f3,
            max_hz=float(spec_max_hz),
            width=int(width),
            height=int(height),
        )
        if not (ok_spec and ok_mel and ok_track):
            return None

        return FormantViz(
            mel_png_rel=f"formant/{mel_name}",
            track_png_rel=f"formant/{track_name}",
            spec_png_rel=f"formant/{spec_name}",
            fs=int(fs_ref),
            order=int(order),
            spec_max_hz=float(spec_max_hz),
            f1_hz=f123[0],
            f2_hz=f123[1],
            f3_hz=f123[2],
            stats=stats,
        )
    except Exception:
        return None


def _notch_filter_biquad(
    signal: np.ndarray,
    fs: int,
    center_hz: float,
    q_factor: float = 25.0,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """
    Apply a 2nd-order notch biquad at center_hz.
    """
    fs = int(fs)
    waveform = np.asarray(signal, dtype=np.float64).ravel()
    if waveform.size == 0 or fs <= 0:
        return waveform.copy()

    frequency = float(center_hz)
    if not np.isfinite(frequency) or frequency <= 0.0 or frequency >= 0.5 * float(fs):
        return waveform.copy()

    q_val = float(max(q_factor, 1.0))
    bw_scale = float(max(bandwidth_scale, 1e-3))
    bandwidth_hz = max(frequency / q_val, 40.0) * bw_scale
    radius = float(np.exp(-np.pi * bandwidth_hz / float(fs)))
    omega = 2.0 * np.pi * frequency / float(fs)
    cos_omega = float(np.cos(omega))

    b0 = 1.0
    b1 = -2.0 * cos_omega
    b2 = 1.0
    a1 = -2.0 * radius * cos_omega
    a2 = radius * radius

    out = np.zeros_like(waveform)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for index in range(int(waveform.size)):
        sample = float(waveform[index])
        y0 = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[index] = y0
        x2 = x1
        x1 = sample
        y2 = y1
        y1 = y0
    return out


def _notch_filter_biquad_track(
    signal: np.ndarray,
    fs: int,
    centers_hz: np.ndarray,
    hop_len: int,
    q_factor: float = 25.0,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """
    Time-varying notch: update center frequency every hop_len samples.

    centers_hz is a per-hop series (typically from formant tracking).
    """
    fs = int(fs)
    waveform = np.asarray(signal, dtype=np.float64).ravel()
    if waveform.size == 0 or fs <= 0:
        return waveform.copy()

    hop_len = int(max(1, hop_len))
    centers = np.asarray(centers_hz, dtype=np.float64).ravel()
    if centers.size == 0:
        return waveform.copy()

    n_samples = int(waveform.size)
    n_blocks = int((n_samples + hop_len - 1) // hop_len)

    # Build per-block center series (extend last value as needed).
    block_centers = np.empty((n_blocks,), dtype=np.float64)
    last_center = float("nan")
    for b in range(n_blocks):
        c = centers[b] if b < int(centers.size) else centers[-1]
        block_centers[b] = float(c)

    # Forward-fill/back-fill NaNs to avoid "gaps" disabling the notch.
    for b in range(n_blocks):
        c = float(block_centers[b])
        if np.isfinite(c):
            last_center = c
        else:
            block_centers[b] = last_center
    if not np.isfinite(float(block_centers[0])):
        finite = block_centers[np.isfinite(block_centers)]
        if finite.size:
            first = float(finite[0])
            for b in range(n_blocks):
                if not np.isfinite(float(block_centers[b])):
                    block_centers[b] = first
                else:
                    break

    if not bool(np.any(np.isfinite(block_centers))):
        return waveform.copy()

    q_val = float(max(q_factor, 1.0))
    bw_scale = float(max(bandwidth_scale, 1e-3))

    out = np.zeros_like(waveform)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0

    pos = 0
    for b in range(n_blocks):
        end = min(n_samples, pos + hop_len)
        center = float(block_centers[b])
        if not np.isfinite(center) or center <= 0.0 or center >= 0.5 * float(fs):
            # identity filter for invalid center
            b0, b1, b2, a1, a2 = 1.0, 0.0, 0.0, 0.0, 0.0
        else:
            bandwidth_hz = max(center / q_val, 40.0) * bw_scale
            radius = float(np.exp(-np.pi * bandwidth_hz / float(fs)))
            omega = 2.0 * np.pi * center / float(fs)
            cos_omega = float(np.cos(omega))
            b0 = 1.0
            b1 = -2.0 * cos_omega
            b2 = 1.0
            a1 = -2.0 * radius * cos_omega
            a2 = radius * radius

        for n in range(pos, end):
            sample = float(waveform[n])
            y0 = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[n] = y0
            x2 = x1
            x1 = sample
            y2 = y1
            y1 = y0
        pos = end
        if pos >= n_samples:
            break

    return out


def _apply_strong_formant_notch(
    signal: np.ndarray,
    fs: int,
    center_hz: float,
) -> np.ndarray:
    """
    Stronger perceptual formant removal than a single narrow notch.

    We cascade several notches at center +/- offsets with mixed Q to widen
    suppression around the formant band so the effect is clearly audible.
    """
    waveform = np.asarray(signal, dtype=np.float64).ravel()
    fs = int(fs)
    center = float(center_hz)
    if waveform.size == 0 or fs <= 0 or (not np.isfinite(center)):
        return waveform.copy()
    if center <= 0.0 or center >= 0.5 * float(fs):
        return waveform.copy()

    out = waveform.copy()
    for q_val in (5.0, 8.0, 12.0):
        out = _notch_filter_biquad(out, fs=fs, center_hz=center, q_factor=q_val, bandwidth_scale=5.0)

    for ratio in (0.92, 0.96, 1.04, 1.08):
        hz = center * ratio
        if hz <= 40.0 or hz >= 0.5 * float(fs) - 40.0:
            continue
        out = _notch_filter_biquad(out, fs=fs, center_hz=hz, q_factor=9.0, bandwidth_scale=5.0)

    return out


def _apply_ultra_formant_notch(
    signal: np.ndarray,
    fs: int,
    center_hz: float,
) -> np.ndarray:
    """
    Much stronger and wider notch profile for clearly audible formant removal.
    """
    waveform = np.asarray(signal, dtype=np.float64).ravel()
    fs = int(fs)
    center = float(center_hz)
    if waveform.size == 0 or fs <= 0 or (not np.isfinite(center)):
        return waveform.copy()
    if center <= 0.0 or center >= 0.5 * float(fs):
        return waveform.copy()

    out = waveform.copy()
    for q_val in (3.5, 5.0, 7.0, 10.0):
        out = _notch_filter_biquad(out, fs=fs, center_hz=center, q_factor=q_val, bandwidth_scale=5.0)

    for ratio in (0.86, 0.90, 0.94, 0.97, 1.03, 1.06, 1.10, 1.14):
        hz = center * ratio
        if hz <= 40.0 or hz >= 0.5 * float(fs) - 40.0:
            continue
        out = _notch_filter_biquad(out, fs=fs, center_hz=hz, q_factor=7.0, bandwidth_scale=5.0)

    # second pass around center to deepen attenuation
    for q_val in (4.0, 6.0):
        out = _notch_filter_biquad(out, fs=fs, center_hz=center, q_factor=q_val, bandwidth_scale=5.0)

    return out


def _apply_strong_formant_notch_track(
    signal: np.ndarray,
    fs: int,
    centers_hz: np.ndarray,
    hop_len: int,
) -> np.ndarray:
    waveform = np.asarray(signal, dtype=np.float64).ravel()
    fs = int(fs)
    if waveform.size == 0 or fs <= 0:
        return waveform.copy()
    centers = np.asarray(centers_hz, dtype=np.float64).ravel()
    if centers.size == 0:
        return waveform.copy()

    out = waveform.copy()
    for q_val in (5.0, 8.0, 12.0):
        out = _notch_filter_biquad_track(
            out,
            fs=fs,
            centers_hz=centers,
            hop_len=int(hop_len),
            q_factor=float(q_val),
            bandwidth_scale=5.0,
        )

    for ratio in (0.92, 0.96, 1.04, 1.08):
        out = _notch_filter_biquad_track(
            out,
            fs=fs,
            centers_hz=centers * float(ratio),
            hop_len=int(hop_len),
            q_factor=9.0,
            bandwidth_scale=5.0,
        )
    return out


def _apply_ultra_formant_notch_track(
    signal: np.ndarray,
    fs: int,
    centers_hz: np.ndarray,
    hop_len: int,
) -> np.ndarray:
    waveform = np.asarray(signal, dtype=np.float64).ravel()
    fs = int(fs)
    if waveform.size == 0 or fs <= 0:
        return waveform.copy()
    centers = np.asarray(centers_hz, dtype=np.float64).ravel()
    if centers.size == 0:
        return waveform.copy()

    out = waveform.copy()
    for q_val in (3.5, 5.0, 7.0, 10.0):
        out = _notch_filter_biquad_track(
            out,
            fs=fs,
            centers_hz=centers,
            hop_len=int(hop_len),
            q_factor=float(q_val),
            bandwidth_scale=5.0,
        )

    for ratio in (0.86, 0.90, 0.94, 0.97, 1.03, 1.06, 1.10, 1.14):
        out = _notch_filter_biquad_track(
            out,
            fs=fs,
            centers_hz=centers * float(ratio),
            hop_len=int(hop_len),
            q_factor=7.0,
            bandwidth_scale=5.0,
        )

    for q_val in (4.0, 6.0):
        out = _notch_filter_biquad_track(
            out,
            fs=fs,
            centers_hz=centers,
            hop_len=int(hop_len),
            q_factor=float(q_val),
            bandwidth_scale=5.0,
        )
    return out


def _extract_style_stats(data: bytes) -> StyleStats:
    h_any, hs = read_header(data)
    br = BitReader(data[hs:])

    frame_len = int(getattr(h_any, "frame_len"))
    subframe_len = int(getattr(h_any, "subframe_len"))
    subframes = frame_len // subframe_len
    rc_bits = int(getattr(h_any, "rc_bits"))
    lpc_order = int(getattr(h_any, "lpc_order"))
    gain_bits_p = int(getattr(h_any, "gain_bits_p"))
    gain_bits_c = int(getattr(h_any, "gain_bits_c"))
    mode_id = int(getattr(h_any, "mode"))
    lag_min = 0
    lag_bits = 0
    pitch_frac_bits = 0
    gp_max = 1.6
    gc_max = 6.0
    pos_bits = pitch.bits_for_pos(subframe_len)

    if isinstance(h_any, BitstreamHeaderV2):
        lag_min = int(h_any.lag_min)
        lag_bits = pitch.bits_for_lag(int(h_any.lag_min), int(h_any.lag_max), frac_bits=0)
        pitch_frac_bits = int(h_any.pitch_frac_bits)
    else:
        lag_min, _ = pitch.lag_bounds(int(h_any.fs), 50.0, 400.0, max_lag_bits=8)
        lag_bits = 8
        gp_max = 1.2
        gc_max = 2.0

    lags: list[float] = []
    gps: list[float] = []
    gcs: list[float] = []

    eof = False
    while True:
        try:
            for _ in range(lpc_order):
                _ = br.read_bits(rc_bits)
        except EOFError:
            break
        for _ in range(subframes):
            try:
                lag_i = int(br.read_bits(lag_bits))
                frac = int(br.read_bits(pitch_frac_bits)) if pitch_frac_bits else 0
                gp_idx = int(br.read_bits(gain_bits_p))
                gc_idx = int(br.read_bits(gain_bits_c))
            except EOFError:
                eof = True
                break

            lag = float(lag_min + lag_i)
            if pitch_frac_bits:
                lag += float(frac) / float(1 << pitch_frac_bits)
            lags.append(lag)
            gps.append(float(gains.dequantize_gain(gp_idx, gain_bits_p, xmin=1e-4, xmax=gp_max)))
            gcs.append(float(gains.dequantize_gain(gc_idx, gain_bits_c, xmin=1e-4, xmax=gc_max)))

            if isinstance(h_any, BitstreamHeaderV2):
                if mode_id == 0:
                    cb_bits = int(int(h_any.celp_codebook_size).bit_length() - 1)
                    for _s in range(int(h_any.celp_stages)):
                        _ = br.read_bits(cb_bits)
                else:
                    for _k in range(int(h_any.acelp_K)):
                        _ = br.read_bits(pos_bits)
                        _ = br.read_bits(int(h_any.acelp_weight_bits))
            else:
                if mode_id == 0:
                    _ = br.read_bits(9)
                else:
                    for _t in range(4):
                        _ = br.read_bits(4)
                        _ = br.read_bits(1)
        if eof:
            break

    if not lags:
        return StyleStats(lag_median=80.0, gp_mean=1.0, gc_mean=1.0)

    return StyleStats(
        lag_median=float(np.median(np.asarray(lags, dtype=np.float64))),
        gp_mean=float(np.mean(np.asarray(gps, dtype=np.float64))),
        gc_mean=float(np.mean(np.asarray(gcs, dtype=np.float64))),
    )


def _style_scale(src: float, tgt: float, alpha: float, lo: float, hi: float) -> float:
    src = float(max(src, 1e-6))
    tgt = float(max(tgt, 1e-6))
    a = float(min(max(alpha, 0.0), 1.0))
    s = float(np.exp(a * np.log(tgt / src)))
    return float(min(max(s, lo), hi))


def _cell_mel_images(
    mel_png: dict[str, str],
    triples: list[tuple[str, str]],
) -> str:
    blocks: list[str] = []
    for wav_name, label in triples:
        ref = mel_png.get(wav_name)
        if ref is None:
            continue
        blocks.append(
            "<div class='melbox'>"
            f"<img class='melspec' src='{html.escape(ref, quote=True)}' alt='mel {html.escape(label, quote=True)}'/>"
            f"<div class='small'>{html.escape(label, quote=True)}</div>"
            "</div>"
        )
    if not blocks:
        return ""
    return "<div class='melrow'>" + "".join(blocks) + "</div>"


def _fmt_hz(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "-"
    return f"{float(v):.0f}"


def _make_html(
    out_html: Path,
    items: list[Item],
    grid_wavs: dict[tuple[str, str], str],
    aug_wavs: dict[tuple[str, str], str],
    aug_specs: list[AugSpec],
    xnf_wavs: dict[tuple[str, str], str],
    xnf_specs: list[AugSpec],
    mel_png: dict[str, str],
    formant_viz: dict[str, FormantViz],
    formant_notch: dict[str, FormantNotch],
    grid_params: dict[tuple[str, str], timbre.TimbreParams],
) -> None:
    title = "CELP Timbre Demo (3x3)"
    style = """
    :root { --bg:#0b0f14; --panel:#121a22; --ink:#e7eef7; --muted:#9fb1c5; --border:#223043; --accent:#4dd4ff; }
    html,body{background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:0;}
    .wrap{max-width:1200px;margin:28px auto;padding:0 18px;}
    h1{font-size:22px;margin:0 0 10px 0;}
    h2{font-size:18px;margin:0 0 10px 0;}
    p{margin:8px 0;color:var(--muted);line-height:1.35;}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px;margin:14px 0;overflow-x:auto;}
    table{width:100%;border-collapse:separate;border-spacing:8px;}
    th,td{background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:12px;padding:10px;vertical-align:top;}
    th{color:var(--muted);font-weight:600;}
    .small{font-size:12px;color:var(--muted);}
    .label{font-size:13px;margin:0 0 6px 0;color:var(--accent);}
    audio{width:100%;}
    .grid th:first-child{width:18%;}
    .k{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px;color:var(--muted);}
    .melrow{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;}
    .melbox{display:flex;flex-direction:column;gap:3px;align-items:center;}
    img.melspec{width:98px;height:62px;object-fit:cover;border:1px solid var(--border);border-radius:6px;background:#000;}
    img.formantspec{width:240px;height:120px;object-fit:cover;border:1px solid var(--border);border-radius:6px;background:#000;}
    """

    def esc(text: str) -> str:
        return html.escape(text, quote=True)

    lines: list[str] = [
        "<!doctype html>",
        "<html><head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f"<style>{style}</style>",
        "</head><body>",
        '<div class="wrap">',
        f"<h1>{esc(title)}</h1>",
        "<p>Pipeline: wav -> encode(v2 ACELP) -> timbre transform (lag/gains/LSF) -> decode wav.</p>",
        "<p>3x3 使用更强 mimic：交叉项会联合迁移 f0、gain 和 LSF，便于听感对比。</p>",
    ]

    lines.extend(
        [
            '<div class="card">',
            "<h2>Reference</h2>",
            "<table>",
            "<tr><th>key</th><th>orig</th><th>recon (no timbre)</th></tr>",
        ]
    )
    for item in items:
        lines.append(
            f"<tr><td class='k'>{esc(item.key)}</td>"
            f"<td><audio controls src='{esc(item.key + '_orig.wav')}'></audio>"
            f"{_cell_mel_images(mel_png, [(item.key + '_orig.wav', 'orig')])}</td>"
            f"<td><audio controls src='{esc(item.key + '_recon.wav')}'></audio>"
            f"{_cell_mel_images(mel_png, [(item.key + '_recon.wav', 'recon')])}</td></tr>"
        )
    lines.extend(["</table>", "</div>"])

    lines.extend(
        [
            '<div class="card">',
            "<h2>Formants (LPC)</h2>",
            "<p class='small'>仅展示 3 条参考音频的 orig/recon。每条音频用 30ms 窗 / 10ms hop 分段做 LPC 包络，估计 F1/F2/F3 轨迹；mel 图叠加轨迹，另提供轨迹图与统计（median + P10/P90）。No F1/F2/F3 的陷波中心跟随分段轨迹（每 10ms 更新）；Std/Ultra 都把陷波带宽在之前基础上扩大 5x。</p>",
            "<table>",
            "<tr><th>key</th><th>type</th><th>audio</th><th>mel + tracks</th><th>tracks</th><th>LPC envelope</th><th>stats (Hz)</th><th>No F1</th><th>No F2</th><th>No F3</th></tr>",
        ]
    )
    for item in items:
        for typ in ("orig", "recon"):
            wav_name = f"{item.key}_{typ}.wav"
            v = formant_viz.get(wav_name)
            notch = formant_notch.get(wav_name)
            if v is None:
                mel_html = "<div class='small'>N/A</div>"
                track_html = "<div class='small'>N/A</div>"
                spec_html = "<div class='small'>N/A</div>"
                stats_html = "<div class='k'>-</div>"
            else:
                mel_html = (
                    f"<img class='formantspec' src='{esc(v.mel_png_rel)}' alt='mel formants {esc(wav_name)}'/>"
                    f"<div class='small'>y 轴真实 Hz（mel 频率轴），彩线=F1/F2/F3</div>"
                )
                track_html = (
                    f"<img class='formantspec' src='{esc(v.track_png_rel)}' alt='formant tracks {esc(wav_name)}'/>"
                    f"<div class='small'>x 轴时间（线性），y 轴真实 Hz，彩线=F1/F2/F3</div>"
                )
                spec_html = (
                    f"<img class='formantspec' src='{esc(v.spec_png_rel)}' alt='lpc spectrum formants {esc(wav_name)}'/>"
                    f"<div class='small'>x 轴真实 Hz（0~{int(round(v.spec_max_hz))}），LPC order={v.order}</div>"
                )

                def _band(m: float | None, p10: float | None, p90: float | None) -> str:
                    if m is None or not np.isfinite(m):
                        return "-"
                    if p10 is None or p90 is None or (not np.isfinite(p10)) or (not np.isfinite(p90)):
                        return f"{float(m):.0f}"
                    return f"{float(m):.0f} ({float(p10):.0f}–{float(p90):.0f})"

                s = v.stats
                stats_html = (
                    "<div class='k'>"
                    f"F1 {esc(_band(s.f1_med_hz, s.f1_p10_hz, s.f1_p90_hz))}<br/>"
                    f"F2 {esc(_band(s.f2_med_hz, s.f2_p10_hz, s.f2_p90_hz))}<br/>"
                    f"F3 {esc(_band(s.f3_med_hz, s.f3_p10_hz, s.f3_p90_hz))}<br/>"
                    f"valid {int(s.n_valid_f1)}/{int(s.n_valid_f2)}/{int(s.n_valid_f3)} of {int(s.n_total)}"
                    "</div>"
                )
            nof1_html = "<div class='small'>N/A</div>"
            nof2_html = "<div class='small'>N/A</div>"
            nof3_html = "<div class='small'>N/A</div>"
            if notch is not None:
                if notch.f1_wav is not None:
                    nof1_html = (
                        f"<div class='small'>Std</div><audio controls src='{esc(notch.f1_wav)}'></audio>"
                        f"{_cell_mel_images(mel_png, [(wav_name, 'base'), (notch.f1_wav, 'std')])}"
                    )
                    if notch.f1_ultra_wav is not None:
                        nof1_html += (
                            f"<div class='small'>Ultra</div><audio controls src='{esc(notch.f1_ultra_wav)}'></audio>"
                            f"{_cell_mel_images(mel_png, [(wav_name, 'base'), (notch.f1_ultra_wav, 'ultra')])}"
                        )
                if notch.f2_wav is not None:
                    nof2_html = (
                        f"<div class='small'>Std</div><audio controls src='{esc(notch.f2_wav)}'></audio>"
                        f"{_cell_mel_images(mel_png, [(wav_name, 'base'), (notch.f2_wav, 'std')])}"
                    )
                    if notch.f2_ultra_wav is not None:
                        nof2_html += (
                            f"<div class='small'>Ultra</div><audio controls src='{esc(notch.f2_ultra_wav)}'></audio>"
                            f"{_cell_mel_images(mel_png, [(wav_name, 'base'), (notch.f2_ultra_wav, 'ultra')])}"
                        )
                if notch.f3_wav is not None:
                    nof3_html = (
                        f"<div class='small'>Std</div><audio controls src='{esc(notch.f3_wav)}'></audio>"
                        f"{_cell_mel_images(mel_png, [(wav_name, 'base'), (notch.f3_wav, 'std')])}"
                    )
                    if notch.f3_ultra_wav is not None:
                        nof3_html += (
                            f"<div class='small'>Ultra</div><audio controls src='{esc(notch.f3_ultra_wav)}'></audio>"
                            f"{_cell_mel_images(mel_png, [(wav_name, 'base'), (notch.f3_ultra_wav, 'ultra')])}"
                        )
            lines.append(
                f"<tr><td class='k'>{esc(item.key)}</td>"
                f"<td class='k'>{esc(typ)}</td>"
                f"<td><audio controls src='{esc(wav_name)}'></audio></td>"
                f"<td>{mel_html}</td>"
                f"<td>{track_html}</td>"
                f"<td>{spec_html}</td>"
                f"<td>{stats_html}</td>"
                f"<td>{nof1_html}</td>"
                f"<td>{nof2_html}</td>"
                f"<td>{nof3_html}</td></tr>"
            )
    lines.extend(["</table>", "</div>"])

    lines.extend(
        [
            '<div class="card">',
            "<h2>3x3 Timbre Mimic Grid</h2>",
            '<table class="grid">',
            "<tr><th>source \\ target</th>",
        ]
    )
    for target in items:
        lines.append(f"<th>{esc(target.key)}<div class='small'>{esc(target.name)}</div></th>")
    lines.append("</tr>")
    for source in items:
        lines.append(f"<tr><th>{esc(source.key)}<div class='small'>{esc(source.name)}</div></th>")
        for target in items:
            wav_name = grid_wavs[(source.key, target.key)]
            p = grid_params[(source.key, target.key)]
            lines.append(
                f"<td><div class='label'>{esc(source.key)} -> {esc(target.key)}</div>"
                f"<div class='small'>lsf_mix={float(p.lsf_mix):.2f}, f0={float(p.f0_scale):.2f}, gp={float(p.gp_scale):.2f}, gc={float(p.gc_scale):.2f}</div>"
                f"<audio controls src='{esc(wav_name)}'></audio>"
                f"{_cell_mel_images(mel_png, [(source.key + '_recon.wav', 'src'), (target.key + '_recon.wav', 'tgt'), (wav_name, 'out')])}</td>"
            )
        lines.append("</tr>")
    lines.extend(["</table>", "</div>"])

    lines.extend(
        [
            '<div class="card">',
            "<h2>Augmentations</h2>",
            "<p class='small'>含 formant 花样（整体上/下、扩/压间距）以及非 formant 花样（f0 与 gp/gc 质感增广）。</p>",
            "<table>",
        ]
    )
    head = ["<tr><th>key</th>"]
    for spec in aug_specs:
        head.append(f"<th>{esc(spec.title)}<div class='small'>{esc(spec.note)}</div></th>")
    head.append("</tr>")
    lines.append("".join(head))

    for item in items:
        row = [f"<tr><td class='k'>{esc(item.key)}</td>"]
        for spec in aug_specs:
            aug_name = aug_wavs[(item.key, spec.tag)]
            row.append(
                f"<td><audio controls src='{esc(aug_name)}'></audio>"
                f"{_cell_mel_images(mel_png, [(item.key + '_recon.wav', 'base'), (aug_name, spec.tag)])}</td>"
            )
        row.append("</tr>")
        lines.append("".join(row))
    lines.extend(["</table>", "</div>"])

    lines.extend(
        [
            '<div class="card">',
            "<h2>Extreme Non-Formant</h2>",
            "<p class='small'>只动 f0/gp/gc，不动 formant_scale / lsf_spread / lsf_mix。</p>",
            "<table>",
        ]
    )
    head2 = ["<tr><th>key</th>"]
    for spec in xnf_specs:
        head2.append(f"<th>{esc(spec.title)}<div class='small'>{esc(spec.note)}</div></th>")
    head2.append("</tr>")
    lines.append("".join(head2))

    for item in items:
        row2 = [f"<tr><td class='k'>{esc(item.key)}</td>"]
        for spec in xnf_specs:
            wav_name = xnf_wavs[(item.key, spec.tag)]
            row2.append(
                f"<td><audio controls src='{esc(wav_name)}'></audio>"
                f"{_cell_mel_images(mel_png, [(item.key + '_recon.wav', 'base'), (wav_name, spec.tag)])}</td>"
            )
        row2.append("</tr>")
        lines.append("".join(row2))

    lines.extend(["</table>", "</div>", "</div></body></html>"])
    out_html.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="../S*X/examples/*.wav")
    parser.add_argument("--out-dir", type=str, default="timbre_demo")
    parser.add_argument("--fs", type=int, default=16000, choices=[8000, 16000])
    parser.add_argument("--lsf-mix", type=float, default=0.85)
    parser.add_argument("--grid-f0-alpha", type=float, default=0.80)
    parser.add_argument("--grid-gain-alpha", type=float, default=0.70)
    parser.add_argument("--mel-width", type=int, default=220)
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _pick_three(args.pattern)
    items = [Item(key=f"s{i}", wav_path=path, name=path.name) for i, path in enumerate(selected)]
    print("selected wavs:")
    for item in items:
        print(f"  {item.key}: {item.wav_path}")

    bitstreams: dict[str, bytes] = {}
    for item in items:
        bitstreams[item.key] = _encode_source(item.wav_path, out_dir, item.key, int(args.fs))

    # Build per-target LSF statistics once, reuse for 3x3 transforms.
    lsf_means: dict[str, np.ndarray] = {}
    style_stats: dict[str, StyleStats] = {}
    for item in items:
        header_any, _ = read_header(bitstreams[item.key])
        lsf_means[item.key] = timbre._target_lsf_mean_from_celpbin(  # type: ignore[attr-defined]
            bitstreams[item.key],
            fs_out=int(args.fs),
            lpc_order=int(getattr(header_any, "lpc_order")),
            min_sep_hz=50.0,
            edge_sep_hz=50.0,
        )
        style_stats[item.key] = _extract_style_stats(bitstreams[item.key])

    grid_wavs: dict[tuple[str, str], str] = {}
    grid_params: dict[tuple[str, str], timbre.TimbreParams] = {}
    for source in items:
        for target in items:
            if source.key == target.key:
                params = timbre.TimbreParams()
            else:
                st_s = style_stats[source.key]
                st_t = style_stats[target.key]
                params = timbre.TimbreParams(
                    f0_scale=_style_scale(
                        src=st_t.lag_median,
                        tgt=st_s.lag_median,
                        alpha=float(args.grid_f0_alpha),
                        lo=0.60,
                        hi=1.80,
                    ),
                    gp_scale=_style_scale(
                        src=st_s.gp_mean,
                        tgt=st_t.gp_mean,
                        alpha=float(args.grid_gain_alpha),
                        lo=0.60,
                        hi=1.80,
                    ),
                    gc_scale=_style_scale(
                        src=st_s.gc_mean,
                        tgt=st_t.gc_mean,
                        alpha=float(args.grid_gain_alpha),
                        lo=0.55,
                        hi=2.20,
                    ),
                    lsf_mix=float(args.lsf_mix),
                    target_lsf_mean=lsf_means[target.key],
                )
            grid_params[(source.key, target.key)] = params
            transformed, _ = timbre.transform_bitstream(
                bitstreams[source.key],
                params,
            )
            out_wav = out_dir / f"grid_{source.key}_to_{target.key}.wav"
            _decode_to_wav(transformed, out_wav)
            grid_wavs[(source.key, target.key)] = out_wav.name

    augment_specs = [
        AugSpec("f0_up", "f0_up", "升基频", timbre.TimbreParams(f0_scale=1.22)),
        AugSpec("f0_down", "f0_down", "降基频", timbre.TimbreParams(f0_scale=0.82)),
        AugSpec("f0_ext_up", "f0_ext_up", "更高基频", timbre.TimbreParams(f0_scale=1.38)),
        AugSpec("f0_ext_down", "f0_ext_down", "更低基频", timbre.TimbreParams(f0_scale=0.72)),
        AugSpec("breathy", "breathy", "gp↓, gc↑", timbre.TimbreParams(gp_scale=0.72, gc_scale=1.35)),
        AugSpec("periodic", "periodic", "gp↑, gc↓", timbre.TimbreParams(gp_scale=1.30, gc_scale=0.72)),
        AugSpec("noisy", "noisy", "gp↓↓, gc↑↑", timbre.TimbreParams(gp_scale=0.58, gc_scale=1.55)),
        AugSpec("formant_up", "formant_up", "整体升 formant", timbre.TimbreParams(formant_scale=1.12)),
        AugSpec("formant_down", "formant_down", "整体降 formant", timbre.TimbreParams(formant_scale=0.90)),
        AugSpec("formant_expand", "formant_expand", "LSF 扩间距", timbre.TimbreParams(lsf_spread=1.30)),
        AugSpec("formant_compress", "formant_compress", "LSF 压间距", timbre.TimbreParams(lsf_spread=0.78)),
        AugSpec(
            "chipmunk",
            "chipmunk",
            "f0↑ + formant↑ + 扩间距",
            timbre.TimbreParams(f0_scale=1.28, formant_scale=1.12, lsf_spread=1.20, gp_scale=0.95, gc_scale=1.12),
        ),
    ]
    aug_wavs: dict[tuple[str, str], str] = {}
    for item in items:
        for spec in augment_specs:
            transformed, _ = timbre.transform_bitstream(bitstreams[item.key], spec.params)
            out_wav = out_dir / f"aug_{item.key}_{spec.tag}.wav"
            _decode_to_wav(transformed, out_wav)
            aug_wavs[(item.key, spec.tag)] = out_wav.name

    extreme_non_formant_specs = [
        AugSpec("f0_ultra_up", "f0_ultra_up", "强升基频", timbre.TimbreParams(f0_scale=1.58)),
        AugSpec("f0_ultra_down", "f0_ultra_down", "强降基频", timbre.TimbreParams(f0_scale=0.62)),
        AugSpec("periodic_hard", "periodic_hard", "gp↑↑, gc↓↓", timbre.TimbreParams(gp_scale=1.62, gc_scale=0.52)),
        AugSpec("noisy_hard", "noisy_hard", "gp↓↓, gc↑↑", timbre.TimbreParams(gp_scale=0.42, gc_scale=1.85)),
        AugSpec(
            "high_buzzy",
            "high_buzzy",
            "f0↑ + gp↑ + gc↑",
            timbre.TimbreParams(f0_scale=1.42, gp_scale=1.28, gc_scale=1.38),
        ),
        AugSpec(
            "low_hollow",
            "low_hollow",
            "f0↓ + gp↓ + gc↑",
            timbre.TimbreParams(f0_scale=0.68, gp_scale=0.70, gc_scale=1.55),
        ),
    ]
    xnf_wavs: dict[tuple[str, str], str] = {}
    for item in items:
        for spec in extreme_non_formant_specs:
            transformed, _ = timbre.transform_bitstream(bitstreams[item.key], spec.params)
            out_wav = out_dir / f"xnf_{item.key}_{spec.tag}.wav"
            _decode_to_wav(transformed, out_wav)
            xnf_wavs[(item.key, spec.tag)] = out_wav.name

    formant_dir = out_dir / "formant"
    formant_dir.mkdir(parents=True, exist_ok=True)
    formant_viz: dict[str, FormantViz] = {}
    reference_wavs = [f"{it.key}_orig.wav" for it in items] + [f"{it.key}_recon.wav" for it in items]
    for wav_name in reference_wavs:
        wav_path = out_dir / wav_name
        if not wav_path.exists():
            continue
        viz = _write_formant_viz_for_wav(wav_path, formant_dir, width=280, height=140)
        if viz is None:
            continue
        formant_viz[wav_name] = viz

    formant_notch: dict[str, FormantNotch] = {}
    for wav_name in reference_wavs:
        wav_path = out_dir / wav_name
        viz = formant_viz.get(wav_name)
        if viz is None or (not wav_path.exists()):
            continue
        signal, fs_in = wav_io.read_wav(wav_path)
        stem = wav_path.stem

        # Recompute tracks for notch so the center can follow the segment-wise trajectory.
        _freq, _env_db, _times_s, f1_tr, f2_tr, f3_tr, _stats, _order, _spec_max_hz = _analyze_lpc_formant_tracks(
            signal,
            fs=int(fs_in),
            order=None,
            preemph=0.97,
            frame_ms=30.0,
            hop_ms=10.0,
            smooth_win=5,
        )
        hop_len = int(max(1, round(float(fs_in) * (10.0 / 1000.0))))

        def _make_notch_file(
            track_hz: np.ndarray,
            fallback_hz: float | None,
            label: str,
            ultra: bool = False,
        ) -> str | None:
            if track_hz.size and bool(np.any(np.isfinite(track_hz))):
                if ultra:
                    filtered = _apply_ultra_formant_notch_track(signal, int(fs_in), centers_hz=track_hz, hop_len=hop_len)
                    out_name = f"{stem}_notch_{label}_ultra.wav"
                else:
                    filtered = _apply_strong_formant_notch_track(signal, int(fs_in), centers_hz=track_hz, hop_len=hop_len)
                    out_name = f"{stem}_notch_{label}.wav"
                wav_io.write_wav(out_dir / out_name, filtered, int(fs_in), clip=True)
                return out_name

            if fallback_hz is None or not np.isfinite(fallback_hz):
                return None
            if ultra:
                filtered = _apply_ultra_formant_notch(signal, int(fs_in), center_hz=float(fallback_hz))
                out_name = f"{stem}_notch_{label}_ultra.wav"
            else:
                filtered = _apply_strong_formant_notch(signal, int(fs_in), center_hz=float(fallback_hz))
                out_name = f"{stem}_notch_{label}.wav"
            wav_io.write_wav(out_dir / out_name, filtered, int(fs_in), clip=True)
            return out_name

        formant_notch[wav_name] = FormantNotch(
            f1_wav=_make_notch_file(f1_tr, viz.f1_hz, "F1"),
            f2_wav=_make_notch_file(f2_tr, viz.f2_hz, "F2"),
            f3_wav=_make_notch_file(f3_tr, viz.f3_hz, "F3"),
            f1_ultra_wav=_make_notch_file(f1_tr, viz.f1_hz, "F1", ultra=True),
            f2_ultra_wav=_make_notch_file(f2_tr, viz.f2_hz, "F2", ultra=True),
            f3_ultra_wav=_make_notch_file(f3_tr, viz.f3_hz, "F3", ultra=True),
        )

    mel_dir = out_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    mel_png: dict[str, str] = {}
    for wav_path in sorted(out_dir.glob("*.wav")):
        png_name = wav_path.stem + ".png"
        png_path = mel_dir / png_name
        if _write_mel_png(wav_path, png_path, width=int(args.mel_width), n_mels=64):
            mel_png[wav_path.name] = f"mel/{png_name}"

    out_html = out_dir / "timbre_demo.html"
    _make_html(
        out_html,
        items,
        grid_wavs,
        aug_wavs,
        augment_specs,
        xnf_wavs,
        extreme_non_formant_specs,
        mel_png,
        formant_viz,
        formant_notch,
        grid_params,
    )
    print(f"wrote html: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
