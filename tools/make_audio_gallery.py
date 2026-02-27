from __future__ import annotations

import argparse
import html
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from celp_codec import bitstream as bs  # noqa: E402
from celp_codec import metrics  # noqa: E402
from celp_codec import pitch  # noqa: E402
from celp_codec import wav_io  # noqa: E402


@dataclass(frozen=True)
class WavItem:
    rel_path: str
    name: str
    size_bytes: int
    samplerate: int | None
    channels: int | None
    frames: int | None
    duration_s: float | None


@dataclass(frozen=True)
class BinItem:
    rel_bin: str
    rel_wav: str | None
    name: str
    file_bytes: int

    version: int
    mode: str
    fs: int
    frame_len: int
    subframe_len: int
    lpc_order: int
    rc_bits: int
    gain_bits_p: int
    gain_bits_c: int
    lag_min: int | None
    lag_max: int | None
    pitch_frac_bits: int | None
    acelp_K: int | None
    acelp_weight_bits: int | None
    celp_codebook_size: int | None
    celp_stages: int | None
    flags: int
    seed: int

    header_bits: int
    payload_bits_per_frame: int
    frames: int
    duration_s: float | None
    payload_bits_used: int
    payload_padding_bits: int
    total_min_bits: int
    file_bits: int

    method_key: str
    method_title: str
    method_meta_lines: tuple[str, ...]


@dataclass
class MethodGroup:
    key: str
    title: str
    meta_lines: tuple[str, ...]
    items: list[BinItem]


@dataclass(frozen=True)
class EvalResult:
    ref_rel: str
    snr_db: float
    seg_snr_db: float
    mel_snr_db: float
    n_samples: int


@dataclass(frozen=True)
class GroupEval:
    eval_count: int
    item_count: int
    refs: tuple[str, ...]
    snr_p50: float | None
    seg_p50: float | None
    mel_p50: float | None
    snr_mean: float | None
    seg_mean: float | None
    mel_mean: float | None
    seg_min: float | None
    seg_max: float | None
    score: float | None


def _wav_info(path: Path) -> tuple[int | None, int | None, int | None, float | None]:
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        return int(info.samplerate), int(info.channels), int(info.frames), float(info.duration)
    except Exception:
        pass

    try:
        import contextlib
        import wave

        with contextlib.closing(wave.open(str(path), "rb")) as w:
            samplerate = int(w.getframerate())
            channels = int(w.getnchannels())
            frames = int(w.getnframes())
        duration_s = float(frames / samplerate) if samplerate > 0 else None
        return samplerate, channels, frames, duration_s
    except Exception:
        return None, None, None, None


def _find_wavs(root: Path, globs: list[str], out_dir: Path) -> dict[str, WavItem]:
    items: dict[str, WavItem] = {}
    for g in globs:
        for p in root.glob(g):
            if not p.is_file() or p.suffix.lower() != ".wav":
                continue
            rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
            samplerate, channels, frames, duration_s = _wav_info(p)
            items[rel] = WavItem(
                rel_path=rel,
                name=p.name,
                size_bytes=int(p.stat().st_size),
                samplerate=samplerate,
                channels=channels,
                frames=frames,
                duration_s=duration_s,
            )
    return items


def _fmt_size(n: int) -> str:
    n = int(n)
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024.0:.1f} KiB"
    return f"{n / (1024.0 * 1024.0):.2f} MiB"


def _fmt_bits(n: int) -> str:
    n = int(n)
    if n < 10_000:
        return f"{n} bits"
    if n < 10_000_000:
        return f"{n / 1000.0:.1f} kbits"
    return f"{n / 1_000_000.0:.2f} Mbits"


def _fmt_dur(d: float | None) -> str:
    if d is None or not (d >= 0.0):
        return "-"
    m = int(d // 60)
    s = d - 60.0 * m
    return f"{m:d}:{s:05.2f}"


def _mode_str(mode_id: int) -> str:
    if int(mode_id) == 0:
        return "celp"
    if int(mode_id) == 1:
        return "acelp"
    return f"mode{int(mode_id)}"


def _v2_budget_and_meta(h: bs.BitstreamHeaderV2) -> tuple[int, int, tuple[str, ...], str]:
    subframes = int(h.frame_len) // int(h.subframe_len)
    lag_bits = pitch.bits_for_lag(int(h.lag_min), int(h.lag_max), frac_bits=0)
    pos_bits = pitch.bits_for_pos(int(h.subframe_len))

    if int(h.mode) == 0:
        cb_size = int(h.celp_codebook_size)
        cb_bits = int(cb_size).bit_length() - 1 if cb_size > 0 else 0
        innov_bits = int(h.celp_stages) * cb_bits
    else:
        innov_bits = int(h.acelp_K) * (pos_bits + int(h.acelp_weight_bits))

    subframe_bits = (
        lag_bits
        + int(h.pitch_frac_bits)
        + int(h.gain_bits_p)
        + int(h.gain_bits_c)
        + innov_bits
    )
    frame_payload_bits = int(h.lpc_order) * int(h.rc_bits) + subframes * subframe_bits

    flags = int(h.flags)
    flag_names = []
    if flags & (1 << 0):
        flag_names.append("POSTFILTER")
    if flags & (1 << 1):
        flag_names.append("LPC_INTERP")
    flags_str = "|".join(flag_names) if flag_names else "none"

    title_parts = [
        f"v2 {_mode_str(h.mode).upper()}",
        f"fs={int(h.fs)}",
        f"frame={int(h.frame_len)}",
        f"subframe={int(h.subframe_len)}",
        f"p={int(h.lpc_order)}",
    ]
    if int(h.mode) == 0:
        title_parts.append(f"cb={int(h.celp_codebook_size)}x{int(h.celp_stages)}")
    else:
        title_parts.append(f"K={int(h.acelp_K)} w={int(h.acelp_weight_bits)}")
    title = " / ".join(title_parts)

    bitrate_kbps = frame_payload_bits / (float(h.frame_len) / float(h.fs)) / 1000.0
    meta = (
        f"flags={flags_str} (0x{flags:02x})",
        "最小 bit 预算（payload / frame）："
        f"RC={int(h.lpc_order)}*{int(h.rc_bits)}={int(h.lpc_order)*int(h.rc_bits)}，"
        f"subframes={subframes}，"
        f"pitch=lag_bits({lag_bits})+frac({int(h.pitch_frac_bits)})，"
        f"gains={int(h.gain_bits_p)}+{int(h.gain_bits_c)}，"
        f"innov={innov_bits} -> per_subframe={subframe_bits}，"
        f"per_frame={frame_payload_bits} bits",
        f"对应 payload 码率（不含 header）：{bitrate_kbps:.2f} kbps",
        f"header 固定：{bs.HEADER_V2_SIZE*8} bits",
    )

    # method_key should group by bit allocation + decode-relevant flags (ignore seed)
    method_key = (
        f"v2|{_mode_str(h.mode)}|fs={int(h.fs)}|fl={int(h.frame_len)}|sl={int(h.subframe_len)}|"
        f"p={int(h.lpc_order)}|rc={int(h.rc_bits)}|gp={int(h.gain_bits_p)}|gc={int(h.gain_bits_c)}|"
        f"lag={int(h.lag_min)}..{int(h.lag_max)}|frac={int(h.pitch_frac_bits)}|"
        f"acelpK={int(h.acelp_K)}|w={int(h.acelp_weight_bits)}|"
        f"celpcb={int(h.celp_codebook_size)}|stg={int(h.celp_stages)}|flags=0x{flags:02x}"
    )

    return bs.HEADER_V2_SIZE * 8, frame_payload_bits, meta, method_key


def _v1_budget_and_meta(h: bs.BitstreamHeaderV1) -> tuple[int, int, tuple[str, ...], str]:
    subframes = int(h.frame_len) // int(h.subframe_len)
    lag_bits = 8  # fixed in v1 spec

    if int(h.mode) == 0:
        # CELP v1: fixed codebook size 512 -> 9 bits
        innov_bits = 9
    else:
        # ACELP v1: 4 tracks * (pos_idx 4 bits + sign 1 bit) = 20 bits
        innov_bits = 20

    subframe_bits = lag_bits + int(h.gain_bits_p) + int(h.gain_bits_c) + innov_bits
    frame_payload_bits = int(h.lpc_order) * int(h.rc_bits) + subframes * subframe_bits

    title = " / ".join(
        [
            f"v1 {_mode_str(h.mode).upper()}",
            f"fs={int(h.fs)}",
            f"frame={int(h.frame_len)}",
            f"subframe={int(h.subframe_len)}",
            f"p={int(h.lpc_order)}",
        ]
    )

    bitrate_kbps = frame_payload_bits / (float(h.frame_len) / float(h.fs)) / 1000.0
    meta = (
        "v1 固定字段：lag_bits=8；innovation: CELP=9bits / ACELP=20bits",
        "最小 bit 预算（payload / frame）："
        f"RC={int(h.lpc_order)}*{int(h.rc_bits)}={int(h.lpc_order)*int(h.rc_bits)}，"
        f"subframes={subframes}，"
        f"pitch=8，gains={int(h.gain_bits_p)}+{int(h.gain_bits_c)}，"
        f"innov={innov_bits} -> per_subframe={subframe_bits}，"
        f"per_frame={frame_payload_bits} bits",
        f"对应 payload 码率（不含 header）：{bitrate_kbps:.2f} kbps",
        f"header 固定：{bs.HEADER_V1_SIZE*8} bits",
    )

    method_key = (
        f"v1|{_mode_str(h.mode)}|fs={int(h.fs)}|fl={int(h.frame_len)}|sl={int(h.subframe_len)}|"
        f"p={int(h.lpc_order)}|rc={int(h.rc_bits)}|gp={int(h.gain_bits_p)}|gc={int(h.gain_bits_c)}"
    )
    return bs.HEADER_V1_SIZE * 8, frame_payload_bits, meta, method_key


def _match_recon_wav(root: Path, bin_path: Path) -> Path | None:
    stem = bin_path.stem
    cand = root / f"{stem}_recon.wav"
    if cand.exists():
        return cand
    cand2 = root / f"{stem}.wav"
    if cand2.exists():
        return cand2
    # last resort: try any matching recon wav
    for p in root.glob(f"{stem}*recon*.wav"):
        if p.is_file():
            return p
    return None


def _parse_celpbin(
    root: Path,
    out_dir: Path,
    p: Path,
) -> BinItem:
    data = p.read_bytes()
    file_bytes = len(data)
    header_any, header_size = bs.read_header(data)

    rel_bin = os.path.relpath(p, out_dir).replace(os.sep, "/")
    wav_path = _match_recon_wav(root, p)
    rel_wav = os.path.relpath(wav_path, out_dir).replace(os.sep, "/") if wav_path else None

    if isinstance(header_any, bs.BitstreamHeaderV2):
        h2 = header_any
        header_bits, payload_bits_per_frame, meta_lines, method_key = _v2_budget_and_meta(h2)
        version = bs.VERSION_V2
        mode = _mode_str(h2.mode)
        fs = int(h2.fs)
        frame_len = int(h2.frame_len)
        subframe_len = int(h2.subframe_len)
        lpc_order = int(h2.lpc_order)
        rc_bits = int(h2.rc_bits)
        gain_bits_p = int(h2.gain_bits_p)
        gain_bits_c = int(h2.gain_bits_c)
        lag_min = int(h2.lag_min)
        lag_max = int(h2.lag_max)
        pitch_frac_bits = int(h2.pitch_frac_bits)
        acelp_K = int(h2.acelp_K) if int(h2.mode) == 1 else None
        acelp_weight_bits = int(h2.acelp_weight_bits) if int(h2.mode) == 1 else None
        celp_codebook_size = int(h2.celp_codebook_size) if int(h2.mode) == 0 else None
        celp_stages = int(h2.celp_stages) if int(h2.mode) == 0 else None
        flags = int(h2.flags)
        seed = int(h2.seed)
        method_title = _method_title_from_v2(h2)
    else:
        h1 = header_any
        header_bits, payload_bits_per_frame, meta_lines, method_key = _v1_budget_and_meta(h1)
        version = bs.VERSION_V1
        mode = _mode_str(h1.mode)
        fs = int(h1.fs)
        frame_len = int(h1.frame_len)
        subframe_len = int(h1.subframe_len)
        lpc_order = int(h1.lpc_order)
        rc_bits = int(h1.rc_bits)
        gain_bits_p = int(h1.gain_bits_p)
        gain_bits_c = int(h1.gain_bits_c)
        lag_min = None
        lag_max = None
        pitch_frac_bits = None
        acelp_K = None
        acelp_weight_bits = None
        celp_codebook_size = None
        celp_stages = None
        flags = 0
        seed = int(h1.seed)
        method_title = _method_title_from_v1(h1)

    payload_bytes = max(0, len(data) - int(header_size))
    payload_bits_avail = payload_bytes * 8
    frames = payload_bits_avail // max(int(payload_bits_per_frame), 1)
    payload_bits_used = frames * int(payload_bits_per_frame)
    payload_padding_bits = max(0, payload_bits_avail - payload_bits_used)
    total_min_bits = int(header_bits) + payload_bits_used
    file_bits = len(data) * 8

    duration_s = None
    if fs > 0 and frame_len > 0:
        duration_s = float(frames * frame_len / float(fs))

    name = p.name
    return BinItem(
        rel_bin=rel_bin,
        rel_wav=rel_wav,
        name=name,
        file_bytes=int(file_bytes),
        version=version,
        mode=mode,
        fs=fs,
        frame_len=frame_len,
        subframe_len=subframe_len,
        lpc_order=lpc_order,
        rc_bits=rc_bits,
        gain_bits_p=gain_bits_p,
        gain_bits_c=gain_bits_c,
        lag_min=lag_min,
        lag_max=lag_max,
        pitch_frac_bits=pitch_frac_bits,
        acelp_K=acelp_K,
        acelp_weight_bits=acelp_weight_bits,
        celp_codebook_size=celp_codebook_size,
        celp_stages=celp_stages,
        flags=flags,
        seed=seed,
        header_bits=int(header_bits),
        payload_bits_per_frame=int(payload_bits_per_frame),
        frames=int(frames),
        duration_s=duration_s,
        payload_bits_used=int(payload_bits_used),
        payload_padding_bits=int(payload_padding_bits),
        total_min_bits=int(total_min_bits),
        file_bits=int(file_bits),
        method_key=method_key,
        method_title=method_title,
        method_meta_lines=tuple(meta_lines),
    )


def _method_title_from_v2(h: bs.BitstreamHeaderV2) -> str:
    mode = _mode_str(h.mode).upper()
    extra = (
        f"K={int(h.acelp_K)} w={int(h.acelp_weight_bits)}"
        if int(h.mode) == 1
        else f"cb={int(h.celp_codebook_size)}x{int(h.celp_stages)}"
    )
    flags = int(h.flags)
    flag_names = []
    if flags & (1 << 1):
        flag_names.append("LPC_INTERP")
    if flags & (1 << 0):
        flag_names.append("POSTFILTER")
    flags_str = "|".join(flag_names) if flag_names else "none"
    return (
        f"v2 {mode} / fs={int(h.fs)} / frame={int(h.frame_len)} / subframe={int(h.subframe_len)} / "
        f"p={int(h.lpc_order)} / {extra} / rc={int(h.rc_bits)} gp={int(h.gain_bits_p)} gc={int(h.gain_bits_c)} / "
        f"flags={flags_str}"
    )


def _method_title_from_v1(h: bs.BitstreamHeaderV1) -> str:
    mode = _mode_str(h.mode).upper()
    return (
        f"v1 {mode} / fs={int(h.fs)} / frame={int(h.frame_len)} / subframe={int(h.subframe_len)} / "
        f"p={int(h.lpc_order)} / rc={int(h.rc_bits)} gp={int(h.gain_bits_p)} gc={int(h.gain_bits_c)}"
    )


def _group_bins(items: list[BinItem]) -> list[MethodGroup]:
    groups: dict[str, MethodGroup] = {}
    for it in items:
        g = groups.get(it.method_key)
        if g is None:
            g = MethodGroup(key=it.method_key, title=it.method_title, meta_lines=it.method_meta_lines, items=[])
            groups[it.method_key] = g
        g.items.append(it)
    out = list(groups.values())
    for g in out:
        g.items.sort(key=lambda x: x.rel_bin.lower())
    out.sort(key=lambda g: g.title.lower())
    return out


def _payload_kbps(rep: BinItem) -> float:
    return float(rep.payload_bits_per_frame) / (float(rep.frame_len) / float(rep.fs)) / 1000.0


class _AudioCache:
    def __init__(self) -> None:
        self._raw: dict[Path, tuple[object, int]] = {}
        self._resampled: dict[tuple[Path, int], object] = {}

    def load(self, path: Path) -> tuple[object, int]:
        path = path.resolve()
        hit = self._raw.get(path)
        if hit is not None:
            return hit
        x, fs = wav_io.read_wav(path)
        out = (x, int(fs))
        self._raw[path] = out
        return out

    def resample(self, path: Path, fs_out: int) -> object:
        path = path.resolve()
        key = (path, int(fs_out))
        hit = self._resampled.get(key)
        if hit is not None:
            return hit
        x, fs_in = self.load(path)
        if int(fs_in) == int(fs_out):
            y = x
        else:
            y = wav_io.resample_to_fs(x, int(fs_in), int(fs_out))
        self._resampled[key] = y
        return y


def _find_reference_candidates(
    root: Path, out_dir: Path, ref: str | None, ref_globs: list[str] | None
) -> list[tuple[str, Path]]:
    if ref:
        p = Path(ref)
        if not p.is_absolute():
            p = (root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--ref not found: {p}")
        rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
        return [(rel, p)]

    globs = ref_globs if ref_globs else ["*_prompt.wav"]
    out: list[tuple[str, Path]] = []
    for g in globs:
        for p in root.glob(g):
            if not p.is_file() or p.suffix.lower() != ".wav":
                continue
            if "recon" in p.name.lower():
                continue
            rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
            out.append((rel, p))

    # de-dupe by resolved path
    seen: set[Path] = set()
    uniq: list[tuple[str, Path]] = []
    for rel, p in sorted(out, key=lambda t: t[0].lower()):
        pr = p.resolve()
        if pr in seen:
            continue
        seen.add(pr)
        uniq.append((rel, pr))
    return uniq


def _eval_item(
    it: BinItem,
    out_dir: Path,
    refs: list[tuple[str, Path]],
    cache: _AudioCache,
) -> EvalResult | None:
    if it.rel_wav is None or not refs:
        return None

    wav_path = (out_dir / it.rel_wav).resolve()
    if not wav_path.exists():
        return None

    y, y_fs = wav_io.read_wav(wav_path)
    target_fs = int(it.fs) if int(it.fs) > 0 else int(y_fs)
    if int(y_fs) != int(target_fs):
        y = wav_io.resample_to_fs(y, int(y_fs), int(target_fs))

    n_expected = int(it.frames) * int(it.frame_len)
    if n_expected > 0:
        y = y[:n_expected]

    best_ref_rel: str | None = None
    best_ref_path: Path | None = None
    best_snr = float("-inf")
    best_seg = float("nan")
    best_n = 0
    for ref_rel, ref_path in refs:
        x = cache.resample(ref_path, target_fs)
        if n_expected > 0:
            x = x[:n_expected]
        n = int(min(getattr(x, "size", 0), getattr(y, "size", 0)))
        if n <= 0:
            continue
        snr = float(metrics.snr_db(x[:n], y[:n]))
        seg = float(metrics.seg_snr_db(x[:n], y[:n], frame_len=int(it.frame_len)))
        if snr > best_snr:
            best_ref_rel = ref_rel
            best_ref_path = ref_path
            best_snr = snr
            best_seg = seg
            best_n = n

    if best_ref_rel is None or best_ref_path is None or best_n <= 0:
        return None

    x_best = cache.resample(best_ref_path, target_fs)
    if n_expected > 0:
        x_best = x_best[:n_expected]
    n = int(min(getattr(x_best, "size", 0), getattr(y, "size", 0)))
    n = int(min(n, best_n))
    if n <= 0:
        return None

    try:
        mel = float(metrics.mel_snr_db(x_best[:n], y[:n], fs=int(target_fs)))
    except Exception:
        mel = float("nan")

    return EvalResult(
        ref_rel=best_ref_rel,
        snr_db=float(best_snr),
        seg_snr_db=float(best_seg),
        mel_snr_db=mel,
        n_samples=int(n),
    )


def _safe_mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _safe_median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(statistics.median(vals))


def _safe_min(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(min(vals))


def _safe_max(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(max(vals))


def _compute_group_evals(
    groups: list[MethodGroup],
    evals_by_rel_bin: dict[str, EvalResult],
    best_by: str,
    lambda_kbps: float,
) -> list[GroupEval]:
    out: list[GroupEval] = []
    for g in groups:
        seg_vals: list[float] = []
        snr_vals: list[float] = []
        mel_vals: list[float] = []
        refs: list[str] = []
        for it in g.items:
            ev = evals_by_rel_bin.get(it.rel_bin)
            if ev is None:
                continue
            seg_vals.append(float(ev.seg_snr_db))
            snr_vals.append(float(ev.snr_db))
            mel_vals.append(float(ev.mel_snr_db))
            refs.append(ev.ref_rel)

        refs_uniq = tuple(sorted(set(refs)))
        seg_p50 = _safe_median(seg_vals)
        snr_p50 = _safe_median(snr_vals)
        mel_p50 = _safe_median(mel_vals)
        seg_mean = _safe_mean(seg_vals)
        snr_mean = _safe_mean(snr_vals)
        mel_mean = _safe_mean(mel_vals)
        seg_min = _safe_min(seg_vals)
        seg_max = _safe_max(seg_vals)

        score: float | None = None
        if g.items and seg_p50 is not None:
            rep = g.items[0]
            kbps = _payload_kbps(rep)
            if best_by == "quality":
                score = float(seg_p50)
            elif best_by == "efficiency":
                score = float(seg_p50) / float(kbps) if kbps > 0.0 else None
            else:
                score = float(seg_p50) - float(lambda_kbps) * float(kbps)

        out.append(
            GroupEval(
                eval_count=len(seg_vals),
                item_count=len(g.items),
                refs=refs_uniq,
                snr_p50=snr_p50,
                seg_p50=seg_p50,
                mel_p50=mel_p50,
                snr_mean=snr_mean,
                seg_mean=seg_mean,
                mel_mean=mel_mean,
                seg_min=seg_min,
                seg_max=seg_max,
                score=score,
            )
        )
    return out


def _choose_best_group_from_scores(groups: list[MethodGroup], group_evals: list[GroupEval]) -> int | None:
    if not groups:
        return None
    if len(groups) != len(group_evals):
        raise ValueError("groups and group_evals length mismatch")

    scored: list[tuple[int, BinItem, GroupEval]] = []
    for i, (g, ge) in enumerate(zip(groups, group_evals)):
        rep = g.items[0] if g.items else None
        if rep is None or ge.score is None:
            continue
        scored.append((i, rep, ge))
    if not scored:
        return None

    def key(t: tuple[int, BinItem, GroupEval]) -> tuple[float, float, float, int]:
        _i, rep, ge = t
        kbps = _payload_kbps(rep)
        # score desc, seg_p50 desc, kbps asc, eval_count desc
        return (
            float(ge.score),
            float(ge.seg_p50 if ge.seg_p50 is not None else float("-inf")),
            -float(kbps),
            int(ge.eval_count),
        )

    return max(scored, key=key)[0]


def _choose_best_quality(groups: list[MethodGroup], group_evals: list[GroupEval]) -> int | None:
    cand: list[tuple[int, float, float]] = []
    for i, (g, ge) in enumerate(zip(groups, group_evals)):
        if not g.items or ge.seg_p50 is None:
            continue
        cand.append((i, float(ge.seg_p50), _payload_kbps(g.items[0])))
    if not cand:
        return None
    # seg desc, kbps asc
    cand.sort(key=lambda t: (t[1], -t[2]), reverse=True)
    return cand[0][0]


def _choose_best_efficiency(groups: list[MethodGroup], group_evals: list[GroupEval]) -> int | None:
    cand: list[tuple[int, float, float, float]] = []
    for i, (g, ge) in enumerate(zip(groups, group_evals)):
        if not g.items or ge.seg_p50 is None:
            continue
        kbps = _payload_kbps(g.items[0])
        if kbps <= 0.0:
            continue
        cand.append((i, float(ge.seg_p50) / float(kbps), float(ge.seg_p50), kbps))
    if not cand:
        return None
    # eff desc, seg desc, kbps asc
    cand.sort(key=lambda t: (t[1], t[2], -t[3]), reverse=True)
    return cand[0][0]


def _render(
    groups: list[MethodGroup],
    group_evals: list[GroupEval],
    evals_by_rel_bin: dict[str, EvalResult],
    unmatched_wavs: list[WavItem],
    title: str,
    subtitle: str,
    best_by: str,
    lambda_kbps: float,
) -> str:
    best_idx = _choose_best_group_from_scores(groups, group_evals)
    best_quality_idx = _choose_best_quality(groups, group_evals)
    best_eff_idx = _choose_best_efficiency(groups, group_evals)

    summary_rows = []
    for i, (g, ge) in enumerate(zip(groups, group_evals)):
        if not g.items:
            continue
        rep = g.items[0]
        total_dur = sum((it.duration_s or 0.0) for it in g.items if it.duration_s is not None)
        payload_kbps = _payload_kbps(rep)

        marks = []
        if best_idx == i:
            marks.append("综合最优")
        if best_quality_idx == i and best_quality_idx != best_idx:
            marks.append("质量最佳")
        if best_eff_idx == i and best_eff_idx not in (best_idx, best_quality_idx):
            marks.append("效率最佳")
        mark_html = "<br/>".join(html.escape(m) for m in marks)

        snr_cell = "-" if ge.snr_p50 is None else f"{ge.snr_p50:.2f}"
        seg_cell = "-" if ge.seg_p50 is None else f"{ge.seg_p50:.2f}"
        mel_cell = "-" if ge.mel_p50 is None else f"{ge.mel_p50:.2f}"
        score_cell = "-" if ge.score is None else f"{ge.score:.2f}"
        eval_cell = f"{ge.eval_count}/{ge.item_count}"
        if not ge.refs:
            refs_cell = "-"
        elif len(ge.refs) <= 2:
            refs_cell = ", ".join(ge.refs)
        else:
            refs_cell = ", ".join([ge.refs[0], ge.refs[1], "…"])

        summary_rows.append(
            "\n".join(
                [
                    f"<tr class=\"sumrow\" data-name=\"{html.escape(g.title.lower())}\">",
                    f"  <td class=\"sum_mark\">{mark_html}</td>",
                    f"  <td class=\"sum_method\"><a href=\"#m{i}\">{html.escape(g.title)}</a></td>",
                    f"  <td class=\"sum_bins\">{len(g.items)}</td>",
                    f"  <td class=\"sum_dur\">{html.escape(_fmt_dur(total_dur))}</td>",
                    f"  <td class=\"sum_bits\">{rep.payload_bits_per_frame}</td>",
                    f"  <td class=\"sum_kbps\">{payload_kbps:.2f}</td>",
                    f"  <td class=\"sum_hdr\">{rep.header_bits}</td>",
                    f"  <td class=\"sum_snr\">{html.escape(snr_cell)}</td>",
                    f"  <td class=\"sum_segsnr\">{html.escape(seg_cell)}</td>",
                    f"  <td class=\"sum_melsnr\">{html.escape(mel_cell)}</td>",
                    f"  <td class=\"sum_eval\">{html.escape(eval_cell)}</td>",
                    f"  <td class=\"sum_score\">{html.escape(score_cell)}</td>",
                    f"  <td class=\"sum_refs\">{html.escape(refs_cell)}</td>",
                    "</tr>",
                ]
            )
        )

    summary_html = (
        "\n".join(summary_rows)
        if summary_rows
        else "<tr><td colspan=\"13\">(no methods)</td></tr>"
    )

    best_callout = ""
    if best_idx is not None and 0 <= best_idx < len(groups) and groups[best_idx].items:
        def _fmt_method(idx: int | None, extra: str) -> str:
            if idx is None or not (0 <= idx < len(groups)) or not groups[idx].items:
                return "-"
            rep2 = groups[idx].items[0]
            kbps2 = _payload_kbps(rep2)
            ge2 = group_evals[idx] if 0 <= idx < len(group_evals) else None
            seg2 = "-" if ge2 is None or ge2.seg_p50 is None else f"{ge2.seg_p50:.2f} dB"
            mel2 = "-" if ge2 is None or ge2.mel_p50 is None else f"{ge2.mel_p50:.2f} dB"
            return (
                f"<a href=\"#m{idx}\"><code>{html.escape(groups[idx].title)}</code></a>"
                f"（segSNR_p50={seg2}，melSNR_p50={mel2}，payload≈{kbps2:.2f} kbps{extra}）"
            )

        if best_by == "quality":
            score_formula = "score = segSNR_p50"
        elif best_by == "efficiency":
            score_formula = "score = segSNR_p50 / payload_kbps"
        else:
            score_formula = f"score = segSNR_p50 - λ * payload_kbps（λ={lambda_kbps:g}）"

        best_ge = group_evals[best_idx] if 0 <= best_idx < len(group_evals) else None
        best_score = "-" if best_ge is None or best_ge.score is None else f"{best_ge.score:.2f}"

        eff_str = "-"
        if best_eff_idx is not None and 0 <= best_eff_idx < len(groups) and groups[best_eff_idx].items:
            rep_eff = groups[best_eff_idx].items[0]
            kbps_eff = _payload_kbps(rep_eff)
            ge_eff = group_evals[best_eff_idx] if 0 <= best_eff_idx < len(group_evals) else None
            if ge_eff is not None and ge_eff.seg_p50 is not None and kbps_eff > 0.0:
                eff_str = f"{(float(ge_eff.seg_p50) / float(kbps_eff)):.3f}"

        best_callout = (
            "<div class=\"best\">"
            f"<div class=\"best-title\">综合最优（{html.escape(best_by)}）</div>"
            "<div class=\"best-body\">"
            f"<div class=\"best-line\">综合评分：<code>{html.escape(score_formula)}</code></div>"
            f"<div class=\"best-line\">综合最优：{_fmt_method(best_idx, f'，score={best_score}')}</div>"
            f"<div class=\"best-line\">质量最佳：{_fmt_method(best_quality_idx, '')}</div>"
            f"<div class=\"best-line\">效率最佳：{_fmt_method(best_eff_idx, f'，segSNR_p50/payload_kbps={eff_str}')}</div>"
            "</div>"
            "</div>"
        )

    sections = []
    for gi, (g, ge) in enumerate(zip(groups, group_evals)):
        rows = []
        for it in g.items:
            bin_url = quote(it.rel_bin)
            wav_cell = "-"
            audio_cell = "-"
            wav_dl = ""
            if it.rel_wav is not None:
                wav_url = quote(it.rel_wav)
                wav_cell = f"<code>{html.escape(it.rel_wav)}</code>"
                audio_cell = f"<audio controls preload=\"none\" src=\"{wav_url}\"></audio>"
                wav_dl = f" | <a href=\"{wav_url}\" download>wav</a>"

            min_bytes = (int(it.total_min_bits) + 7) // 8
            overhead_bits = int(it.file_bits) - int(it.total_min_bits)
            overhead_bytes = int(it.file_bytes) - int(min_bytes)
            bits_lines = [
                f"min_total={_fmt_bits(it.total_min_bits)} (≈{min_bytes} bytes)",
                f"file={it.file_bytes} bytes ({_fmt_bits(it.file_bits)})",
                f"overhead={overhead_bits} bits ({overhead_bytes} bytes)",
                f"(header={it.header_bits} + payload={_fmt_bits(it.payload_bits_used)})",
                f"payload/frame={it.payload_bits_per_frame} bits",
                f"padding={it.payload_padding_bits} bits",
            ]
            bits_html = "<br/>".join(html.escape(s) for s in bits_lines)

            q_html = "-"
            ev = evals_by_rel_bin.get(it.rel_bin)
            if ev is not None:
                q_lines = [
                    f"ref={ev.ref_rel}",
                    f"SNR={ev.snr_db:.2f} dB",
                    f"segSNR={ev.seg_snr_db:.2f} dB",
                    f"melSNR={ev.mel_snr_db:.2f} dB",
                ]
                q_html = "<br/>".join(html.escape(s) for s in q_lines)

            data_name = " ".join(
                [
                    g.title.lower(),
                    it.rel_bin.lower(),
                    (it.rel_wav.lower() if it.rel_wav else ""),
                ]
            )
            rows.append(
                "\n".join(
                    [
                        f"<tr class=\"row\" data-name=\"{html.escape(data_name)}\">",
                        f"  <td class=\"name\"><code>{html.escape(it.rel_bin)}</code></td>",
                        f"  <td class=\"meta\">frames={it.frames}<br/>dur={html.escape(_fmt_dur(it.duration_s))}</td>",
                        f"  <td class=\"bits\">{bits_html}</td>",
                        f"  <td class=\"q\">{q_html}</td>",
                        f"  <td class=\"wav\">{wav_cell}</td>",
                        f"  <td class=\"player\">{audio_cell}</td>",
                        "  <td class=\"dl\">"
                        f"<a href=\"{bin_url}\" download>bin</a>{wav_dl}"
                        "</td>",
                        "</tr>",
                    ]
                )
            )

        rows_html = "\n".join(rows) if rows else "<tr><td colspan=\"7\">(no items)</td></tr>"
        meta_lines = list(g.meta_lines)
        if ge.eval_count > 0 and g.items:
            rep = g.items[0]
            kbps = _payload_kbps(rep)
            refs = ", ".join(ge.refs) if ge.refs else "-"
            snr_p50 = "-" if ge.snr_p50 is None else f"{ge.snr_p50:.2f} dB"
            seg_p50 = "-" if ge.seg_p50 is None else f"{ge.seg_p50:.2f} dB"
            mel_p50 = "-" if ge.mel_p50 is None else f"{ge.mel_p50:.2f} dB"
            score = "-" if ge.score is None else f"{ge.score:.2f}"
            meta_lines.append(
                f"测评：eval={ge.eval_count}/{ge.item_count}，refs={refs}，"
                f"SNR_p50={snr_p50}，segSNR_p50={seg_p50}，melSNR_p50={mel_p50}，payload≈{kbps:.2f} kbps，score={score}"
            )
        meta_html = "<br/>".join(html.escape(x) for x in meta_lines)
        sections.append(
            f"""
    <section class="method" id="m{gi}">
      <h2>{html.escape(g.title)}</h2>
      <p class="method-meta">{meta_html}</p>
      <table>
        <thead>
	          <tr>
	            <th>bitstream</th>
	            <th>长度</th>
	            <th>最小 bit 数（含必要信息）</th>
	            <th>质量（SNR/segSNR/melSNR）</th>
	            <th>recon wav</th>
	            <th>播放</th>
	            <th></th>
	          </tr>
        </thead>
        <tbody>
{rows_html}
        </tbody>
      </table>
    </section>
"""
        )

    unmatched_section = ""
    if unmatched_wavs:
        rows = []
        for it in unmatched_wavs:
            rel_url = quote(it.rel_path)
            meta = []
            if it.samplerate is not None:
                meta.append(f"{it.samplerate} Hz")
            if it.channels is not None:
                meta.append(f"{it.channels} ch")
            if it.duration_s is not None:
                meta.append(_fmt_dur(it.duration_s))
            meta_str = " / ".join(meta) if meta else "-"
            data_name = it.rel_path.lower()
            rows.append(
                "\n".join(
                    [
                        f"<tr class=\"row\" data-name=\"{html.escape(data_name)}\">",
                        f"  <td class=\"name\"><code>{html.escape(it.rel_path)}</code></td>",
                        f"  <td class=\"meta\">{html.escape(meta_str)}</td>",
                        f"  <td class=\"size\">{html.escape(_fmt_size(it.size_bytes))}</td>",
                        f"  <td class=\"player\"><audio controls preload=\"none\" src=\"{rel_url}\"></audio></td>",
                        f"  <td class=\"dl\"><a href=\"{rel_url}\" download>download</a></td>",
                        "</tr>",
                    ]
                )
            )
        rows_html = "\n".join(rows)
        unmatched_section = f"""
    <section class="method">
      <h2>未匹配的 WAV</h2>
      <p class="method-meta">这些 wav 没有在同目录找到对应的 *.celpbin，因此无法计算“压缩最小 bit 数”。</p>
      <table>
        <thead>
          <tr>
            <th>文件</th>
            <th>信息</th>
            <th>大小</th>
            <th>播放</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
{rows_html}
        </tbody>
      </table>
    </section>
"""

    sections_html = "\n".join(sections) if sections else "<p>(no bitstreams found)</p>"

    return f"""<!doctype html>
<html lang="zh-Hans">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title)}</title>
    <style>
      :root {{
        color-scheme: light dark;
        --bg: #0b0d12;
        --fg: #e9eef5;
        --muted: #a9b3c4;
        --card: rgba(255,255,255,0.04);
        --border: rgba(255,255,255,0.10);
        --accent: #7aa2ff;
      }}
      body {{
        margin: 0;
        padding: 24px;
        background: var(--bg);
        color: var(--fg);
        font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }}
      h1 {{
        margin: 0 0 6px 0;
        font-size: 22px;
        letter-spacing: 0.2px;
      }}
      h2 {{
        margin: 18px 0 8px 0;
        font-size: 16px;
        letter-spacing: 0.1px;
      }}
      .sub {{
        margin: 0 0 16px 0;
        color: var(--muted);
      }}
      .bar {{
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
        padding: 12px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 14px;
        position: sticky;
        top: 0;
        backdrop-filter: blur(6px);
      }}
      input[type="search"] {{
        flex: 1;
        min-width: 280px;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(0,0,0,0.25);
        color: var(--fg);
        outline: none;
      }}
      button {{
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(0,0,0,0.25);
        color: var(--fg);
        cursor: pointer;
      }}
      button:hover {{
        border-color: rgba(255,255,255,0.18);
      }}
      .hint {{
        color: var(--muted);
        font-size: 12px;
      }}
      .method-meta {{
        margin: 0 0 10px 0;
        color: var(--muted);
        font-size: 12px;
      }}
      .best {{
        padding: 12px;
        border-radius: 12px;
        border: 1px solid rgba(122,162,255,0.45);
        background: rgba(122,162,255,0.08);
        margin: 10px 0 14px 0;
      }}
      .best-title {{
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .best-body {{
        color: var(--fg);
      }}
      .best-line {{
        margin: 4px 0;
        color: var(--fg);
      }}
      section.method {{
        margin-bottom: 18px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
      }}
      th, td {{
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        vertical-align: top;
      }}
      th {{
        text-align: left;
        font-weight: 600;
        color: var(--muted);
        background: rgba(0,0,0,0.18);
      }}
      tr:last-child td {{
        border-bottom: 0;
      }}
      code {{
        color: var(--fg);
      }}
      a {{
        color: var(--accent);
        text-decoration: none;
      }}
      a:hover {{
        text-decoration: underline;
      }}
      .meta, .size, .bits, .q {{
        white-space: nowrap;
        color: var(--muted);
        font-variant-numeric: tabular-nums;
      }}
      audio {{
        width: min(560px, 100%);
      }}
      td.player {{
        min-width: 320px;
      }}
      table.summary {{
        margin-bottom: 12px;
      }}
      table.summary th, table.summary td {{
        vertical-align: middle;
      }}
      .sum_mark {{
        white-space: nowrap;
        color: var(--fg);
      }}
      .sum_bins, .sum_dur, .sum_bits, .sum_kbps, .sum_hdr {{
        white-space: nowrap;
        color: var(--muted);
        font-variant-numeric: tabular-nums;
      }}
      .sum_snr, .sum_segsnr, .sum_melsnr, .sum_eval, .sum_score, .sum_refs {{
        white-space: nowrap;
        color: var(--muted);
        font-variant-numeric: tabular-nums;
      }}
    </style>
  </head>
  <body>
    <h1>{html.escape(title)}</h1>
    <p class="sub">{html.escape(subtitle)}</p>

    <div class="bar">
      <input id="q" type="search" placeholder="按方法/文件名过滤（例如：v2 acelp / lpc_interp / out10）" />
      <button id="stop">Stop all</button>
      <span class="hint" id="count"></span>
    </div>

{best_callout}

    <section class="method" id="summary">
      <h2>方法横向对比（按 bitstream header 分组）</h2>
      <p class="method-meta">表中 “payload/frame” 与 “payload kbps” 由 header 推导，代表最小必要 payload 比特预算（不含文件尾部 padding）。若目录存在参考 WAV（默认 <code>*_prompt.wav</code>，或用 <code>--ref</code> 指定），会自动计算 SNR/segSNR，并按综合评分选出“综合最优”。</p>
      <table class="summary">
        <thead>
          <tr>
            <th></th>
            <th>方法</th>
            <th>bins</th>
            <th>总时长</th>
            <th>payload/frame (bits)</th>
            <th>payload (kbps)</th>
            <th>header (bits)</th>
            <th>SNR_p50 (dB)</th>
            <th>segSNR_p50 (dB)</th>
            <th>melSNR_p50 (dB)</th>
            <th>eval</th>
            <th>score</th>
            <th>refs</th>
          </tr>
        </thead>
        <tbody>
{summary_html}
        </tbody>
      </table>
    </section>

{sections_html}
{unmatched_section}

    <script>
      const rows = Array.from(document.querySelectorAll('tr.row, tr.sumrow'));
      const audios = Array.from(document.querySelectorAll('audio'));
      const q = document.getElementById('q');
      const count = document.getElementById('count');

      function updateCount() {{
        const vis = rows.filter(r => r.style.display !== 'none').length;
        count.textContent = `显示 ${{vis}} / ${{rows.length}}`;
      }}

      function filter() {{
        const term = (q.value || '').trim().toLowerCase();
        for (const r of rows) {{
          const name = (r.dataset.name || '');
          r.style.display = (!term || name.includes(term)) ? '' : 'none';
        }}
        updateCount();
      }}

      q.addEventListener('input', filter);
      document.getElementById('stop').addEventListener('click', () => {{
        for (const a of audios) {{
          a.pause();
          try {{ a.currentTime = 0; }} catch (e) {{}}
        }}
      }});

      // Keep only one audio playing at a time.
      for (const a of audios) {{
        a.addEventListener('play', () => {{
          for (const b of audios) {{
            if (b !== a) b.pause();
          }}
        }});
      }}

      updateCount();
    </script>
  </body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate a playable HTML page for recon WAVs, grouped by method.")
    p.add_argument("--root", type=str, default=".", help="Directory to scan (default: .)")
    p.add_argument(
        "--glob-bin",
        dest="glob_bins",
        action="append",
        default=None,
        help="Glob for bitstreams (repeatable). Default: *.celpbin",
    )
    p.add_argument(
        "--glob-wav",
        dest="glob_wavs",
        action="append",
        default=None,
        help="Glob for wavs (repeatable). Default: *recon*.wav",
    )
    # Back-compat: --glob is an alias for --glob-wav
    p.add_argument("--glob", dest="glob_wav_alias", action="append", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out", type=str, default="recon_gallery.html", help="Output HTML path")
    p.add_argument("--title", type=str, default="重建 WAV 播放页（按方法分组）", help="HTML title")
    p.add_argument("--ref", type=str, default=None, help="Reference WAV for metrics (optional).")
    p.add_argument(
        "--ref-glob",
        dest="ref_globs",
        action="append",
        default=None,
        help="Glob for reference WAV candidates (repeatable). Default: *_prompt.wav",
    )
    p.add_argument(
        "--best-by",
        type=str,
        choices=["balanced", "quality", "efficiency"],
        default="balanced",
        help="How to choose '综合最优' when metrics are available.",
    )
    p.add_argument(
        "--lambda-kbps",
        type=float,
        default=0.15,
        help="Balanced score: score = segSNR_p50 - lambda_kbps * payload_kbps",
    )
    p.add_argument("--no-metrics", action="store_true", help="Skip computing SNR/segSNR.")
    args = p.parse_args(argv)

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()
    out_dir = out_path.parent

    glob_bins = args.glob_bins if args.glob_bins else ["*.celpbin"]
    glob_wavs = args.glob_wavs if args.glob_wavs else ["*recon*.wav"]
    if args.glob_wav_alias:
        glob_wavs.extend(args.glob_wav_alias)

    wavs_by_rel = _find_wavs(root, glob_wavs, out_dir=out_dir)

    bin_items: list[BinItem] = []
    for g in glob_bins:
        for pth in root.glob(g):
            if not pth.is_file() or pth.suffix.lower() != ".celpbin":
                continue
            try:
                bin_items.append(_parse_celpbin(root, out_dir, pth))
            except Exception as e:
                rel = os.path.relpath(pth, out_dir).replace(os.sep, "/")
                print(f"skip {rel}: {e}")

    groups = _group_bins(bin_items)

    evals_by_rel_bin: dict[str, EvalResult] = {}
    if not args.no_metrics:
        try:
            refs = _find_reference_candidates(root, out_dir, ref=args.ref, ref_globs=args.ref_globs)
        except Exception as e:
            refs = []
            print(f"metrics disabled: {e}")

        if refs:
            cache = _AudioCache()
            for it in bin_items:
                ev = _eval_item(it, out_dir=out_dir, refs=refs, cache=cache)
                if ev is not None:
                    evals_by_rel_bin[it.rel_bin] = ev

    group_evals = _compute_group_evals(
        groups,
        evals_by_rel_bin=evals_by_rel_bin,
        best_by=str(args.best_by),
        lambda_kbps=float(args.lambda_kbps),
    )

    # Determine unmatched wavs (those not referenced by any bin item).
    used_wavs = {it.rel_wav for it in bin_items if it.rel_wav is not None}
    unmatched = [w for rel, w in wavs_by_rel.items() if rel not in used_wavs]
    unmatched.sort(key=lambda it: it.rel_path.lower())

    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    if args.best_by == "quality":
        score_desc = "score=segSNR_p50"
    elif args.best_by == "efficiency":
        score_desc = "score=segSNR_p50/payload_kbps"
    else:
        score_desc = f"score=segSNR_p50-λ*payload_kbps（λ={float(args.lambda_kbps):g}）"

    subtitle = (
        f"生成时间：{ts}。"
        "“最小 bit 数”= header_bits + frames * payload_bits_per_frame（包含解码所需的所有必要字段）。"
        f"综合评分（{args.best_by}）：{score_desc}。"
        "如浏览器不允许 file:// 引用音频，建议在该目录运行：python3 -m http.server"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render(
            groups,
            group_evals=group_evals,
            evals_by_rel_bin=evals_by_rel_bin,
            unmatched_wavs=unmatched,
            title=args.title,
            subtitle=subtitle,
            best_by=str(args.best_by),
            lambda_kbps=float(args.lambda_kbps),
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path} (methods={len(groups)} bins={len(bin_items)} wav_unmatched={len(unmatched)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
