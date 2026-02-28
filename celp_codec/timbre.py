from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gains, lpc, pitch, wav_io
from .bitstream import BitReader, BitWriter, BitstreamHeaderV1, BitstreamHeaderV2, read_header
from .lsf import lpc_to_lsf, lsf_to_lpc, mix_lsf, spread_lsf, warp_lsf


MODE_CELP = 0
MODE_ACELP = 1

V1_PITCH_MIN_HZ = 50.0
V1_PITCH_MAX_HZ = 400.0
V1_LAG_BITS = 8
V1_CELP_CODEBOOK_SIZE = 512
V1_CELP_CB_BITS = 9
V1_TRACKS = 4

V1_GP_MAX = 1.2
V1_GC_MAX = 2.0
V2_GP_MAX = 1.6
V2_GC_MAX = 6.0


@dataclass(frozen=True)
class TimbreParams:
    seed: int = 1234
    f0_scale: float = 1.0
    gp_scale: float = 1.0
    gc_scale: float = 1.0
    formant_scale: float = 1.0
    lsf_spread: float = 1.0
    lsf_mix: float = 0.0
    target: str | None = None
    target_lsf_mean: np.ndarray | None = None
    min_sep_hz: float = 50.0
    edge_sep_hz: float = 50.0


def _gain_ranges_for_header(h: BitstreamHeaderV1 | BitstreamHeaderV2) -> tuple[float, float]:
    if isinstance(h, BitstreamHeaderV1):
        return V1_GP_MAX, V1_GC_MAX
    return V2_GP_MAX, V2_GC_MAX


def _v1_lag_bounds(fs: int) -> tuple[int, int]:
    return pitch.lag_bounds(
        int(fs),
        V1_PITCH_MIN_HZ,
        V1_PITCH_MAX_HZ,
        max_lag_bits=V1_LAG_BITS,
    )


def _v1_track_positions(subframe_len: int) -> list[np.ndarray]:
    L = int(subframe_len)
    out = [np.arange(t, L, V1_TRACKS, dtype=np.int64) for t in range(V1_TRACKS)]
    for cand in out:
        if cand.size > 16:
            raise ValueError("v1 ACELP requires <=16 positions per track (4-bit pos_idx).")
    return out


def _target_lsf_mean_from_celpbin(
    data: bytes,
    fs_out: int,
    lpc_order: int,
    min_sep_hz: float,
    edge_sep_hz: float,
) -> np.ndarray:
    h_any, hs = read_header(data)
    if int(getattr(h_any, "lpc_order", 0)) != int(lpc_order):
        raise ValueError("--target lpc_order must match input lpc_order for lsf-mix.")

    fs_tgt = int(getattr(h_any, "fs", 0))
    if fs_tgt <= 0:
        raise ValueError("Invalid target fs.")

    br = BitReader(data[hs:])
    vals: list[np.ndarray] = []

    while True:
        try:
            rc_idx = np.array(
                [br.read_bits(int(getattr(h_any, "rc_bits"))) for _ in range(int(lpc_order))],
                dtype=np.int64,
            )
        except EOFError:
            break

        k = lpc.dequantize_reflection_coeffs(rc_idx, int(getattr(h_any, "rc_bits")))
        a = lpc.step_up(k)
        a[0] = 1.0
        try:
            w = lpc_to_lsf(a, fs=fs_tgt)
        except Exception:
            w = None

        # Skip payload for this frame (subframes) so we can read next rc.
        frame_len = int(getattr(h_any, "frame_len"))
        subframe_len = int(getattr(h_any, "subframe_len"))
        if frame_len <= 0 or subframe_len <= 0 or frame_len % subframe_len != 0:
            raise ValueError("Invalid target frame/subframe length.")
        subframes = frame_len // subframe_len

        if isinstance(h_any, BitstreamHeaderV2):
            lag_bits = pitch.bits_for_lag(int(h_any.lag_min), int(h_any.lag_max), frac_bits=0)
            pos_bits = pitch.bits_for_pos(subframe_len)
            if int(h_any.mode) == MODE_CELP:
                cb_bits = int(int(h_any.celp_codebook_size).bit_length() - 1)
                for _ in range(subframes):
                    _ = br.read_bits(lag_bits)
                    if int(h_any.pitch_frac_bits):
                        _ = br.read_bits(int(h_any.pitch_frac_bits))
                    _ = br.read_bits(int(h_any.gain_bits_p))
                    _ = br.read_bits(int(h_any.gain_bits_c))
                    for _s in range(int(h_any.celp_stages)):
                        _ = br.read_bits(cb_bits)
            else:
                K = int(h_any.acelp_K)
                w_bits = int(h_any.acelp_weight_bits)
                for _ in range(subframes):
                    _ = br.read_bits(lag_bits)
                    if int(h_any.pitch_frac_bits):
                        _ = br.read_bits(int(h_any.pitch_frac_bits))
                    _ = br.read_bits(int(h_any.gain_bits_p))
                    _ = br.read_bits(int(h_any.gain_bits_c))
                    for _k in range(K):
                        _ = br.read_bits(pos_bits)
                        _ = br.read_bits(w_bits)
        else:
            track_pos = _v1_track_positions(subframe_len)
            if int(h_any.mode) == MODE_CELP:
                for _ in range(subframes):
                    _ = br.read_bits(V1_LAG_BITS)
                    _ = br.read_bits(int(h_any.gain_bits_p))
                    _ = br.read_bits(int(h_any.gain_bits_c))
                    _ = br.read_bits(V1_CELP_CB_BITS)
            else:
                for _ in range(subframes):
                    _ = br.read_bits(V1_LAG_BITS)
                    _ = br.read_bits(int(h_any.gain_bits_p))
                    _ = br.read_bits(int(h_any.gain_bits_c))
                    for _t, _cand in enumerate(track_pos):
                        _ = br.read_bits(4)
                        _ = br.read_bits(1)

        if w is not None:
            # Map target LSF to the output fs (Hz-preserving) before averaging.
            f_hz = w * float(fs_tgt) / (2.0 * float(np.pi))
            w_out = 2.0 * float(np.pi) * f_hz / float(fs_out)
            w_out = np.clip(w_out, 1e-6, float(np.pi) - 1e-6)
            w_out = np.sort(w_out)
            w_out = warp_lsf(w_out, fs=fs_out, scale=1.0, min_sep_hz=min_sep_hz, edge_sep_hz=edge_sep_hz)
            vals.append(w_out)

    if not vals:
        raise ValueError("Failed to extract any valid LSF frames from --target.")
    M = np.stack(vals, axis=0)
    return np.mean(M, axis=0).astype(np.float64)


def _target_lsf_mean_from_wav(
    path: str,
    fs_out: int,
    frame_len: int,
    lpc_order: int,
    preemph: float = 0.97,
    min_sep_hz: float = 50.0,
    edge_sep_hz: float = 50.0,
) -> np.ndarray:
    x, fs_in = wav_io.read_wav(path)
    x = wav_io.resample_to_fs(x, int(fs_in), int(fs_out))
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("Empty target wav.")

    pad = (-x.size) % int(frame_len)
    if pad:
        x = np.concatenate([x, np.zeros((pad,), dtype=np.float64)])

    vals: list[np.ndarray] = []
    prev = np.zeros((lpc_order,), dtype=np.int64)
    for fi in range(x.size // int(frame_len)):
        fr = x[fi * int(frame_len) : (fi + 1) * int(frame_len)]
        # LPC analysis similar to encoder: preemph + window + Levinson (no quantization).
        if preemph != 0.0:
            fr = fr.copy()
            fr[1:] = fr[1:] - float(preemph) * fr[:-1]
        win = np.hamming(fr.size).astype(np.float64)
        frw = fr * win
        r = lpc.autocorrelation(frw, lpc_order)
        _, k = lpc.levinson_durbin(r, lpc_order)
        if not np.all(np.isfinite(k)) or r[0] <= 1e-10:
            # fallback to prev quantized reflection domain if available
            k = lpc.dequantize_reflection_coeffs(prev, bits=7)
        a = lpc.step_up(k)
        a[0] = 1.0
        try:
            w = lpc_to_lsf(a, fs=fs_out)
        except Exception:
            continue
        w = warp_lsf(w, fs=fs_out, scale=1.0, min_sep_hz=min_sep_hz, edge_sep_hz=edge_sep_hz)
        vals.append(w)

        prev = lpc.quantize_reflection_coeffs(k, bits=7)

    if not vals:
        raise ValueError("Failed to extract any valid LSF frames from target wav.")
    M = np.stack(vals, axis=0)
    return np.mean(M, axis=0).astype(np.float64)


def transform_bitstream(
    data: bytes,
    params: TimbreParams,
    dump_json_path: str | None = None,
) -> tuple[bytes, dict | None]:
    """
    Transform a CELP/ACELP bitstream in the parameter domain.

    Returns:
      out_bytes, debug_dict_or_None
    """
    header_any, header_size = read_header(data)
    header_bytes = data[:header_size]
    br = BitReader(data[header_size:])
    bw = BitWriter()

    fs = int(getattr(header_any, "fs"))
    frame_len = int(getattr(header_any, "frame_len"))
    subframe_len = int(getattr(header_any, "subframe_len"))
    lpc_order = int(getattr(header_any, "lpc_order"))
    rc_bits = int(getattr(header_any, "rc_bits"))
    mode_id = int(getattr(header_any, "mode"))

    if fs <= 0 or frame_len <= 0 or subframe_len <= 0 or lpc_order <= 0:
        raise ValueError("Invalid bitstream header.")
    if frame_len % subframe_len != 0:
        raise ValueError("frame_len must be divisible by subframe_len")
    subframes = frame_len // subframe_len

    # Lag bounds for v1 are implicit; v2 has explicit lag range.
    if isinstance(header_any, BitstreamHeaderV2):
        lag_min = int(header_any.lag_min)
        lag_max = int(header_any.lag_max)
        lag_bits = pitch.bits_for_lag(lag_min, lag_max, frac_bits=0)
        pitch_frac_bits = int(header_any.pitch_frac_bits)
    else:
        lag_min, lag_max = _v1_lag_bounds(fs)
        lag_bits = V1_LAG_BITS
        pitch_frac_bits = 0

    gp_max, gc_max = _gain_ranges_for_header(header_any)

    # Optional target style statistics for LSF mixing.
    target_lsf_mean: np.ndarray | None = None
    if params.lsf_mix and params.target_lsf_mean is not None:
        target_lsf_mean = np.asarray(params.target_lsf_mean, dtype=np.float64).ravel().copy()
    elif params.lsf_mix and params.target:
        tgt = str(params.target)
        if os.path.splitext(tgt)[1].lower() == ".celpbin":
            target_lsf_mean = _target_lsf_mean_from_celpbin(
                Path(tgt).read_bytes(),
                fs_out=fs,
                lpc_order=lpc_order,
                min_sep_hz=float(params.min_sep_hz),
                edge_sep_hz=float(params.edge_sep_hz),
            )
        elif os.path.splitext(tgt)[1].lower() == ".wav":
            target_lsf_mean = _target_lsf_mean_from_wav(
                tgt,
                fs_out=fs,
                frame_len=frame_len,
                lpc_order=lpc_order,
                preemph=0.97,
                min_sep_hz=float(params.min_sep_hz),
                edge_sep_hz=float(params.edge_sep_hz),
            )
        else:
            raise ValueError("--target must be .celpbin or .wav")

    debug: dict | None = None
    if dump_json_path is not None:
        debug = {
            "fs": fs,
            "mode": "acelp" if mode_id == MODE_ACELP else "celp",
            "version": 2 if isinstance(header_any, BitstreamHeaderV2) else 1,
            "params": {
                "seed": int(params.seed),
                "f0_scale": float(params.f0_scale),
                "gp_scale": float(params.gp_scale),
                "gc_scale": float(params.gc_scale),
                "formant_scale": float(params.formant_scale),
                "lsf_spread": float(params.lsf_spread),
                "lsf_mix": float(params.lsf_mix),
                "target": str(params.target) if params.target else None,
            },
            "frames": [],
        }

    frame_index = 0
    while True:
        try:
            rc_idx = np.array([br.read_bits(rc_bits) for _ in range(lpc_order)], dtype=np.int64)
        except EOFError:
            break

        rc_idx_out = rc_idx.copy()
        lsf_ok = True
        if (
            float(params.formant_scale) != 1.0
            or float(params.lsf_spread) != 1.0
            or float(params.lsf_mix) != 0.0
        ):
            k = lpc.dequantize_reflection_coeffs(rc_idx, rc_bits)
            a = lpc.step_up(k)
            a[0] = 1.0
            try:
                w = lpc_to_lsf(a, fs=fs)
                if float(params.formant_scale) != 1.0:
                    w = warp_lsf(
                        w,
                        fs=fs,
                        scale=float(params.formant_scale),
                        min_sep_hz=float(params.min_sep_hz),
                        edge_sep_hz=float(params.edge_sep_hz),
                    )
                if float(params.lsf_spread) != 1.0:
                    w = spread_lsf(
                        w,
                        fs=fs,
                        spread=float(params.lsf_spread),
                        min_sep_hz=float(params.min_sep_hz),
                        edge_sep_hz=float(params.edge_sep_hz),
                    )
                if float(params.lsf_mix) != 0.0:
                    if target_lsf_mean is None:
                        raise ValueError("--lsf-mix requires --target")
                    w = mix_lsf(
                        w,
                        target_lsf_mean,
                        mix=float(params.lsf_mix),
                        fs=fs,
                        min_sep_hz=float(params.min_sep_hz),
                        edge_sep_hz=float(params.edge_sep_hz),
                    )
                a2 = lsf_to_lpc(w)
                k2 = lpc.step_down(a2)
                rc_idx_out = lpc.quantize_reflection_coeffs(k2, rc_bits)
            except Exception:
                lsf_ok = False
                rc_idx_out = rc_idx.copy()

        for v in rc_idx_out.tolist():
            bw.write_bits(int(v), rc_bits)

        frame_dbg = None
        if debug is not None:
            frame_dbg = {
                "frame_index": int(frame_index),
                "rc_idx_in": rc_idx.tolist(),
                "rc_idx_out": rc_idx_out.tolist(),
                "lsf_ok": bool(lsf_ok),
                "subframes": [],
            }
            debug["frames"].append(frame_dbg)

        # Subframe loop
        for si in range(int(subframes)):
            try:
                lag_i = int(br.read_bits(lag_bits))
                frac = int(br.read_bits(pitch_frac_bits)) if pitch_frac_bits else 0
                gp_idx = int(br.read_bits(int(getattr(header_any, "gain_bits_p"))))
                gc_idx = int(br.read_bits(int(getattr(header_any, "gain_bits_c"))))
            except EOFError:
                break

            lag = lag_min + lag_i
            lag_new = int(lag)
            if float(params.f0_scale) != 1.0:
                s = float(params.f0_scale)
                if not np.isfinite(s) or s <= 0.0:
                    raise ValueError("--f0-scale must be > 0")
                lag_new = int(np.rint(float(lag) / s))
            lag_new = int(min(max(lag_new, lag_min), lag_max))
            lag_i_new = int(lag_new - lag_min)

            gp_new_idx = gp_idx
            gc_new_idx = gc_idx
            if float(params.gp_scale) != 1.0 or float(params.gc_scale) != 1.0:
                gp = gains.dequantize_gain(gp_idx, int(getattr(header_any, "gain_bits_p")), xmin=1e-4, xmax=gp_max)
                gc = gains.dequantize_gain(gc_idx, int(getattr(header_any, "gain_bits_c")), xmin=1e-4, xmax=gc_max)
                gp2 = float(gp) * float(params.gp_scale)
                gc2 = float(gc) * float(params.gc_scale)
                gp2 = float(min(max(gp2, 0.0), gp_max))
                gc2 = float(min(max(gc2, 0.0), gc_max))
                gp_new_idx = gains.quantize_gain(
                    gp2, int(getattr(header_any, "gain_bits_p")), xmin=1e-4, xmax=gp_max
                )
                gc_new_idx = gains.quantize_gain(
                    gc2, int(getattr(header_any, "gain_bits_c")), xmin=1e-4, xmax=gc_max
                )

            bw.write_bits(lag_i_new, lag_bits)
            if pitch_frac_bits:
                bw.write_bits(frac, pitch_frac_bits)
            bw.write_bits(int(gp_new_idx), int(getattr(header_any, "gain_bits_p")))
            bw.write_bits(int(gc_new_idx), int(getattr(header_any, "gain_bits_c")))

            # Pass-through innovation bits
            if isinstance(header_any, BitstreamHeaderV2):
                if mode_id == MODE_CELP:
                    cb_bits = int(int(header_any.celp_codebook_size).bit_length() - 1)
                    for _s in range(int(header_any.celp_stages)):
                        idx = int(br.read_bits(cb_bits))
                        bw.write_bits(idx, cb_bits)
                else:
                    K = int(header_any.acelp_K)
                    pos_bits = pitch.bits_for_pos(subframe_len)
                    w_bits = int(header_any.acelp_weight_bits)
                    for _k in range(K):
                        pos = int(br.read_bits(pos_bits))
                        wi = int(br.read_bits(w_bits))
                        bw.write_bits(pos, pos_bits)
                        bw.write_bits(wi, w_bits)
            else:
                if mode_id == MODE_CELP:
                    idx = int(br.read_bits(V1_CELP_CB_BITS))
                    bw.write_bits(idx, V1_CELP_CB_BITS)
                else:
                    for _t in range(V1_TRACKS):
                        pos_idx = int(br.read_bits(4))
                        sign = int(br.read_bits(1))
                        bw.write_bits(pos_idx, 4)
                        bw.write_bits(sign, 1)

            if frame_dbg is not None:
                frame_dbg["subframes"].append(
                    {
                        "subframe_index": int(si),
                        "lag_in": int(lag),
                        "lag_out": int(lag_new),
                        "gp_idx_in": int(gp_idx),
                        "gp_idx_out": int(gp_new_idx),
                        "gc_idx_in": int(gc_idx),
                        "gc_idx_out": int(gc_new_idx),
                    }
                )

        frame_index += 1

    out_bytes = header_bytes + bw.get_bytes()
    if dump_json_path is not None and debug is not None:
        with open(dump_json_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
    return out_bytes, debug
