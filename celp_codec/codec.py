from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from . import acelp, celp_codebook, filters, gains, lpc, metrics, pitch
from .bitstream import (
    BitReader,
    BitWriter,
    BitstreamHeaderV2,
    VERSION_V2,
    read_header,
)


MODE_CELP = 0
MODE_ACELP = 1

FLAG_POSTFILTER = 1 << 0
FLAG_LPC_INTERP = 1 << 1


@dataclass
class CodecConfig:
    mode: str = "acelp"  # "celp" | "acelp"
    fs: int = 16000
    frame_ms: int = 20
    subframe_ms: int = 5
    lpc_order: int | None = None
    lpc_preemph: float = 0.97
    lpc_interp: bool = True
    pitch_min_hz: float = 50.0
    pitch_max_hz: float = 400.0
    pitch_frac_bits: int = 0
    dp_pitch: bool = True
    dp_topk: int = 10
    dp_lambda: float = 0.05
    rc_bits: int = 10
    gain_bits_p: int = 7
    gain_bits_c: int = 7
    gp_max: float = 1.6
    gc_max: float = 6.0
    seed: int = 1234
    clip: bool = True
    postfilter: bool = False
    # CELP innovation (v2)
    celp_codebook_size: int = 2048
    celp_stages: int = 2
    # ACELP innovation (v2): sparse weighted pulses
    acelp_K: int | None = None
    acelp_weight_bits: int = 5
    acelp_solver: str = "omp"  # "ista" | "omp"
    ista_iters: int = 60
    ista_lambda: float = 0.02

    def frame_len(self) -> int:
        return int(round(self.fs * (self.frame_ms / 1000.0)))

    def subframe_len(self) -> int:
        return int(round(self.fs * (self.subframe_ms / 1000.0)))

    def subframes_per_frame(self) -> int:
        fl = self.frame_len()
        sl = self.subframe_len()
        if fl % sl != 0:
            raise ValueError("frame_len must be divisible by subframe_len")
        return fl // sl

    def resolved_lpc_order(self) -> int:
        if self.lpc_order is not None:
            return int(self.lpc_order)
        return 10 if int(self.fs) <= 8000 else 16

    def resolved_acelp_K(self) -> int:
        if self.acelp_K is not None:
            return int(self.acelp_K)
        # High-quality default: ~L/4 pulses
        return max(1, self.subframe_len() // 4)


def _mode_id(mode: str) -> int:
    m = mode.lower()
    if m == "celp":
        return MODE_CELP
    if m == "acelp":
        return MODE_ACELP
    raise ValueError(f"Unknown mode: {mode}")


def _fft_lipschitz_from_h(h: np.ndarray) -> float:
    h = np.asarray(h, dtype=np.float64).ravel()
    n = 1
    target = max(2 * h.size, 8)
    while n < target:
        n *= 2
    Hf = np.fft.rfft(h, n=n)
    Lc = float(np.max(np.abs(Hf) ** 2))
    return max(Lc, 1e-12)


def _analyze_lpc(
    frame: np.ndarray, order: int, rc_bits: int, prev_idx: np.ndarray | None, preemph: float
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(frame, dtype=np.float64).ravel()
    if x.size == 0:
        idx = np.zeros((order,), dtype=np.int64)
        a = np.zeros((order + 1,), dtype=np.float64)
        a[0] = 1.0
        return a, idx

    if preemph != 0.0:
        x = filters.preemphasis(x, float(preemph))

    win = np.hamming(x.size).astype(np.float64)
    xw = x * win
    r = lpc.autocorrelation(xw, order)
    _, k = lpc.levinson_durbin(r, order)

    if not np.all(np.isfinite(k)) or r[0] <= 1e-10:
        idx = prev_idx.copy() if prev_idx is not None else np.zeros((order,), dtype=np.int64)
    else:
        idx = lpc.quantize_reflection_coeffs(k, rc_bits)

    k_hat = lpc.dequantize_reflection_coeffs(idx, rc_bits)
    a_hat = lpc.step_up(k_hat)
    a_hat[0] = 1.0
    return a_hat, idx.astype(np.int64)


def _build_weighting_filters(
    a: np.ndarray, gamma1: float = 0.94, gamma2: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    W(z) = A_gamma1(z) / A_gamma2(z)
    F(z) = W(z) / A(z) = A_gamma1(z) / (A_gamma2(z)*A(z))
    """
    A = np.asarray(a, dtype=np.float64).ravel()
    A_g1 = filters.bandwidth_expand(A, gamma1)
    A_g2 = filters.bandwidth_expand(A, gamma2)
    denom_F = np.convolve(A_g2, A)
    return A_g1, A_g2, denom_F


def _innov_celp_shape(
    residual: np.ndarray,
    H: np.ndarray,
    codebook: np.ndarray,
    stages: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, list[int]]:
    """
    Multi-stage greedy CELP innovation:
      c = sum_s C[idx_s]  (no per-stage gains), then normalize.
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.float64)
    C = np.asarray(codebook, dtype=np.float64)
    stages = int(stages)
    if stages <= 0:
        raise ValueError("stages must be > 0")

    # Precompute codeword responses in weighted domain for this subframe
    Y = C @ H.T  # (M, L)
    idxs: list[int] = []
    c_sum = np.zeros((C.shape[1],), dtype=np.float64)
    rr = r.copy()

    for _ in range(stages):
        dots = Y @ rr
        energies = np.sum(Y * Y, axis=1) + eps
        scores = np.where(dots > 0.0, (dots * dots) / energies, 0.0)
        idx = int(np.argmax(scores))
        idxs.append(idx)
        c_sum += C[idx]
        # Update residual with optimal gain along this codeword response (analysis-only)
        g = float(dots[idx] / energies[idx])
        rr = rr - g * Y[idx]

    norm = float(np.linalg.norm(c_sum) + eps)
    c_shape = c_sum / norm
    return c_shape, idxs


def _innov_acelp_shape(
    residual: np.ndarray,
    H: np.ndarray,
    K: int,
    weight_bits: int,
    solver: str,
    tau: float,
    lam: float,
    iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sparse weighted-pulse innovation:
      - solve for continuous c (ISTA) or use it as proxy
      - choose top-K support
      - LS-refine weights
      - normalize shape to max|w|=1
      - quantize/dequantize the shape weights to `weight_bits`

    Returns:
      positions(int64,K), weight_idx(int64,K), c_shape_hat(float64,L)
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.float64)
    L = int(r.size)
    K = int(min(max(int(K), 1), L))
    weight_bits = int(weight_bits)

    if solver == "ista":
        c_cont = acelp.ista_lasso(r, H, tau=tau, lam=lam, iters=iters)
        pos = acelp.topk_support(c_cont, K)
        w = acelp.refine_weights_ls(r, H, pos)
    elif solver == "omp":
        pos, w = acelp.omp_support(r, H, K)
    else:
        raise ValueError(f"Unknown acelp_solver: {solver}")
    w_shape = acelp.normalize_shape(w)

    w_idx = acelp.quantize_unit(w_shape, weight_bits)
    w_hat = acelp.dequantize_unit(w_idx, weight_bits)
    c_shape_hat = acelp.support_to_vector(pos, w_hat, length=L)
    return pos, w_idx, c_shape_hat


def encode_samples(
    x_in: np.ndarray,
    cfg: CodecConfig,
    dump_json_path: str | None = None,
) -> tuple[bytes, np.ndarray, dict | None, dict]:
    """
    Encode float64 mono samples in [-1,1].

    Returns:
      bitstream_bytes, recon_samples(float64), debug_dict_or_None, stats
    """
    t0 = time.time()

    mode_id = _mode_id(cfg.mode)
    fs = int(cfg.fs)
    frame_len = cfg.frame_len()
    subframe_len = cfg.subframe_len()
    subframes = cfg.subframes_per_frame()
    lpc_order = cfg.resolved_lpc_order()
    K = cfg.resolved_acelp_K()

    if frame_len <= 0 or subframe_len <= 0:
        raise ValueError("Invalid frame/subframe length.")

    lag_min, lag_max = pitch.lag_bounds(fs, cfg.pitch_min_hz, cfg.pitch_max_hz, max_lag_bits=None)
    lag_bits = pitch.bits_for_lag(lag_min, lag_max, frac_bits=0)
    pos_bits = pitch.bits_for_pos(subframe_len)

    if mode_id == MODE_CELP:
        if cfg.celp_codebook_size <= 0 or (cfg.celp_codebook_size & (cfg.celp_codebook_size - 1)) != 0:
            raise ValueError("--celp-codebook-size must be a power of 2.")
        if cfg.celp_stages <= 0:
            raise ValueError("--celp-stages must be > 0.")

    flags = 0
    if cfg.postfilter:
        flags |= FLAG_POSTFILTER
    if cfg.lpc_interp:
        flags |= FLAG_LPC_INTERP

    header = BitstreamHeaderV2(
        mode=mode_id,
        fs=fs,
        frame_len=int(frame_len),
        subframe_len=int(subframe_len),
        lpc_order=int(lpc_order),
        rc_bits=int(cfg.rc_bits),
        gain_bits_p=int(cfg.gain_bits_p),
        gain_bits_c=int(cfg.gain_bits_c),
        lag_min=int(lag_min),
        lag_max=int(lag_max),
        pitch_frac_bits=int(cfg.pitch_frac_bits),
        acelp_K=int(K) if mode_id == MODE_ACELP else 0,
        acelp_weight_bits=int(cfg.acelp_weight_bits) if mode_id == MODE_ACELP else 0,
        flags=int(flags),
        celp_codebook_size=int(cfg.celp_codebook_size) if mode_id == MODE_CELP else 0,
        celp_stages=int(cfg.celp_stages) if mode_id == MODE_CELP else 0,
        seed=int(cfg.seed),
    )

    bw = BitWriter()

    debug: dict | None = None
    if dump_json_path is not None:
        debug = {"fs": fs, "mode": cfg.mode.lower(), "version": VERSION_V2, "frames": []}

    x = np.asarray(x_in, dtype=np.float64).ravel()
    x = np.clip(x, -1.0, 1.0)
    pad = (-x.size) % frame_len
    if pad:
        x = np.concatenate([x, np.zeros((pad,), dtype=np.float64)])

    n_frames = x.size // frame_len
    total_samples = int(x.size)

    mem_syn = np.zeros((lpc_order,), dtype=np.float64)
    mem_w = np.zeros((lpc_order,), dtype=np.float64)
    mem_F = np.zeros((2 * lpc_order,), dtype=np.float64)
    exc_buf = np.zeros((lag_max + subframe_len + 2,), dtype=np.float64)
    prev_rc_idx = np.zeros((lpc_order,), dtype=np.int64)
    prev_k_hat = np.zeros((lpc_order,), dtype=np.float64)

    codebook = None
    cb_bits = 0
    if mode_id == MODE_CELP:
        codebook = celp_codebook.generate_codebook(cfg.seed, cfg.celp_codebook_size, subframe_len)
        cb_bits = int(np.log2(cfg.celp_codebook_size))

    recon = np.zeros((total_samples,), dtype=np.float64)

    for fi in range(int(n_frames)):
        frame = x[fi * frame_len : (fi + 1) * frame_len]

        _, rc_idx = _analyze_lpc(frame, lpc_order, cfg.rc_bits, prev_rc_idx, preemph=cfg.lpc_preemph)
        prev_rc_idx = rc_idx
        for idx in rc_idx.tolist():
            bw.write_bits(int(idx), cfg.rc_bits)

        zeros_sf = np.zeros((subframe_len,), dtype=np.float64)
        k_cur_hat = lpc.dequantize_reflection_coeffs(rc_idx, cfg.rc_bits)
        if fi == 0:
            prev_k_hat = k_cur_hat.copy()

        frame_dbg = None
        if debug is not None:
            frame_dbg = {"frame_index": fi, "rc_idx": rc_idx.tolist(), "subframes": []}
            debug["frames"].append(frame_dbg)

        if cfg.dp_pitch:
            mem_w0 = mem_w.copy()
            mem_F0 = mem_F.copy()
            exc0 = exc_buf.copy()

            cand_lags: list[np.ndarray] = []
            cand_scores: list[np.ndarray] = []

            mem_w_tmp = mem_w0.copy()
            mem_F_tmp = mem_F0.copy()
            exc_tmp = exc0.copy()

            for si in range(subframes):
                if cfg.lpc_interp:
                    alpha = float((si + 1) / subframes)
                    k_sf = (1.0 - alpha) * prev_k_hat + alpha * k_cur_hat
                else:
                    k_sf = k_cur_hat
                a_sf = lpc.step_up(k_sf)
                a_sf[0] = 1.0
                A_g1, A_g2, denom_F = _build_weighting_filters(a_sf)
                h = filters.impulse_response(A_g1, denom_F, subframe_len)
                H = filters.conv_matrix(h)

                s = frame[si * subframe_len : (si + 1) * subframe_len]
                s_w, mem_w_tmp = filters.iir_filter(A_g1, A_g2, s, zi=mem_w_tmp)
                y_free, _ = filters.iir_filter(A_g1, denom_F, zeros_sf, zi=mem_F_tmp)
                d = s_w - y_free

                lags_k, scores_k = pitch.topk_pitch_candidates(
                    d, H, exc_tmp, lag_min, lag_max, topk=cfg.dp_topk
                )
                cand_lags.append(lags_k)
                cand_scores.append(scores_k)

                # Provisional state advance using pitch-only excitation (fast)
                lag_g = int(lags_k[0]) if lags_k.size else int(lag_min)
                ep = exc_tmp[-lag_g - subframe_len : -lag_g].copy()
                yp = H @ ep
                gp = gains.estimate_gain(d, yp, max_gain=cfg.gp_max)
                e = gp * ep
                _, mem_F_tmp = filters.iir_filter(A_g1, denom_F, e, zi=mem_F_tmp)
                exc_tmp[:-subframe_len] = exc_tmp[subframe_len:]
                exc_tmp[-subframe_len:] = e

            lags_path = pitch.viterbi_smooth_lags(cand_lags, cand_scores, cfg.dp_lambda)
        else:
            lags_path = []

        for si in range(subframes):
            if cfg.lpc_interp:
                alpha = float((si + 1) / subframes)
                k_sf = (1.0 - alpha) * prev_k_hat + alpha * k_cur_hat
            else:
                k_sf = k_cur_hat
            a_sf = lpc.step_up(k_sf)
            a_sf[0] = 1.0
            A_g1, A_g2, denom_F = _build_weighting_filters(a_sf)
            h = filters.impulse_response(A_g1, denom_F, subframe_len)
            H = filters.conv_matrix(h)
            tau = 1.0 / _fft_lipschitz_from_h(h)

            s = frame[si * subframe_len : (si + 1) * subframe_len]
            s_w, mem_w = filters.iir_filter(A_g1, A_g2, s, zi=mem_w)
            y_free, _ = filters.iir_filter(A_g1, denom_F, zeros_sf, zi=mem_F)
            d = s_w - y_free

            if cfg.dp_pitch:
                lag = int(lags_path[si])
            else:
                lags_1, _ = pitch.topk_pitch_candidates(d, H, exc_buf, lag_min, lag_max, topk=1)
                lag = int(lags_1[0]) if lags_1.size else int(lag_min)

            ep = exc_buf[-lag - subframe_len : -lag].copy()
            yp = H @ ep

            # Innovation analysis uses a fast pre-gain estimate
            gp_pre = gains.estimate_gain(d, yp, max_gain=cfg.gp_max)
            r0 = d - gp_pre * yp

            if mode_id == MODE_CELP:
                assert codebook is not None
                c_shape_hat, idxs = _innov_celp_shape(r0, H, codebook, stages=cfg.celp_stages)
                innov_bits = {"cb_idx": idxs}
                y_c = H @ c_shape_hat
            else:
                pos, w_idx, c_shape_hat = _innov_acelp_shape(
                    r0,
                    H,
                    K=K,
                    weight_bits=cfg.acelp_weight_bits,
                    solver=cfg.acelp_solver,
                    tau=tau,
                    lam=cfg.ista_lambda,
                    iters=cfg.ista_iters,
                )
                innov_bits = {"pos": pos, "w_idx": w_idx}
                y_c = H @ c_shape_hat

            gp, gc = gains.estimate_gains_joint(d, yp, y_c, gp_max=cfg.gp_max, gc_max=cfg.gc_max)
            gp_idx = gains.quantize_gain(gp, cfg.gain_bits_p, xmin=1e-4, xmax=cfg.gp_max)
            gc_idx = gains.quantize_gain(gc, cfg.gain_bits_c, xmin=1e-4, xmax=cfg.gc_max)
            gp_hat = gains.dequantize_gain(gp_idx, cfg.gain_bits_p, xmin=1e-4, xmax=cfg.gp_max)
            gc_hat = gains.dequantize_gain(gc_idx, cfg.gain_bits_c, xmin=1e-4, xmax=cfg.gc_max)

            lag_idx = int(lag - lag_min)
            bw.write_bits(lag_idx, lag_bits)
            if cfg.pitch_frac_bits:
                # Reserved for future fractional delay (currently always 0)
                bw.write_bits(0, cfg.pitch_frac_bits)
            bw.write_bits(int(gp_idx), cfg.gain_bits_p)
            bw.write_bits(int(gc_idx), cfg.gain_bits_c)

            if mode_id == MODE_CELP:
                for idx in innov_bits["cb_idx"]:
                    bw.write_bits(int(idx), cb_bits)
            else:
                pos_arr = np.asarray(innov_bits["pos"], dtype=np.int64)
                w_arr = np.asarray(innov_bits["w_idx"], dtype=np.int64)
                if pos_arr.size != K:
                    # Pad with zeros deterministically
                    pos_arr = np.pad(pos_arr, (0, K - pos_arr.size), constant_values=0)
                    w_arr = np.pad(w_arr, (0, K - w_arr.size), constant_values=(1 << cfg.acelp_weight_bits) // 2)
                for p, wi in zip(pos_arr.tolist(), w_arr.tolist()):
                    bw.write_bits(int(p), pos_bits)
                    bw.write_bits(int(wi), cfg.acelp_weight_bits)

            c = gc_hat * c_shape_hat
            e = gp_hat * ep + c

            _, mem_F = filters.iir_filter(A_g1, denom_F, e, zi=mem_F)
            s_hat, mem_syn = filters.iir_filter(np.array([1.0]), a_sf, e, zi=mem_syn)

            recon[
                fi * frame_len + si * subframe_len : fi * frame_len + (si + 1) * subframe_len
            ] = s_hat

            exc_buf[:-subframe_len] = exc_buf[subframe_len:]
            exc_buf[-subframe_len:] = e

            if frame_dbg is not None:
                sf = {"lag": int(lag), "gp_idx": int(gp_idx), "gc_idx": int(gc_idx)}
                if mode_id == MODE_CELP:
                    sf["celp"] = {"cb_idx": [int(i) for i in innov_bits["cb_idx"]]}
                else:
                    sf["acelp"] = {
                        "K": int(K),
                        "pulses": [
                            {"pos": int(p), "w_idx": int(wi)} for p, wi in zip(pos_arr.tolist(), w_arr.tolist())
                        ],
                    }
                frame_dbg["subframes"].append(sf)

        prev_k_hat = k_cur_hat.copy()

    bitstream_bytes = header.to_bytes() + bw.get_bytes()

    if cfg.clip:
        recon = np.clip(recon, -1.0, 1.0).astype(np.float64, copy=False)

    if dump_json_path is not None and debug is not None:
        with open(dump_json_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)

    t1 = time.time()
    stats = {
        "version": VERSION_V2,
        "frames": int(n_frames),
        "samples": int(total_samples),
        "seconds": float(total_samples / float(fs)),
        "payload_bits": int(bw.bits_written),
        "total_bits": int(len(header.to_bytes()) * 8 + bw.bits_written),
        "encode_seconds": float(t1 - t0),
    }
    return bitstream_bytes, recon, debug, stats


def decode_bitstream(data: bytes, clip: bool | None = None) -> tuple[np.ndarray, BitstreamHeaderV2, dict]:
    header_any, header_size = read_header(data)
    if not isinstance(header_any, BitstreamHeaderV2):
        raise ValueError("This build expects v2 bitstreams; please re-encode.")
    header = header_any
    br = BitReader(data[header_size:])

    if header.fs <= 0 or header.frame_len <= 0 or header.subframe_len <= 0:
        raise ValueError("Invalid header parameters.")

    mode_id = int(header.mode)
    frame_len = int(header.frame_len)
    subframe_len = int(header.subframe_len)
    subframes = frame_len // subframe_len
    lag_min = int(header.lag_min)
    lag_max = int(header.lag_max)
    lag_bits = pitch.bits_for_lag(lag_min, lag_max, frac_bits=0)
    pos_bits = pitch.bits_for_pos(subframe_len)

    if clip is None:
        clip = True

    mem_syn = np.zeros((header.lpc_order,), dtype=np.float64)
    exc_buf = np.zeros((lag_max + subframe_len + 2,), dtype=np.float64)
    prev_k_hat = np.zeros((header.lpc_order,), dtype=np.float64)
    lpc_interp = bool(int(header.flags) & FLAG_LPC_INTERP)

    codebook = None
    cb_bits = 0
    if mode_id == MODE_CELP:
        size = int(header.celp_codebook_size)
        codebook = celp_codebook.generate_codebook(header.seed, size, subframe_len)
        cb_bits = int(np.log2(size))

    out: list[np.ndarray] = []
    frames_decoded = 0
    K = int(header.acelp_K) if mode_id == MODE_ACELP else 0
    w_bits = int(header.acelp_weight_bits) if mode_id == MODE_ACELP else 0

    while True:
        try:
            rc_idx = [br.read_bits(header.rc_bits) for _ in range(header.lpc_order)]
        except EOFError:
            break

        k_cur_hat = lpc.dequantize_reflection_coeffs(np.array(rc_idx, dtype=np.int64), header.rc_bits)
        if frames_decoded == 0:
            prev_k_hat = k_cur_hat.copy()

        for si in range(subframes):
            if lpc_interp:
                alpha = float((si + 1) / subframes)
                k_sf = (1.0 - alpha) * prev_k_hat + alpha * k_cur_hat
            else:
                k_sf = k_cur_hat
            a_hat = lpc.step_up(k_sf)
            a_hat[0] = 1.0
            try:
                lag_i = br.read_bits(lag_bits)
                frac = br.read_bits(header.pitch_frac_bits) if header.pitch_frac_bits else 0
                gp_idx = br.read_bits(header.gain_bits_p)
                gc_idx = br.read_bits(header.gain_bits_c)
            except EOFError:
                return _finalize_decode(out, header, frames_decoded, clip)

            lag = lag_min + int(lag_i)
            if lag < 1:
                lag = 1
            if lag > lag_max:
                lag = lag_max

            # Integer-lag adaptive vector (fractional delay reserved for future)
            ep = exc_buf[-lag - subframe_len : -lag].copy()

            gp_hat = gains.dequantize_gain(gp_idx, header.gain_bits_p, xmin=1e-4, xmax=1.6)
            gc_hat = gains.dequantize_gain(gc_idx, header.gain_bits_c, xmin=1e-4, xmax=6.0)

            if mode_id == MODE_CELP:
                assert codebook is not None
                try:
                    idxs = [br.read_bits(cb_bits) for _ in range(int(header.celp_stages))]
                except EOFError:
                    return _finalize_decode(out, header, frames_decoded, clip)
                c = np.zeros((subframe_len,), dtype=np.float64)
                for idx in idxs:
                    c += codebook[int(idx)]
                c = c / (np.linalg.norm(c) + 1e-12)
            else:
                pos = np.empty((K,), dtype=np.int64)
                w_idx = np.empty((K,), dtype=np.int64)
                try:
                    for i in range(K):
                        pos[i] = int(br.read_bits(pos_bits))
                        w_idx[i] = int(br.read_bits(w_bits))
                except EOFError:
                    return _finalize_decode(out, header, frames_decoded, clip)
                w = acelp.dequantize_unit(w_idx, w_bits)
                c = acelp.support_to_vector(pos, w, length=subframe_len)

            e = gp_hat * ep + gc_hat * c
            s_hat, mem_syn = filters.iir_filter(np.array([1.0]), a_hat, e, zi=mem_syn)
            out.append(s_hat)

            exc_buf[:-subframe_len] = exc_buf[subframe_len:]
            exc_buf[-subframe_len:] = e

        frames_decoded += 1
        prev_k_hat = k_cur_hat.copy()

    return _finalize_decode(out, header, frames_decoded, clip)


def _finalize_decode(
    out_chunks: list[np.ndarray], header: BitstreamHeaderV2, frames_decoded: int, clip: bool
) -> tuple[np.ndarray, BitstreamHeaderV2, dict]:
    y = np.concatenate(out_chunks) if out_chunks else np.zeros((0,), dtype=np.float64)
    if clip:
        y = np.clip(y, -1.0, 1.0)
    return y, header, {"frames": int(frames_decoded)}


def roundtrip_metrics(x: np.ndarray, y: np.ndarray, frame_len: int) -> dict:
    return {
        "snr_db": metrics.snr_db(x, y),
        "seg_snr_db": metrics.seg_snr_db(x, y, frame_len=frame_len),
    }
