from __future__ import annotations

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def estimate_gain(target: np.ndarray, vec: np.ndarray, max_gain: float, eps: float = 1e-12) -> float:
    target = np.asarray(target, dtype=np.float64).ravel()
    vec = np.asarray(vec, dtype=np.float64).ravel()
    denom = float(np.dot(vec, vec) + eps)
    num = float(np.dot(target, vec))
    g = num / denom
    if not np.isfinite(g):
        g = 0.0
    return clamp(g, 0.0, max_gain)


def estimate_gains_joint(
    d: np.ndarray,
    yp: np.ndarray,
    yc: np.ndarray,
    gp_max: float,
    gc_max: float,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """
    Solve min ||d - gp*yp - gc*yc||^2 (2x2 least squares) then clamp to [0, max].
    Falls back to sequential if ill-conditioned.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    yp = np.asarray(yp, dtype=np.float64).ravel()
    yc = np.asarray(yc, dtype=np.float64).ravel()

    a11 = float(np.dot(yp, yp) + eps)
    a22 = float(np.dot(yc, yc) + eps)
    a12 = float(np.dot(yp, yc))
    b1 = float(np.dot(d, yp))
    b2 = float(np.dot(d, yc))

    det = a11 * a22 - a12 * a12
    if not np.isfinite(det) or det <= eps:
        gp = estimate_gain(d, yp, gp_max, eps=eps)
        r = d - gp * yp
        gc = estimate_gain(r, yc, gc_max, eps=eps)
        return gp, gc

    gp = (b1 * a22 - b2 * a12) / det
    gc = (b2 * a11 - b1 * a12) / det
    if not np.isfinite(gp):
        gp = 0.0
    if not np.isfinite(gc):
        gc = 0.0
    return clamp(gp, 0.0, gp_max), clamp(gc, 0.0, gc_max)


def quantize_gain(
    g: float, bits: int, xmin: float, xmax: float, eps: float = 1e-12
) -> int:
    """
    Log-domain quantizer with a reserved zero level.

    idx=0 -> 0.0
    idx=1..(2^bits-1) -> log-spaced in [xmin, xmax]
    """
    bits = int(bits)
    if bits <= 1:
        raise ValueError("bits must be >= 2 for this quantizer")
    levels = 1 << bits

    g = float(g)
    if not np.isfinite(g) or g <= eps:
        return 0

    g = min(max(g, xmin), xmax)
    log_xmin = float(np.log(xmin))
    log_xmax = float(np.log(xmax))
    ratio = (float(np.log(g)) - log_xmin) / (log_xmax - log_xmin + eps)
    ratio = min(max(ratio, 0.0), 1.0)
    idx = 1 + int(round(ratio * (levels - 2)))
    return int(min(max(idx, 1), levels - 1))


def dequantize_gain(
    idx: int, bits: int, xmin: float, xmax: float, eps: float = 1e-12
) -> float:
    bits = int(bits)
    if bits <= 1:
        raise ValueError("bits must be >= 2 for this quantizer")
    levels = 1 << bits

    idx = int(idx)
    if idx <= 0:
        return 0.0
    idx = min(idx, levels - 1)

    ratio = (idx - 1) / float(levels - 2)
    log_xmin = float(np.log(xmin))
    log_xmax = float(np.log(xmax))
    g = float(np.exp(log_xmin + ratio * (log_xmax - log_xmin)))
    if not np.isfinite(g):
        g = 0.0
    return g
