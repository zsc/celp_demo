from __future__ import annotations

import numpy as np

from .filters import soft_threshold


def ista_lasso(
    residual: np.ndarray,
    H: np.ndarray,
    tau: float,
    lam: float,
    iters: int,
) -> np.ndarray:
    """
    Solve min_c 0.5*||r - Hc||^2 + lam*||c||_1 via ISTA.
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.float64)
    L = int(r.size)
    c = np.zeros((L,), dtype=np.float64)

    tau = float(tau)
    lam = float(lam)
    iters = int(iters)

    for _ in range(iters):
        y = H @ c
        grad = H.T @ (y - r)
        c = soft_threshold(c - tau * grad, tau * lam)
    return c


def topk_support(c_cont: np.ndarray, k: int) -> np.ndarray:
    c = np.asarray(c_cont, dtype=np.float64).ravel()
    k = int(k)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-np.abs(c), kind="mergesort")
    sel = order[: min(k, order.size)]
    sel = np.asarray(sel, dtype=np.int64)
    # Deterministic bitstream order: increasing position
    return np.sort(sel)


def refine_weights_ls(residual: np.ndarray, H: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Given support positions, refine pulse weights by least squares:
      min_w || r - H[:,pos] @ w ||^2
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.int64).ravel()
    if pos.size == 0:
        return np.zeros((0,), dtype=np.float64)
    A = H[:, pos]
    w, *_ = np.linalg.lstsq(A, r, rcond=None)
    w = np.asarray(w, dtype=np.float64).ravel()
    w[~np.isfinite(w)] = 0.0
    return w


def omp_support(
    residual: np.ndarray,
    H: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Matching Pursuit (OMP) support selection for:
      min_c ||r - Hc||^2  s.t. ||c||_0 <= k

    Returns:
      positions (sorted int64), weights (float64) for those positions.
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.float64)
    L = int(r.size)
    k = int(min(max(int(k), 0), L))
    if k == 0 or L == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)

    selected: list[int] = []
    rr = r.copy()

    for _ in range(k):
        corr = H.T @ rr
        abs_corr = np.abs(corr)
        if selected:
            abs_corr[np.array(selected, dtype=np.int64)] = -1.0
        pos = int(np.argmax(abs_corr))
        if pos < 0 or pos >= L:
            break
        selected.append(pos)

        A = H[:, selected]
        w, *_ = np.linalg.lstsq(A, r, rcond=None)
        w = np.asarray(w, dtype=np.float64).ravel()
        w[~np.isfinite(w)] = 0.0
        rr = r - A @ w

    pos_sorted = np.array(sorted(selected), dtype=np.int64)
    if pos_sorted.size == 0:
        return pos_sorted, np.zeros((0,), dtype=np.float64)
    w_sorted = refine_weights_ls(r, H, pos_sorted)
    return pos_sorted, w_sorted


def normalize_shape(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        return w
    s = float(np.max(np.abs(w)))
    if not np.isfinite(s) or s <= eps:
        return np.zeros_like(w)
    return w / s


def quantize_unit(values: np.ndarray, bits: int) -> np.ndarray:
    """
    Uniform quantizer for [-1,1] -> uint in [0, 2^bits-1].
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    bits = int(bits)
    if bits <= 0:
        raise ValueError("bits must be > 0")
    levels = (1 << bits) - 1
    x = np.clip(v, -1.0, 1.0)
    idx = np.rint((x + 1.0) * 0.5 * levels).astype(np.int64)
    return np.clip(idx, 0, levels)


def dequantize_unit(idxs: np.ndarray, bits: int) -> np.ndarray:
    idx = np.asarray(idxs, dtype=np.int64).ravel()
    bits = int(bits)
    if bits <= 0:
        raise ValueError("bits must be > 0")
    levels = (1 << bits) - 1
    x = (idx.astype(np.float64) / levels) * 2.0 - 1.0
    return np.clip(x, -1.0, 1.0)


def support_to_vector(positions: np.ndarray, weights: np.ndarray, length: int) -> np.ndarray:
    L = int(length)
    pos = np.asarray(positions, dtype=np.int64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if pos.size != w.size:
        raise ValueError("positions and weights must have same length")
    c = np.zeros((L,), dtype=np.float64)
    if pos.size:
        c[pos] = w
    return c
