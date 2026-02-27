from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def lag_min_from_fs(fs: int, pitch_max_hz: float) -> int:
    return int(np.floor(float(fs) / float(pitch_max_hz)))


def lag_bounds(
    fs: int, pitch_min_hz: float, pitch_max_hz: float, max_lag_bits: int | None = None
) -> tuple[int, int]:
    lag_min = lag_min_from_fs(fs, pitch_max_hz)
    lag_max = int(np.ceil(float(fs) / float(pitch_min_hz)))
    if max_lag_bits is not None:
        max_range = (1 << int(max_lag_bits)) - 1
        lag_max = min(lag_max, lag_min + max_range)
    if lag_min < 1:
        lag_min = 1
    if lag_max < lag_min:
        lag_max = lag_min
    return lag_min, lag_max


def bits_for_lag(lag_min: int, lag_max: int, frac_bits: int = 0) -> int:
    """
    Bits needed to encode (lag, frac) where:
      lag in [lag_min, lag_max], frac in [0, 2^frac_bits-1]
    """
    lag_min = int(lag_min)
    lag_max = int(lag_max)
    frac_bits = int(frac_bits)
    span = max(1, (lag_max - lag_min + 1) * (1 << frac_bits))
    return int(np.ceil(np.log2(span))) if span > 1 else 1


def bits_for_pos(subframe_len: int) -> int:
    L = int(subframe_len)
    if L <= 1:
        return 1
    return int(np.ceil(np.log2(L)))


def topk_pitch_candidates(
    d: np.ndarray,
    H: np.ndarray,
    exc_buf: np.ndarray,
    lag_min: int,
    lag_max: int,
    topk: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pitch scores for all lags and return top-k (lags, scores).

    score(lag) = (d·y)^2 / (y·y + eps) with y = H @ e_p(lag), only if d·y>0.
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    H = np.asarray(H, dtype=np.float64)
    exc_buf = np.asarray(exc_buf, dtype=np.float64).ravel()

    L = int(d.size)
    N = int(exc_buf.size)
    lags = np.arange(int(lag_min), int(lag_max) + 1, dtype=np.int64)
    if lags.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)

    windows = sliding_window_view(exc_buf, window_shape=L)  # (N-L+1, L)
    starts = (N - lags - L).astype(np.int64)
    starts = np.clip(starts, 0, windows.shape[0] - 1)
    ep_mat = windows[starts]  # (#lags, L)
    yp_mat = ep_mat @ H.T

    dots = yp_mat @ d
    energies = np.sum(yp_mat * yp_mat, axis=1) + eps
    scores = np.where(dots > 0.0, (dots * dots) / energies, 0.0)

    k = int(min(max(int(topk), 1), scores.size))
    order = np.argsort(-scores, kind="mergesort")[:k]
    return lags[order], scores[order]


def viterbi_smooth_lags(
    cand_lags: list[np.ndarray], cand_scores: list[np.ndarray], lam: float
) -> list[int]:
    T = len(cand_lags)
    if T == 0:
        return []

    lam = float(lam)
    costs: list[np.ndarray] = []
    back: list[np.ndarray] = []

    c0 = -np.asarray(cand_scores[0], dtype=np.float64)
    costs.append(c0)
    back.append(-np.ones_like(c0, dtype=np.int64))

    for t in range(1, T):
        lags_t = np.asarray(cand_lags[t], dtype=np.int64)
        scores_t = np.asarray(cand_scores[t], dtype=np.float64)
        prev_lags = np.asarray(cand_lags[t - 1], dtype=np.int64)
        prev_costs = costs[t - 1]

        ct = np.empty_like(scores_t, dtype=np.float64)
        bt = np.empty_like(lags_t, dtype=np.int64)

        for i in range(lags_t.size):
            lag = int(lags_t[i])
            best_j = 0
            best_cost = float("inf")
            for j in range(prev_lags.size):
                diff = lag - int(prev_lags[j])
                c = float(prev_costs[j] + lam * (diff * diff))
                if c < best_cost - 1e-18:
                    best_cost = c
                    best_j = j
                elif c == best_cost:
                    if int(prev_lags[j]) < int(prev_lags[best_j]):
                        best_j = j
            ct[i] = -float(scores_t[i]) + best_cost
            bt[i] = best_j

        costs.append(ct)
        back.append(bt)

    last_costs = costs[-1]
    end_i = int(np.argmin(last_costs))

    path = [0] * T
    path[T - 1] = int(np.asarray(cand_lags[T - 1], dtype=np.int64)[end_i])
    idx = end_i
    for t in range(T - 1, 0, -1):
        idx = int(back[t][idx])
        path[t - 1] = int(np.asarray(cand_lags[t - 1], dtype=np.int64)[idx])

    return path
