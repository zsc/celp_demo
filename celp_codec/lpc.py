from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    order = int(order)
    if order < 0:
        raise ValueError("order must be >= 0")
    r = np.zeros((order + 1,), dtype=np.float64)
    if x.size == 0:
        return r
    for k in range(order + 1):
        r[k] = float(np.dot(x[k:], x[: x.size - k]))
    return r


def levinson_durbin(r: np.ndarray, order: int, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Levinson-Durbin recursion for autocorrelation sequence r[0..order].

    Returns:
      a: LPC polynomial coefficients (length order+1, a[0]=1)
      k: reflection coefficients (length order)
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    order = int(order)
    if r.size < order + 1:
        raise ValueError("r must have at least order+1 elements")

    a = np.zeros((order + 1,), dtype=np.float64)
    a[0] = 1.0
    k = np.zeros((order,), dtype=np.float64)

    if not np.isfinite(r[0]) or r[0] <= eps:
        return a, k

    E = float(r[0])
    for i in range(1, order + 1):
        acc = float(r[i])
        for j in range(1, i):
            acc += float(a[j] * r[i - j])
        ki = -acc / (E + eps)
        ki = float(np.clip(ki, -0.9999, 0.9999))
        k[i - 1] = ki

        a_prev = a.copy()
        a[i] = ki
        for j in range(1, i):
            a[j] = a_prev[j] + ki * a_prev[i - j]

        E *= 1.0 - ki * ki
        if not np.isfinite(E) or E <= eps:
            break

    return a, k


def step_up(k: np.ndarray) -> np.ndarray:
    """Step-up recursion: reflection coeffs -> LPC a (a[0]=1)."""
    k = np.asarray(k, dtype=np.float64).ravel()
    order = int(k.size)
    a = np.zeros((order + 1,), dtype=np.float64)
    a[0] = 1.0
    for i in range(1, order + 1):
        ki = float(np.clip(k[i - 1], -0.9999, 0.9999))
        a_prev = a.copy()
        a[i] = ki
        for j in range(1, i):
            a[j] = a_prev[j] + ki * a_prev[i - j]
    return a


def quantize_reflection_coeffs(k: np.ndarray, bits: int, vmax: float = 2.5) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64).ravel()
    bits = int(bits)
    if bits <= 0:
        raise ValueError("bits must be > 0")

    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    k = np.clip(k, -0.9999, 0.9999)
    v = np.arctanh(k)
    v = np.clip(v, -vmax, vmax)

    levels = (1 << bits) - 1
    idx = np.rint((v + vmax) / (2.0 * vmax) * levels).astype(np.int64)
    idx = np.clip(idx, 0, levels)
    return idx.astype(np.int64)


def dequantize_reflection_coeffs(idx: np.ndarray, bits: int, vmax: float = 2.5) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).ravel()
    bits = int(bits)
    if bits <= 0:
        raise ValueError("bits must be > 0")
    levels = (1 << bits) - 1
    v = (idx.astype(np.float64) / levels) * (2.0 * vmax) - vmax
    k = np.tanh(v)
    k = np.clip(k, -0.9999, 0.9999)
    return k.astype(np.float64)

