from __future__ import annotations

import numpy as np


def preemphasis(x: np.ndarray, coeff: float) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    if coeff == 0.0 or x64.size == 0:
        return x64.copy()
    y = np.empty_like(x64)
    y[0] = x64[0]
    y[1:] = x64[1:] - coeff * x64[:-1]
    return y


def deemphasis(y: np.ndarray, coeff: float) -> np.ndarray:
    y64 = np.asarray(y, dtype=np.float64)
    if coeff == 0.0 or y64.size == 0:
        return y64.copy()
    x = np.empty_like(y64)
    x[0] = y64[0]
    for i in range(1, y64.size):
        x[i] = y64[i] + coeff * x[i - 1]
    return x


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def iir_filter(
    b: np.ndarray, a: np.ndarray, x: np.ndarray, zi: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Direct Form II transposed IIR filtering (SciPy lfilter-style state).

    - a[0] is normalized to 1
    - state zi has length max(len(a), len(b)) - 1
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()

    if b.size == 0 or a.size == 0:
        raise ValueError("Empty filter coefficients.")
    if not np.isfinite(a[0]) or a[0] == 0.0:
        raise ValueError("Invalid a[0] for IIR filter.")

    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    n = int(max(a.size, b.size))
    if a.size < n:
        a = np.pad(a, (0, n - a.size))
    if b.size < n:
        b = np.pad(b, (0, n - b.size))

    if n == 1:
        return b[0] * x, np.zeros((0,), dtype=np.float64)

    if zi is None:
        z = np.zeros((n - 1,), dtype=np.float64)
    else:
        z = np.asarray(zi, dtype=np.float64).ravel().copy()
        if z.shape != (n - 1,):
            raise ValueError(f"zi must have shape {(n - 1,)}; got {z.shape}.")

    y = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        xi = x[i]
        yi = b[0] * xi + z[0]
        y[i] = yi
        for j in range(1, n - 1):
            z[j - 1] = b[j] * xi + z[j] - a[j] * yi
        z[n - 2] = b[n - 1] * xi - a[n - 1] * yi

    return y, z


def bandwidth_expand(a: np.ndarray, gamma: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    out = a.copy()
    if out.size <= 1:
        return out
    g = float(gamma)
    pows = g ** np.arange(out.size, dtype=np.float64)
    out[1:] *= pows[1:]
    out[0] = 1.0
    return out


def impulse_response(b: np.ndarray, a: np.ndarray, length: int) -> np.ndarray:
    length = int(length)
    if length <= 0:
        return np.zeros((0,), dtype=np.float64)
    x = np.zeros((length,), dtype=np.float64)
    x[0] = 1.0
    h, _ = iir_filter(b, a, x)
    return h


def conv_trunc(h: np.ndarray, x: np.ndarray, length: int) -> np.ndarray:
    y = np.convolve(np.asarray(h, dtype=np.float64), np.asarray(x, dtype=np.float64))
    if y.size >= length:
        return y[:length]
    out = np.zeros((length,), dtype=np.float64)
    out[: y.size] = y
    return out


def conv_matrix(h: np.ndarray) -> np.ndarray:
    """
    Build lower-triangular Toeplitz convolution matrix H such that:
      y = H @ x  ==  conv(h, x)[:L]   (with L = len(h) = len(x))
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    L = int(h.size)
    H = np.zeros((L, L), dtype=np.float64)
    for n in range(L):
        H[n, : n + 1] = h[n::-1]
    return H


def shifted_atom(h: np.ndarray, pos: int) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64).ravel()
    L = int(h.size)
    pos = int(pos)
    atom = np.zeros((L,), dtype=np.float64)
    if 0 <= pos < L:
        atom[pos:] = h[: L - pos]
    return atom

