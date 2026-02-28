from __future__ import annotations

import numpy as np


def _deflate_1_plus_zinv(poly: np.ndarray) -> np.ndarray:
    """
    Divide poly(z^-1) by (1 + z^-1).

    poly is in z^-1 coefficient order: poly[k] is coeff for z^-k.
    """
    p = np.asarray(poly, dtype=np.float64).ravel()
    if p.size < 2:
        raise ValueError("poly too short to deflate")
    m = int(p.size - 1)
    q = np.empty((m,), dtype=np.float64)
    q[0] = p[0]
    for k in range(1, m):
        q[k] = p[k] - q[k - 1]
    # p[m] should equal q[m-1] (within numerical noise); ignore remainder.
    return q


def _deflate_1_minus_zinv(poly: np.ndarray) -> np.ndarray:
    """
    Divide poly(z^-1) by (1 - z^-1).

    poly is in z^-1 coefficient order: poly[k] is coeff for z^-k.
    """
    p = np.asarray(poly, dtype=np.float64).ravel()
    if p.size < 2:
        raise ValueError("poly too short to deflate")
    m = int(p.size - 1)
    q = np.empty((m,), dtype=np.float64)
    q[0] = p[0]
    for k in range(1, m):
        q[k] = p[k] + q[k - 1]
    # p[m] should equal -q[m-1] (within numerical noise); ignore remainder.
    return q


def _eval_poly_zinv(poly: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Evaluate poly(z^-1) at z=e^{j*omega}.
    Returns complex values for each omega.
    """
    c = np.asarray(poly, dtype=np.float64).ravel()
    w = np.asarray(omega, dtype=np.float64).ravel()
    k = np.arange(c.size, dtype=np.float64)
    E = np.exp(-1j * np.outer(w, k))
    return E @ c


def _bisect_root(poly: np.ndarray, lo: float, hi: float, iters: int = 40) -> float:
    """
    Bisection root refine on a real-valued function obtained from poly on unit circle.
    """
    flo = float(_eval_poly_zinv(poly, np.array([lo]))[0].real)
    fhi = float(_eval_poly_zinv(poly, np.array([hi]))[0].real)
    if flo == 0.0:
        return float(lo)
    if fhi == 0.0:
        return float(hi)
    if flo * fhi > 0.0:
        return float(0.5 * (lo + hi))

    a = float(lo)
    b = float(hi)
    fa = flo
    fb = fhi
    for _ in range(int(iters)):
        m = 0.5 * (a + b)
        fm = float(_eval_poly_zinv(poly, np.array([m]))[0].real)
        if fa * fm <= 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return float(0.5 * (a + b))


def _roots_on_unit_circle(poly: np.ndarray, n_roots: int, grid: int = 4096) -> np.ndarray:
    """
    Find roots in (0, pi) for a deflated LSP polynomial using sign-change bracketing.
    """
    n_roots = int(n_roots)
    if n_roots <= 0:
        return np.zeros((0,), dtype=np.float64)

    grid = int(max(grid, 256))
    w = np.linspace(0.0, float(np.pi), grid + 1, dtype=np.float64)
    f = _eval_poly_zinv(poly, w).real.astype(np.float64, copy=False)

    roots: list[float] = []
    for i in range(grid):
        f0 = float(f[i])
        f1 = float(f[i + 1])
        if f0 == 0.0:
            r = float(w[i])
            if 0.0 < r < float(np.pi):
                roots.append(r)
        if f0 * f1 < 0.0:
            r = _bisect_root(poly, float(w[i]), float(w[i + 1]))
            if 0.0 < r < float(np.pi):
                roots.append(r)
        if len(roots) >= n_roots:
            break

    out = np.array(roots[:n_roots], dtype=np.float64)
    out.sort()
    return out


def stabilize_lsf(
    lsf: np.ndarray,
    fs: int,
    min_sep_hz: float = 50.0,
    edge_sep_hz: float = 50.0,
) -> np.ndarray:
    """
    Enforce:
      - 0 < lsf_0 < ... < lsf_{p-1} < pi
      - minimal separation (in Hz) and edge margin (in Hz)
    """
    x = np.asarray(lsf, dtype=np.float64).ravel()
    fs = int(fs)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if x.size == 0:
        return x.copy()

    nyq = 0.5 * float(fs)
    min_sep = float(max(min_sep_hz, 1e-6))
    edge = float(max(edge_sep_hz, 1e-6))

    f = x * float(fs) / (2.0 * float(np.pi))
    f = np.clip(f, edge, nyq - edge)
    f = np.sort(f)

    # forward pass
    f[0] = max(f[0], edge)
    for i in range(1, f.size):
        f[i] = max(f[i], f[i - 1] + min_sep)

    # backward pass
    f[-1] = min(f[-1], nyq - edge)
    for i in range(f.size - 2, -1, -1):
        f[i] = min(f[i], f[i + 1] - min_sep)

    f = np.clip(f, edge, nyq - edge)
    w = 2.0 * float(np.pi) * f / float(fs)
    w = np.clip(w, 1e-6, float(np.pi) - 1e-6)
    w = np.sort(w)
    return w.astype(np.float64)


def warp_lsf(
    lsf: np.ndarray,
    fs: int,
    scale: float,
    min_sep_hz: float = 50.0,
    edge_sep_hz: float = 50.0,
) -> np.ndarray:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be > 0")
    x = np.asarray(lsf, dtype=np.float64).ravel()
    if x.size == 0:
        return x.copy()

    f = x * float(fs) / (2.0 * float(np.pi))
    f = f * scale
    w = 2.0 * float(np.pi) * f / float(fs)
    return stabilize_lsf(w, fs=fs, min_sep_hz=min_sep_hz, edge_sep_hz=edge_sep_hz)


def spread_lsf(
    lsf: np.ndarray,
    fs: int,
    spread: float,
    pivot_hz: float | None = None,
    min_sep_hz: float = 50.0,
    edge_sep_hz: float = 50.0,
) -> np.ndarray:
    """
    Expand/compress LSF spacing around a pivot frequency.

    spread > 1.0  : larger spacing (brighter / smaller-vocal tendency)
    spread < 1.0  : tighter spacing (darker / larger-vocal tendency)
    """
    s = float(spread)
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("spread must be > 0")

    x = np.asarray(lsf, dtype=np.float64).ravel()
    if x.size == 0:
        return x.copy()

    f = x * float(fs) / (2.0 * float(np.pi))
    if pivot_hz is None:
        pivot = float(np.mean(f))
    else:
        pivot = float(pivot_hz)

    f2 = pivot + s * (f - pivot)
    w = 2.0 * float(np.pi) * f2 / float(fs)
    return stabilize_lsf(w, fs=fs, min_sep_hz=min_sep_hz, edge_sep_hz=edge_sep_hz)


def mix_lsf(
    lsf_src: np.ndarray,
    lsf_tgt: np.ndarray,
    mix: float,
    fs: int,
    min_sep_hz: float = 50.0,
    edge_sep_hz: float = 50.0,
) -> np.ndarray:
    m = float(mix)
    if not np.isfinite(m):
        raise ValueError("mix must be finite")
    m = float(min(max(m, 0.0), 1.0))
    a = np.asarray(lsf_src, dtype=np.float64).ravel()
    b = np.asarray(lsf_tgt, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("lsf_src and lsf_tgt must have the same length")
    w = (1.0 - m) * a + m * b
    return stabilize_lsf(w, fs=fs, min_sep_hz=min_sep_hz, edge_sep_hz=edge_sep_hz)


def lpc_to_lsf(a: np.ndarray, fs: int, grid: int = 4096) -> np.ndarray:
    """
    Convert LPC A(z) (a[0]=1) to LSF (radians, length p).

    Notes:
      - currently supports even order p (common in this repo: 10 or 16)
      - uses the canonical P/Q root decomposition
    """
    del grid  # kept for API compatibility

    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size < 2:
        raise ValueError("a must have at least 2 coefficients")
    if a[0] == 0.0 or not np.isfinite(a[0]):
        raise ValueError("Invalid a[0]")
    if a[0] != 1.0:
        a = a / float(a[0])

    p = int(a.size - 1)
    if p % 2 != 0:
        raise ValueError("lpc_to_lsf currently requires even LPC order")

    # Build P(x), Q(x) in x=z^-1:
    #   P(x)=A(x)+x^{p+1}A(1/x), Q(x)=A(x)-x^{p+1}A(1/x)
    P = np.zeros((p + 2,), dtype=np.float64)
    Q = np.zeros((p + 2,), dtype=np.float64)
    for k in range(p + 1):
        ak = float(a[k])
        P[k] += ak
        P[p + 1 - k] += ak
        Q[k] += ak
        Q[p + 1 - k] -= ak

    rootsP = np.roots(P[::-1])
    rootsQ = np.roots(Q[::-1])

    def _extract_half(roots: np.ndarray, trivial: complex) -> np.ndarray:
        roots = np.asarray(roots, dtype=np.complex128).ravel()
        if roots.size < (p // 2 + 1):
            raise ValueError("Insufficient P/Q roots.")
        rm = int(np.argmin(np.abs(roots - trivial)))
        roots = np.delete(roots, rm)

        ang = np.angle(roots[np.imag(roots) > 1e-8])
        ang = ang[(ang > 1e-6) & (ang < float(np.pi) - 1e-6)]
        vals = np.sort(ang.astype(np.float64, copy=False))
        if vals.size == p // 2:
            return vals

        # Fallback for numerically near-real conjugates.
        all_ang = np.abs(np.angle(roots)).astype(np.float64, copy=False)
        all_ang = all_ang[(all_ang > 1e-6) & (all_ang < float(np.pi) - 1e-6)]
        all_ang.sort()
        dedup: list[float] = []
        for v in all_ang.tolist():
            if not dedup or abs(v - dedup[-1]) > 1e-4:
                dedup.append(float(v))
        vals = np.array(dedup, dtype=np.float64)
        if vals.size != p // 2:
            raise ValueError("Failed to recover expected LSF roots.")
        return vals

    rootsP_w = _extract_half(rootsP, trivial=-1.0 + 0.0j)
    rootsQ_w = _extract_half(rootsQ, trivial=1.0 + 0.0j)

    out = np.empty((p,), dtype=np.float64)
    out[0::2] = rootsP_w
    out[1::2] = rootsQ_w
    out = np.clip(out, 1e-6, float(np.pi) - 1e-6)
    return np.sort(out)


def lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
    """
    Convert LSF (radians) back to LPC A(z) coefficients (a[0]=1).

    Currently supports even order.
    """
    w = np.asarray(lsf, dtype=np.float64).ravel()
    if w.size == 0:
        return np.array([1.0], dtype=np.float64)
    p = int(w.size)
    if p % 2 != 0:
        raise ValueError("lsf_to_lpc currently requires even order")
    w = np.sort(np.clip(w, 1e-6, float(np.pi) - 1e-6))

    P_ws = w[0::2]
    Q_ws = w[1::2]

    P = np.array([1.0], dtype=np.float64)
    for wi in P_ws.tolist():
        c = float(np.cos(float(wi)))
        P = np.convolve(P, np.array([1.0, -2.0 * c, 1.0], dtype=np.float64))
    P = np.convolve(P, np.array([1.0, 1.0], dtype=np.float64))

    Q = np.array([1.0], dtype=np.float64)
    for wi in Q_ws.tolist():
        c = float(np.cos(float(wi)))
        Q = np.convolve(Q, np.array([1.0, -2.0 * c, 1.0], dtype=np.float64))
    Q = np.convolve(Q, np.array([1.0, -1.0], dtype=np.float64))

    A_full = 0.5 * (P + Q)
    a = np.asarray(A_full[: p + 1], dtype=np.float64).ravel()
    if a.size != p + 1:
        raise ValueError("Internal LSF->LPC size mismatch")
    a[0] = 1.0
    a[~np.isfinite(a)] = 0.0
    return a
