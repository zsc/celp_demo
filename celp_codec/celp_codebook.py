from __future__ import annotations

import numpy as np


def generate_codebook(seed: int, size: int, subframe_len: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    C = rng.standard_normal((int(size), int(subframe_len)), dtype=np.float64)
    norms = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    C = C / norms
    return C.astype(np.float64, copy=False)


def search_codebook(
    residual: np.ndarray, yc_mat: np.ndarray, eps: float = 1e-12
) -> int:
    """
    Choose codeword index maximizing (r·y)^2/(y·y) with non-negative gain.
    yc_mat: shape (M, L) where each row is y_c = H * C[m] in weighted domain.
    """
    r = np.asarray(residual, dtype=np.float64).ravel()
    Y = np.asarray(yc_mat, dtype=np.float64)

    dots = Y @ r
    energies = np.sum(Y * Y, axis=1) + eps
    scores = np.where(dots > 0.0, (dots * dots) / energies, 0.0)
    return int(np.argmax(scores))

