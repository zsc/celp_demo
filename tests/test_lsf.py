import unittest

import numpy as np

from celp_codec import lpc, lsf


class TestLsf(unittest.TestCase):
    def test_lpc_lsf_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        for order, fs in ((10, 8000), (16, 16000)):
            for _ in range(80):
                k = rng.uniform(-0.7, 0.7, size=(order,))
                a = lpc.step_up(k)
                a[0] = 1.0
                w = lsf.lpc_to_lsf(a, fs=fs)
                a2 = lsf.lsf_to_lpc(w)
                self.assertLess(float(np.max(np.abs(a - a2))), 1e-6)

    def test_warp_mix_keep_lsf_valid(self) -> None:
        fs = 16000
        base = np.linspace(150.0, 0.5 * fs - 200.0, 16, dtype=np.float64)
        base_w = 2.0 * np.pi * base / fs
        a = lsf.stabilize_lsf(base_w, fs=fs, min_sep_hz=50.0, edge_sep_hz=50.0)

        warped = lsf.warp_lsf(a, fs=fs, scale=1.08, min_sep_hz=50.0, edge_sep_hz=50.0)
        spread = lsf.spread_lsf(a, fs=fs, spread=1.2, min_sep_hz=50.0, edge_sep_hz=50.0)
        mixed = lsf.mix_lsf(spread, warped, mix=0.4, fs=fs, min_sep_hz=50.0, edge_sep_hz=50.0)

        self.assertTrue(np.all(np.isfinite(mixed)))
        self.assertTrue(np.all(np.diff(mixed) > 0.0))
        self.assertGreater(float(np.max(np.abs(spread - a))), 1e-5)
        a2 = lsf.lsf_to_lpc(mixed)
        self.assertTrue(np.all(np.isfinite(a2)))
        self.assertEqual(a2.shape, (17,))


if __name__ == "__main__":
    unittest.main()
