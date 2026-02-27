import unittest

import numpy as np

from celp_codec import lpc


class TestLpcStability(unittest.TestCase):
    def test_rc_quant_dequant_stable(self) -> None:
        rng = np.random.default_rng(0)
        k = rng.uniform(-0.95, 0.95, size=(10,))
        idx = lpc.quantize_reflection_coeffs(k, bits=7)
        k_hat = lpc.dequantize_reflection_coeffs(idx, bits=7)
        self.assertTrue(np.all(np.abs(k_hat) < 1.0))
        a_hat = lpc.step_up(k_hat)
        self.assertTrue(np.isfinite(a_hat).all())
        self.assertEqual(a_hat.shape, (11,))


if __name__ == "__main__":
    unittest.main()

