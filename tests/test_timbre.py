import unittest

import numpy as np

from celp_codec import codec, pitch, timbre
from celp_codec.bitstream import BitReader, BitWriter, BitstreamHeaderV1


class TestTimbre(unittest.TestCase):
    def test_identity_transform_v1(self) -> None:
        fs = 8000
        t = np.arange(int(0.08 * fs)) / fs
        x = 0.5 * np.sin(2 * np.pi * 160 * t) + 0.02 * np.random.default_rng(0).standard_normal(t.shape)
        x = np.clip(x, -1.0, 1.0)

        cfg = codec.CodecConfig(
            mode="acelp",
            fs=fs,
            frame_ms=20,
            subframe_ms=5,
            lpc_order=10,
            rc_bits=7,
            gain_bits_p=5,
            gain_bits_c=5,
            seed=1234,
            dp_pitch=False,
            lpc_interp=False,
        )
        bs_in, _, _, _ = codec.encode_samples_v1(x, cfg)

        out, _ = timbre.transform_bitstream(bs_in, timbre.TimbreParams())
        self.assertEqual(out, bs_in)

    def test_identity_transform_v2(self) -> None:
        fs = 8000
        t = np.arange(int(0.06 * fs)) / fs
        x = 0.55 * np.sin(2 * np.pi * 180 * t) + 0.02 * np.random.default_rng(1).standard_normal(t.shape)
        x = np.clip(x, -1.0, 1.0)

        cfg = codec.CodecConfig(
            mode="acelp",
            fs=fs,
            dp_pitch=False,
            lpc_interp=False,
            acelp_solver="ista",
            ista_iters=10,
            ista_lambda=0.02,
            acelp_K=4,
            seed=1234,
        )
        bs_in, _, _, _ = codec.encode_samples(x, cfg)

        out, _ = timbre.transform_bitstream(bs_in, timbre.TimbreParams())
        self.assertEqual(out, bs_in)

    def test_f0_scale_v1_changes_lag_bits(self) -> None:
        # Craft a tiny v1 ACELP bitstream with a known mid-range lag.
        h = BitstreamHeaderV1(
            mode=1,
            fs=8000,
            frame_len=160,
            subframe_len=40,
            lpc_order=10,
            rc_bits=7,
            gain_bits_p=5,
            gain_bits_c=5,
            seed=1234,
        )
        lag_min, lag_max = pitch.lag_bounds(8000, 50.0, 400.0, max_lag_bits=8)
        self.assertEqual(lag_min, 20)
        self.assertEqual(lag_max, 160)

        lag = 80
        lag_idx = lag - lag_min
        bw = BitWriter()
        for _ in range(h.lpc_order):
            bw.write_bits(64, h.rc_bits)
        for _ in range(h.frame_len // h.subframe_len):
            bw.write_bits(int(lag_idx), 8)
            bw.write_bits(10, h.gain_bits_p)
            bw.write_bits(10, h.gain_bits_c)
            for _t in range(4):
                bw.write_bits(0, 4)
                bw.write_bits(0, 1)
        bs_in = h.to_bytes() + bw.get_bytes()

        out, _ = timbre.transform_bitstream(bs_in, timbre.TimbreParams(f0_scale=2.0))
        _, hs = (h, len(h.to_bytes()))
        br = BitReader(out[hs:])
        _ = [br.read_bits(h.rc_bits) for _ in range(h.lpc_order)]
        lag_idx_out = int(br.read_bits(8))

        self.assertNotEqual(lag_idx_out, lag_idx)


if __name__ == "__main__":
    unittest.main()

