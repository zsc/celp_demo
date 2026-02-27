import tempfile
import unittest
from pathlib import Path

import numpy as np

from celp_codec import codec, wav_io


class TestRoundtrip(unittest.TestCase):
    def test_roundtrip_acelp(self) -> None:
        fs = 8000
        t = np.arange(int(0.3 * fs)) / fs
        # simple voiced-like signal (sine + weak noise)
        x = 0.6 * np.sin(2 * np.pi * 180 * t) + 0.05 * np.random.default_rng(0).standard_normal(t.shape)
        x = np.clip(x, -1.0, 1.0)

        cfg = codec.CodecConfig(mode="acelp", fs=fs, dp_pitch=False, acelp_solver="ista", seed=1234)
        bs, y_enc, _, st = codec.encode_samples(x, cfg)
        y_dec, header, _ = codec.decode_bitstream(bs, clip=True)

        # decode output should be finite and roughly same length (padding allowed)
        self.assertTrue(np.isfinite(y_dec).all())
        self.assertGreaterEqual(y_dec.size, x.size)

        # roundtrip via file IO smoke
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_wav = td / "in.wav"
            out_bin = td / "out.celpbin"
            out_wav = td / "out.wav"
            wav_io.write_wav(in_wav, x, fs, clip=True)
            out_bin.write_bytes(bs)
            y2, _, _ = codec.decode_bitstream(out_bin.read_bytes(), clip=True)
            wav_io.write_wav(out_wav, y2, fs, clip=True)
            self.assertTrue(out_wav.exists())


if __name__ == "__main__":
    unittest.main()
