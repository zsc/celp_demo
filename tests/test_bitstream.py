import unittest

from celp_codec.bitstream import (
    BitReader,
    BitWriter,
    BitstreamHeaderV2,
    HEADER_V2_SIZE,
    VERSION_V2,
    read_header,
)


class TestBitstream(unittest.TestCase):
    def test_bits_roundtrip(self) -> None:
        bw = BitWriter()
        vals = [(3, 2), (255, 8), (5, 3), (17, 5), (0, 1)]
        for v, n in vals:
            bw.write_bits(v, n)
        data = bw.get_bytes()
        br = BitReader(data)
        out = [br.read_bits(n) for _, n in vals]
        self.assertEqual(out, [v for v, _ in vals])

    def test_header_pack_unpack(self) -> None:
        h = BitstreamHeaderV2(
            mode=1,
            fs=8000,
            frame_len=160,
            subframe_len=40,
            lpc_order=10,
            rc_bits=10,
            gain_bits_p=7,
            gain_bits_c=7,
            seed=1234,
            lag_min=20,
            lag_max=160,
            pitch_frac_bits=0,
            acelp_K=10,
            acelp_weight_bits=5,
            flags=0,
            celp_codebook_size=0,
            celp_stages=0,
        )
        b = h.to_bytes()
        self.assertEqual(len(b), HEADER_V2_SIZE)
        h2, hs = read_header(b + b"payload")
        self.assertEqual(hs, HEADER_V2_SIZE)
        self.assertEqual(h2.mode, h.mode)
        self.assertEqual(h2.fs, h.fs)
        self.assertEqual(h2.frame_len, h.frame_len)
        self.assertEqual(h2.subframe_len, h.subframe_len)
        self.assertEqual(h2.lpc_order, h.lpc_order)
        self.assertEqual(h2.rc_bits, h.rc_bits)
        self.assertEqual(h2.gain_bits_p, h.gain_bits_p)
        self.assertEqual(h2.gain_bits_c, h.gain_bits_c)
        self.assertEqual(h2.seed, h.seed)
        self.assertEqual(h2.lag_min, h.lag_min)
        self.assertEqual(h2.lag_max, h.lag_max)
        self.assertEqual(h2.acelp_K, h.acelp_K)
        self.assertEqual(h2.acelp_weight_bits, h.acelp_weight_bits)
        self.assertEqual(h2.flags, h.flags)
        self.assertEqual(h2.celp_codebook_size, h.celp_codebook_size)
        self.assertEqual(h2.celp_stages, h.celp_stages)


if __name__ == "__main__":
    unittest.main()
