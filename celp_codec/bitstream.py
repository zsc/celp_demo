from __future__ import annotations

import base64
import struct
from dataclasses import dataclass


MAGIC = b"CLP1"
VERSION_V1 = 1
VERSION_V2 = 2

HEADER_V1_FMT = "<4sBBIHHBBBBI8s"
HEADER_V1_SIZE = struct.calcsize(HEADER_V1_FMT)  # 30 bytes

# v2 adds flexible pitch/innovation parameters (still little-endian).
# Fields:
#   magic(4s), version(B), mode(B),
#   fs(I), frame_len(H), subframe_len(H),
#   lpc_order(B), rc_bits(B), gain_bits_p(B), gain_bits_c(B),
#   lag_min(H), lag_max(H), pitch_frac_bits(B),
#   acelp_K(B), acelp_weight_bits(B), flags(B),
#   celp_codebook_size(H), celp_stages(B), reserved(B),
#   seed(I)
HEADER_V2_FMT = "<4sBBIHHBBBBHHBBBBHBBI"
HEADER_V2_SIZE = struct.calcsize(HEADER_V2_FMT)  # 34 bytes


@dataclass
class BitstreamHeaderV1:
    mode: int
    fs: int
    frame_len: int
    subframe_len: int
    lpc_order: int
    rc_bits: int
    gain_bits_p: int
    gain_bits_c: int
    seed: int

    def to_bytes(self) -> bytes:
        reserved = bytes(8)
        return struct.pack(
            HEADER_V1_FMT,
            MAGIC,
            VERSION_V1,
            int(self.mode),
            int(self.fs),
            int(self.frame_len),
            int(self.subframe_len),
            int(self.lpc_order),
            int(self.rc_bits),
            int(self.gain_bits_p),
            int(self.gain_bits_c),
            int(self.seed),
            reserved,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BitstreamHeaderV1":
        if len(data) < HEADER_V1_SIZE:
            raise ValueError("Bitstream too short for v1 header.")
        (
            magic,
            version,
            mode,
            fs,
            frame_len,
            subframe_len,
            lpc_order,
            rc_bits,
            gain_bits_p,
            gain_bits_c,
            seed,
            reserved,
        ) = struct.unpack(HEADER_V1_FMT, data[:HEADER_V1_SIZE])

        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        if version != VERSION_V1:
            raise ValueError(f"Not a v1 header (version={version}).")

        return cls(
            mode=int(mode),
            fs=int(fs),
            frame_len=int(frame_len),
            subframe_len=int(subframe_len),
            lpc_order=int(lpc_order),
            rc_bits=int(rc_bits),
            gain_bits_p=int(gain_bits_p),
            gain_bits_c=int(gain_bits_c),
            seed=int(seed),
        )


@dataclass
class BitstreamHeaderV2:
    mode: int
    fs: int
    frame_len: int
    subframe_len: int
    lpc_order: int
    rc_bits: int
    gain_bits_p: int
    gain_bits_c: int
    lag_min: int
    lag_max: int
    pitch_frac_bits: int
    acelp_K: int
    acelp_weight_bits: int
    flags: int
    celp_codebook_size: int
    celp_stages: int
    seed: int

    def to_bytes(self) -> bytes:
        return struct.pack(
            HEADER_V2_FMT,
            MAGIC,
            VERSION_V2,
            int(self.mode),
            int(self.fs),
            int(self.frame_len),
            int(self.subframe_len),
            int(self.lpc_order),
            int(self.rc_bits),
            int(self.gain_bits_p),
            int(self.gain_bits_c),
            int(self.lag_min),
            int(self.lag_max),
            int(self.pitch_frac_bits),
            int(self.acelp_K),
            int(self.acelp_weight_bits),
            int(self.flags),
            int(self.celp_codebook_size),
            int(self.celp_stages),
            0,
            int(self.seed),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "BitstreamHeaderV2":
        if len(data) < HEADER_V2_SIZE:
            raise ValueError("Bitstream too short for v2 header.")
        (
            magic,
            version,
            mode,
            fs,
            frame_len,
            subframe_len,
            lpc_order,
            rc_bits,
            gain_bits_p,
            gain_bits_c,
            lag_min,
            lag_max,
            pitch_frac_bits,
            acelp_K,
            acelp_weight_bits,
            flags,
            celp_codebook_size,
            celp_stages,
            reserved,
            seed,
        ) = struct.unpack(HEADER_V2_FMT, data[:HEADER_V2_SIZE])

        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        if version != VERSION_V2:
            raise ValueError(f"Not a v2 header (version={version}).")

        return cls(
            mode=int(mode),
            fs=int(fs),
            frame_len=int(frame_len),
            subframe_len=int(subframe_len),
            lpc_order=int(lpc_order),
            rc_bits=int(rc_bits),
            gain_bits_p=int(gain_bits_p),
            gain_bits_c=int(gain_bits_c),
            lag_min=int(lag_min),
            lag_max=int(lag_max),
            pitch_frac_bits=int(pitch_frac_bits),
            acelp_K=int(acelp_K),
            acelp_weight_bits=int(acelp_weight_bits),
            flags=int(flags),
            celp_codebook_size=int(celp_codebook_size),
            celp_stages=int(celp_stages),
            seed=int(seed),
        )


def read_header(data: bytes) -> tuple[BitstreamHeaderV1 | BitstreamHeaderV2, int]:
    if len(data) < 6:
        raise ValueError("Bitstream too short for magic/version.")
    if data[:4] != MAGIC:
        raise ValueError(f"Bad magic: {data[:4]!r}")
    version = data[4]
    if version == VERSION_V1:
        return BitstreamHeaderV1.from_bytes(data), HEADER_V1_SIZE
    if version == VERSION_V2:
        return BitstreamHeaderV2.from_bytes(data), HEADER_V2_SIZE
    raise ValueError(f"Unsupported version: {version}")


class BitWriter:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._bitpos = 0
        self.bits_written = 0

    def write_bits(self, value: int, nbits: int) -> None:
        value = int(value)
        nbits = int(nbits)
        if nbits < 0:
            raise ValueError("nbits must be >= 0")
        for i in range(nbits):
            bit = (value >> i) & 1
            if self._bitpos == 0:
                self._buf.append(0)
            if bit:
                self._buf[-1] |= 1 << self._bitpos
            self._bitpos += 1
            self.bits_written += 1
            if self._bitpos == 8:
                self._bitpos = 0

    def get_bytes(self) -> bytes:
        return bytes(self._buf)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._idx = 0
        self._bitpos = 0

    def read_bits(self, nbits: int) -> int:
        nbits = int(nbits)
        if nbits < 0:
            raise ValueError("nbits must be >= 0")
        value = 0
        for i in range(nbits):
            if self._idx >= len(self._data):
                raise EOFError("Unexpected EOF in bitstream.")
            byte = self._data[self._idx]
            bit = (byte >> self._bitpos) & 1
            value |= int(bit) << i
            self._bitpos += 1
            if self._bitpos == 8:
                self._bitpos = 0
                self._idx += 1
        return value


def bytes_hex_prefix(data: bytes, nbytes: int) -> str:
    n = int(min(max(nbytes, 0), len(data)))
    return data[:n].hex()


def bytes_base64_prefix(data: bytes, nbytes: int) -> str:
    n = int(min(max(nbytes, 0), len(data)))
    return base64.b64encode(data[:n]).decode("ascii")
