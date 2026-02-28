from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import bitstream, codec, wav_io


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bitstream-version", type=int, default=2, choices=[1, 2])
    p.add_argument("--fs", type=int, default=16000, choices=[8000, 16000])
    p.add_argument("--frame-ms", type=int, default=20)
    p.add_argument("--subframe-ms", type=int, default=5)

    p.add_argument("--lpc-order", type=int, default=None)
    p.add_argument("--lpc-preemph", type=float, default=0.97)
    p.add_argument("--rc-bits", type=int, default=10)

    p.add_argument("--pitch-min-hz", type=float, default=50.0)
    p.add_argument("--pitch-max-hz", type=float, default=400.0)
    p.add_argument("--pitch-frac-bits", type=int, default=0, choices=[0, 1, 2])

    p.add_argument("--dp-pitch", type=str, default="on", choices=["on", "off"])
    p.add_argument("--dp-topk", type=int, default=10)
    p.add_argument("--dp-lambda", type=float, default=0.05)

    p.add_argument("--gain-bits-p", type=int, default=7)
    p.add_argument("--gain-bits-c", type=int, default=7)

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--postfilter", action=argparse.BooleanOptionalAction, default=False)

    # CELP innovation (v2)
    p.add_argument("--celp-codebook-size", type=int, default=2048)
    p.add_argument("--celp-stages", type=int, default=2)

    # ACELP innovation (v2)
    p.add_argument("--acelp-K", type=int, default=None)
    p.add_argument("--acelp-weight-bits", type=int, default=5)
    p.add_argument("--acelp-solver", type=str, default="omp", choices=["ista", "omp"])
    p.add_argument("--ista-iters", type=int, default=60)
    p.add_argument("--ista-lambda", type=float, default=0.02)


def _cfg_from_args(args: argparse.Namespace) -> codec.CodecConfig:
    return codec.CodecConfig(
        mode=args.mode,
        fs=int(args.fs),
        frame_ms=int(args.frame_ms),
        subframe_ms=int(args.subframe_ms),
        lpc_order=args.lpc_order,
        lpc_preemph=float(args.lpc_preemph),
        pitch_min_hz=float(args.pitch_min_hz),
        pitch_max_hz=float(args.pitch_max_hz),
        pitch_frac_bits=int(args.pitch_frac_bits),
        dp_pitch=(args.dp_pitch == "on"),
        dp_topk=int(args.dp_topk),
        dp_lambda=float(args.dp_lambda),
        rc_bits=int(args.rc_bits),
        gain_bits_p=int(args.gain_bits_p),
        gain_bits_c=int(args.gain_bits_c),
        seed=int(args.seed),
        clip=bool(args.clip),
        postfilter=bool(args.postfilter),
        celp_codebook_size=int(args.celp_codebook_size),
        celp_stages=int(args.celp_stages),
        acelp_K=args.acelp_K,
        acelp_weight_bits=int(args.acelp_weight_bits),
        acelp_solver=str(args.acelp_solver),
        ista_iters=int(args.ista_iters),
        ista_lambda=float(args.ista_lambda),
    )


def _print_summary(cfg: codec.CodecConfig, stats: dict, m: dict, wall_s: float) -> None:
    bitrate_kbps = stats["total_bits"] / max(stats["seconds"], 1e-9) / 1000.0
    print(
        f"mode={cfg.mode} fs={cfg.fs} ver={stats.get('version','?')} frames={stats['frames']} "
        f"bitrate={bitrate_kbps:.2f} kbps "
        f"SNR={m['snr_db']:.2f} dB segSNR={m['seg_snr_db']:.2f} dB "
        f"time={wall_s:.2f}s"
    )


def _encode_dispatch(
    x: object,
    cfg: codec.CodecConfig,
    version: int,
    dump_json_path: str | None,
) -> tuple[bytes, object, dict | None, dict]:
    ver = int(version)
    if ver == 1:
        return codec.encode_samples_v1(x, cfg, dump_json_path=dump_json_path)
    if ver == 2:
        return codec.encode_samples(x, cfg, dump_json_path=dump_json_path)
    raise ValueError(f"Unsupported bitstream version: {ver}")


def cmd_roundtrip(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)

    x, fs_in = wav_io.read_wav(args.input)
    x = wav_io.resample_to_fs(x, fs_in, cfg.fs)

    t0 = time.time()
    bit_bytes, _, _, stats = _encode_dispatch(
        x,
        cfg,
        version=int(args.bitstream_version),
        dump_json_path=args.dump_json,
    )
    Path(args.out_bitstream).write_bytes(bit_bytes)

    if args.print_hex is not None:
        print("hex:", bitstream.bytes_hex_prefix(bit_bytes, args.print_hex))
    if args.print_base64 is not None:
        print("base64:", bitstream.bytes_base64_prefix(bit_bytes, args.print_base64))

    y, header, _ = codec.decode_bitstream(bit_bytes, clip=cfg.clip)
    wav_io.write_wav(args.out_wav, y, header.fs, clip=cfg.clip)

    m = codec.roundtrip_metrics(x, y, frame_len=cfg.frame_len())
    t1 = time.time()
    _print_summary(cfg, stats, m, wall_s=t1 - t0)
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    x, fs_in = wav_io.read_wav(args.input)
    x = wav_io.resample_to_fs(x, fs_in, cfg.fs)
    bit_bytes, _, _, stats = _encode_dispatch(
        x,
        cfg,
        version=int(args.bitstream_version),
        dump_json_path=args.dump_json,
    )
    Path(args.out).write_bytes(bit_bytes)
    if args.print_hex is not None:
        print("hex:", bitstream.bytes_hex_prefix(bit_bytes, args.print_hex))
    if args.print_base64 is not None:
        print("base64:", bitstream.bytes_base64_prefix(bit_bytes, args.print_base64))
    bitrate_kbps = stats["total_bits"] / max(stats["seconds"], 1e-9) / 1000.0
    print(f"wrote {args.out} frames={stats['frames']} bitrate={bitrate_kbps:.2f} kbps")
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    data = Path(args.input).read_bytes()
    y, header, st = codec.decode_bitstream(data, clip=args.clip)
    wav_io.write_wav(args.out, y, header.fs, clip=args.clip)
    print(f"decoded frames={st['frames']} fs={header.fs} -> {args.out}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    x, fsx = wav_io.read_wav(args.x)
    y, fsy = wav_io.read_wav(args.y)
    if fsx != fsy:
        y = wav_io.resample_to_fs(y, fsy, fsx)
    m = codec.roundtrip_metrics(x, y, frame_len=int(round(fsx * 0.02)))
    print(f"SNR={m['snr_db']:.2f} dB segSNR={m['seg_snr_db']:.2f} dB")
    return 0


def _out9_preset_config() -> codec.CodecConfig:
    return codec.CodecConfig(
        mode="acelp",
        fs=8000,
        frame_ms=20,
        subframe_ms=5,
        lpc_order=10,
        lpc_preemph=0.97,
        lpc_interp=False,
        pitch_min_hz=50.0,
        pitch_max_hz=400.0,
        pitch_frac_bits=0,
        dp_pitch=True,
        dp_topk=10,
        dp_lambda=0.2,
        rc_bits=7,
        gain_bits_p=5,
        gain_bits_c=5,
        gp_max=1.2,
        gc_max=2.0,
        seed=1234,
        clip=True,
        postfilter=False,
    )


def cmd_out9(args: argparse.Namespace) -> int:
    cfg = _out9_preset_config()
    cfg.clip = bool(args.clip)

    x, fs_in = wav_io.read_wav(args.input)
    x = wav_io.resample_to_fs(x, fs_in, cfg.fs)

    t0 = time.time()
    bit_bytes, _, _, stats = codec.encode_samples_v1(x, cfg, dump_json_path=args.dump_json)
    Path(args.out_bitstream).write_bytes(bit_bytes)

    if args.print_hex is not None:
        print("hex:", bitstream.bytes_hex_prefix(bit_bytes, args.print_hex))
    if args.print_base64 is not None:
        print("base64:", bitstream.bytes_base64_prefix(bit_bytes, args.print_base64))

    y, header, _ = codec.decode_bitstream(bit_bytes, clip=cfg.clip)
    wav_io.write_wav(args.out_wav, y, int(header.fs), clip=cfg.clip)

    m = codec.roundtrip_metrics(x, y, frame_len=cfg.frame_len())
    t1 = time.time()
    _print_summary(cfg, stats, m, wall_s=t1 - t0)
    print("preset=out9 version=1 mode=acelp fs=8000 frame_len=160 subframe_len=40 p=10 rc=7 gp=5 gc=5 seed=1234")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="celpcodec")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("roundtrip", help="encode + decode")
    pr.add_argument("--in", dest="input", required=True)
    pr.add_argument("--mode", required=True, choices=["celp", "acelp"])
    pr.add_argument("--out-bitstream", dest="out_bitstream", required=True)
    pr.add_argument("--out-wav", dest="out_wav", required=True)
    pr.add_argument("--dump-json", dest="dump_json", default=None)
    pr.add_argument("--print-hex", dest="print_hex", type=int, default=None)
    pr.add_argument("--print-base64", dest="print_base64", type=int, default=None)
    _add_common_args(pr)
    pr.set_defaults(func=cmd_roundtrip)

    pe = sub.add_parser("encode", help="encode wav -> bitstream")
    pe.add_argument("--in", dest="input", required=True)
    pe.add_argument("--mode", required=True, choices=["celp", "acelp"])
    pe.add_argument("--out", required=True)
    pe.add_argument("--dump-json", dest="dump_json", default=None)
    pe.add_argument("--print-hex", dest="print_hex", type=int, default=None)
    pe.add_argument("--print-base64", dest="print_base64", type=int, default=None)
    _add_common_args(pe)
    pe.set_defaults(func=cmd_encode)

    pd = sub.add_parser("decode", help="decode bitstream -> wav")
    pd.add_argument("--in", dest="input", required=True)
    pd.add_argument("--out", required=True)
    pd.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)
    pd.set_defaults(func=cmd_decode)

    pm = sub.add_parser("metrics", help="compute SNR/segSNR between wavs")
    pm.add_argument("--x", required=True)
    pm.add_argument("--y", required=True)
    pm.set_defaults(func=cmd_metrics)

    po = sub.add_parser("out9", help="v1 ACELP preset (out9.celpbin style)")
    po.add_argument("--in", dest="input", required=True)
    po.add_argument("--out-bitstream", dest="out_bitstream", default="out9.celpbin")
    po.add_argument("--out-wav", dest="out_wav", default="out9_recon.wav")
    po.add_argument("--dump-json", dest="dump_json", default=None)
    po.add_argument("--print-hex", dest="print_hex", type=int, default=None)
    po.add_argument("--print-base64", dest="print_base64", type=int, default=None)
    po.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)
    po.set_defaults(func=cmd_out9)

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return int(args.func(args))
