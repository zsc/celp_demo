from __future__ import annotations

import argparse
import html
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from celp_codec import codec, gains, metrics, pitch, timbre, wav_io  # noqa: E402
from celp_codec.bitstream import BitReader, BitstreamHeaderV1, BitstreamHeaderV2, read_header  # noqa: E402


@dataclass(frozen=True)
class Item:
    key: str
    wav_path: Path
    name: str


@dataclass(frozen=True)
class StyleStats:
    lag_median: float
    gp_mean: float
    gc_mean: float


@dataclass(frozen=True)
class AugSpec:
    tag: str
    title: str
    note: str
    params: timbre.TimbreParams


def _pick_three(pattern: str) -> list[Path]:
    pat = os.path.expanduser(str(pattern))
    if not os.path.isabs(pat):
        pat = str((REPO_ROOT / pat).resolve())
    paths = [Path(p) for p in sorted(glob(pat)) if p.lower().endswith(".wav")]
    if not paths:
        raise FileNotFoundError(f"No wavs matched pattern: {pattern!r}")

    prefer = ["en_happy_prompt.wav", "fear_zh_female_prompt.wav", "whisper_prompt.wav"]
    by_name = {p.name: p for p in paths}
    preferred = [by_name[n] for n in prefer if n in by_name]
    if len(preferred) >= 3:
        return preferred[:3]

    uniq: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        uniq.append(path)
        if len(uniq) >= 3:
            break
    if len(uniq) < 3:
        raise FileNotFoundError(f"Need at least 3 wavs, got {len(uniq)} from {pattern!r}")
    return uniq


def _encode_source(wav_path: Path, out_dir: Path, key: str, fs: int) -> bytes:
    x, fs_in = wav_io.read_wav(wav_path)
    x = wav_io.resample_to_fs(x, int(fs_in), int(fs))

    cfg = codec.CodecConfig(
        mode="acelp",
        fs=int(fs),
        frame_ms=20,
        subframe_ms=5,
        dp_pitch=True,
        dp_topk=10,
        dp_lambda=0.05,
        lpc_interp=True,
        rc_bits=10,
        gain_bits_p=7,
        gain_bits_c=7,
        seed=1234,
        acelp_solver="ista",
        ista_iters=25,
        ista_lambda=0.02,
        acelp_K=10,
        acelp_weight_bits=5,
    )
    bitstream_bytes, recon, _, _ = codec.encode_samples(x, cfg)

    wav_io.write_wav(out_dir / f"{key}_orig.wav", x, int(fs), clip=True)
    wav_io.write_wav(out_dir / f"{key}_recon.wav", recon, int(fs), clip=True)
    (out_dir / f"{key}.celpbin").write_bytes(bitstream_bytes)
    return bitstream_bytes


def _decode_to_wav(out_bytes: bytes, out_wav: Path) -> None:
    y, header, _ = codec.decode_bitstream(out_bytes, clip=True)
    wav_io.write_wav(out_wav, y, int(getattr(header, "fs")), clip=True)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    if len(tag) != 4:
        raise ValueError("PNG chunk tag must be 4 bytes")
    crc = zlib.crc32(tag)
    crc = zlib.crc32(data, crc)
    crc &= 0xFFFFFFFF
    return len(data).to_bytes(4, "big") + tag + data + crc.to_bytes(4, "big")


def _png_encode_rgb8(rgb: np.ndarray) -> bytes:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    comp = zlib.compress(raw, level=6)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")


def _mel_spectrogram_db(
    x: np.ndarray,
    fs: int,
    n_mels: int = 64,
    win_ms: float = 25.0,
    hop_ms: float = 10.0,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    fs = int(fs)
    win_length = int(round(float(fs) * (float(win_ms) / 1000.0)))
    hop_length = int(round(float(fs) * (float(hop_ms) / 1000.0)))
    win_length = int(max(16, win_length))
    hop_length = int(max(1, hop_length))
    n_fft = 1
    while n_fft < win_length:
        n_fft <<= 1
    n_fft = int(max(256, n_fft))

    power = metrics._stft_power(x, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    fb = metrics.mel_filterbank(fs=fs, n_fft=n_fft, n_mels=int(n_mels), fmin=float(fmin), fmax=fmax)
    mel = power @ fb.T
    mel_db = 10.0 * np.log10(mel + 1e-12)
    return mel_db


def _resample_time(m: np.ndarray, width: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    width = int(width)
    n_frames, n_mels = int(m.shape[0]), int(m.shape[1])
    if n_frames == width:
        return m
    if n_frames == 1:
        return np.repeat(m, width, axis=0)
    t0 = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    t1 = np.linspace(0.0, 1.0, width, dtype=np.float64)
    out = np.empty((width, n_mels), dtype=np.float64)
    for i in range(n_mels):
        out[:, i] = np.interp(t1, t0, m[:, i])
    return out


def _write_mel_png(wav_path: Path, png_path: Path, width: int = 220, n_mels: int = 64) -> bool:
    try:
        x, fs = wav_io.read_wav(wav_path)
        mel_db = _mel_spectrogram_db(x, fs=fs, n_mels=int(n_mels))
        mel_db = _resample_time(mel_db, width=int(width))
        img = mel_db.T[::-1, :]
        lo = float(np.percentile(img, 5.0))
        hi = float(np.percentile(img, 95.0))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo + 1e-9:
            lo = float(np.min(img))
            hi = float(np.max(img))
        if hi <= lo + 1e-9:
            hi = lo + 1.0
        norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        u8 = np.round(norm * 255.0).astype(np.uint8)
        rgb = np.repeat(u8[:, :, None], 3, axis=2)
        png_path.write_bytes(_png_encode_rgb8(rgb))
        return True
    except Exception:
        return False


def _extract_style_stats(data: bytes) -> StyleStats:
    h_any, hs = read_header(data)
    br = BitReader(data[hs:])

    frame_len = int(getattr(h_any, "frame_len"))
    subframe_len = int(getattr(h_any, "subframe_len"))
    subframes = frame_len // subframe_len
    rc_bits = int(getattr(h_any, "rc_bits"))
    lpc_order = int(getattr(h_any, "lpc_order"))
    gain_bits_p = int(getattr(h_any, "gain_bits_p"))
    gain_bits_c = int(getattr(h_any, "gain_bits_c"))
    mode_id = int(getattr(h_any, "mode"))
    lag_min = 0
    lag_bits = 0
    pitch_frac_bits = 0
    gp_max = 1.6
    gc_max = 6.0
    pos_bits = pitch.bits_for_pos(subframe_len)

    if isinstance(h_any, BitstreamHeaderV2):
        lag_min = int(h_any.lag_min)
        lag_bits = pitch.bits_for_lag(int(h_any.lag_min), int(h_any.lag_max), frac_bits=0)
        pitch_frac_bits = int(h_any.pitch_frac_bits)
    else:
        lag_min, _ = pitch.lag_bounds(int(h_any.fs), 50.0, 400.0, max_lag_bits=8)
        lag_bits = 8
        gp_max = 1.2
        gc_max = 2.0

    lags: list[float] = []
    gps: list[float] = []
    gcs: list[float] = []

    eof = False
    while True:
        try:
            for _ in range(lpc_order):
                _ = br.read_bits(rc_bits)
        except EOFError:
            break
        for _ in range(subframes):
            try:
                lag_i = int(br.read_bits(lag_bits))
                frac = int(br.read_bits(pitch_frac_bits)) if pitch_frac_bits else 0
                gp_idx = int(br.read_bits(gain_bits_p))
                gc_idx = int(br.read_bits(gain_bits_c))
            except EOFError:
                eof = True
                break

            lag = float(lag_min + lag_i)
            if pitch_frac_bits:
                lag += float(frac) / float(1 << pitch_frac_bits)
            lags.append(lag)
            gps.append(float(gains.dequantize_gain(gp_idx, gain_bits_p, xmin=1e-4, xmax=gp_max)))
            gcs.append(float(gains.dequantize_gain(gc_idx, gain_bits_c, xmin=1e-4, xmax=gc_max)))

            if isinstance(h_any, BitstreamHeaderV2):
                if mode_id == 0:
                    cb_bits = int(int(h_any.celp_codebook_size).bit_length() - 1)
                    for _s in range(int(h_any.celp_stages)):
                        _ = br.read_bits(cb_bits)
                else:
                    for _k in range(int(h_any.acelp_K)):
                        _ = br.read_bits(pos_bits)
                        _ = br.read_bits(int(h_any.acelp_weight_bits))
            else:
                if mode_id == 0:
                    _ = br.read_bits(9)
                else:
                    for _t in range(4):
                        _ = br.read_bits(4)
                        _ = br.read_bits(1)
        if eof:
            break

    if not lags:
        return StyleStats(lag_median=80.0, gp_mean=1.0, gc_mean=1.0)

    return StyleStats(
        lag_median=float(np.median(np.asarray(lags, dtype=np.float64))),
        gp_mean=float(np.mean(np.asarray(gps, dtype=np.float64))),
        gc_mean=float(np.mean(np.asarray(gcs, dtype=np.float64))),
    )


def _style_scale(src: float, tgt: float, alpha: float, lo: float, hi: float) -> float:
    src = float(max(src, 1e-6))
    tgt = float(max(tgt, 1e-6))
    a = float(min(max(alpha, 0.0), 1.0))
    s = float(np.exp(a * np.log(tgt / src)))
    return float(min(max(s, lo), hi))


def _cell_mel_images(
    mel_png: dict[str, str],
    triples: list[tuple[str, str]],
) -> str:
    blocks: list[str] = []
    for wav_name, label in triples:
        ref = mel_png.get(wav_name)
        if ref is None:
            continue
        blocks.append(
            "<div class='melbox'>"
            f"<img class='melspec' src='{html.escape(ref, quote=True)}' alt='mel {html.escape(label, quote=True)}'/>"
            f"<div class='small'>{html.escape(label, quote=True)}</div>"
            "</div>"
        )
    if not blocks:
        return ""
    return "<div class='melrow'>" + "".join(blocks) + "</div>"


def _make_html(
    out_html: Path,
    items: list[Item],
    grid_wavs: dict[tuple[str, str], str],
    aug_wavs: dict[tuple[str, str], str],
    aug_specs: list[AugSpec],
    xnf_wavs: dict[tuple[str, str], str],
    xnf_specs: list[AugSpec],
    mel_png: dict[str, str],
    grid_params: dict[tuple[str, str], timbre.TimbreParams],
) -> None:
    title = "CELP Timbre Demo (3x3)"
    style = """
    :root { --bg:#0b0f14; --panel:#121a22; --ink:#e7eef7; --muted:#9fb1c5; --border:#223043; --accent:#4dd4ff; }
    html,body{background:var(--bg);color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:0;}
    .wrap{max-width:1200px;margin:28px auto;padding:0 18px;}
    h1{font-size:22px;margin:0 0 10px 0;}
    h2{font-size:18px;margin:0 0 10px 0;}
    p{margin:8px 0;color:var(--muted);line-height:1.35;}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px;margin:14px 0;overflow-x:auto;}
    table{width:100%;border-collapse:separate;border-spacing:8px;}
    th,td{background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:12px;padding:10px;vertical-align:top;}
    th{color:var(--muted);font-weight:600;}
    .small{font-size:12px;color:var(--muted);}
    .label{font-size:13px;margin:0 0 6px 0;color:var(--accent);}
    audio{width:100%;}
    .grid th:first-child{width:18%;}
    .k{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px;color:var(--muted);}
    .melrow{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;}
    .melbox{display:flex;flex-direction:column;gap:3px;align-items:center;}
    img.melspec{width:98px;height:62px;object-fit:cover;border:1px solid var(--border);border-radius:6px;background:#000;}
    """

    def esc(text: str) -> str:
        return html.escape(text, quote=True)

    lines: list[str] = [
        "<!doctype html>",
        "<html><head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{esc(title)}</title>",
        f"<style>{style}</style>",
        "</head><body>",
        '<div class="wrap">',
        f"<h1>{esc(title)}</h1>",
        "<p>Pipeline: wav -> encode(v2 ACELP) -> timbre transform (lag/gains/LSF) -> decode wav.</p>",
        "<p>3x3 使用更强 mimic：交叉项会联合迁移 f0、gain 和 LSF，便于听感对比。</p>",
    ]

    lines.extend(
        [
            '<div class="card">',
            "<h2>Reference</h2>",
            "<table>",
            "<tr><th>key</th><th>orig</th><th>recon (no timbre)</th></tr>",
        ]
    )
    for item in items:
        lines.append(
            f"<tr><td class='k'>{esc(item.key)}</td>"
            f"<td><audio controls src='{esc(item.key + '_orig.wav')}'></audio>"
            f"{_cell_mel_images(mel_png, [(item.key + '_orig.wav', 'orig')])}</td>"
            f"<td><audio controls src='{esc(item.key + '_recon.wav')}'></audio>"
            f"{_cell_mel_images(mel_png, [(item.key + '_recon.wav', 'recon')])}</td></tr>"
        )
    lines.extend(["</table>", "</div>"])

    lines.extend(
        [
            '<div class="card">',
            "<h2>3x3 Timbre Mimic Grid</h2>",
            '<table class="grid">',
            "<tr><th>source \\ target</th>",
        ]
    )
    for target in items:
        lines.append(f"<th>{esc(target.key)}<div class='small'>{esc(target.name)}</div></th>")
    lines.append("</tr>")
    for source in items:
        lines.append(f"<tr><th>{esc(source.key)}<div class='small'>{esc(source.name)}</div></th>")
        for target in items:
            wav_name = grid_wavs[(source.key, target.key)]
            p = grid_params[(source.key, target.key)]
            lines.append(
                f"<td><div class='label'>{esc(source.key)} -> {esc(target.key)}</div>"
                f"<div class='small'>lsf_mix={float(p.lsf_mix):.2f}, f0={float(p.f0_scale):.2f}, gp={float(p.gp_scale):.2f}, gc={float(p.gc_scale):.2f}</div>"
                f"<audio controls src='{esc(wav_name)}'></audio>"
                f"{_cell_mel_images(mel_png, [(source.key + '_recon.wav', 'src'), (target.key + '_recon.wav', 'tgt'), (wav_name, 'out')])}</td>"
            )
        lines.append("</tr>")
    lines.extend(["</table>", "</div>"])

    lines.extend(
        [
            '<div class="card">',
            "<h2>Augmentations</h2>",
            "<p class='small'>含 formant 花样（整体上/下、扩/压间距）以及非 formant 花样（f0 与 gp/gc 质感增广）。</p>",
            "<table>",
        ]
    )
    head = ["<tr><th>key</th>"]
    for spec in aug_specs:
        head.append(f"<th>{esc(spec.title)}<div class='small'>{esc(spec.note)}</div></th>")
    head.append("</tr>")
    lines.append("".join(head))

    for item in items:
        row = [f"<tr><td class='k'>{esc(item.key)}</td>"]
        for spec in aug_specs:
            aug_name = aug_wavs[(item.key, spec.tag)]
            row.append(
                f"<td><audio controls src='{esc(aug_name)}'></audio>"
                f"{_cell_mel_images(mel_png, [(item.key + '_recon.wav', 'base'), (aug_name, spec.tag)])}</td>"
            )
        row.append("</tr>")
        lines.append("".join(row))
    lines.extend(["</table>", "</div>"])

    lines.extend(
        [
            '<div class="card">',
            "<h2>Extreme Non-Formant</h2>",
            "<p class='small'>只动 f0/gp/gc，不动 formant_scale / lsf_spread / lsf_mix。</p>",
            "<table>",
        ]
    )
    head2 = ["<tr><th>key</th>"]
    for spec in xnf_specs:
        head2.append(f"<th>{esc(spec.title)}<div class='small'>{esc(spec.note)}</div></th>")
    head2.append("</tr>")
    lines.append("".join(head2))

    for item in items:
        row2 = [f"<tr><td class='k'>{esc(item.key)}</td>"]
        for spec in xnf_specs:
            wav_name = xnf_wavs[(item.key, spec.tag)]
            row2.append(
                f"<td><audio controls src='{esc(wav_name)}'></audio>"
                f"{_cell_mel_images(mel_png, [(item.key + '_recon.wav', 'base'), (wav_name, spec.tag)])}</td>"
            )
        row2.append("</tr>")
        lines.append("".join(row2))

    lines.extend(["</table>", "</div>", "</div></body></html>"])
    out_html.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="../S*X/examples/*.wav")
    parser.add_argument("--out-dir", type=str, default="timbre_demo")
    parser.add_argument("--fs", type=int, default=16000, choices=[8000, 16000])
    parser.add_argument("--lsf-mix", type=float, default=0.85)
    parser.add_argument("--grid-f0-alpha", type=float, default=0.80)
    parser.add_argument("--grid-gain-alpha", type=float, default=0.70)
    parser.add_argument("--mel-width", type=int, default=220)
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _pick_three(args.pattern)
    items = [Item(key=f"s{i}", wav_path=path, name=path.name) for i, path in enumerate(selected)]
    print("selected wavs:")
    for item in items:
        print(f"  {item.key}: {item.wav_path}")

    bitstreams: dict[str, bytes] = {}
    for item in items:
        bitstreams[item.key] = _encode_source(item.wav_path, out_dir, item.key, int(args.fs))

    # Build per-target LSF statistics once, reuse for 3x3 transforms.
    lsf_means: dict[str, np.ndarray] = {}
    style_stats: dict[str, StyleStats] = {}
    for item in items:
        header_any, _ = read_header(bitstreams[item.key])
        lsf_means[item.key] = timbre._target_lsf_mean_from_celpbin(  # type: ignore[attr-defined]
            bitstreams[item.key],
            fs_out=int(args.fs),
            lpc_order=int(getattr(header_any, "lpc_order")),
            min_sep_hz=50.0,
            edge_sep_hz=50.0,
        )
        style_stats[item.key] = _extract_style_stats(bitstreams[item.key])

    grid_wavs: dict[tuple[str, str], str] = {}
    grid_params: dict[tuple[str, str], timbre.TimbreParams] = {}
    for source in items:
        for target in items:
            if source.key == target.key:
                params = timbre.TimbreParams()
            else:
                st_s = style_stats[source.key]
                st_t = style_stats[target.key]
                params = timbre.TimbreParams(
                    f0_scale=_style_scale(
                        src=st_t.lag_median,
                        tgt=st_s.lag_median,
                        alpha=float(args.grid_f0_alpha),
                        lo=0.60,
                        hi=1.80,
                    ),
                    gp_scale=_style_scale(
                        src=st_s.gp_mean,
                        tgt=st_t.gp_mean,
                        alpha=float(args.grid_gain_alpha),
                        lo=0.60,
                        hi=1.80,
                    ),
                    gc_scale=_style_scale(
                        src=st_s.gc_mean,
                        tgt=st_t.gc_mean,
                        alpha=float(args.grid_gain_alpha),
                        lo=0.55,
                        hi=2.20,
                    ),
                    lsf_mix=float(args.lsf_mix),
                    target_lsf_mean=lsf_means[target.key],
                )
            grid_params[(source.key, target.key)] = params
            transformed, _ = timbre.transform_bitstream(
                bitstreams[source.key],
                params,
            )
            out_wav = out_dir / f"grid_{source.key}_to_{target.key}.wav"
            _decode_to_wav(transformed, out_wav)
            grid_wavs[(source.key, target.key)] = out_wav.name

    augment_specs = [
        AugSpec("f0_up", "f0_up", "升基频", timbre.TimbreParams(f0_scale=1.22)),
        AugSpec("f0_down", "f0_down", "降基频", timbre.TimbreParams(f0_scale=0.82)),
        AugSpec("f0_ext_up", "f0_ext_up", "更高基频", timbre.TimbreParams(f0_scale=1.38)),
        AugSpec("f0_ext_down", "f0_ext_down", "更低基频", timbre.TimbreParams(f0_scale=0.72)),
        AugSpec("breathy", "breathy", "gp↓, gc↑", timbre.TimbreParams(gp_scale=0.72, gc_scale=1.35)),
        AugSpec("periodic", "periodic", "gp↑, gc↓", timbre.TimbreParams(gp_scale=1.30, gc_scale=0.72)),
        AugSpec("noisy", "noisy", "gp↓↓, gc↑↑", timbre.TimbreParams(gp_scale=0.58, gc_scale=1.55)),
        AugSpec("formant_up", "formant_up", "整体升 formant", timbre.TimbreParams(formant_scale=1.12)),
        AugSpec("formant_down", "formant_down", "整体降 formant", timbre.TimbreParams(formant_scale=0.90)),
        AugSpec("formant_expand", "formant_expand", "LSF 扩间距", timbre.TimbreParams(lsf_spread=1.30)),
        AugSpec("formant_compress", "formant_compress", "LSF 压间距", timbre.TimbreParams(lsf_spread=0.78)),
        AugSpec(
            "chipmunk",
            "chipmunk",
            "f0↑ + formant↑ + 扩间距",
            timbre.TimbreParams(f0_scale=1.28, formant_scale=1.12, lsf_spread=1.20, gp_scale=0.95, gc_scale=1.12),
        ),
    ]
    aug_wavs: dict[tuple[str, str], str] = {}
    for item in items:
        for spec in augment_specs:
            transformed, _ = timbre.transform_bitstream(bitstreams[item.key], spec.params)
            out_wav = out_dir / f"aug_{item.key}_{spec.tag}.wav"
            _decode_to_wav(transformed, out_wav)
            aug_wavs[(item.key, spec.tag)] = out_wav.name

    extreme_non_formant_specs = [
        AugSpec("f0_ultra_up", "f0_ultra_up", "强升基频", timbre.TimbreParams(f0_scale=1.58)),
        AugSpec("f0_ultra_down", "f0_ultra_down", "强降基频", timbre.TimbreParams(f0_scale=0.62)),
        AugSpec("periodic_hard", "periodic_hard", "gp↑↑, gc↓↓", timbre.TimbreParams(gp_scale=1.62, gc_scale=0.52)),
        AugSpec("noisy_hard", "noisy_hard", "gp↓↓, gc↑↑", timbre.TimbreParams(gp_scale=0.42, gc_scale=1.85)),
        AugSpec(
            "high_buzzy",
            "high_buzzy",
            "f0↑ + gp↑ + gc↑",
            timbre.TimbreParams(f0_scale=1.42, gp_scale=1.28, gc_scale=1.38),
        ),
        AugSpec(
            "low_hollow",
            "low_hollow",
            "f0↓ + gp↓ + gc↑",
            timbre.TimbreParams(f0_scale=0.68, gp_scale=0.70, gc_scale=1.55),
        ),
    ]
    xnf_wavs: dict[tuple[str, str], str] = {}
    for item in items:
        for spec in extreme_non_formant_specs:
            transformed, _ = timbre.transform_bitstream(bitstreams[item.key], spec.params)
            out_wav = out_dir / f"xnf_{item.key}_{spec.tag}.wav"
            _decode_to_wav(transformed, out_wav)
            xnf_wavs[(item.key, spec.tag)] = out_wav.name

    mel_dir = out_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    mel_png: dict[str, str] = {}
    for wav_path in sorted(out_dir.glob("*.wav")):
        png_name = wav_path.stem + ".png"
        png_path = mel_dir / png_name
        if _write_mel_png(wav_path, png_path, width=int(args.mel_width), n_mels=64):
            mel_png[wav_path.name] = f"mel/{png_name}"

    out_html = out_dir / "timbre_demo.html"
    _make_html(
        out_html,
        items,
        grid_wavs,
        aug_wavs,
        augment_specs,
        xnf_wavs,
        extreme_non_formant_specs,
        mel_png,
        grid_params,
    )
    print(f"wrote html: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
