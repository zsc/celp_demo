# CELP/ACELP 研究型编解码器（Python CLI）

这个仓库提供离线语音编解码：输入 `WAV`，输出压缩比特流 `*.celpbin` 和重建 `WAV`。

重点支持两条路线：
- `v2`：当前默认的效果优先实验路线（支持更高自由度参数）
- `v1`：固定结构路线（用于复现 `out9.celpbin` 这类配置）

## 1. 你要的 out9 预设（推荐入口）

直接用：

```bash
python3 -m celp_codec out9 \
  --in en_happy_prompt.wav \
  --out-bitstream out9.celpbin \
  --out-wav out9_recon.wav \
  --dump-json out9.json
```

安装脚本后也可用：

```bash
celpcodec out9 --in en_happy_prompt.wav
```

`out9` 子命令固定输出为 **bitstream v1**，并锁定你指定的头字段：

- bitstream 版本：`v1`（`magic=CLP1`, `version=1`）
- mode：`ACELP`（`mode=1`）
- 采样率：`fs=8000`
- 帧长/子帧长：`frame_len=160`（20ms），`subframe_len=40`（5ms）
- LPC：`lpc_order=10`，`rc_bits=7`
- 增益量化：`gain_bits_p=5`，`gain_bits_c=5`
- seed：`1234`

## 2. v1（out9）编码约定

注意：v1 header 不包含全部算法超参，解码可复现依赖固定约定。当前实现固定为：

- Pitch lag 编码：`8 bits`，`lag = lag_min + idx`
  - `lag_min/lag_max` 由 `fs` 和 `[50Hz, 400Hz]` 推导，并限制在 8bit 覆盖范围内
- ACELP 创新：`4 tracks`，每轨 `1` 个脉冲，脉冲幅度固定 `±1`
  - 每轨写入 `pos_idx(4 bits) + sign(1 bit)`
- CELP（v1）创新：固定随机码本大小 `512`，每子帧 `9 bits` 索引
- 增益反量化范围：
  - `g_p in [0, 1.2]`
  - `g_c in [0, 2.0]`
- 感知加权滤波：`gamma1=0.9`, `gamma2=0.6`
- v1 解码路径固定不做 LPC 跨子帧插值

## 3. out9 的最小 bit 预算

在 `fs=8000, frame_len=160, subframe_len=40, lpc_order=10, rc_bits=7, gp=5, gc=5, mode=acelp(v1)` 下：

- 每帧 LPC：`10 * 7 = 70 bits`
- 每子帧：
  - `pitch_lag_idx`: `8 bits`
  - `gp_idx`: `5 bits`
  - `gc_idx`: `5 bits`
  - `innovation`: `4 * (4+1) = 20 bits`
  - 合计：`38 bits/subframe`
- 每帧 4 子帧：`4 * 38 = 152 bits`
- 每帧 payload：`70 + 152 = 222 bits`
- payload 码率：`222 / 0.02 = 11100 bps = 11.1 kbps`
- v1 header 固定：`30 bytes = 240 bits`

## 4. 命令行用法

### 4.1 一键编码+解码

```bash
python3 -m celp_codec roundtrip \
  --in en_happy_prompt.wav \
  --mode acelp \
  --bitstream-version 2 \
  --out-bitstream out.celpbin \
  --out-wav out_recon.wav
```

### 4.2 指定 v1 编码（非 out9 预设入口）

```bash
python3 -m celp_codec encode \
  --in en_happy_prompt.wav \
  --mode acelp \
  --bitstream-version 1 \
  --fs 8000 --frame-ms 20 --subframe-ms 5 \
  --lpc-order 10 --rc-bits 7 --gain-bits-p 5 --gain-bits-c 5 \
  --seed 1234 \
  --out out9_like.celpbin
```

### 4.3 只解码（只依赖 bitstream）

```bash
python3 -m celp_codec decode --in out9.celpbin --out decoded.wav
```

### 4.4 out9 风格 + `fs=16000` 的推荐配置

当前 `out9` 子命令固定为 `fs=8000`。如果要保留 out9 的 v1 风格并切到 16k，请用 `roundtrip/encode`，并把子帧改为 `4ms`：

```bash
python3 -m celp_codec roundtrip \
  --in en_happy_prompt.wav \
  --mode acelp --bitstream-version 1 \
  --fs 16000 --frame-ms 20 --subframe-ms 4 \
  --lpc-order 10 --rc-bits 7 --gain-bits-p 5 --gain-bits-c 5 \
  --seed 1234 --dp-pitch on --dp-topk 10 --dp-lambda 0.2 \
  --out-bitstream out9_16k.celpbin --out-wav out9_16k_recon.wav
```

说明：
- 这会得到 v1 header：`fs=16000, frame_len=320, subframe_len=64, p=10, rc_bits=7, gain_bits_p=5, gain_bits_c=5, seed=1234`。
- 不建议 `subframe-ms=5`（80 点），因为 v1 ACELP 每轨位置索引固定是 `4 bits`，16k/5ms 下单轨候选数会超过 16，当前实现会报错。
- 如果必须使用 `5ms@16k`，需要扩展 v1 格式（例如把 `pos_idx` 从 4bit 提到 5bit），这将不再是当前 out9 兼容定义。

## 5. “是否有信息泄漏”快速自测

目标：确认 `celpbin -> wav` 只依赖 `celpbin` 本身。

```bash
# 1) 先生成 v1 out9 bitstream
python3 -m celp_codec out9 --in en_happy_prompt.wav --out-bitstream leak_test.celpbin --out-wav a.wav

# 2) 再单独 decode 同一 bitstream
python3 -m celp_codec decode --in leak_test.celpbin --out b.wav

# 3) 比较输出文件是否完全一致（应一致）
shasum -a 256 a.wav b.wav
cmp a.wav b.wav && echo "OK: decode only uses celpbin"
```

如果 `cmp` 返回一致，说明解码结果由 bitstream 完全决定（同版本实现下）。

## 6. 项目结构

```text
celp_codec/
  cli.py
  codec.py
  bitstream.py
  lpc.py
  filters.py
  pitch.py
  acelp.py
  celp_codebook.py
  gains.py
  metrics.py
  wav_io.py
tools/
  make_audio_gallery.py
tests/
  test_bitstream.py
  test_lpc_stability.py
  test_roundtrip.py
```

## 7. 依赖与安装

- Python `>=3.9`
- `numpy`
- `scipy`
- `soundfile`

```bash
pip install -e .
```

## 8. 回归测试

```bash
python3 -m unittest discover -v
```

## 9. bitstream 兼容说明

- `decode` 现在同时支持 `v1` 和 `v2`。
- `roundtrip/encode` 默认输出 `v2`；可通过 `--bitstream-version 1` 切换到 v1。
- `out9` 子命令固定输出 `v1 ACELP`（用于稳定复现 out9 配置）。

## 10. `timbre_demo` 实验（参数域音色变换）

`timbre_demo` 的目标是：在 **仅修改 bitstream 参数**（不触碰原始 wav）的前提下做音色相关增广，并产出可直接试听与对照的 HTML。

### 10.1 实验入口

```bash
python3 tools/make_timbre_demo.py \
  --pattern "../S*X/examples/*.wav" \
  --out-dir timbre_demo \
  --fs 16000 \
  --lsf-mix 0.9 \
  --grid-f0-alpha 0.9 \
  --grid-gain-alpha 0.8 \
  --mel-width 220
```

默认会优先选取 3 条样本（若存在）：
- `en_happy_prompt.wav`
- `fear_zh_female_prompt.wav`
- `whisper_prompt.wav`

### 10.2 实验流程

1. 先把 3 条输入编码为 `v2 ACELP`（16k）得到 `s0/s1/s2.celpbin`。
2. 从每条 bitstream 抽取风格统计（`lag` 中位数、`gp/gc` 均值）和目标 `LSF` 均值。
3. 生成 `3x3` mimic：
   - 对角线：identity（不变换）
   - 非对角：联合迁移 `f0 + gp/gc + LSF`
4. 生成增广：
   - 常规增广（包含 formant 与非 formant）
   - `Extreme Non-Formant`（仅 `f0/gp/gc`）
5. 所有输出 wav 自动生成 mel 图，写入 `timbre_demo/mel/*.png`，并合成 `timbre_demo/timbre_demo.html`。

### 10.3 HTML 页面结构

`timbre_demo/timbre_demo.html` 包含四节：
- `Reference`：`orig` 与 `recon` 对照
- `3x3 Timbre Mimic Grid`：每格显示 `src/tgt/out` 三张 mel
- `Augmentations`：常规增广（含 formant）
- `Extreme Non-Formant`：极端非 formant 预设（仅 `f0/gp/gc`）

### 10.4 关键预设说明

`Augmentations`（常规）示例：
- 非 formant：`f0_up/down`、`f0_ext_up/down`、`breathy`、`periodic`、`noisy`
- formant：`formant_up/down`、`formant_expand/compress`、`chipmunk`

`Extreme Non-Formant`（单独一节，明确不动 formant）：
- `f0_ultra_up` / `f0_ultra_down`
- `periodic_hard`（`gp↑↑, gc↓↓`）
- `noisy_hard`（`gp↓↓, gc↑↑`）
- `high_buzzy`（`f0↑ + gp↑ + gc↑`）
- `low_hollow`（`f0↓ + gp↓ + gc↑`）

### 10.5 强度调参建议

- `3x3` 更像目标音色：提高 `--lsf-mix`、`--grid-f0-alpha`、`--grid-gain-alpha`
- 更极端高音/花栗鼠倾向：提高 `f0_scale`，并配合 `formant_scale>1` / `lsf_spread>1`
- 只做非 formant 质感变化：只调 `f0_scale/gp_scale/gc_scale`，保持 `formant_scale=1, lsf_spread=1, lsf_mix=0`

## 11. 单文件 timbre 变换 CLI（`celpbin -> celpbin`）

可直接对某个 bitstream 做参数域变换：

```bash
python3 -m celp_codec timbre \
  --in out9.celpbin \
  --out-bitstream out9_timbre.celpbin \
  --out-wav out9_timbre.wav \
  --f0-scale 1.25 \
  --gp-scale 0.75 \
  --gc-scale 1.35 \
  --formant-scale 1.05 \
  --lsf-spread 1.15 \
  --dump-json out9_timbre.json
```

参数作用（子帧级）：
- `--f0-scale`：通过缩放 `lag` 改变基频感知
- `--gp-scale`：缩放自适应码本增益（周期成分）
- `--gc-scale`：缩放固定码本增益（噪声/细节成分）
- `--formant-scale`：整体 formant 上/下移
- `--lsf-spread`：formant 间距扩张/压缩
- `--lsf-mix` + `--target`：向目标音色 LSF 统计靠近
