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
