# CELP/ACELP 研究型编解码器（Python CLI，效果优先 / 非标准）

离线语音编解码器：输入 WAV → 输出压缩比特流 `*.celpbin` + 解码重建 WAV。  
实现目标是 **结构对齐 CELP/ACELP 的关键思想（LPC + Pitch + AbS 闭环）**，但 **不受任何现成标准兼容性束缚**，优先追求效果（允许编码端更慢、更“优化”）。

## 当前技术状态（实现清单）

已实现（v2 bitstream）：

- CLI 子命令：`roundtrip / encode / decode / metrics`
- 两种模式：`--mode celp`、`--mode acelp`
- AbS（analysis-by-synthesis）闭环：
  - 感知加权 `W(z)=A_γ1/A_γ2`（默认 `γ1=0.94, γ2=0.6`）
  - 加权合成映射 `F(z)=W(z)/A(z)`，子帧内用冲激响应卷积矩阵 `H` 做快速搜索/优化
  - 编码端维护合成滤波器状态、加权滤波器状态、激励缓冲区（编解码一致）
- LPC（稳定性优先）：
  - autocorrelation + Levinson-Durbin
  - reflection coeffs `k_i` 的 `atanh/tanh` 量化与反量化（保证 `|k_i|<1`）
  - **跨帧 LPC 插值（默认开启）**：用上一帧与当前帧的 reflection coeffs 做子帧级线性插值，再 step-up 得到子帧 `A(z)`（显著降低帧边界爆音/峰值放大）
- Pitch（自适应码本）：
  - 子帧逐 lag 搜索（相关最大化）
  - 可选 DP/Viterbi 跨子帧平滑（默认 `on`），支持 `--dp-topk/--dp-lambda`
- 固定码本（创新）：
  - CELP：seed 可复现高斯随机码本（默认 2048）+ 多阶段贪心叠加（`--celp-stages`）
  - ACELP（v2）：**稀疏“带权脉冲”激励**（每子帧 `K` 个脉冲，编码位置 + 量化权重）
    - `--acelp-solver omp`：OMP（贪心稀疏最小二乘，默认，质量更好但更慢）
    - `--acelp-solver ista`：LASSO(ℓ1) ISTA + top-K 支撑集 + LS 精修（更接近“近似凸优化”路线）
- 增益：
  - `g_p/g_c` 用 2×2 联合最小二乘估计，再 log-domain 量化（`idx=0` 保留为 0 增益）
- Bitstream：
  - `CLP1` + **version=2** header（小端）
  - bit packing：**LSB-first within byte**
  - header 里带 `flags`（目前用到：`LPC_INTERP` 开关；`POSTFILTER` 预留）
- Debug：
  - `--dump-json` 输出逐帧/逐子帧索引与参数（lag、增益索引、创新脉冲参数等）
  - `--print-hex` / `--print-base64`
- 单元测试：bitstream、LPC 稳定性、端到端 roundtrip（`unittest`）

当前限制/说明（仍在迭代）：

- **暂未实现分数延迟 pitch**（`--pitch-frac-bits` 目前仅保留字段；bitstream 中写 0，解码端读出但不生效）
- `--postfilter` 的 bitstream flag 已预留，但 **解码端尚未实现 postfilter**
- 暂无 VAD/CNG/DTX；静音段可能仍有编码噪声（欢迎继续做效果向改进）
- 这是研究型实现：bitstream 语义可能继续演进（建议以当前仓库版本配套解码）

## 依赖与环境

- Python >= 3.9
- 依赖：`numpy`（必选），`scipy`（高质量重采样），`soundfile`（更通用音频读写）

本仓库当前测评环境（2026-02-27）：

- Python 3.9.6
- numpy 2.0.2
- scipy 1.13.1
- soundfile 0.13.1

## 用法

```bash
# roundtrip: 生成 out.celpbin + out_recon.wav（推荐：ACELP + OMP）
python3 -m celp_codec roundtrip --in en_happy_prompt.wav --mode acelp \
  --out-bitstream out.celpbin --out-wav out_recon.wav \
  --dp-pitch on --acelp-solver omp --dump-json out.json --print-hex 64

# 只解码
python3 -m celp_codec decode --in out.celpbin --out decoded.wav

# 只编码（不解码）
python3 -m celp_codec encode --in en_happy_prompt.wav --mode celp --out out_celp.celpbin

# 计算指标（两个 wav）
python3 -m celp_codec metrics --x en_happy_prompt.wav --y out_recon.wav
```

如果你想安装成命令行 `celpcodec`：

```bash
pip install -e .
celpcodec roundtrip --in en_happy_prompt.wav --mode acelp --out-bitstream out.celpbin --out-wav out.wav
```

## 关键参数（常用）

- `--fs {8000,16000}`（默认 16000）
- `--mode {celp,acelp}`
- `--dp-pitch {on,off}` / `--dp-topk` / `--dp-lambda`
- `--acelp-solver {omp,ista}`（默认 `omp`）
- `--acelp-K`：每子帧脉冲数（默认 `subframe_len/4`，16kHz 时 `L=80 -> K=20`；码率随之上升）
- `--acelp-weight-bits`：脉冲权重 bit 数（默认 5）
- `--ista-iters` / `--ista-lambda`：ISTA 收敛强度（更慢但可能更好）
- `--seed`：CELP 随机码本 seed（写入 header，保证可复现）
- `--dump-json out.json`：逐帧 debug 信息

## Bitstream 格式（v2）

Header：见 `celp_codec/bitstream.py`（固定 34 bytes，小端）。核心字段：

- `magic=b"CLP1"`, `version=2`, `mode={0:celp,1:acelp}`
- `fs/frame_len/subframe_len`
- `lpc_order/rc_bits/gain_bits_p/gain_bits_c`
- `lag_min/lag_max/pitch_frac_bits`
- `acelp_K/acelp_weight_bits`
- `celp_codebook_size/celp_stages`
- `flags`：
  - bit0：`POSTFILTER`（预留）
  - bit1：`LPC_INTERP`（当前默认开启）

Payload（按帧顺序直到 EOF）：

- 每帧：
  - `p` 个 reflection coeff index（每个 `rc_bits`）
  - `N_subframes = frame_len/subframe_len` 个子帧，每子帧：
    - `pitch_lag_idx`（`lag_bits`，由 `lag_min..lag_max` 自适应决定）
    - `pitch_frac`（`pitch_frac_bits`，当前保留）
    - `gp_idx` / `gc_idx`
    - innovation：
      - CELP：每阶段 `cb_idx`（`log2(celp_codebook_size)` bits）
      - ACELP：`K` 次重复：`pos`（`ceil(log2(L))` bits）+ `w_idx`（`acelp_weight_bits`）

Bit packing：**LSB-first within byte**（读写严格一致）。

## 测评结果（本仓库样例，v2 / 16kHz / ACELP+OMP）

统一配置（除输入音频不同）：

- `fs=16000, frame=20ms, subframe=5ms`
- `rc_bits=10, gain_bits_p=7, gain_bits_c=7`
- `K=20 (默认), acelp_weight_bits=5`
- `dp_pitch=on, acelp_solver=omp`

| 音频 | 时长 (s) | 码率 (kbps) | SNR (dB) | segSNR (dB) | 编码耗时 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `en_happy_prompt.wav` | 7.44 | 60.64 | 10.51 | 16.76 | 18.54 |
| `fear_zh_female_prompt.wav` | 5.04 | 60.65 | 11.94 | 16.34 | 12.56 |
| `whisper_prompt.wav` | 7.33 | 60.64 | 16.43 | 18.49 | 18.30 |
| `speed_prompt.wav` | 3.96 | 60.67 | 11.21 | 16.62 | 9.79 |
| `vad_prompt.wav` | 8.58 | 60.63 | 14.49 | 13.59 | 21.20 |

> 注：SNR/segSNR 仅作为回归 smoke 指标；主目标仍是主观听感与伪影控制。

## 对比播放页（HTML）

把目录里的 `*.celpbin` + `*_recon.wav` 汇总成一个可播放的对比页，并按 bitstream header（即“解码所需字段”）分组：

```bash
python3 tools/make_audio_gallery.py --root . --out recon_gallery.html
python3 -m http.server
```

- 默认会在目录内用 `*_prompt.wav` 自动匹配参考音频并计算 `SNR/segSNR`，同时用综合评分选出“综合最优”：
  - `score = segSNR_p50 - λ * payload_kbps`（默认 `λ=0.15`，可用 `--lambda-kbps` 调整）
- 如需固定参考文件（避免多条 prompt 混在一起）：`--ref en_happy_prompt.wav`
- 也可切换选择规则：`--best-by {balanced,quality,efficiency}`

## 测试

```bash
python3 -m unittest discover -v
```
