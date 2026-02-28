# 音色变换（Timbre Augmentation / Target Timbre）功能 SPEC（基于 CELP/ACELP 参数域）

## 0. 背景与目标

当前仓库已经有可工作的 CELP/ACELP 编解码器（`*.celpbin` bitstream + 解码重建 WAV）。本 SPEC 在此基础上新增“音色变换”能力，用于：

- **数据增广**：在不改变语音内容（大体保持可懂度）的前提下，生成多种音色/发声方式的变体。
- **目标音色模拟**：给定一个“目标音色”（可来自参考 WAV 或参考 `celpbin`），把源语音映射到更接近目标的 timbre。

允许的控制手段（可单独或组合）：

1. **改 F0 / 激励（excitation）**：通过修改自适应码本 lag（pitch）和/或增益、创新激励结构改变发声特征（清亮/沙哑/气声等）。
2. **改 LSF/LSP（谱包络）**：通过修改 LPC 频谱包络（formant/共振峰位置与整体明暗）改变 timbre。

核心工程要求：

- 以 **参数域变换** 为主：输入 `celpbin`，输出新的 `celpbin`，再解码得到变化后的 WAV。
- **无信息泄漏**（针对 `celpbin -> wav`）：当输入是 `celpbin` 时，输出 WAV 必须仅由 `celpbin` 与显式变换参数决定；同一输入 + 同一参数 + 同一 seed -> 输出 bitstream 完全一致。
- 支持 `v1` 与 `v2` bitstream（至少：`v1 ACELP(out9 风格)` + `v2 ACELP`；CELP 支持可做但不强制）。

## 1. 非目标（明确不做）

- 不追求与任何标准（AMR/G.729）兼容或 bit-exact。
- 不把它做成“语音转换 SOTA”（不做高保真声纹迁移、内容-音色解耦网络等）。
- 不做实时流式。

## 2. 术语

- `F0`：基频。对 CELP 而言主要由 pitch lag 决定，近似 `F0 ≈ fs / lag`。
- `excitation`：激励 `e[n] = g_p * e_p[n] + g_c * c[n]`。
  - `e_p`：自适应码本（过去激励延迟）
  - `c`：创新（固定码本 / 代数脉冲）
- `LSF/LSP`：
  - LSF（line spectral frequencies）/ LSP（line spectral pairs）是 LPC 的稳定表示。
  - 通过移动 LSF/LSP 可实现“formant shift / bright/dark”一类 timbre 变化。

## 3. 用户接口（CLI）

新增子命令（基础命令）：`timbre`

### 3.1 timbre：对 bitstream 做参数域变换

```bash
celpcodec timbre \
  --in in.celpbin \
  --out-bitstream out_style.celpbin \
  --out-wav out_style.wav \
  [--seed 1234] \
  [--f0-scale 1.10] \
  [--gp-scale 0.90 --gc-scale 1.15] \
  [--formant-scale 1.05] \
  [--lsf-mix 0.30 --target target.celpbin] \
  [--dump-json dbg.json]
```

参数（建议实现的最小集合）：

- 输入输出：
  - `--in`: 输入 `*.celpbin`（v1/v2）
  - `--out-bitstream`: 输出变换后的 `*.celpbin`
  - `--out-wav`: 可选，直接解码输出 WAV（便于试听）
  - `--dump-json`: 可选，输出逐帧/逐子帧变换前后的参数摘要
- 随机性：
  - `--seed`: 影响 jitter/随机增强；默认 1234；必须保证确定性
- F0/激励变换：
  - `--f0-scale`: `>0`，默认 1.0。`>1` 提高音高（lag 变小），`<1` 降低音高（lag 变大）
  - `--gp-scale`, `--gc-scale`: 对 `g_p/g_c` 的缩放（在反量化域缩放后再量化）
  - 可选增强项（后续迭代）：`--lag-jitter`, `--innov-jitter`, `--voicing {breathy,clear,...}`
- LSF/LSP 变换：
  - `--formant-scale`: 默认 1.0。对 LSF 频率做整体缩放（配合稳定性约束）
  - `--lsf-mix`: `0..1`，默认 0。把源 LSF 向目标 LSF 插值
  - `--target`: 参考目标音色，可为 `target.celpbin`（优先）或 `target.wav`（可选支持）

输出行为：

- `out_style.celpbin` 与 `in.celpbin` 版本一致（v1 输入 -> v1 输出；v2 输入 -> v2 输出）。
- 对同一输入，`--seed` + 参数固定时，输出 bitstream 必须稳定（字节级一致）。

### 3.2 augment：批量生成增广样本（可选）

```bash
celpcodec augment --in in.celpbin --out-dir aug/ --n 8 \
  --f0-scale-min 0.85 --f0-scale-max 1.15 \
  --formant-scale-min 0.95 --formant-scale-max 1.08 \
  --seed 2026
```

`augment` 是对 `timbre` 的批处理封装：为每个样本派生子 seed，生成 `out_i.celpbin` 和 `out_i.wav`。

## 4. 变换总流程（必须保证可逆性/稳定性）

输入：`in.celpbin`（v1/v2）  
输出：`out.celpbin`（同版本） + 可选 `out.wav`

1. 解析 header，确定版本、模式、`fs/frame_len/subframe_len/lpc_order/rc_bits/gain_bits/...`。
2. 逐帧读取 payload（RC indices + 子帧参数）。
3. 对每帧/子帧执行变换（可组合）：
   - `pitch/lag` 变换（F0）
   - `g_p/g_c` 变换（voicing/气声/清亮）
   - 创新结构 jitter（可选）
   - LSF/LSP 变换（谱包络）
4. 把变换后的参数重新 bit-pack 成新的 payload，拼接原 header（或按需要微调 seed 字段）。
5. 可选：立即解码并写 WAV（用于试听/验收）。

强约束：

- 参数必须在合法范围内（lag、索引 bits 上限、pos 范围、增益索引范围）。
- LSF/LSP 变换必须保证合成滤波器稳定，解码过程中不出现 NaN/Inf。

## 5. F0 / 激励变换（最低可用实现）

### 5.1 F0 缩放（只改 pitch lag）

对于每个子帧的 pitch lag：

- `lag_new = clamp(round(lag / f0_scale), lag_min, lag_max)`
  - `f0_scale > 1`：提高音高（lag 变小）
  - `f0_scale < 1`：降低音高（lag 变大）

注意事项：

- `v1`：`lag_bits=8` 固定；`lag_min/lag_max` 由 `fs` 与 `[50,400]Hz` 推导，并限制在 8-bit span。
- `v2`：`lag_min/lag_max` 来自 header；`lag_bits` 由范围计算。
- 可选做平滑：对子帧 `lag_new` 做轻量 median/DP，避免过大跳变导致的“抖音”感。

### 5.2 激励“清亮/气声”控制（改 g_p/g_c）

对每个子帧：

- 读取 `gp_idx/gc_idx` -> 反量化得到 `g_p, g_c`
- 施加缩放：
  - `g_p' = clamp(g_p * gp_scale, 0, gp_max)`
  - `g_c' = clamp(g_c * gc_scale, 0, gc_max)`
- 重新量化为索引（保持原 bits）

建议提供两个预设（可选实现）：  

- `breathy`（更气声/更噪）：`gp_scale < 1`, `gc_scale > 1`
- `clear`（更清亮/更谐波）：`gp_scale > 1`, `gc_scale < 1`

约束：

- 需要按 bitstream 版本约定使用正确的 `(gp_max, gc_max)`（例如 `v1 out9` 采用 `[1.2, 2.0]`）。
- 可选做能量守恒：在缩放后对 `(g_p', g_c')` 做整体归一化以减少响度漂移（不强制）。

### 5.3 创新 jitter（可选）

用于增强“嘶声/粗糙度/随机性”，但容易损伤可懂度，默认应关闭。

- v1 ACELP（4-track）：对每轨 `pos_idx` 做 `±1` 小扰动（概率 `p`），或小概率翻转 sign。
- v2 ACELP（稀疏带权脉冲）：对少量脉冲做位置/权重扰动（保持范围）。

必须保证确定性：所有随机选择只依赖 `--seed` + `(frame_idx, subframe_idx, track_idx, pulse_idx)`。

## 6. LSF/LSP 变换（用于 timbre / formant）

### 6.1 为什么要 LSF/LSP

只改 F0 往往像“变调”，但 timbre（音色）更大程度由 **谱包络/共振峰** 决定。LPC 的 LSF/LSP 表示天然适合做“formant shift / bright-dim / speaker-color”类变化。

### 6.2 最小实现：formant-scale（不依赖目标）

对每帧（或每子帧，取决于 bitstream LPC 更新粒度）：

1. `rc_idx -> k_hat -> a_hat`（已有代码）
2. `a_hat -> lsf`（新增：LPC->LSF/LSP 转换）
3. 变换：
   - `lsf' = warp(lsf, scale=formant_scale)`  
     推荐在 Hz 或 mel 上做缩放，然后映射回 rad：
     - `f = lsf * fs/(2π)` -> `f' = clamp(f*scale, fmin, fs/2 - fmin)` -> `lsf' = 2π f'/fs`
4. 稳定性修正（必须）：
   - 单调递增 + 最小间隔 `min_sep_hz`（例如 50Hz）
   - 边界：`lsf'[0] >= sep`, `lsf'[p-1] <= π - sep`
5. `lsf' -> a' -> k' -> rc_idx'`（新增：LSF->LPC + LPC->reflection(step-down)）
6. 写回 bitstream

如果某帧转换失败（数值问题/根求解失败），该帧应回退：保持原 `rc_idx` 不变。

### 6.3 目标音色：LSF mix（target timbre）

支持 `--target target.celpbin`（优先）：

1. 解析目标 bitstream，提取每帧的 `rc_idx`，转换为 `lsf_tgt`。
2. 汇总目标统计（最小实现取全局平均）：
   - `lsf_tgt_avg = mean(lsf_tgt over frames)`
3. 对源：
   - `lsf' = (1 - lsf_mix) * lsf_src + lsf_mix * lsf_tgt_avg`
4. 稳定性修正 + 回写同上。

可选增强（后续迭代）：

- 不只用均值：可对每个系数做方差匹配（`z-score` 对齐）或按 voiced/unvoiced 分开统计。
- 同时做 F0 匹配：用目标的 lag 统计给出 `f0_scale` 建议值（例如 median 匹配）。

支持 `--target target.wav`（可选）：

- 方案 A：先用现有 codec 把 `target.wav` 编码成临时 `celpbin`，再按上面提取 LSF（实现简单，代价是编码较慢）。
- 方案 B：直接对 `target.wav` 做 LPC->LSF 分析统计（更快，但需要额外实现 pitch/voicing 分类则更复杂）。

## 7. 工程实现建议（模块与接口）

新增模块建议：

- `celp_codec/timbre.py`
  - `transform_bitstream(data: bytes, params: TimbreParams) -> bytes`
  - `extract_style_stats(data_or_wav, ...) -> StyleStats`
  - `apply_f0_scale(...)`, `apply_gain_scale(...)`, `apply_lsf_transform(...)`
- `celp_codec/lsf.py`（或 `lsp.py`）
  - `lpc_to_lsf(a, fs) -> lsf(rad)`
  - `lsf_to_lpc(lsf) -> a`
  - `lpc_to_reflection(a) -> k`（step-down / Schur）

CLI：

- `celp_codec/cli.py` 增加子命令 `timbre`（必要）和 `augment`（可选）。

## 8. 验收与测试（必须自动化）

### 8.1 正确性

1. `timbre` 对 `v1 out9.celpbin` 与 `v2` 样例均可运行，不崩溃。
2. 输出 `out_style.celpbin` 能被 `decode` 独立解码为 WAV。
3. 变换后 bitstream 参数仍合法：
   - lag/pos/gain/rc 索引范围合法
   - 解码无 NaN/Inf

### 8.2 确定性 / 无信息泄漏

对于输入为 `celpbin` 的模式：

1. 同一输入 `in.celpbin` + 同一参数 + 同一 `--seed`：
   - `out_style.celpbin` 必须字节级一致（`shasum` 不变）
2. `timbre` 产生的 WAV 与“单独 decode 输出 bitstream”完全一致：

```bash
celpcodec timbre --in in.celpbin --out-bitstream o.celpbin --out-wav a.wav --seed 7 --f0-scale 1.1
celpcodec decode --in o.celpbin --out b.wav
cmp a.wav b.wav
```

### 8.3 质量 sanity（不做硬指标，但要有回归）

- 变换后输出不应大面积爆音/发散。
- 提供 debug.json 记录变换前后 lag、gains、LSF 的摘要（均值/分位数），便于定位异常。

## 9. 版本与兼容策略

- `timbre` 不引入新的 bitstream version：输入 v1/v2 -> 输出同版本。
- 所有变换均在现有 payload 字段范围内完成，保证“只要有 decoder 就能解码”。
