# SPEC: Python CELP/ACELP Toy Codec CLI（输入 WAV → 输出压缩比特流 + 重建 WAV）

> 面向 gemini-cli / codex 代码生成：该 SPEC 要求实现一个**可运行**的“教学/研究型” CELP 与 ACELP 编解码器（不对齐任何现成标准，但结构与关键算法一致），并在 CLI 中支持**搜索 / DP / 近似凸优化**等估计策略。  
> 目标是：给定一段语音 WAV，输出（1）压缩后的比特流文件（可选 JSON/hex dump）和（2）解码重建 WAV。

---

## 0. 术语约定

- `Fs`: 采样率（默认 8000 Hz）。
- `frame`: 20 ms 帧（默认 160 点 @8k）。
- `subframe`: 5 ms 子帧（默认 40 点 @8k），每帧 4 个子帧。
- `p`: LPC 阶数（默认 10 @8k）。
- **自适应码本**（adaptive / pitch / LTP）: 过去激励的延迟版本，参数是 `lag`（可选分数延迟）。
- **固定码本**（fixed / innovation）:
  - CELP 模式：随机/伪随机矢量码本（索引）。
  - ACELP 模式：代数稀疏脉冲码本（脉冲位置 + 符号）。
- **AbS**: analysis-by-synthesis（闭环合成分析）。

---

## 1. 项目目标与非目标

### 1.1 目标（MVP 必须满足）
1. 提供 Python CLI：输入一段 WAV，输出：
   - 压缩比特流文件：`*.celpbin`（无论 CELP/ACELP 模式都用同一容器，header 标记 mode）
   - 重建 WAV：`*_recon.wav`
   - 控制台打印：编码参数、帧数、bitrate、简单客观指标（SNR / Segmental SNR）与耗时
2. 支持两种模式：
   - `--mode celp`
   - `--mode acelp`
3. 估计算法显式覆盖：
   - 搜索（pitch、创新码本）
   - DP（跨子帧 pitch path 平滑）
   - 近似凸优化（ACELP：用 L1/LASSO 松弛求稀疏，再投影到代数码本约束）
4. 解码器可独立运行：`decode` 命令只依赖 bitstream 输出 WAV。
5. 结果可复现：相同输入 + 相同 seed → bitstream 完全一致（浮点误差允许在可控范围内；必要时关键步骤用 float64）。

### 1.2 非目标（明确不做）
- 不追求与 AMR/G.729 等标准 bit-exact 或音质对齐。
- 不实现复杂的后滤波（postfilter）/舒适噪声（CNG）/DTX（可留 TODO，但不影响 MVP）。
- 不做实时流式 API（仅离线文件）。

---

## 2. 输入输出约束

### 2.1 WAV 输入
- 支持：PCM 16-bit（优先）、单声道；若双声道则 downmix 为 mono。
- 若采样率不是 `Fs`（默认 8000），需重采样到 `Fs`（`scipy.signal.resample_poly` 优先；否则退化到线性插值）。
- 归一化：读入转 float32/float64，范围 [-1, 1]。

### 2.2 输出
- `*.celpbin`：自定义 bitstream（见第 6 章）。
- `*_recon.wav`：16-bit PCM，采样率 = `Fs`。
- 可选：
  - `--dump-json out.json`：逐帧/逐子帧输出索引与中间量（pitch、增益、码本参数等）。
  - `--print-hex`：控制台打印 bitstream 前 N 字节 hex。
  - `--print-base64`：控制台打印 base64（可限制长度）。

---

## 3. CLI 设计

### 3.1 命令总览
提供可执行入口（任一满足即可）：
- `python -m celp_codec ...`
- 或安装后 `celpcodec ...`

支持子命令：
1. `roundtrip`：**一条命令完成 encode + decode**（用户的主要入口）
2. `encode`
3. `decode`
4. `metrics`（可选，但建议实现，便于验收）

### 3.2 roundtrip
```bash
celpcodec roundtrip \
  --in input.wav \
  --mode celp|acelp \
  --out-bitstream out.celpbin \
  --out-wav out_recon.wav \
  [--fs 8000] [--seed 1234] \
  [--dp-pitch on|off] \
  [--acelp-solver greedy|ista] \
  [--dump-json debug.json] \
  [--print-hex 64]
````

### 3.3 encode

```bash
celpcodec encode --in input.wav --mode acelp --out out.celpbin [options...]
```

### 3.4 decode

```bash
celpcodec decode --in out.celpbin --out decoded.wav
```

### 3.5 关键可调参数（都要有默认值）

* `--fs {8000,16000}`（默认 8000；16000 允许但可先“实验性”）
* `--frame-ms 20`（固定 20，允许参数化）
* `--subframe-ms 5`（固定 5）
* `--lpc-order p`（默认 10 @8k；16k 可默认 16）
* `--preemph 0.97`（0 表示关闭）
* `--pitch-min-hz 50 --pitch-max-hz 400`（自动转 lag 范围）
* `--dp-pitch on|off`（默认 on）
* `--dp-topk 10`（每子帧保留 top-k pitch 候选）
* `--dp-lambda 0.2`（transition penalty 系数，见 5.4）
* `--celp-codebook-size 512`（CELP 模式固定码本大小）
* `--acelp-K 4`（ACELP 子帧脉冲数：默认 4）
* `--acelp-tracks 4`（默认 4 轨）
* `--acelp-solver greedy|ista`（默认 greedy）
* `--ista-iters 30 --ista-lambda 0.02`（近似凸优化用）
* `--gain-bits-p 5 --gain-bits-c 5`（自适应/创新增益量化比特）
* `--rc-bits 7`（reflection coefficients 每阶量化比特数）
* `--clip`（重建输出前 clip 到 [-1,1]）

---

## 4. 编解码总体结构（AbS 闭环）

### 4.1 模型

每个子帧激励：

* `e[n] = g_p * e_p[n] + g_c * c[n]`

  * `e_p[n]`：从过去激励缓冲区按 lag（可分数延迟）取样
  * `c[n]`：固定码本向量（CELP）或代数稀疏脉冲（ACELP）

合成：

* `s_hat[n] = synth_filter( 1 / A(z), e[n] )`

感知加权误差：

* 采用加权滤波器 `W(z)`，目标最小化：

  * `E = Σ || W(z) * (s - s_hat) ||^2`

### 4.2 关键实现要求

* 编码端与解码端必须共享：

  * 量化后的 LPC 参数（用 reflection coefficients 传输以保证稳定）
  * 相同的码本生成规则（CELP 伪随机码本由 seed + 参数派生）
  * 相同的 bitstream 解析方式
* 必须维护状态：

  * 合成滤波器状态（IIR memory）
  * 激励缓冲区（至少覆盖 `max_lag + subframe_len + 1`）
  * 可选：加权滤波器状态（若实现为 IIR/FIR）

---

## 5. 估计/优化算法细节（搜索 / DP / 近似凸优化）

### 5.1 LPC 分析与传输（稳定性优先：用 reflection coefficients）

#### 5.1.1 分析

* 每帧对预加重后的信号 `x` 做 autocorrelation：

  * `r[k] = Σ_{n=k..N-1} x[n] x[n-k]`
* 用 Levinson-Durbin 求解得到：

  * LPC 系数 `a[1..p]`
  * reflection coeffs `k[1..p]`（PARCOR）
* 约束与防炸：

  * 若预测误差能量过小或出现数值异常，退化为 `a=0`（全通）或使用前一帧 `k`。

#### 5.1.2 量化（必须保证 |k_i|<1）

* 对每个 `k_i` 做 atanh 变换：

  * `v_i = 0.5 * ln((1+k_i)/(1-k_i))`  （数值上可用 `np.arctanh(k_i)`）
* 裁剪：`v_i ∈ [-Vmax, Vmax]`，建议 `Vmax=2.5`
* 均匀量化到 `rc_bits`：

  * `idx = round( (v_i + Vmax) / (2*Vmax) * (2^B - 1) )`
* 解码端反量化：

  * `v_i_hat = idx/(2^B-1) * 2*Vmax - Vmax`
  * `k_i_hat = tanh(v_i_hat)`
* 用 step-up recursion 从 `k_hat` 恢复 LPC `a_hat`（保证稳定）。

> 验收点：任何帧解码出的 `A(z)` 必须稳定（IIR 不发散），且数值 NaN/Inf 不允许出现。

### 5.2 感知加权滤波器 W(z)

* 采用带宽扩展（bandwidth expansion）：

  * `A_γ(z) = 1 + Σ a_i * (γ^i) z^{-i}`
* 设 `γ1=0.9, γ2=0.6`：

  * `W(z) = A_{γ1}(z) / A_{γ2}(z)`
* 组合滤波器（用于快速评估）：

  * `F(z) = W(z) / A(z)`
  * 即把激励映射到“加权域的合成输出”。

实现建议：

* 每个子帧计算 `h = impulse_response(F, L=subframe_len)`：

  * 输入单位冲激 `δ[0]=1`，其它为 0，用 `lfilter` 得到前 L 点。
* 后续将 `H*c` 视为 `conv(h, c)` 截取长度 L（Toeplitz 乘法的卷积实现）。

### 5.3 自适应码本（pitch）搜索（逐子帧搜索）

#### 5.3.1 候选 lag 范围

* 由 `pitch-min-hz / pitch-max-hz` 转换：

  * `lag_min = floor(Fs / pitch_max_hz)`
  * `lag_max = ceil(Fs / pitch_min_hz)`
* 默认 @8k：`lag ∈ [20, 160]`

#### 5.3.2 代价函数（用于排序/DP）

对每个候选 lag：

1. 从 excitation buffer 取 `e_p`（长度 L）
2. 计算 `y_p = H * e_p`（即 `conv(h, e_p)` 截取 L）
3. 对目标向量 `d`（见 5.5）：

   * 最优 `g_p* = (d·y_p) / (y_p·y_p + eps)`
   * 残差能量（越小越好）：

     * `J_p(lag) = || d - g_p* y_p ||^2`
   * 等价可用 “相关最大化”：

     * `score = (d·y_p)^2 / (y_p·y_p + eps)`（越大越好）

### 5.4 跨子帧 pitch 的 DP（Viterbi）平滑（必须实现）

目的：避免每子帧独立搜索导致 pitch 抖动，尤其在弱 voiced/噪声段。

对一帧 4 个子帧，做：

1. 对每个子帧独立计算所有 lag 的 `score`，取 top-k（`--dp-topk`）形成候选集合 `C_t`
2. 进行 DP，路径代价：

   * `Cost(t, lag) = -score(t, lag) + min_{lag'} [ Cost(t-1, lag') + λ * (lag - lag')^2 ]`
   * `λ = --dp-lambda`
3. 回溯得到最优 lag 序列 `[lag_0..lag_3]`

实现要点：

* `top-k` 候选是为了把 DP 状态数控制在 `4 * K` 的量级。
* 允许加入 hard constraint：`|lag_t - lag_{t-1}| <= Δ`（可选）。

验收点：

* 开启 `--dp-pitch on` 时，输出 debug.json 中相邻子帧 lag 跳变显著减少（统计上可见）。

### 5.5 目标向量 d 的构造（加权域）

实现上允许“简化但一致”的 AbS 版本：

* 先将原始子帧语音 `s` 通过加权滤波器得到 `s_w = W(z)*s`
* 将已知（已确定的）部分激励通过 `F(z)` 得到加权域合成 `y_known`
* 目标：

  * `d = s_w - y_known`

**最小要求**：`y_known` 至少要包含 IIR 状态延续的影响（合成滤波器 memory），否则每子帧边界会出现明显伪影。

建议实现策略：

* 维护合成滤波器状态 `mem_A`（IIR）
* 在编码端的候选评估中，通过卷积 `H * e_candidate` 近似加权合成输出（对短子帧足够）
* 真实输出更新时，再用 IIR `lfilter` 跑一遍得到 `s_hat` 并更新状态（保证编解码一致）。

---

## 5.6 固定码本：CELP 模式（随机码本 + 搜索）

### 5.6.1 码本生成（必须可复现）

* 使用 `numpy.random.default_rng(seed)` 生成 `M x L` 的高斯码本：

  * `C[m] ~ N(0,1)`，并对每个码字归一化 `||C[m]||=1`
* `seed` 来自 CLI `--seed`，并写入 header，确保解码可重建相同码本（或直接把 seed 作为编码参数存储）。

### 5.6.2 搜索（相关最大化）

在 pitch 已确定（含 `g_p`）后，定义残差目标：

* `r = d - g_p * y_p`

对每个码字：

1. `y_c = H * C[m]`
2. 最优创新增益：

   * `g_c* = (r·y_c)/(y_c·y_c + eps)`
3. 代价：

   * `J(m) = || r - g_c* y_c ||^2`
     取 `J(m)` 最小的 m。

优化建议（非必须但建议）：

* 预计算 `y_c` 或 `y_c` 的能量（对每帧 `h` 不同，预计算有限；可以每子帧按需计算）。
* 使用向量化加速（numpy batch dot）。

---

## 5.7 固定码本：ACELP 模式（代数稀疏脉冲）

### 5.7.1 代数码本结构（必须实现）

* 子帧长度 `L=40`
* `T = --acelp-tracks`（默认 4 轨）
* 轨定义：

  * track `t` 的允许位置集合：`P_t = { t + T*k | k=0..(L/T - 1) }`
  * 默认 T=4 → 每轨 10 个位置
* 脉冲数 `K = --acelp-K`（默认 4）
* 约束（默认实现）：

  * 每轨恰好 1 个脉冲（因此 K=T=4）
  * 脉冲幅度固定为 ±1（符号位传输）
* 码字构造：

  * `c[n] = Σ_{t=0..T-1} s_t * δ[n - p_t]`
  * 其中 `p_t ∈ P_t`，`s_t ∈ {+1,-1}`

> 说明：此结构易于 bit 打包（每轨位置 4 bits + 符号 1 bit），同时保留 ACELP 的“稀疏 + 轨道”特性。

### 5.7.2 Greedy 搜索（必须实现：默认 solver）

在残差目标 `r` 上，用匹配追踪式策略找每轨最佳脉冲：

* 预计算每个候选位置的“原子响应”：

  * `atom[pos] = shift(h, pos)` 截取长度 L
  * 因为 `H*δ[n-pos]` 就是 `h` 的移位。
* 对每轨 t：

  * 计算相关：

    * `corr[pos] = r · atom[pos]`
  * 选择 `pos* = argmax_{pos∈P_t} |corr[pos]|`
  * 符号：`s = sign(corr[pos*])`
  * 临时构造该轨脉冲并更新 `r ← r - α * s * atom[pos*]`

    * α 可取 1（仅用于选择），或用局部最小二乘估计（建议）。

完成 K 个脉冲后：

* 构造 `c`（脉冲向量）
* 计算 `y_c = H*c`
* 最优 `g_c* = (r0 · y_c)/(y_c·y_c+eps)`（其中 r0 为未扣创新前的残差目标）
* 得到最终创新增益并进入量化。

### 5.7.3 近似凸优化（必须实现：`--acelp-solver ista`）

目的：把“选脉冲位置”的组合优化近似为 L1 稀疏回归，再投影回代数结构。

#### (1) 连续稀疏求解（LASSO）

设卷积算子 `H` 由 `h` 定义，求：

* `min_c  0.5 || r - Hc ||^2 + λ ||c||_1`

实现 ISTA/FISTA：

* 梯度：`∇ = H^T(Hc - r)`

  * `Hc` 用卷积实现
  * `H^T x` 用与 `h` 反转后的相关实现（或卷积）
* 步长 `τ`：

  * 用 FFT 估计 Lipschitz 常数：`L = max |FFT(h)|^2`（零填充到 >=2L）
  * `τ = 1/(L+eps)`
* 软阈值：

  * `soft(x, θ)=sign(x)*max(|x|-θ,0)`
* 迭代 `N=--ista-iters`

输出连续解 `c_cont`（长度 L）。

#### (2) 投影到代数码本约束（必须）

把 `c_cont` 投影到 “每轨一个脉冲 ±1”：

* 对每轨 t：

  * 在 `P_t` 中选 `p_t = argmax |c_cont[p]|`
  * `s_t = sign(c_cont[p_t])`（若为 0 则取 +1）
* 构造离散 `c_alg`：在 `p_t` 处置为 `s_t`，其余为 0。

之后同 greedy：

* `y_c = H*c_alg`
* `g_c*` 由最小二乘得到

> 验收点：`--acelp-solver ista` 可跑通并生成可解码 bitstream；在多数语音样本上，SNR 不应明显差于 greedy（允许小幅波动）。

---

## 5.8 增益估计与量化（必须）

对确定的 `y_p` 与 `y_c`，用最小二乘估计增益（允许顺序估计）：

建议顺序法：

1. `g_p = clamp( (d·y_p)/(y_p·y_p+eps), 0, g_p_max )`
2. `r = d - g_p*y_p`
3. `g_c = clamp( (r·y_c)/(y_c·y_c+eps), 0, g_c_max )`

量化（对数域更稳健）：

* `q(x) = round( (log(x+eps)-log(xmin)) / (log(xmax)-log(xmin)) * (2^B-1) )`
* 解码反量化：

  * `x_hat = exp( q/(2^B-1)*(log(xmax)-log(xmin)) + log(xmin) ) - eps`

默认范围建议：

* `g_p ∈ [0.0, 1.2]`
* `g_c ∈ [0.0, 2.0]`

---

## 6. Bitstream 格式（必须严格实现）

### 6.1 总体：字节序与 bit 打包

* 采用 **小端字节序**存储多字节整数（uint16/uint32）。
* **bit packing 规则**：建议 “LSB-first within byte”（实现简单），但必须在 SPEC 中固定一种并在读写一致。
* 必须实现 `BitWriter` / `BitReader`：

  * `write_bits(value, nbits)`
  * `read_bits(nbits)`

### 6.2 Header（固定字段）

| 字段           |      类型 | 说明                         |
| ------------ | ------: | -------------------------- |
| magic        | 4 bytes | ASCII: `b"CLP1"`           |
| version      |   uint8 | `1`                        |
| mode         |   uint8 | `0=celp, 1=acelp`          |
| fs           |  uint32 | 采样率                        |
| frame_len    |  uint16 | 默认 160                     |
| subframe_len |  uint16 | 默认 40                      |
| lpc_order    |   uint8 | 默认 10                      |
| rc_bits      |   uint8 | 默认 7                       |
| gain_bits_p  |   uint8 | 默认 5                       |
| gain_bits_c  |   uint8 | 默认 5                       |
| seed         |  uint32 | CELP 码本 seed（ACELP 也写入以统一） |
| reserved     | 8 bytes | 全 0，便于扩展                   |

### 6.3 每帧 payload（bit-level）

按帧顺序写入，直到文件结束。

#### (A) LPC reflection coeff indices

* `p` 个索引，每个 `rc_bits` bits

#### (B) 每个子帧字段（共 4 个）

最小字段：

* `pitch_lag_idx`：8 bits（表示 lag = lag_min + idx；若范围更大需自动增 bits，但 MVP 固定 8 bits 并限制范围）
* `gp_idx`：gain_bits_p
* `gc_idx`：gain_bits_c
* innovation：

  * CELP：`cb_idx` = log2(M) bits（M 默认 512 → 9 bits）
  * ACELP：每轨：

    * `pos_idx`：4 bits（0..9 对应该轨的第几个位置）
    * `sign`：1 bit（0=+1,1=-1）

> 文件末尾无需 padding 以外的结束标记；以 EOF 结束。解码时用 header 已知参数与 frame_len 计算应生成样本数；不足则丢弃残余。

---

## 7. 解码流程（必须与编码一致）

对每帧：

1. 读取并反量化 `k_hat` → 恢复 `a_hat`
2. 每子帧：

   * 从 bitstream 读取 `pitch_lag`
   * 从激励缓冲取 `e_p`
   * 从 innovation 字段恢复 `c`
   * 反量化 `g_p, g_c`
   * 得到 `e = g_p e_p + g_c c`
   * 通过合成滤波器 `1/A(z)` 得到 `s_hat`（维护 IIR 状态）
   * 写入输出 PCM buffer
   * 更新 excitation buffer（把 e append）

---

## 8. 工程实现与代码结构（建议强制）

### 8.1 目录结构（建议照做）

```
celp_codec/
  __init__.py
  cli.py               # argparse/typer，子命令 roundtrip/encode/decode/metrics
  wav_io.py            # 读写wav，重采样，mono处理
  bitstream.py         # BitWriter/BitReader + header + frame IO
  lpc.py               # autocorr, levinson_durbin, step_up, rc quant/dequant
  filters.py           # preemph, bandwidth expansion, impulse response, conv helpers
  pitch.py             # pitch candidates, scoring, DP/Viterbi
  celp_codebook.py     # 伪随机码本生成 + 搜索
  acelp.py             # algebraic structure + greedy + ISTA + projection
  gains.py             # gain estimation + quantization
  codec.py             # Encoder/Decoder 主流程
  metrics.py           # SNR, segSNR, runtime stats
tests/
  test_roundtrip.py
  test_bitstream.py
  test_lpc_stability.py
pyproject.toml         # 依赖与 entrypoint
README.md              # 运行示例
```

### 8.2 依赖要求

* 必选：`numpy`
* 建议：`scipy`（resample_poly, lfilter）
* WAV IO：优先 `soundfile`；若不可用则用标准库 `wave`（仅 PCM16）。

---

## 9. 指标与验收标准（必须可自动化测试）

### 9.1 功能验收（必须）

1. `roundtrip` 生成两个文件：bitstream + recon wav
2. `decode` 能单独从 bitstream 生成 wav，长度与 `roundtrip` 的 recon 一致（允许末尾 padding 差 < 1 frame）
3. 任何输入都不应崩溃；非法 bitstream 要给出清晰错误（magic/version 校验）

### 9.2 数值/稳定性验收（必须）

* IIR 合成滤波器不发散；输出不出现 NaN/Inf
* LPC 反射系数量化后仍满足稳定性（|k|<1）
* 输出 PCM 不溢出：写 WAV 前 clip 到 [-1, 1]

### 9.3 质量基线（建议，用于回归）

对一段“以语音为主”的样本：

* CELP / ACELP 至少达到：

  * 全局 SNR > 8 dB（仅作为 smoke test，避免退化到纯噪声）
* DP 打开相对关闭：

  * pitch lag 的平均跳变幅度应下降（在 debug.json 可统计）

---

## 10. Debug/可解释输出（强烈建议）

`--dump-json` 输出结构示例：

```json
{
  "fs": 8000,
  "mode": "acelp",
  "frames": [
    {
      "frame_index": 0,
      "rc_idx": [ ... ],
      "subframes": [
        {
          "lag": 58,
          "gp_idx": 12,
          "gc_idx": 7,
          "acelp": {
            "tracks": [
              {"track":0, "pos": 12, "sign": +1},
              ...
            ]
          }
        }
      ]
    }
  ]
}
```

---

## 11. 关键伪代码（编码端）

```python
for each frame:
  x = get_frame_samples()

  # 1) LPC analysis -> reflection coeffs k -> quantize idx_k
  k = levinson_reflection(x_windowed, order=p)
  idx_k = quantize_rc(k)

  # 2) reconstruct A(z) from dequantized k_hat (encoder must mirror decoder)
  k_hat = dequantize_rc(idx_k)
  a_hat = step_up(k_hat)

  # 3) build weighting/synthesis impulse response h for F(z)=W(z)/A(z)
  h = impulse_response_weighted(a_hat)

  for each subframe:
    d = build_weighted_target(subframe_speech, states, a_hat, h)

    # 4) pitch candidates + DP across 4 subframes (if enabled)
    lag = choose_pitch_via_dp_or_greedy(d, excitation_buffer, h)

    ep = get_adaptive_vector(excitation_buffer, lag)
    yp = conv(h, ep)

    gp = ls_gain(d, yp)
    gp_idx = quantize_gain(gp)

    r = d - dequant(gp_idx)*yp

    # 5) innovation search
    if mode == CELP:
        cb_idx, yc = celp_search(r, h, codebook)
        gc = ls_gain(r, yc)
        innov_bits = cb_idx
    else:
        if solver == greedy:
            pulses = acelp_greedy(r, h, tracks)
        else:
            c_cont = ista_lasso(r, h, lam, iters)
            pulses = project_to_tracks(c_cont, tracks)
        c = build_pulse_vector(pulses)
        yc = conv(h, c)
        gc = ls_gain(r, yc)
        innov_bits = pulses_to_bits(pulses)

    gc_idx = quantize_gain(gc)

    # 6) synthesize in time domain and update states/buffers
    e = deq(gp_idx)*ep + deq(gc_idx)*c
    s_hat = lfilter_synthesis(a_hat, e, mem)
    write_output(s_hat)
    update_excitation_buffer(e)
```

---

## 12. 交付清单（最终仓库必须包含）

* 可执行 CLI（`roundtrip/encode/decode` 至少三者）
* `*.celpbin` bitstream 的读写实现与文档
* 单元测试 + 一个端到端测试
* README：包含至少两条运行示例与参数说明

---

## 13. README 示例（必须能跑）

```bash
# roundtrip: 生成 out.celpbin + out_recon.wav
celpcodec roundtrip --in demo.wav --mode acelp \
  --out-bitstream out.celpbin --out-wav out_recon.wav \
  --dp-pitch on --acelp-solver ista --dump-json out.json

# 只解码
celpcodec decode --in out.celpbin --out decoded.wav
```

---

```

如果你希望这个 SPEC 更贴近某个现成标准（比如 **G.729 的 CS-ACELP** 或 **AMR-NB ACELP** 的轨道/脉冲配置、比特分配、分数延迟等），我也可以在不要求 bit-exact 的前提下，把 **轨道数、脉冲数、bit allocation、lag 分数插值** 改成更“标准味”的版本（同时仍保留 DP 与 ISTA 两条估计路线）。

