# Prompt: compare_symbolic_fft_ho3d.py

## 目标

新建脚本 `compare_symbolic_fft_ho3d.py`，在标准 3D 谐振子上对比两种
Chebyshev explosion filter 实现：

| 方法 | f(H) 来源 | Ritz 用的 H |
|------|-----------|------------|
| **symbolic** | Julia 执行符号展开 Σ c_n H^n sin(kr+b) | H_FFT |
| **fft** | Python 3 项递推直接用 H_FFT | H_FFT |

两种方法共用同一组随机平面波 {k, b}，最终 Ritz 值应趋于一致。
用于验证符号路径的正确性，并比较两者的速度。

---

## 符号约定（无 ω, m 参数）

H = −½ ∇² + ½(x² + y² + z²)

ψ_0 = sin(k·r + b)，k = (kx, ky, kz)

过滤器：f(H) = T_m(aH + b_sc)，a = 2/(E_hi − E_lo)，b_sc = −(E_hi + E_lo)/(E_hi − E_lo)

---

## CLI 参数

```
--N_sweep    "10:14"  或 "10,12,14"   # N 的列表
--box_L      5.0                       # 名义半箱长
--cheb_m     5
--E_lo       -3.0
--E_hi       20.0
--n_random   400                       # 随机平面波数量
--n_print    20                        # 打印/保存的 Ritz 值数量
--seed       42
--k_max      0.0                       # 0=自动取 π/d (Nyquist)
--svd_tol    1e-4
--cache_dir  ho3d_symbolic_cache       # pkl 和 .jl 的缓存目录
--julia_exe  julia
--out_json   symbolic_fft_ho3d.json
```

---

## ⚠️ 关键细节：linspace 网格约定

**正确做法**（与 compare_fd_fft_explosion.py、compare_symbolic_explosion_ho3d.py 完全一致）：

```python
def make_grid(N: int, box_L: float):
    d   = 2.0 * box_L / N          # 网格间距 = 2*box_L / N
    L   = (N - 1) * d / 2          # 实际半长 = (N-1)/N * box_L
    x1d = np.linspace(-L, L, N)    # N 个点，spacing 恰好等于 d
    return d, x1d
```

**错误做法**（过去曾犯的错误，精度差异极大，务必避免）：

```python
# ❌ 不要用这个！
x1d = np.linspace(-box_L, box_L, N)
# 问题：这给出 N 个点，间距 = 2*box_L/(N-1) ≠ 2*box_L/N
# FFT k 向量 = 2π * fftfreq(N, d)，d 错了动能就全错
```

原因：FFT 隐含周期为 N*d。linspace(-box_L, box_L, N) 实际间距是
2*box_L/(N-1)，比正确值偏大一个 N/(N-1) 因子，导致动能系统偏低。

验证：print(x1d[1] - x1d[0]) 应等于 d = 2*box_L/N，而非 2*box_L/(N-1)。

---

## 流程

### Step 1 — H^n pkl 缓存（复用 ensure_H_powers_cache）

与 compare_symbolic_explosion_ho3d.py 完全一致，复用其 `ensure_H_powers_cache`
函数（omega=1 硬编码，V = ½(x²+y²+z²)）。

缓存目录：`cache_dir / H_power_{n}.pkl`，n = 0..cheb_m。

### Step 2 — Chebyshev 系数

```python
from symbolic_code.chebyshev_filter import chebyshev_coeffs_transformed
a    =  2.0 / (E_hi - E_lo)
b_sc = -(E_hi + E_lo) / (E_hi - E_lo)
coeffs = chebyshev_coeffs_transformed(cheb_m, a=a, b=b_sc)
# 返回 [c_0, c_1, ..., c_m]，使得 T_m(aH+b_sc) = Σ c_n H^n
```

打印出 coeffs 供用户检查。

### Step 3 — Julia .jl 脚本缓存（复用 ensure_julia_script）

与 compare_symbolic_explosion_ho3d.py 完全一致，复用其 `ensure_julia_script`。
缓存名：`julia_filter_m{m}_a{a_key}_b{b_key}.jl`（编码 E_lo/E_hi）。

### Step 4 — 逐 N 扫描

对每个 N：

#### 4a. 构建网格与 H_FFT

```python
d, x1d = make_grid(N, box_L)                # 见上方正确做法
X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
V3  = 0.5 * (X**2 + Y**2 + Z**2)
k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
T_k = (k1d[:,None,None]**2 + k1d[None,:,None]**2 + k1d[None,None,:]**2) / 2.0

def apply_H_fft(v):
    p = v.reshape(N, N, N)
    return (np.fft.ifftn(T_k * np.fft.fftn(p)).real + V3 * p).ravel()
```

#### 4b. 参考特征值（eigsh on H_FFT）

```python
E_ref, _ = eigsh(H_linop, k=min(10, N**3-2), which='SA')
```

#### 4c. 随机平面波

```python
rng    = np.random.default_rng(seed)
k_max  = k_max_arg if k_max_arg > 0 else np.pi / d
k_vals = rng.uniform(-k_max, k_max, (n_random, 3))
b_vals = rng.uniform(0.0, 2*np.pi, n_random)
```

#### 4d. **Symbolic 路径**（Julia 评估）

复用 `julia_eval_filter(jl_path, x1d, k_vals, b_vals, julia_exe)` → `C_f_sym` shape `(n_random, N, N, N)`。

#### 4e. **FFT 路径**（Python 3 项递推）

对每个随机平面波 (k, b) 直接在格点上构造 sin(k·r+b) 并应用 Chebyshev 递推：

```python
def chebyshev_recurrence(H_apply, psi0, a, b_sc, m):
    """Apply T_m(a*H + b_sc) to psi0 via 3-term recurrence."""
    Hs = lambda phi: a * H_apply(phi) + b_sc * phi
    if m == 0:
        return psi0.copy()
    y_prev = psi0.copy()
    y_curr = Hs(psi0)
    for _ in range(2, m + 1):
        y_next = 2.0 * Hs(y_curr) - y_prev
        y_prev, y_curr = y_curr, y_next
    return y_curr
```

```python
C_f_fft = np.zeros((n_random, N**3), dtype=float)
for i, (k, b) in enumerate(zip(k_vals, b_vals)):
    phase = k[0]*X + k[1]*Y + k[2]*Z + b
    psi0  = np.sin(phase).ravel()
    C_f_fft[i] = chebyshev_recurrence(apply_H_fft, psi0, a, b_sc, cheb_m)
```

#### 4f. Ritz（两种方法各自）

```python
# Symbolic
basis_sym = C_f_sym.reshape(n_random, N**3).T        # (N³, n_random)
E_sym, _, rank_sym = svd_rayleigh_ritz_op(basis_sym, apply_H_fft, svd_tol, n_print, True)

# FFT
basis_fft = C_f_fft.T                                # (N³, n_random)
E_fft, _, rank_fft = svd_rayleigh_ritz_op(basis_fft, apply_H_fft, svd_tol, n_print, True)
```

#### 4g. 误差对比打印

对比 E_sym vs E_fft vs E_ref vs 精确解（用 ho3d_exact_levels）。

---

## 输出

### 终端
每个 N 打印：
- d, k_max, N³
- E_ref (eigsh)
- Ritz evals（symbolic, fft）各 n_print 个
- max|E_sym − E_fft|（两方法内部一致性误差）
- max|E_sym − exact| 和 max|E_fft − exact|
- 各步耗时

### JSON（out_json）
```json
{
  "params": { "N_list", "box_L", "cheb_m", "E_lo", "E_hi", "a", "b_sc", "coeffs", ... },
  "sweep": [
    {
      "N": 10,
      "d": ...,
      "E_ref": [...],
      "t_ref_s": ...,
      "symbolic": { "ritz_evals": [...], "rank": ..., "t_julia_s": ..., "t_ritz_s": ... },
      "fft":      { "ritz_evals": [...], "rank": ..., "t_filter_s": ..., "t_ritz_s": ... },
      "max_err_sym_vs_fft": ...,
      "max_err_sym_vs_exact": ...,
      "max_err_fft_vs_exact": ...
    },
    ...
  ]
}
```

---

## 复用的函数（直接 import，不重复实现）

| 函数 | 来源 |
|------|------|
| `ensure_H_powers_cache` | `compare_symbolic_explosion_ho3d` |
| `ensure_julia_script` | `compare_symbolic_explosion_ho3d` |
| `julia_eval_filter` | `compare_symbolic_explosion_ho3d` |
| `ho3d_exact_levels` | `compare_symbolic_explosion_ho3d` |
| `make_grid`, `make_T_k`, `apply_H_fft` | `compare_symbolic_explosion_ho3d` |
| `chebyshev_coeffs_transformed` | `symbolic_code.chebyshev_filter` |
| `svd_rayleigh_ritz_op` | `filter_core` |
| `parse_N_sweep` | `compare_symbolic_explosion_ho3d` |

> **注意**：`ensure_H_powers_cache` 中硬编码 V = ½(x²+y²+z²)（omega=1，无 ω 参数），符合需求。

---

## 不需要实现的东西

- 不需要生成额外图表（JSON 足够）
- 不需要新的符号计算函数
- 不需要修改 symbolic_code/ 下任何文件

---

## 待验证

1. **linspace 正确性**：运行后 print `x1d[1]-x1d[0]` 应等于 `2*box_L/N`。
2. **一致性**：E_sym 和 E_fft 的 Ritz 值应高度接近（max|ΔE| < 1e-3 量级）。
3. **精度趋势**：随 N 增大，误差应单调减小。
4. **缓存命中**：第二次运行应跳过 H^n 计算和 .jl 生成，直接加载缓存。
