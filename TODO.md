# TODO：统一 fft_code 与 rbf_core 中重合的 filter / SVD 模块

## 进度

- [x] Step 0 — 写入本 TODO.md（追踪文件）
- [ ] Step 1 — `main.py`：滤波调用迁移到 `filter_core`
- [ ] Step 2 — `main.py`：SVD/RR 调用迁移到 `filter_core`
- [ ] Step 3 — `fft_code/hamiltonian.py`：`apply_filter_H` / `apply_filter_H_all` 改为 thin wrapper
- [ ] Step 4 — `fft_code/rayleigh_ritz.py`：`svd_rayleigh_ritz` 改为 thin wrapper
- [ ] Step 5 — `compare_fft_rbf_filter_qd.py`：确认并统一 filter / RR 调用

---

## 背景与目标

`filter_core.py` 已提供算符无关的 Newton 滤波和 Rayleigh-Ritz 函数，
`run_rbf_filter.py` 已正确使用它们。但 `main.py`（FFT 路径）仍直接调用
`fft_code/hamiltonian.py` 和 `fft_code/rayleigh_ritz.py` 中的 FFT 特定版本，
导致相同算法存在两份实现。

---

## 精确重合点

### 重合 1：Newton 递推滤波

| 现有函数 | 所在文件 | 状态 |
|----------|----------|------|
| `apply_filter_H(psi, V, nodes, coeffs, par, T_k_diagonal)` | `fft_code/hamiltonian.py` | ❌ FFT 特定，与 filter_core 重合 |
| `apply_filter_H_all(psi, V, nodes, an, par, T_k_diagonal)` | `fft_code/hamiltonian.py` | ❌ FFT 特定，与 filter_core 重合 |
| `apply_filter_H_op(H_apply, psi, nodes, coeffs, par)` | `filter_core.py` | ✅ 算符无关，保留 |
| `apply_filter_H_all_op(H_apply, psi, nodes, an, par)` | `filter_core.py` | ✅ 算符无关，保留 |

### 重合 2：SVD + Rayleigh-Ritz

| 现有函数 | 所在文件 | 状态 |
|----------|----------|------|
| `svd_rayleigh_ritz(filtered_psi_matrix, x_grid, V, Nx, Ny, Nz, T_k_diagonal, ...)` | `fft_code/rayleigh_ritz.py` | ❌ FFT 特定 |
| `svd_rayleigh_ritz_op(basis_mat, H_apply, ...)` | `filter_core.py` | ✅ 算符无关，保留 |

---

## 步骤详情

### Step 1：`main.py` 滤波调用迁移

在 `run()` 函数内，构建 `T_k_diagonal` 之后定义闭包：

```python
H_apply_fft = lambda psi: apply_H(psi, V, T_k_diagonal)
```

替换滤波调用：
```python
# 非 split_bandpass 分支
# 旧：apply_filter_H_all(psi_rand, V, samp, an, par, T_k_diagonal)
psi_filt_all = apply_filter_H_all_op(H_apply_fft, psi_rand, samp, an, par)

# split_bandpass 分支
# 旧：apply_filter_H(psi_hi_all[ie], V, samp, an_lo[ie], par, T_k_diagonal)
apply_filter_H_op(H_apply_fft, psi_hi_all[ie], samp, an_lo[ie], par)
```

修改 import（删除旧 filter 函数，保留 apply_H）：
```python
# 删除：apply_filter_H, apply_filter_H_all
# 新增（from filter_core）：apply_filter_H_op, apply_filter_H_all_op
```

### Step 2：`main.py` SVD/RR 调用迁移

`filtered_psi_matrix` shape = `(ms*n_random, Nx, Ny, Nz)`；
`svd_rayleigh_ritz_op` 要求 `basis_mat` shape = `(n_grid, n_basis)`：

```python
n_grid    = Nx * Ny * Nz
basis_mat = filtered_psi_matrix.reshape(ist.ms * n_random, n_grid).T  # (n_grid, n_basis)

H_apply_flat = lambda psi_flat: apply_H(
    psi_flat.reshape(Nx, Ny, Nz), V, T_k_diagonal).ravel()

energies, Ur, rank = svd_rayleigh_ritz_op(
    basis_mat, H_apply_flat,
    svd_tol=cfg.get("svd_tol", 1e-3),
    max_energies=cfg.get("max_energies", 200),
    hermitian=True,
)
```

### Step 3：`fft_code/hamiltonian.py` thin wrapper

```python
def apply_filter_H_all(psi, V, nodes, an, par, T_k_diagonal):
    """Thin wrapper → filter_core.apply_filter_H_all_op."""
    from filter_core import apply_filter_H_all_op
    return apply_filter_H_all_op(lambda p: apply_H(p, V, T_k_diagonal),
                                  psi, nodes, an, par)

def apply_filter_H(psi, V, nodes, coeffs, par, T_k_diagonal):
    """Thin wrapper → filter_core.apply_filter_H_op."""
    from filter_core import apply_filter_H_op
    return apply_filter_H_op(lambda p: apply_H(p, V, T_k_diagonal),
                              psi, nodes, coeffs, par)
```

### Step 4：`fft_code/rayleigh_ritz.py` thin wrapper

```python
def svd_rayleigh_ritz(filtered_psi_matrix, x_grid, V, Nx, Ny, Nz,
                      T_k_diagonal, svd_tol=1e-3, max_energies=200):
    """Thin wrapper → filter_core.svd_rayleigh_ritz_op."""
    from filter_core import svd_rayleigh_ritz_op
    n_grid    = Nx * Ny * Nz
    basis_mat = filtered_psi_matrix.reshape(-1, n_grid).T
    H_apply   = lambda psi_flat: apply_H(
        psi_flat.reshape(Nx, Ny, Nz), V, T_k_diagonal).ravel()
    return svd_rayleigh_ritz_op(basis_mat, H_apply, svd_tol, max_energies,
                                 hermitian=True)
```

### Step 5：`compare_fft_rbf_filter_qd.py`

grep 并统一所有 `apply_filter_H_all` / `svd_rayleigh_ritz` 调用。

---

## 不变约束

- `fft_code/filter_coeff.py` 接口不改
- `filter_core.py` 公开接口不改
- `run_rbf_filter.py` 不改（已正确）
- `fft_code/hamiltonian.py::apply_H` 不改

## 验证命令

```bash
# 确认无文件直接调用旧 filter 函数（除 wrapper 内部）
grep -rn "apply_filter_H_all\|apply_filter_H\b" . --include="*.py" \
  | grep -v "fft_code/hamiltonian.py" | grep -v "filter_core.py"

# 确认 svd_rayleigh_ritz 调用已迁移
grep -rn "svd_rayleigh_ritz\b" . --include="*.py" \
  | grep -v "fft_code/rayleigh_ritz.py" | grep -v "filter_core.py"

# 回归测试
python main.py --set nc=50 --set n_random=1 --set tag=after_refactor
```
