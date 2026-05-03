# TODO：统一 fft_code 与 rbf_core 中重合的 filter / SVD 模块

## 进度

- [x] Step 0 — 写入本 TODO.md（追踪文件）
- [x] Step 1 — `main.py`：滤波调用迁移到 `filter_core`
- [x] Step 2 — `main.py`：SVD/RR 调用迁移到 `filter_core`
- [x] Step 3 — `fft_code/hamiltonian.py`：`apply_filter_H` / `apply_filter_H_all` 改为 thin wrapper
- [ ] Step 4 — `fft_code/rayleigh_ritz.py`：`svd_rayleigh_ritz` 改为 thin wrapper
- [ ] Step 5 — `compare_fft_rbf_filter_qd.py`：确认并统一 filter / RR 调用

---

## 背景与目标

`filter_core.py` 已提供算符无关的 Newton 滤波和 Rayleigh-Ritz 函数，
`run_rbf_filter.py` 已正确使用它们。但 `main.py`（FFT 路径）仗直接调用
`fft_code/hamiltonian.py` 和 `fft_code/rayleigh_ritz.py` 中的 FFT 特定版本，
导致相同算法存在两份实现。

---

## 精确重合点

### 重合 1：Newton 递推滤波

| 现有函数 | 所在文件 | 状态 |
|----------|----------|------|
| `apply_filter_H(psi, V, nodes, coeffs, par, T_k_diagonal)` | `fft_code/hamiltonian.py` | ✅ 已改为 thin wrapper |
| `apply_filter_H_all(psi, V, nodes, an, par, T_k_diagonal)` | `fft_code/hamiltonian.py` | ✅ 已改为 thin wrapper |
| `apply_filter_H_op(H_apply, psi, nodes, coeffs, par)` | `filter_core.py` | ✅ 算符无关，保留 |
| `apply_filter_H_all_op(H_apply, psi, nodes, an, par)` | `filter_core.py` | ✅ 算符无关，保留 |

### 重合 2：SVD + Rayleigh-Ritz

| 现有函数 | 所在文件 | 状态 |
|----------|----------|------|
| `svd_rayleigh_ritz(filtered_psi_matrix, x_grid, V, Nx, Ny, Nz, T_k_diagonal, ...)` | `fft_code/rayleigh_ritz.py` | ⏳ Step 4 待完成 |
| `svd_rayleigh_ritz_op(basis_mat, H_apply, ...)` | `filter_core.py` | ✅ 算符无关，保留 |

---

## 待完成内容

### Step 4：`fft_code/rayleigh_ritz.py` thin wrapper

```python
def svd_rayleigh_ritz(filtered_psi_matrix, x_grid, V, Nx, Ny, Nz,
                      T_k_diagonal, svd_tol=1e-3, max_energies=200):
    """Thin wrapper → filter_core.svd_rayleigh_ritz_op."""
    from filter_core import svd_rayleigh_ritz_op
    from fft_code.hamiltonian import apply_H
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
grep -rn "apply_filter_H_all\|apply_filter_H\b" . --include="*.py" \
  | grep -v "fft_code/hamiltonian.py" | grep -v "filter_core.py"

grep -rn "svd_rayleigh_ritz\b" . --include="*.py" \
  | grep -v "fft_code/rayleigh_ritz.py" | grep -v "filter_core.py"

python main.py --set nc=50 --set n_random=1 --set tag=after_refactor
```
