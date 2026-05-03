# TODO：统一 fft_code 与 rbf_core 中重合的 filter / SVD 模块

## 进度

- [x] Step 0 — 写入本 TODO.md（追踪文件）
- [x] Step 1 — `main.py`：滤波调用迁移到 `filter_core`
- [x] Step 2 — `main.py`：SVD/RR 调用迁移到 `filter_core`
- [x] Step 3 — `fft_code/hamiltonian.py`：`apply_filter_H` / `apply_filter_H_all` 改为 thin wrapper
- [x] Step 4 — `fft_code/rayleigh_ritz.py`：`svd_rayleigh_ritz` 改为 thin wrapper
- [ ] Step 5 — `compare_fft_rbf_filter_qd.py`：确认并统一 filter / RR 调用

---

## 背景与目标

`filter_core.py` 已提供算符无关的 Newton 滤波和 Rayleigh-Ritz 函数，
`run_rbf_filter.py` 已正确使用它们。`main.py`、`fft_code/hamiltonian.py`、
`fft_code/rayleigh_ritz.py` 已完成迁移，仅剩 Step 5。

---

## Step 5 内容

grep `compare_fft_rbf_filter_qd.py`，确认并统一所有
`apply_filter_H_all` / `svd_rayleigh_ritz` 调用。

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
