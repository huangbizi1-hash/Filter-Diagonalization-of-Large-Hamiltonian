# TODO：统一 fft_code 与 rbf_core 中重合的 filter / SVD 模块

## 进度

- [x] Step 0 — 写入本 TODO.md（追踪文件）
- [x] Step 1 — `main.py`：滤波调用迁移到 `filter_core`
- [x] Step 2 — `main.py`：SVD/RR 调用迁移到 `filter_core`
- [x] Step 3 — `fft_code/hamiltonian.py`：`apply_filter_H` / `apply_filter_H_all` 改为 thin wrapper
- [x] Step 4 — `fft_code/rayleigh_ritz.py`：`svd_rayleigh_ritz` 改为 thin wrapper
- [x] Step 5 — `compare_fft_rbf_filter_qd.py`：已正确使用 `filter_core`，无需修改

✅ **全部任务完成**

---

## 最终状态总结

### 改动文件

| 文件 | 改动内容 |
|------|------|
| `main.py` | 移除 `apply_filter_H_all`/`apply_filter_H`/`svd_rayleigh_ritz` 直接调用；改用 `filter_core` 的三个通用函数 |
| `fft_code/hamiltonian.py` | `apply_filter_H` 和 `apply_filter_H_all` 改为 thin wrapper 委托到 `filter_core` |
| `fft_code/rayleigh_ritz.py` | `svd_rayleigh_ritz` 改为 thin wrapper 委托到 `filter_core` |

### 未改动文件

| 文件 | 原因 |
|------|------|
| `filter_core.py` | 目标模块，不改接口 |
| `run_rbf_filter.py` | 已正确使用 `filter_core` |
| `compare_fft_rbf_filter_qd.py` | 已从 `filter_core` 导入 `svd_rayleigh_ritz_op`；`_apply_filter_with_blowup_guard` 是带裂变诊断的**特化扩展**，不是重复代码 |
| `fft_code/filter_coeff.py` | 公开接口不改 |

---

## 验证命令

```bash
# 确认无文件直接调用旧 filter 函数（wrapper 内部除外）
grep -rn "apply_filter_H_all\|apply_filter_H\b" . --include="*.py" \
  | grep -v "fft_code/hamiltonian.py" | grep -v "filter_core.py"

# 确认 svd_rayleigh_ritz 调用已迁移
grep -rn "svd_rayleigh_ritz\b" . --include="*.py" \
  | grep -v "fft_code/rayleigh_ritz.py" | grep -v "filter_core.py"

# 回归测试
python main.py --set nc=50 --set n_random=1 --set tag=after_refactor
```
