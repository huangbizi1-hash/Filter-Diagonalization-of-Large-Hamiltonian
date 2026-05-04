# Modularisation TODO

## Task A — filter_core refactor (DONE)

All FFT / RBF filter+SVD code unified in `filter_core.py`.
`fft_code/hamiltonian.py` and `fft_code/rayleigh_ritz.py` are now thin wrappers.
`main.py` calls `filter_core` directly.

---

## Task B — `rbf_core.py` → `rbf_code/` package (DONE)

All 8 steps completed on branch `claude/add-main-comments-bFS0b`.

| Step | File | Status |
|------|------|---------|
| 1 | `rbf_code/config.py`     | ✅ done |
| 2 | `rbf_code/periodic.py`   | ✅ done |
| 3 | `rbf_code/nodes.py`      | ✅ done |
| 4 | `rbf_code/laplacian.py`  | ✅ done |
| 5 | `rbf_code/io_qd.py`      | ✅ done |
| 6 | `rbf_code/eigensolve.py` | ✅ done |
| 7 | `rbf_code/__init__.py`   | ✅ done |
| 8 | `rbf_core.py` (thin wrapper) | ✅ done |

### Package layout

```
rbf_code/
  __init__.py     — re-exports all public names
  config.py       — RBFConfig, RBFProblem, IterationRecord, constants, helpers
  periodic.py     — fractional-coord geometry (wrap, unique, distances, adaptive filter)
  nodes.py        — Poisson-disc, sphere, atom-augmented, conv-cell node generators
  laplacian.py    — Hamiltonian assembly, weight matrices, node quality metrics
  io_qd.py        — Gaussian cube I/O, build_qd_problem entry point
  eigensolve.py   — eigensolvers, parameter sweeps, iterate_hamiltonian
rbf_core.py       — thin wrapper (backward compat, imports everything from rbf_code)
```

### Backward compatibility

`rbf_core.py` re-exports every public name unchanged.
All existing callers (`run_rbf_filter.py`, `compare_fft_rbf_filter_qd.py`, etc.)
require **zero changes**.
