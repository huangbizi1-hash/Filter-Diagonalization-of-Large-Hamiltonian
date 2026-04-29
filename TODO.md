# RBF-FD JDQMR solver path — implementation progress

## Goal
Add `solver` parameter to `run_rbf_filter.py` so that after RBF-FD builds
the sparse H matrix, the user can choose:
- `solver="filter"` (default): existing Newton-filter + SVD/RR path
- `solver="jdqmr"`: call PRIMME JDQMR directly on `problem._H_sparse` — no FFT, no filter

## Steps
- [x] Step 1/3 — Add JDQMR params to CONFIG; update docstring examples
- [ ] Step 2/3 — Restructure `run()`: branch on `solver_type`, add JDQMR path
- [ ] Step 3/3 — Update plot + JSON save for both paths; delete TODO.md

## New CONFIG keys (step 1)
```
"solver"              : "filter" | "jdqmr"
"jdqmr_n_levels"     : int        (default 20)
"jdqmr_target"       : float|None (default None → SA)
"jdqmr_tol"          : float      (default 1e-6)
"jdqmr_ncv"          : int|None   (default None → max(80, 2*n))
"jdqmr_maxBlockSize" : int        (default 1)
"jdqmr_verbose"      : int        (default 0)
```
