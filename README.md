# Filter-Diagonalization-of-Large-Hamiltonian
Built filter diagonalization code in several approaches. Also implemented other numerical solvers. 
实现了标准的在格点上的滤波对角化代码，用于计算量子点哈密顿的能谱。

## FFT vs FD (HO3D, JDQMR) CLI

This repo provides two equivalent entrypoints:

- `python cli_compare_fft_fd_ho3d_jdqmr.py ...`
- `python scripts/cli_compare_fft_fd_ho3d_jdqmr.py ...`

Example:

```bash
python scripts/cli_compare_fft_fd_ho3d_jdqmr.py \
  --N 32 \
  --n-levels 20 \
  --fd-order 8 \
  --timing-repeats 500 \
  --out results/fft_fd_ho3d_jdqmr.json
```

## Troubleshooting on clusters

### 1) `index.lock` blocks `git pull` / `git reset`

If you see:

`Unable to create .../.git/index.lock: File exists`

run:

```bash
# in repo root
ps -ef | grep "[g]it"          # check whether a git process is still running
rm -f .git/index.lock           # remove stale lock if no active git process
git fetch origin
git reset --hard origin/<your-branch>
```

### 2) `can't open file ... scripts/cli_compare_fft_fd_ho3d_jdqmr.py`

This usually means your local checkout is not updated to a commit that contains this file.
After fixing `index.lock`, update branch and verify:

```bash
git fetch origin
# choose one branch you want to track, e.g. work or your PR branch
git checkout <your-branch>
git reset --hard origin/<your-branch>

# verify file exists
ls scripts/cli_compare_fft_fd_ho3d_jdqmr.py
```
