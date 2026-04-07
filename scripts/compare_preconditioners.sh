#!/bin/bash -l
#SBATCH -J precond_compare
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

# ========== 比较预条件子性能 ==========
# N=64，n_levels=1，blocksize=1
# target: -0.17 / -0.20 / -0.22 / -0.25
# 预条件子: None / Jacobi / TPA / ShiftedKinetic
# 结果保存在 precond_results/

mkdir -p precond_results

srun python -u compare_preconditioners.py

echo "预条件子比较完成，结果保存在 precond_results/"
