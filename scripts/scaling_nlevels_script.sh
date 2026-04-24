#!/bin/bash -l
#SBATCH -J scaling_nlevels
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

# ========== 结果输出目录 ==========
mkdir -p scaling_results

# ========== 运行 scaling 实验 ==========
# T_wall / N_H 关于 n_levels 的 scaling
# n_levels in [1,3,5,...,19]，N=64，target=-0.22，blocksize=1
srun python -u scaling_nlevels.py

echo "scaling_nlevels 实验完成，结果保存在 scaling_results/"
