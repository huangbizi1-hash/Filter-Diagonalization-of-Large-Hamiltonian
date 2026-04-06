#!/bin/bash -l
#SBATCH -J scaling_ngrid
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
# T_wall / N_H 关于 N_grid 的 scaling
# N in [64,66,68,70,72,74]，固定 d=d_base，4 个 target 曲线
srun python -u scaling_ngrid.py

echo "scaling_ngrid 实验完成，结果保存在 scaling_results/"
