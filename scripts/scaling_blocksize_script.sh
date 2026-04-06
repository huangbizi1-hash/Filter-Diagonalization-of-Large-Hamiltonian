#!/bin/bash -l
#SBATCH -J scaling_blocksize
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
# T_wall / N_H 关于 blocksize 的 scaling
# blocksize in [1,3,5,...,19]，N=64，target=-0.22，n_levels=10
srun python -u scaling_blocksize.py

echo "scaling_blocksize 实验完成，结果保存在 scaling_results/"
