#!/bin/bash -l
#SBATCH -J compare_fd_filter_localPot
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

mkdir -p fd_results

# 使用默认 localPot.cube（N=64）
srun python -u compare_fd_filter.py

echo "compare_fd_filter (localPot) 完成，结果保存在 fd_results/"
