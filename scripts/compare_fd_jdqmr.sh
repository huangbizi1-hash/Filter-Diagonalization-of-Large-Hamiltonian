#!/bin/bash -l
#SBATCH -J compare_fd_jdqmr
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

mkdir -p fd_results

srun python -u compare_fd_jdqmr.py

echo "compare_fd_jdqmr 完成，结果保存在 fd_results/"
