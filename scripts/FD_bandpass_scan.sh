#!/bin/bash -l
#SBATCH -J fd_bandpass_scan
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

# ========== 运行滤波对角化 ==========
# filter_type : bandpass
# E1=0.05（半宽），beta=50（边沿陡峭系数）
# n_random=100，El 从 -0.18 到 -0.15，间距 0.001（共 31 个滤波中心）
# 结果由 main.py 内置逻辑存入 results/

srun python -u main.py \
    --set filter_type=bandpass \
    --set E1=0.05 \
    --set beta=50 \
    --set n_random=100 \
    --set "El_list=[-0.18,-0.179,-0.178,-0.177,-0.176,-0.175,-0.174,-0.173,-0.172,-0.171,-0.17,-0.169,-0.168,-0.167,-0.166,-0.165,-0.164,-0.163,-0.162,-0.161,-0.16,-0.159,-0.158,-0.157,-0.156,-0.155,-0.154,-0.153,-0.152,-0.151,-0.15]"

echo "滤波对角化完成，结果已存入 results/"
