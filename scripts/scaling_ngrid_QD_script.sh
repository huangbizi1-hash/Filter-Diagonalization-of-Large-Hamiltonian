#!/bin/bash -l
#SBATCH -J scaling_ngrid_QD
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

# ========== 生成 QD Cube 文件（若已存在则跳过） ==========
if [ ! -d "QD_Outputs" ] || [ -z "$(ls QD_Outputs/QD_R*.cube 2>/dev/null)" ]; then
    echo "QD_Outputs 不存在或为空，正在生成 Cube 文件..."
    python -u generate_QD_cubes.py
else
    echo "QD_Outputs 已存在，跳过生成步骤。"
fi

mkdir -p scaling_results

# ========== 运行 N_grid scaling 实验 ==========
# 使用真实 InAs QD 几何（d=0.625 Bohr，N 随 QD 半径变化）
# target: -0.17 / -0.20 / -0.22 / -0.25，n_levels=1，blocksize=1
srun python -u scaling_ngrid_QD.py

echo "scaling_ngrid_QD 实验完成，结果保存在 scaling_results/"
