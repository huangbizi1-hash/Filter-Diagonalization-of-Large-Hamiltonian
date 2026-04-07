#!/bin/bash -l
#SBATCH -J scaling_QD_GD_Jacobi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

if [ ! -d "QD_Outputs" ] || [ -z "$(ls QD_Outputs/QD_R*.cube 2>/dev/null)" ]; then
    echo "QD_Outputs 不存在或为空，正在生成 Cube 文件..."
    python -u generate_QD_cubes.py
else
    echo "QD_Outputs 已存在，跳过生成步骤。"
fi

mkdir -p scaling_results
srun python -u scaling_ngrid_QD_primme_gd_jacobi.py

echo "PRIMME_GD + Jacobi scaling 完成，结果保存在 scaling_results/"
