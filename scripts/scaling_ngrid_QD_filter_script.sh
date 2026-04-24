#!/bin/bash -l
#SBATCH -J scaling_QD_filter
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -A m4868
#SBATCH --output=%x_R${QD_RADIUS}-%j.out

# 用法（通过 --export 传入 QD_RADIUS）：
#   sbatch --export=ALL,QD_RADIUS=11 scripts/scaling_ngrid_QD_filter_script.sh
#   sbatch --export=ALL,QD_RADIUS=17 scripts/scaling_ngrid_QD_filter_script.sh

if [ -z "${QD_RADIUS}" ]; then
    echo "错误：需要设置 QD_RADIUS 环境变量。"
    echo "用法: sbatch --export=ALL,QD_RADIUS=17 scripts/scaling_ngrid_QD_filter_script.sh"
    exit 1
fi

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

if [ ! -d "QD_Outputs" ] || [ -z "$(ls QD_Outputs/QD_R*.cube 2>/dev/null)" ]; then
    echo "QD_Outputs 不存在或为空，正在生成 Cube 文件..."
    python -u generate_QD_cubes.py
fi

mkdir -p scaling_results

echo "运行 scaling_ngrid_QD_filter.py --qd-radius ${QD_RADIUS}"
srun python -u scaling_ngrid_QD_filter.py --qd-radius ${QD_RADIUS}

echo "scaling_ngrid_QD_filter QD_R${QD_RADIUS} 完成，结果保存在 scaling_results/"
