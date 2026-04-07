#!/bin/bash -l
#SBATCH -J compare_fd_filter_qd
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -A m4868
#SBATCH --output=%x_R${QD_RADIUS}-%j.out

# 用法（通过 --export 传入 QD_RADIUS）：
#   sbatch --export=ALL,QD_RADIUS=11 scripts/compare_fd_filter_qd.sh
#   sbatch --export=ALL,QD_RADIUS=17 scripts/compare_fd_filter_qd.sh
#   ...

if [ -z "${QD_RADIUS}" ]; then
    echo "错误：需要设置 QD_RADIUS 环境变量。"
    echo "用法: sbatch --export=ALL,QD_RADIUS=17 scripts/compare_fd_filter_qd.sh"
    exit 1
fi

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

# 如果 QD_Outputs 不存在则先生成
if [ ! -d "QD_Outputs" ] || [ -z "$(ls QD_Outputs/QD_R*.cube 2>/dev/null)" ]; then
    echo "QD_Outputs 不存在或为空，正在生成 Cube 文件..."
    python -u generate_QD_cubes.py
fi

mkdir -p fd_results

echo "运行 compare_fd_filter.py --qd-radius ${QD_RADIUS}"
srun python -u compare_fd_filter.py --qd-radius ${QD_RADIUS}

echo "compare_fd_filter QD_R${QD_RADIUS} 完成，结果保存在 fd_results/"
