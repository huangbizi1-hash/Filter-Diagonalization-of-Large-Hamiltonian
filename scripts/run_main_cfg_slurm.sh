#!/bin/bash -l
#SBATCH -J fd_main_cfg
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

# 用法示例：
#   sbatch --export=ALL,CFG=fft_config_qd13.json scripts/run_main_cfg_slurm.sh
#   sbatch --export=ALL,CFG=configs/fft_config_qd13.json,CPUS=64,TIME_LIMIT=36:00:00 scripts/run_main_cfg_slurm.sh

set -euo pipefail

CFG=${CFG:-}
CPUS=${CPUS:-${SLURM_CPUS_PER_TASK:-32}}
TIME_LIMIT=${TIME_LIMIT:-}

if [ -z "${CFG}" ]; then
    echo "错误：需要通过环境变量 CFG 指定配置文件路径。"
    echo "示例：sbatch --export=ALL,CFG=fft_config_qd13.json scripts/run_main_cfg_slurm.sh"
    exit 1
fi

if [ ! -f "${CFG}" ]; then
    echo "错误：配置文件不存在 -> ${CFG}"
    echo "当前目录：$(pwd)"
    exit 1
fi

module load conda
conda activate primme_env

cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

if [ ! -f "${CFG}" ]; then
    echo "错误：在仓库目录下仍找不到配置文件 -> ${CFG}"
    exit 1
fi

export OMP_NUM_THREADS=${CPUS}
export MKL_NUM_THREADS=${CPUS}
export OPENBLAS_NUM_THREADS=${CPUS}

echo "============================================================"
echo "Start time      : $(date '+%F %T')"
echo "Host            : $(hostname)"
echo "SLURM_JOB_ID    : ${SLURM_JOB_ID:-N/A}"
echo "Config          : ${CFG}"
echo "OMP threads     : ${OMP_NUM_THREADS}"
if [ -n "${TIME_LIMIT}" ]; then
    echo "Note            : TIME_LIMIT=${TIME_LIMIT} 仅记录，不会动态修改 SBATCH 时长"
fi
echo "Run command     : python -u main.py --cfg ${CFG}"
echo "============================================================"

srun -n 1 -c "${CPUS}" python -u main.py --cfg "${CFG}"

echo "============================================================"
echo "Finished at $(date '+%F %T')"
echo "输出目录请查看 main.py 打印的 Output directory"
echo "============================================================"
