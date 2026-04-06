#!/bin/bash -l
#SBATCH -J jdqmr_narrow_n50_b1
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

# ========== 结果输出目录 ==========
OUTPUT_DIR="jdqmr_results"
mkdir -p "${OUTPUT_DIR}"

# ========== 时间戳 + 脚本标题，用于结果文件命名 ==========
SCRIPT_TITLE="jdqmr_narrow_n50_block1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/${SCRIPT_TITLE}_${TIMESTAMP}.json"

# ========== 运行扫描 ==========
# 区间 [-0.17, -0.16]，e_step=0.5 → 单个扫描点 -0.17
# 每次求 50 个本征值，maxBlockSize=1
srun python -u scan_jdqmr_fft.py \
    --e_min -0.17 \
    --e_max -0.16 \
    --e_step 0.5 \
    --n_levels 50 \
    --maxBlockSize 1 \
    --n 64 \
    --r_cut 7.0 \
    --cube_file localPot.cube \
    --params_file gaussian_fit_params.json \
    --job_title "${SCRIPT_TITLE}" \
    --output "${OUTPUT_FILE}"

echo "结果已保存至：${OUTPUT_FILE}"
