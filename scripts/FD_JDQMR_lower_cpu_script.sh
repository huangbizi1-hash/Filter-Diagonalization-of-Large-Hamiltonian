#!/bin/bash -l
#SBATCH -J scan_JDQMR_lower
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
SCRIPT_TITLE="JDQMR_lower_wide_n1_block1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/${SCRIPT_TITLE}_${TIMESTAMP}.json"

# ========== 运行扫描 ==========
srun python -u scan_jdqmr_fft.py \
    --e_min -0.29 \
    --e_max -0.17 \
    --e_step 0.001 \
    --n_levels 1 \
    --maxBlockSize 1 \
    --n 64 \
    --r_cut 7.0 \
    --cube_file localPot.cube \
    --params_file gaussian_fit_params.json \
    --job_title "${SCRIPT_TITLE}" \
    --output "${OUTPUT_FILE}"

echo "结果已保存至：${OUTPUT_FILE}"
