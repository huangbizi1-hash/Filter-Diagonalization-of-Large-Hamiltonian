#!/bin/bash -l
#SBATCH -J CG_scan_lower
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out
module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian
# ========== 时间戳，用于结果文件命名 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="CG_scan_results_lower_${TIMESTAMP}.json"
# ========== 运行扫描 ==========
srun python -u scan_gaussian_energy.py \
    --delta_e 5e-4 \
    --maxiter 2000 \
    --e_min -0.29 \
    --e_max -0.17 \
    --e_step 0.001 \
    --n 64 \
    --r_cut 7.0 \
    --cube_file localPot.cube \
    --params_file gaussian_fit_params.json \
    --output "${OUTPUT_FILE}"
echo "结果已保存至：${OUTPUT_FILE}"
