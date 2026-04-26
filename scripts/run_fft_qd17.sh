#!/bin/bash -l
#SBATCH -J fft_qd17
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -A m4868
#SBATCH --output=%x-%j.out

module load conda
conda activate primme_env
cd /pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian

srun python -u main.py --cfg fft_config_qd17.json

echo "FFT filter diagonalisation QD_R17 complete. Results in results/"
