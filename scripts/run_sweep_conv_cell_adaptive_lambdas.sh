#!/usr/bin/env bash
set -euo pipefail

# 扫描参数：
#   --conv-cell-adaptive-lambda-grad: logspace(0.1, 10.0, 5)
#   --conv-cell-adaptive-lambda-lap : logspace(0.1, 10.0, 5)

python sweep_conv_cell_adaptive_lambdas.py \
  --system qd --qd-radius 17 \
  --el -0.18 --nc 500 --n-random 1 --dE 300.0 --V-min -5.0 --fft-kinetic-cut 20.0 \
  --rbf-node-method conv_cell \
  --conv-cell-template-mode fcc_refined --conv-cell-domain-shape cube --rbf-stencil-radius 1.60 \
  --conv-cell-fcc-scale-factor 8 \
  --conv-cell-fcc-origin-frac 0.0 0.0 0.0 \
  --conv-cell-fcc-atom-refine-factor 16 \
  --conv-cell-fcc-atom-radius-frac 0.00 \
  --conv-cell-a 11.4523 --conv-cell-d-min-frac 0.02 --conv-cell-n-random 300 --conv-cell-seed 42 \
  --conv-cell-adaptive-random \
  --conv-cell-adaptive-grid-n 48 \
  --conv-cell-adaptive-candidate-multiplier 12 \
  --rbf-phi ga --rbf-eps 0.6 --rbf-order 0 --rbf-v-source gaussian_direct \
  --quality-probe-method uniform --quality-probe-n 200000 \
  --save-nodes rbf_nodes/
