#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/run_sweep_conv_cell_radius.sh
#   bash scripts/run_sweep_conv_cell_radius.sh --dry-run

python sweep_compare_fft_rbf_filter_qd_radius.py "$@"
