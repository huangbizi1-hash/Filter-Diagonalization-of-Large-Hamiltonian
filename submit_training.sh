#!/bin/bash
# submit_training.sh — 并行提交三个 chain 训练任务
# 用法: bash submit_training.sh
# 日志输出到 logs/chain{1,3,5}.log

set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_job() {
    local chain_len=$1
    local log="$LOG_DIR/chain${chain_len}.log"
    echo "Launching chain_len=${chain_len}  →  ${log}"
    nohup python run_gnn.py \
        --mode train \
        --wf_type sine \
        --k_max 3 \
        --epochs 200000 \
        --save_every 1000 \
        --chain_len "${chain_len}" \
        --kinetic_cutoff 30.0 \
        > "${log}" 2>&1 &
    echo "  PID: $!"
}

run_job 1
run_job 3
run_job 5

echo ""
echo "All 3 jobs submitted.  Monitor with:"
echo "  tail -f logs/chain1.log"
echo "  tail -f logs/chain3.log"
echo "  tail -f logs/chain5.log"
echo ""
echo "Or watch all at once:"
echo "  tail -f logs/chain{1,3,5}.log"
