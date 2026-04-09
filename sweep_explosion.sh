#!/bin/bash
# sweep_explosion.sh
# 系统测试：多阶 vs 单阶 Chebyshev explosion，扫描 n_states。
#
# 实验设计
# --------
#   multistage : n_stages=5, E_lower=-0.5 -0.4 -0.3 -0.2 -0.1
#   single     : n_stages=1, E_lower=-0.1
#   n_states   : 128, 256, 512, 1024
#   共 8 个任务，顺序运行，日志写到 logs_sweep/
#
# 用法
#   bash sweep_explosion.sh            # 顺序运行
#   bash sweep_explosion.sh --dry-run  # 仅打印命令，不执行

set -euo pipefail

COMMON="--m 20 --k_max 2.0 --E_upper 33.0 --N 64 --svd_tol 1e-3"
ROOT="results_sweep"
LOGDIR="logs_sweep"
DRY=0

if [[ "${1:-}" == "--dry-run" ]]; then DRY=1; fi

mkdir -p "$LOGDIR" "$ROOT"

run_job() {
    local tag=$1; shift
    local out="${ROOT}/${tag}"
    local log="${LOGDIR}/${tag}.log"
    local cmd="python explosion_qd.py $COMMON $* --out_dir ${out}"
    echo "── ${tag}"
    echo "   cmd : ${cmd}"
    echo "   log : ${log}"
    if [[ $DRY -eq 0 ]]; then
        mkdir -p "${out}"
        echo "=== ${tag} ===" > "${log}"
        echo "cmd: ${cmd}"   >> "${log}"
        echo ""               >> "${log}"
        { time ${cmd}; } >> "${log}" 2>&1
        echo "   done  (exit $?)"
    fi
    echo ""
}

echo "=========================================="
echo "  Chebyshev Explosion Sweep"
echo "  COMMON: ${COMMON}"
echo "  DRY_RUN: ${DRY}"
echo "=========================================="
echo ""

for n in 128 256 512 1024; do

    # ── 多阶滤波（5 阶）──
    run_job "multistage_n${n}" \
        --n_stages 5 \
        --E_lower -0.5 -0.4 -0.3 -0.2 -0.1 \
        --n_states ${n}

    # ── 单阶滤波（直接 E_lower=-0.1）──
    run_job "single_n${n}" \
        --n_stages 1 \
        --E_lower -0.1 \
        --n_states ${n}

done

echo "=========================================="
echo "  All jobs done. Results in ${ROOT}/"
echo "  Summary: python summarize_sweep.py"
echo "=========================================="
