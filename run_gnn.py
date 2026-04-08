"""
run_gnn.py — Hamiltonian GNN 统一入口

从仓库根目录运行：

    # 纯差分基准测试（无 PyTorch，可独立运行）
    python run_gnn.py --mode test_fd

    # 完整流程（baseline + 训练 + 测试，需要 PyTorch）
    python run_gnn.py --mode all --wf_type gaussian --epochs 5000 --save_every 500

    # 仅训练（正弦波）
    python run_gnn.py --mode train --wf_type sine --k_max 3 --epochs 5000

    # 仅测试最新 run
    python run_gnn.py --mode test_gnn --d_test 2 --n_steps 10

    # 指定 run 目录测试
    python run_gnn.py --mode test_gnn --run_dir gnn_models/20240101_120000

输出目录（不纳入版本控制）：
    gnn_models/                  ← 已加入 .gitignore
      <timestamp>/
        config.json
        epoch_NNNNN.pt
        loss_history.json
        loss_curve.png
        gnn_test_energy.png
      baseline_test.png
      fd_baseline_numpy.png      ← test_fd 模式输出
"""

import os
import argparse

OUTPUT_ROOT = os.path.dirname(os.path.abspath(__file__))   # 仓库根目录


def main():
    parser = argparse.ArgumentParser(
        description="Hamiltonian GNN — 入口脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="all",
                        choices=["train", "test_baseline", "test_gnn",
                                 "test_fd", "test_ho", "all"],
                        help="test_fd: FD baseline on random wfs (no PyTorch); "
                             "test_ho: HO ground state correctness test (no PyTorch)")

    # 波函数
    parser.add_argument("--wf_type", default="gaussian",
                        choices=["gaussian", "sine"],
                        help="训练/测试波函数类型")
    parser.add_argument("--k_max", type=int, default=2,
                        help="正弦波最大波数（仅 wf_type=sine 时有效）")

    # 训练超参数
    parser.add_argument("--hidden_dim",      type=int,   default=64)
    parser.add_argument("--epochs",          type=int,   default=5000)
    parser.add_argument("--batch_per_epoch", type=int,   default=10)
    parser.add_argument("--save_every",      type=int,   default=500)
    parser.add_argument("--lr",              type=float, default=1e-3)

    # 测试参数
    parser.add_argument("--omega", type=float, default=1.0,
                        help="HO frequency for test_ho mode")
    parser.add_argument("--kinetic_cutoff", type=float, default=30.0,
                        help="Kinetic energy cutoff T(k)<=cutoff in FFT operator")
    parser.add_argument("--n_steps", type=int, default=8,
                        help="Max number of H applications in test")
    parser.add_argument("--n_test",  type=int, default=5,
                        help="测试波函数数量")
    parser.add_argument("--d_test",  type=int, default=1,
                        help="checkpoint 选取间隔")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="test_gnn 模式下指定已有 run 目录；"
                             "不指定则自动选最新 run")

    args = parser.parse_args()

    run_dir = args.run_dir   # 训练后会更新

    # ── 纯差分基准（无 PyTorch）──
    if args.mode == "test_fd":
        from gnn_code.fd_baseline import test_fd_baseline
        test_fd_baseline(
            n_steps=args.n_steps,
            n_test=args.n_test,
            wf_type=args.wf_type,
            k_max=args.k_max,
            output_root=OUTPUT_ROOT,
        )
        return

    # ── HO ground state correctness test (no PyTorch) ──
    if args.mode == "test_ho":
        from gnn_code.fd_baseline import test_ho_groundstate
        test_ho_groundstate(
            n_steps=args.n_steps,
            omega=args.omega,
            kinetic_cutoff=args.kinetic_cutoff,
            output_root=OUTPUT_ROOT,
        )
        return

    # ── 以下模式需要 PyTorch ──
    from gnn_code import train, test_baseline, test_gnn_from_run

    # ── baseline 测试（torch 版）──
    if args.mode in ("test_baseline", "all"):
        test_baseline(
            n_steps=args.n_steps,
            n_test=args.n_test,
            wf_type=args.wf_type,
            k_max=args.k_max,
            output_root=OUTPUT_ROOT,
        )

    # ── 训练 ──
    if args.mode in ("train", "all"):
        _, run_dir, _ = train(
            wf_type=args.wf_type,
            k_max=args.k_max,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_per_epoch=args.batch_per_epoch,
            save_every=args.save_every,
            lr=args.lr,
            output_root=OUTPUT_ROOT,
        )

    # ── GNN 测试 ──
    if args.mode in ("test_gnn", "all"):
        if run_dir is None:
            gnn_models_dir = os.path.join(OUTPUT_ROOT, "gnn_models")
            candidates = sorted([
                d for d in os.listdir(gnn_models_dir)
                if os.path.isdir(os.path.join(gnn_models_dir, d))
                and d[0].isdigit()   # 时间戳目录
            ])
            if not candidates:
                raise RuntimeError(
                    "gnn_models/ 下未找到任何 run 目录。"
                    "请先训练或用 --run_dir 指定目录。")
            run_dir = os.path.join(gnn_models_dir, candidates[-1])
            print(f"Auto-selected run_dir: {run_dir}")

        test_gnn_from_run(
            run_dir=run_dir,
            n_steps=args.n_steps,
            n_test=args.n_test,
            d_test=args.d_test,
            wf_type=args.wf_type,
            k_max=args.k_max,
        )


if __name__ == "__main__":
    main()
