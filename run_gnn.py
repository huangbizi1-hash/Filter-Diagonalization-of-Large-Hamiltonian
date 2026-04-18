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
                                 "test_fd", "test_ho", "test_gnn_ho",
                                 "gen_dataset", "test_filter", "benchmark",
                                 "timing", "all"],
                        help="test_fd: FD baseline on random wfs (no PyTorch); "
                             "test_ho: HO ground state correctness test (no PyTorch); "
                             "test_gnn_ho: HO ground state test with GNN comparison "
                             "(energy + wavefunction similarity vs n); "
                             "gen_dataset: generate fixed k-grid sine-wave dataset; "
                             "test_filter: filter diagonalization with GNN vs FD on sparse grid; "
                             "benchmark: compare multiple GNN models (--run_dirs) vs FFT/FD baseline; "
                             "timing: measure single H-apply time for all architectures (no training)")

    # 波函数
    parser.add_argument("--wf_type", default="gaussian",
                        choices=["gaussian", "sine", "pm1"],
                        help="波函数类型：gaussian/sine 用于训练/测试；pm1 仅用于 gen_dataset")
    parser.add_argument("--k_max", type=int, default=2,
                        help="正弦波最大波数（仅 wf_type=sine 或 gen_dataset 时有效）")

    # 固定数据集
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="预生成数据集目录；训练时指定则从磁盘加载，"
                             "不指定则按 wf_type 动态生成")
    parser.add_argument("--n_k", type=int, default=10,
                        help="gen_dataset 模式（wf_type=sine/k-grid）：每轴 k 取值点数（共 n_k³ 个态）")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="gen_dataset 模式（wf_type=pm1/gaussian/sine 随机）：生成样本数")
    parser.add_argument("--seed", type=int, default=None,
                        help="gen_dataset 随机数据集的随机种子（默认不固定）")

    # 图结构
    parser.add_argument("--graph_type", default="cube",
                        choices=["cube", "cross"],
                        help="cube: 3×3×3 Mehrstellen（原有）; "
                             "cross: 高阶十字星FD + correction立方体（两套边）")
    parser.add_argument("--fd_order", type=int, default=4,
                        help="graph_type=cross 时，FD 差分阶数（偶数，默认 4）")
    parser.add_argument("--n_co", type=int, default=3,
                        help="graph_type=cross 时，correction 立方体边长（奇数，默认 3）")

    # 模型类型
    parser.add_argument("--model_type", default="gnn",
                        choices=["gnn", "so3"],
                        help="gnn: 原有 MLP correction（HamiltonianGNN / HamiltonianGNN_Cross）; "
                             "so3: SO(3) 启发的 l=0+l=1 修正（SO3HamiltonianNet，需 graph_type=cross）")
    parser.add_argument("--radial_hidden_dim", type=int, default=32,
                        help="model_type=so3 时径向 MLP 隐藏层宽度（默认 32）")

    # 训练超参数
    parser.add_argument("--hidden_dim",      type=int,   default=64)
    parser.add_argument("--epochs",          type=int,   default=5000)
    parser.add_argument("--batch_per_epoch", type=int,   default=10,
                        help="Number of optimizer steps per epoch")
    parser.add_argument("--batch_size",      type=int,   default=1,
                        help="Number of independent psi's averaged per optimizer step")
    parser.add_argument("--save_every",      type=int,   default=500)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--chain_len",       type=int,   default=1,
                        help="H-application chain length per training sample "
                             "(1 = original single-step; >1 = chain training)")
    parser.add_argument("--chain_mode",     type=str,   default="teacher",
                        choices=["teacher", "auto"],
                        help="teacher: each step uses FFT reference input (teacher forcing); "
                             "auto: each step uses normalised GNN output from previous step "
                             "(autoregressive, trains model on its own output distribution)")
    parser.add_argument("--chain_bptt",    action="store_true", default=False,
                        help="[auto mode only] allow gradients to flow across chain steps "
                             "(full BPTT); default is detach between steps. "
                             "Has no effect in teacher mode.")
    parser.add_argument("--kinetic_cutoff",  type=float, default=30.0,
                        help="Kinetic energy cap T(k)<=cutoff in FFT operator (Ha)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for training: auto=use CUDA if available, "
                             "cpu=force CPU, cuda=force CUDA (fails if unavailable)")

    # filter 测试参数
    parser.add_argument("--filter_nc",       type=int,   default=5000,
                        help="test_filter: Newton filter order (H-applies per random vector)")
    parser.add_argument("--filter_el_list",  type=float, nargs='+',
                        default=[-0.17],
                        help="test_filter: one or more target energies El (Ha), "
                             "e.g. --filter_el_list -0.25 -0.243 -0.18")
    parser.add_argument("--filter_n_random", type=int,   default=64,
                        help="test_filter: number of random starting vectors")
    parser.add_argument("--filter_svd_tol",  type=float, default=1e-3,
                        help="test_filter: SVD rank truncation threshold for Rayleigh-Ritz")
    parser.add_argument("--filter_cube",     type=str,   default=None,
                        help="test_filter: path to QD cube file (default: localPot.cube)")
    parser.add_argument("--filter_params",   type=str,   default=None,
                        help="test_filter: path to Gaussian fit params JSON "
                             "(default: gaussian_fit_params.json)")
    parser.add_argument("--filter_vmin",     type=float, default=None,
                        help="test_filter: 谱窗口下界 Vmin (Ha)；默认 -5.0（与 --Vmin 等价）")
    parser.add_argument("--filter_de",       type=float, default=None,
                        help="test_filter: 谱窗口宽度 dE (Ha)；默认 50.0（与 --dE 等价）")
    parser.add_argument("--Vmin",            type=float, default=None,
                        help="test_filter: 谱窗口下界（Ha），同 --filter_vmin，与 fft filter 保持一致")
    parser.add_argument("--dE",              type=float, default=None,
                        help="test_filter: 谱窗口宽度（Ha），同 --filter_de，与 fft filter 保持一致")

    # 其他测试参数
    parser.add_argument("--omega", type=float, default=1.0,
                        help="HO frequency for test_ho mode")
    parser.add_argument("--n_steps", type=int, default=8,
                        help="Max number of H applications in test")
    parser.add_argument("--n_test",  type=int, default=5,
                        help="测试波函数数量")
    parser.add_argument("--d_test",  type=int, default=1,
                        help="checkpoint 选取间隔")
    parser.add_argument("--run_name", type=str, default=None,
                        help="训练输出文件夹名（gnn_models/<run_name>）；"
                             "不指定则使用时间戳")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="test_gnn / test_filter 模式下指定已有 run 目录；"
                             "不指定则自动选最新 run")
    parser.add_argument("--run_dirs", nargs="+", default=None,
                        help="benchmark 模式：要比较的 GNN run 目录列表，"
                             "例如 gnn_models/chain1_teacher gnn_models/chain2_teacher")
    parser.add_argument("--n_timing_reps", type=int, default=20,
                        help="benchmark 模式：单次 H-apply 计时重复次数（取中位数）")

    # timing 模式专用参数
    parser.add_argument("--timing_hidden_dims", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128, 256],
                        help="timing 模式：测试的 hidden_dim 列表（GNN-cube / GNN-cross）")
    parser.add_argument("--timing_radial_hidden_dims", type=int, nargs="+",
                        default=None,
                        help="timing 模式：SO3 radial_hidden_dim 列表；"
                             "不指定则与 --timing_hidden_dims 相同")
    parser.add_argument("--timing_n_reps", type=int, default=100,
                        help="timing 模式：计时重复次数（均值，默认 100）")
    parser.add_argument("--timing_n_warmup", type=int, default=10,
                        help="timing 模式：热身次数（不计入统计，默认 10）")
    parser.add_argument("--timing_description", type=str, default="",
                        help="timing 模式：写入 JSON description 字段的说明文字")

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

    # ── 生成固定数据集（无 PyTorch）──
    if args.mode == "gen_dataset":
        out_dir = args.dataset_dir or os.path.join(OUTPUT_ROOT, "gnn_dataset")
        if args.wf_type == 'pm1' or (args.wf_type in ('gaussian', 'sine') and args.n_samples):
            # 随机态数据集：pm1 / gaussian / sine，共 n_samples 个独立样本
            from gnn_code.dataset import generate_random_dataset
            generate_random_dataset(
                wf_type=args.wf_type,
                n_samples=args.n_samples,
                chain_len=args.chain_len,
                kinetic_cutoff=args.kinetic_cutoff,
                out_dir=out_dir,
                seed=args.seed,
            )
        else:
            # 默认：均匀 k-grid 正弦波（n_k³ 个固定态）
            from gnn_code.dataset import generate_k_grid_dataset
            generate_k_grid_dataset(
                k_max=args.k_max,
                n_k=args.n_k,
                chain_len=args.chain_len,
                kinetic_cutoff=args.kinetic_cutoff,
                out_dir=out_dir,
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

    # ── filter 测试（GNN vs FD，需要 PyTorch + fft_code）──
    if args.mode == "test_filter":
        if args.run_dir is None:
            gnn_models_dir = os.path.join(OUTPUT_ROOT, "gnn_models")
            candidates = sorted([
                d for d in os.listdir(gnn_models_dir)
                if os.path.isdir(os.path.join(gnn_models_dir, d)) and d[0].isdigit()
            ])
            if not candidates:
                raise RuntimeError("gnn_models/ 下未找到任何 run 目录。")
            run_dir = os.path.join(gnn_models_dir, candidates[-1])
            print(f"Auto-selected run_dir: {run_dir}")
        else:
            run_dir = args.run_dir
        from gnn_code.test_filter import test_gnn_filter
        # --Vmin/--dE 与 --filter_vmin/--filter_de 等价，短名优先
        vmin_val = args.Vmin if args.Vmin is not None else args.filter_vmin
        de_val   = args.dE   if args.dE   is not None else args.filter_de
        kw = {}
        if args.filter_cube is not None: kw['cube_file']   = args.filter_cube
        if args.filter_params is not None: kw['params_file'] = args.filter_params
        if vmin_val is not None: kw['vmin'] = vmin_val
        if de_val   is not None: kw['d_e']  = de_val
        test_gnn_filter(
            run_dir=run_dir,
            nc=args.filter_nc,
            el_list=args.filter_el_list,
            n_random=args.filter_n_random,
            svd_tol=args.filter_svd_tol,
            output_root=run_dir,
            device=args.device,
            **kw,
        )
        return

    # ── benchmark 模式：多模型精度-效率对比 ──
    if args.mode == "benchmark":
        if not args.run_dirs:
            raise ValueError(
                "--run_dirs 必须指定至少一个 GNN run 目录，"
                "例如：--run_dirs gnn_models/chain1_teacher gnn_models/chain2_teacher")
        from gnn_code.benchmark import benchmark_models
        vmin_val = args.Vmin if args.Vmin is not None else args.filter_vmin
        de_val   = args.dE   if args.dE   is not None else args.filter_de
        kw = {}
        if args.filter_cube   is not None: kw["cube_file"]   = args.filter_cube
        if args.filter_params is not None: kw["params_file"] = args.filter_params
        if vmin_val is not None: kw["vmin"] = vmin_val
        if de_val   is not None: kw["d_e"]  = de_val
        benchmark_models(
            run_dirs      = args.run_dirs,
            nc            = args.filter_nc,
            el            = args.filter_el_list[0],
            n_random      = args.filter_n_random,
            svd_tol       = args.filter_svd_tol,
            n_timing_reps = args.n_timing_reps,
            device        = args.device,
            output_root   = OUTPUT_ROOT,
            **kw,
        )
        return

    # ── timing 模式：H-apply 计时基准 ──
    if args.mode == "timing":
        from gnn_code.timing_benchmark import time_h_apply
        kw = {}
        if args.filter_cube   is not None: kw["cube_file"]   = args.filter_cube
        if args.filter_params is not None: kw["params_file"] = args.filter_params
        time_h_apply(
            n_reps             = args.timing_n_reps,
            n_warmup           = args.timing_n_warmup,
            hidden_dims        = args.timing_hidden_dims,
            radial_hidden_dims = args.timing_radial_hidden_dims,
            fd_order           = args.fd_order,
            n_co               = args.n_co,
            device             = args.device,
            output_root        = OUTPUT_ROOT,
            description        = args.timing_description,
            **kw,
        )
        return

    # ── 以下模式需要 PyTorch ──
    from gnn_code import train, test_baseline, test_gnn_from_run, test_gnn_ho

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
            batch_size=args.batch_size,
            save_every=args.save_every,
            lr=args.lr,
            chain_len=args.chain_len,
            chain_mode=args.chain_mode,
            chain_bptt=args.chain_bptt,
            kinetic_cutoff=args.kinetic_cutoff,
            output_root=OUTPUT_ROOT,
            run_name=args.run_name,
            device=args.device,
            dataset_dir=args.dataset_dir,
            graph_type=args.graph_type,
            fd_order=args.fd_order,
            n_co=args.n_co,
            model_type=args.model_type,
            radial_hidden_dim=args.radial_hidden_dim,
        )

    # ── HO + GNN 测试（能量 + 相似度序列）──
    if args.mode == "test_gnn_ho":
        if run_dir is None:
            gnn_models_dir = os.path.join(OUTPUT_ROOT, "gnn_models")
            candidates = sorted([
                d for d in os.listdir(gnn_models_dir)
                if os.path.isdir(os.path.join(gnn_models_dir, d))
                and d[0].isdigit()
            ])
            if not candidates:
                raise RuntimeError(
                    "gnn_models/ 下未找到任何 run 目录。"
                    "请先训练或用 --run_dir 指定目录。")
            run_dir = os.path.join(gnn_models_dir, candidates[-1])
            print(f"Auto-selected run_dir: {run_dir}")
        test_gnn_ho(
            run_dir=run_dir,
            n_steps=args.n_steps,
            omega=args.omega,
            kinetic_cutoff=args.kinetic_cutoff,
        )
        return

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
