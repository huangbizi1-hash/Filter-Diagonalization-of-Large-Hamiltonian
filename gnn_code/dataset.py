"""
dataset.py — Fixed dataset generation and loading for GNN training.

两种生成函数：

generate_k_grid_dataset(k_max, n_k, ...)
    均匀 k-grid 正弦波：n_k³ 个固定态（跳过 k=0）。

generate_random_dataset(wf_type, n_samples, ...)
    随机态：n_samples 个独立随机样本。
    wf_type='pm1'      → 每个细网格格点独立随机 ±1
    wf_type='gaussian' → 3 个随机高斯波包叠加
    wf_type='sine'     → 3-6 个随机正弦模式叠加

两种函数输出格式相同，均可被 WavefunctionDataset 加载。

Storage (one .npz per sample in out_dir/):
  psi_sparse : float32  [chain_len, N_sparse³]
  target     : float32  [chain_len, N_sparse³]

Loading:
  try_load_dataset()  tries full in-memory; falls back to lazy disk on MemoryError.
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .physics import X, Y, Z, L, N, N_sparse
from .data import generate_chain, gen_fine_wavefunction


# ── generation ──────────────────────────────────────────────────────────────

def generate_k_grid_dataset(
    k_max:          int,
    n_k:            int,
    chain_len:      int   = 1,
    kinetic_cutoff: float = 30.0,
    out_dir:        str   = "dataset",
    k_shift:        float = 0.0,
) -> int:
    """
    Generate n_k³ sine-wave wavefunctions on a uniform k-grid and save to out_dir.

    k vectors: all (kx, ky, kz) with each component in
               (np.linspace(-k_max, k_max, n_k) + k_shift) * 2π/L.

    k_shift (in the same units as k_max) shifts the entire k-grid, allowing
    non-overlapping test sets when set to 2*k_max + small_epsilon.

    The zero wavefunction (all k=0 after shift) is skipped automatically.

    Returns the number of samples actually saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    dk     = 2 * np.pi / L
    k_vals = (np.linspace(-k_max, k_max, n_k) + k_shift) * dk

    count = 0
    for kx in k_vals:
        for ky in k_vals:
            for kz in k_vals:
                psi_fine = np.sin(kx * X + ky * Y + kz * Z)
                if np.max(np.abs(psi_fine)) < 1e-10:   # identically zero → skip
                    continue

                pairs = generate_chain(psi_fine, chain_len, kinetic_cutoff)
                if not pairs:
                    continue

                psi_arr = np.stack([p[0].flatten().astype(np.float32) for p in pairs])
                tgt_arr = np.stack([p[1].flatten().astype(np.float32) for p in pairs])

                np.savez_compressed(
                    os.path.join(out_dir, f"sample_{count:05d}.npz"),
                    psi_sparse=psi_arr,
                    target=tgt_arr,
                )
                count += 1

    meta = dict(
        k_max=k_max, n_k=n_k, k_shift=k_shift,
        chain_len=chain_len, kinetic_cutoff=kinetic_cutoff,
        n_samples=count, N=N,
    )
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset: {count} samples saved → {out_dir}")
    return count


# ── 随机态数据集（pm1 / gaussian / sine，n_samples 个独立样本）────────────────

def generate_random_dataset(
    wf_type:        str   = 'pm1',
    n_samples:      int   = 1000,
    chain_len:      int   = 1,
    kinetic_cutoff: float = 30.0,
    out_dir:        str   = "dataset",
    seed:           int   = None,
) -> int:
    """
    生成 n_samples 个独立随机波函数样本并保存到 out_dir。

    wf_type
    -------
    'pm1'      每个细网格格点独立随机 ±1，下采样到稀疏网格后仍为 ±1 值。
               适合训练 GNN filter 处理平坦初态（覆盖所有频率分量）。
    'gaussian' 3 个随机中心 / 宽度的高斯波包叠加。
    'sine'     3-6 个随机 k 矢量的正弦模式叠加。

    所有类型均通过 FFT 精确计算 H|ψ⟩（细网格）再下采样，
    与 generate_k_grid_dataset 输出格式相同，可直接用于训练。

    Parameters
    ----------
    wf_type        : 初态类型，'pm1' | 'gaussian' | 'sine'
    n_samples      : 样本数量
    chain_len      : 每个样本存储的 H 链长（≥1，与训练 --chain_len 匹配）
    kinetic_cutoff : FFT 动能截断（Ha）
    out_dir        : 输出目录
    seed           : 随机种子（None = 不固定）

    Returns
    -------
    int : 实际保存的样本数
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    count = 0
    for idx in range(n_samples):
        psi_fine = gen_fine_wavefunction(wf_type=wf_type, rng=rng)
        if np.max(np.abs(psi_fine)) < 1e-10:
            continue

        pairs = generate_chain(psi_fine, chain_len, kinetic_cutoff)
        if not pairs:
            continue

        psi_arr = np.stack([p[0].flatten().astype(np.float32) for p in pairs])
        tgt_arr = np.stack([p[1].flatten().astype(np.float32) for p in pairs])

        np.savez_compressed(
            os.path.join(out_dir, f"sample_{count:05d}.npz"),
            psi_sparse=psi_arr,
            target=tgt_arr,
        )
        count += 1
        if (count) % 100 == 0:
            print(f"  {count}/{n_samples} samples saved...", flush=True)

    meta = dict(
        wf_type=wf_type, n_samples=count,
        chain_len=chain_len, kinetic_cutoff=kinetic_cutoff,
        seed=seed, N=N,
    )
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset ({wf_type}): {count} samples saved → {out_dir}")
    return count


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class WavefunctionDataset(Dataset):
    """
    PyTorch Dataset backed by .npz files in data_dir.

    Each item is a list of (psi_tensor, target_tensor) pairs of length
    ≤ chain_len, with tensors shaped [N_sparse³, 1] (GNN format).

    Parameters
    ----------
    data_dir   : directory containing sample_*.npz files
    in_memory  : if True, all arrays are read into RAM at construction
    """

    def __init__(self, data_dir: str, in_memory: bool = True):
        self.data_dir = data_dir
        files = sorted(f for f in os.listdir(data_dir) if f.startswith("sample_") and f.endswith(".npz"))
        self.files = [os.path.join(data_dir, f) for f in files]

        if not self.files:
            raise RuntimeError(f"No sample_*.npz files found in {data_dir}")

        self.in_memory = in_memory
        self._cache: list | None = None

        if in_memory:
            self._cache = []
            for path in self.files:
                d = np.load(path)
                self._cache.append({
                    "psi_sparse": d["psi_sparse"].copy(),
                    "target":     d["target"].copy(),
                })

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> list:
        if self.in_memory:
            d = self._cache[idx]
        else:
            raw = np.load(self.files[idx])
            d = {"psi_sparse": raw["psi_sparse"], "target": raw["target"]}

        pairs = []
        for k in range(len(d["psi_sparse"])):
            psi = torch.tensor(d["psi_sparse"][k], dtype=torch.float32).unsqueeze(-1)
            tgt = torch.tensor(d["target"][k],     dtype=torch.float32).unsqueeze(-1)
            pairs.append((psi, tgt))
        return pairs


# ── smart loader ─────────────────────────────────────────────────────────────

def try_load_dataset(data_dir: str) -> WavefunctionDataset:
    """
    Attempt full in-memory load; fall back to lazy disk on MemoryError.
    """
    try:
        ds = WavefunctionDataset(data_dir, in_memory=True)
        print(f"Dataset loaded in memory: {len(ds)} samples from {data_dir}")
        return ds
    except MemoryError:
        print("WARNING: MemoryError — switching to lazy disk loading.")
        ds = WavefunctionDataset(data_dir, in_memory=False)
        print(f"Dataset (lazy disk): {len(ds)} samples from {data_dir}")
        return ds
