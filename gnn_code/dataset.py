"""
dataset.py — Fixed dataset generation and loading for GNN training.

Dataset: uniform k-grid sine wavefunctions.
  k values per axis: np.linspace(-k_max, k_max, n_k) * 2π/L  →  n_k³ states
  Each sample stores chain_len (psi_sparse, H_target) pairs.

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

from .physics import X_f, Y_f, Z_f, L, N_sparse
from .data import generate_chain


# ── generation ──────────────────────────────────────────────────────────────

def generate_k_grid_dataset(
    k_max:          int,
    n_k:            int,
    chain_len:      int   = 1,
    kinetic_cutoff: float = 30.0,
    out_dir:        str   = "dataset",
) -> int:
    """
    Generate n_k³ sine-wave wavefunctions on a uniform k-grid and save to out_dir.

    k vectors: all (kx, ky, kz) with each component in
               np.linspace(-k_max, k_max, n_k) * 2π/L.

    The zero wavefunction (kx=ky=kz=0) is skipped automatically.

    Returns the number of samples actually saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    dk     = 2 * np.pi / L
    k_vals = np.linspace(-k_max, k_max, n_k) * dk

    count = 0
    for kx in k_vals:
        for ky in k_vals:
            for kz in k_vals:
                psi_fine = np.sin(kx * X_f + ky * Y_f + kz * Z_f)
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
        k_max=k_max, n_k=n_k,
        chain_len=chain_len, kinetic_cutoff=kinetic_cutoff,
        n_samples=count, N_sparse=N_sparse,
    )
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset: {count} samples saved → {out_dir}")
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
