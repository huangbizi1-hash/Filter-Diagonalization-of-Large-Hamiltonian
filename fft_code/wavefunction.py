"""
波函数归一化与随机初态生成。
"""
import random
from typing import Optional

import numpy as np


def normalize_psi(psi: np.ndarray) -> np.ndarray:
    return psi / np.sqrt(np.sum(np.abs(psi)**2))


def random_sine_psi(X, Y, Z, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """随机正弦叠加波函数。"""
    if rng is not None:
        kx_val, ky_val, kz_val, b_val = rng.uniform(-1.0, 1.0, 4)
    else:
        kx_val = random.uniform(-1.0, 1.0)
        ky_val = random.uniform(-1.0, 1.0)
        kz_val = random.uniform(-1.0, 1.0)
        b_val  = random.uniform(-1.0, 1.0)
    psi = np.sin(kx_val * X + ky_val * Y + kz_val * Z + b_val) + 0.0j
    return normalize_psi(psi)


def random_pm1_psi(X, Y, Z, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """格点上独立随机 ±1 波函数（归一化后作为滤波初态）。"""
    if rng is not None:
        signs = rng.choice([-1, 1], size=X.shape)
    else:
        signs = np.array([random.choice([-1, 1]) for _ in range(X.size)],
                         dtype=float).reshape(X.shape)
    psi = signs.astype(complex)
    return normalize_psi(psi)
