"""
测试用势能构建（谐振子、调和阱、高斯凹坑）。

真实材料势能（gaussian_files 类型）由根目录
gaussian_potential_builder.GaussianPotentialBuilder 负责提供。
"""
from typing import Dict, List, Tuple, Any

import numpy as np

from .grid import build_grid


def build_ho3d(X, Y, Z, omega: float = 1.0) -> np.ndarray:
    """V = ½ω²r²（三维谐振子）。"""
    return 0.5 * omega**2 * (X**2 + Y**2 + Z**2)


def build_harmonic_well(X, Y, Z, center_locations: List, width: float) -> np.ndarray:
    """截断调和阱（可多中心叠加）。"""
    V = np.zeros(X.shape)
    V_max = 0.5 * width**2
    for center in center_locations:
        c = np.asarray(center)
        r2 = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2
        inside = np.sqrt(r2) < width
        V[inside] += 0.5 * r2[inside]
    V[(V == 0) | (V >= V_max)] = V_max
    return V


def build_gaussian_blob(X, Y, Z, h: float = -5.0, mu: float = 0.1) -> np.ndarray:
    """V = h·exp(-μr²)。"""
    return h * np.exp(-mu * (X**2 + Y**2 + Z**2))


def build_potential_from_config(cfg: Dict[str, Any], N: int) -> Tuple:
    """
    根据 cfg['potential']['type'] 分派到对应的势能构建器。

    gaussian_files 类型调用 GaussianPotentialBuilder（根目录模块）。

    返回 x, y, z, X, Y, Z, x_grid, V
    """
    pot   = cfg["potential"]
    ptype = pot["type"]

    if ptype == "ho3d":
        x, y, z, X, Y, Z, x_grid = build_grid(N, pot["d"])
        V = build_ho3d(X, Y, Z, omega=pot["omega"])

    elif ptype == "harmonic_well":
        x, y, z, X, Y, Z, x_grid = build_grid(N, pot["d"])
        V = build_harmonic_well(X, Y, Z,
                                center_locations=pot["center_locations"],
                                width=pot["width"])

    elif ptype == "gaussian_blob":
        x, y, z, X, Y, Z, x_grid = build_grid(N, pot["d"])
        V = build_gaussian_blob(X, Y, Z, h=pot["h"], mu=pot["mu"])

    elif ptype == "gaussian_files":
        from gaussian_potential_builder import GaussianPotentialBuilder
        builder = GaussianPotentialBuilder(
            cube_file=pot["cube_file"],
            params_file=pot["params_file"],
            r_cut=pot["r_cut"],
        )
        x, y, z, V = builder.build_potential(N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        x_grid  = [x, y, z]

    else:
        raise ValueError(f"Unknown potential type: '{ptype}'")

    return x, y, z, X, Y, Z, x_grid, V
