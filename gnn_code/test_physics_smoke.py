import numpy as np

from gnn_code import physics


def test_make_grid_shape():
    x, y, z = physics._make_grid(4)
    assert x.shape == (4, 4, 4)
    assert y.shape == (4, 4, 4)
    assert z.shape == (4, 4, 4)


def test_fft_energy_is_finite_for_constant_state():
    psi = np.ones((physics.N_fine, physics.N_fine, physics.N_fine), dtype=float)
    energy = physics.fft_energy(psi)
    assert np.isfinite(energy)
