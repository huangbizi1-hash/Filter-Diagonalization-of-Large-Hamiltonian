# Filter interpolation plotting snippet + complete CLI

Below is the extracted plotting snippet (from `main.py` + `fft_code/plotting.py`) and a **complete runnable CLI**.

The x-axis limit is controlled by `plot_interval` in config, passed as `interval`, and applied by:

```python
ax.set_xlim(interval[0], interval[1])
```

## 1) Paper-ready plotting snippet

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_filter_interpolation_for_paper(
    El_list,
    an,
    samp,
    dt,
    Vmin,
    dE,
    interval,   # usually cfg["plot_interval"], e.g. [-0.35, 0.0]
    out_png="filter_interpolation_window.png",
    fontsize=14,
    figsize=(8, 5),
    dpi=300,
    lw=2.0,
):
    """Plot true Gaussian filter and Newton interpolation curves."""

    def filter_func(x, El):
        return np.sqrt(dt / np.pi) * np.exp(-(x - El) ** 2 * dt)

    def _scaled_eval(x_eval, nodes, coeffs):
        x = 4.0 * (x_eval - Vmin) / dE - 2.0
        result = coeffs[0]
        basis = 1.0
        for j in range(1, len(nodes)):
            basis *= (x - nodes[j - 1])
            result += coeffs[j] * basis
        return result

    x_plot = np.linspace(interval[0], interval[1], 1000)
    fig, ax = plt.subplots(figsize=figsize)

    for ie, El in enumerate(El_list):
        y_true = filter_func(x_plot, El)
        y_interp = np.array([_scaled_eval(x, samp, an[ie]) for x in x_plot])

        ax.plot(x_plot, y_true, lw=lw, label="Gaussian window" if ie == 0 else "")
        ax.plot(x_plot, y_interp, "--", lw=lw, label="Newton interpolation" if ie == 0 else "")

    ax.set_xlabel("Energy (Hartree)", fontsize=fontsize)
    ax.set_ylabel(r"$f(E)$", fontsize=fontsize)
    ax.set_title("Filter interpolation", fontsize=fontsize + 1)
    ax.tick_params(axis="both", labelsize=fontsize - 1)

    # x-axis lim from config plot_interval
    ax.set_xlim(interval[0], interval[1])

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fontsize - 2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
```

## 2) Complete CLI (exactly runnable)

```bash
python -c "import json, numpy as np; \
CONFIG = {\
    'out_root': 'results',\
    'tag': 'gaussian_run',\
    'potential': {\
        'type': 'gaussian_files',\
        'd': 0.5,\
        'cube_file': '/pscratch/sd/b/bizi/3Dtest/Filter-Diagonalization-of-Large-Hamiltonian/QD_Outputs/QD_R13.cube',\
        'params_file': 'gaussian_fit_params.json',\
        'r_cut': 7.0\
    },\
    'N': 64,\
    'nc': 2000,\
    'dE': 35.0,\
    'Vmin': -3.0,\
    'El_list': list(np.arange(-0.23, -0.17, 0.01)),\
    'filter_type': 'gaussian',\
    'samp_method': 'ashkenazy',\
    'plot_window_bands': {\
        'target': [-0.22, -0.13],\
        'gap': [-0.20, -0.15]\
    },\
    'n_random': 256,\
    'seed': 42,\
    'svd_tol': 1e-3,\
    'max_energies': 200,\
    'kinetic_cut': 30.0,\
    'print_every_filter': 1,\
    'interval_samp_enhance': [-0.22, -0.13],\
    'interpolation_tolerance': 1e-3,\
    'enhance_step': 1,\
    'max_enhance_iters': 0,\
    'enhance_density_factor': 1,\
    'plot_interval': [-0.35, 0.0],\
    'initial_state_type': 'pm1'\
}; \
CONFIG['dt'] = 300.0; \
json.dump(CONFIG, open('fft_config_qd_0.json','w'), indent=4)"

python main.py --cfg fft_config_qd_0.json
```

## 3) Your x-axis `lim` value in this CLI

For the above config, x-axis limit is:

```python
plot_interval = [-0.35, 0.0]
# => ax.set_xlim(-0.35, 0.0)
```


## 4) How to adjust fontsize/title from CLI config

Add this block in your `CONFIG`:

```python
'plot_filter_interpolation_style': {
    'title': 'Filter interpolation (paper figure)',
    'title_fontsize': 18,
    'label_fontsize': 16,
    'tick_fontsize': 14,
    'legend_fontsize': 13,
    'figsize': [9, 5]
}
```

Then run the same:

```bash
python main.py --cfg fft_config_qd_0.json
```

Now these style options are read by `main.py` and applied in `fft_code/plotting.py` when drawing `filter_interpolation.png`.
