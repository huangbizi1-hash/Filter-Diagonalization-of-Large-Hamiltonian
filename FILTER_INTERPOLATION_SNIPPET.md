# Filter interpolation plotting snippet

This snippet is extracted from `main.py` + `fft_code/plotting.py` and can be pasted into a paper-figure script.  
The x-axis limit comes from `interval` (usually from `cfg["plot_interval"]`), i.e. `ax.set_xlim(interval[0], interval[1])`.

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
    interval,   # <- usually cfg["plot_interval"], e.g. [-0.35, 0.0]
    out_png="filter_interpolation_window.png",
    fontsize=14,
    figsize=(8, 5),
):
    """Plot true Gaussian filter and Newton interpolation curves."""

    # default Gaussian filter used in the project
    def filter_func(x, El):
        return np.sqrt(dt / np.pi) * np.exp(-(x - El) ** 2 * dt)

    # evaluate Newton interpolation on physical-energy x
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

        ax.plot(x_plot, y_true, lw=2.0, label="Gaussian window" if ie == 0 else "")
        ax.plot(x_plot, y_interp, "--", lw=2.0, label="Newton interpolation" if ie == 0 else "")

    # ===== publication style knobs =====
    ax.set_xlabel("Energy (Hartree)", fontsize=fontsize)
    ax.set_ylabel(r"$f(E)$", fontsize=fontsize)
    ax.set_title("Filter interpolation", fontsize=fontsize + 1)
    ax.tick_params(axis="both", labelsize=fontsize - 1)

    # x-axis limits in original code:
    # ax.set_xlim(interval[0], interval[1])
    ax.set_xlim(interval[0], interval[1])

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fontsize - 2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

## In your CLI/config

- `filter_interpolation.png` is produced by:
  - `python main.py --cfg fft_config_qd_0.json`
- x-axis `lim` parameter is:
  - `plot_interval` in config JSON
  - then passed to plotting as `interval`
  - finally applied via `ax.set_xlim(interval[0], interval[1])`
