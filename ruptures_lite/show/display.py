r"""Display. Ported from ruptures (BSD-2-Clause).

matplotlib is imported lazily so that the rest of the package works in
environments where matplotlib is not installed.
"""

from __future__ import annotations

from itertools import cycle

import numpy as np

from ruptures_lite.utils import pairwise

COLOR_CYCLE = ["#4286f4", "#f44174"]


class MatplotlibMissingError(RuntimeError):
    pass


def display(
    signal,
    true_chg_pts,
    computed_chg_pts=None,
    computed_chg_pts_color="k",
    computed_chg_pts_linewidth=3,
    computed_chg_pts_linestyle="--",
    computed_chg_pts_alpha=1.0,
    **kwargs
):
    """Display a signal and the change points in alternating colors.

    Returns:
        tuple: (figure, axarr)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MatplotlibMissingError(
            "This feature requires the optional dependency matplotlib, you can "
            "install it using `pip install matplotlib`."
        )

    if type(signal) != np.ndarray:
        # Try to get array from a Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    matplotlib_options = {"figsize": (10, 2 * n_features)}
    matplotlib_options.update(kwargs)

    fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
    if n_features == 1:
        axarr = [axarr]

    for axe, sig in zip(axarr, signal.T):
        color_cycle = cycle(COLOR_CYCLE)
        axe.plot(range(n_samples), sig)

        bkps = [0] + sorted(true_chg_pts)
        alpha = 0.2
        for (start, end), col in zip(pairwise(bkps), color_cycle):
            axe.axvspan(max(0, start - 0.5), end - 0.5, facecolor=col, alpha=alpha)
        if computed_chg_pts is not None:
            for bkp in computed_chg_pts:
                if bkp != 0 and bkp < n_samples:
                    axe.axvline(
                        x=bkp - 0.5,
                        color=computed_chg_pts_color,
                        linewidth=computed_chg_pts_linewidth,
                        linestyle=computed_chg_pts_linestyle,
                        alpha=computed_chg_pts_alpha,
                    )

    fig.tight_layout()
    return fig, axarr
