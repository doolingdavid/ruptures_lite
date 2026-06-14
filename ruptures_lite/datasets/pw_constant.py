"""Piecewise constant signal (with noise). Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

import numpy as np

from ruptures_lite.utils import draw_bkps


def pw_constant(
    n_samples=200, n_features=1, n_bkps=3, noise_std=None, delta=(1, 10), seed=None
):
    """Return a piecewise constant signal and the associated changepoints.

    Returns:
        tuple: signal of shape (n_samples, n_features), list of breakpoints
    """
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    signal = np.empty((n_samples, n_features), dtype=float)
    tt_ = np.arange(n_samples)
    delta_min, delta_max = delta
    center = np.zeros(n_features)
    rng = np.random.default_rng(seed=seed)
    for ind in np.split(tt_, bkps):
        if ind.size > 0:
            jump = rng.uniform(delta_min, delta_max, size=n_features)
            spin = rng.choice([-1, 1], n_features)
            center += jump * spin
            signal[ind] = center

    if noise_std is not None:
        noise = rng.normal(size=signal.shape) * noise_std
        signal = signal + noise

    return signal, bkps
