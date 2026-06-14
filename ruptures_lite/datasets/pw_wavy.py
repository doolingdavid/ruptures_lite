"""Piecewise sinusoidal (pw_wavy). Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

from itertools import cycle

import numpy as np

from ruptures_lite.utils import draw_bkps


def pw_wavy(n_samples=200, n_bkps=3, noise_std=None, seed=None):
    """Return a 1D piecewise wavy signal and the associated changepoints.

    Returns:
        tuple: signal of shape (n_samples,), list of breakpoints
    """
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    f1 = np.array([0.075, 0.1])
    f2 = np.array([0.1, 0.125])
    freqs = np.zeros((n_samples, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n_samples)

    signal = np.sum([np.sin(2 * np.pi * tt * f) for f in freqs.T], axis=0)

    if noise_std is not None:
        rng = np.random.default_rng(seed=seed)
        noise = rng.normal(scale=noise_std, size=signal.shape)
        signal += noise

    return signal, bkps
