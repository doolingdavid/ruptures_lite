r"""Shift in linear model. Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

import numpy as np

from .pw_constant import pw_constant


def pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=None, seed=None):
    """Return a piecewise linear signal and the associated changepoints.

    Returns:
        tuple: signal of shape (n_samples, n_features+1), list of breakpoints
    """
    rng = np.random.default_rng(seed=seed)
    covar = rng.normal(size=(n_samples, n_features))
    linear_coeff, bkps = pw_constant(
        n_samples=n_samples,
        n_bkps=n_bkps,
        n_features=n_features,
        noise_std=None,
        seed=seed,
    )
    var = np.sum(linear_coeff * covar, axis=1)
    if noise_std is not None:
        var += rng.normal(scale=noise_std, size=var.shape)
    signal = np.c_[var, covar]
    return signal, bkps
