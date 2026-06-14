"""Draw a random partition.

Ported verbatim from ``ruptures.utils.drawbkps`` (BSD-2-Clause). See LICENSE.
"""

from __future__ import annotations

import numpy as np


def draw_bkps(n_samples=100, n_bkps=3, seed=None):
    """Draw a random partition with the specified number of samples and changes."""
    rng = np.random.default_rng(seed=seed)
    alpha = np.ones(n_bkps + 1) / (n_bkps + 1) * 2000
    bkps = np.cumsum(rng.dirichlet(alpha) * n_samples).astype(int).tolist()
    bkps[-1] = n_samples
    return bkps
