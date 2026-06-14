r"""Rank-based cost function (CostRank). Ported from ruptures (BSD-2-Clause).

Note: ruptures uses ``scipy.stats.mstats.rankdata(signal, axis=0)``. The ``axis``
keyword is not available in older SciPy releases, so we apply the 1-D
``scipy.stats.rankdata`` column by column. For ordinary (non-masked) arrays this
produces ranks identical to ``mstats.rankdata(..., axis=0)`` (average method for
ties), preserving numerical parity with ruptures.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import pinv, LinAlgError
from scipy.stats import rankdata

from ruptures_lite.base import BaseCost
from ruptures_lite.costs import NotEnoughPoints


def _rankdata_axis0(signal):
    """Column-wise average ranks; equivalent to mstats.rankdata(signal, axis=0)."""
    ranks = np.empty(signal.shape, dtype=float)
    for j in range(signal.shape[1]):
        ranks[:, j] = rankdata(signal[:, j])
    return ranks


class CostRank(BaseCost):
    r"""Rank-based cost function."""

    model = "rank"

    def __init__(self):
        """Initialize the object."""
        self.inv_cov = None
        self.ranks = None
        self.min_size = 2

    def fit(self, signal):
        """Set parameters of the instance."""
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        obs, vars = signal.shape

        # Convert signal data into ranks in the range [1, n]
        ranks = _rankdata_axis0(signal)
        # Center the ranks into the range [-(n+1)/2, (n+1)/2]
        centered_ranks = ranks - ((obs + 1) / 2)
        # Sigma is the covariance of these ranks. If scalar, reshape to 1x1.
        cov = np.cov(centered_ranks, rowvar=False, bias=True).reshape(vars, vars)

        # Use the pseudoinverse to handle linear dependencies
        # see Lung-Yut-Fong, A., Levy-Leduc, C., & Cappe, O. (2015)
        try:
            self.inv_cov = pinv(cov)
        except LinAlgError as e:
            raise LinAlgError(
                "The covariance matrix of the rank signal is not invertible and the "
                "pseudo-inverse computation did not converge."
            ) from e
        self.ranks = centered_ranks
        self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size`).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        mean = np.reshape(np.mean(self.ranks[start:end], axis=0), (-1, 1))
        return -(end - start) * mean.T @ self.inv_cov @ mean
