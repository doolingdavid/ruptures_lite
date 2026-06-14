r"""Kernelized mean change (rbf). Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures_lite.base import BaseCost
from ruptures_lite.exceptions import NotEnoughPoints


class CostRbf(BaseCost):
    r"""Kernel cost function (rbf kernel)."""

    model = "rbf"

    def __init__(self, gamma=None):
        """Initialize the object."""
        self.min_size = 1
        self.gamma = gamma
        self._gram = None

    @property
    def gram(self):
        """Generate the gram matrix (lazy loading).

        Only access this after a ``.fit()`` (otherwise ``self.signal`` is undefined).
        """
        if self._gram is None:
            K = pdist(self.signal, metric="sqeuclidean")
            if self.gamma is None:
                self.gamma = 1.0
                # median heuristics
                K_median = np.median(K)
                if K_median != 0:
                    self.gamma = 1 / K_median
            K *= self.gamma
            np.clip(K, 1e-2, 1e2, K)  # clip to avoid exponential under/overflow
            self._gram = np.exp(squareform(-K))
        return self._gram

    def fit(self, signal):
        """Set parameters of the instance."""
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        # If gamma is None, set it using the median heuristic (lazy gram access).
        if self.gamma is None:
            self.gram
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size`).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
