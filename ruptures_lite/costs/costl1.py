r"""CostL1 (least absolute deviation). Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

import numpy as np

from ruptures_lite.base import BaseCost
from ruptures_lite.costs import NotEnoughPoints


class CostL1(BaseCost):
    r"""Least absolute deviation."""

    model = "l1"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 2

    def fit(self, signal):
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size`).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub = self.signal[start:end]
        med = np.median(sub, axis=0)
        return abs(sub - med).sum()
