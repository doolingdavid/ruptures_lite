r"""Continuous linear change. Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

import numpy as np

from ruptures_lite.base import BaseCost
from ruptures_lite.costs import NotEnoughPoints


class CostCLinear(BaseCost):
    r"""Piecewise linear approximation with a continuity constraint."""

    model = "clinear"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 3

    def fit(self, signal):
        """Set parameters of the instance."""
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

        if start == 0:
            start = 1

        sub = self.signal[start:end]
        slope = (self.signal[end - 1] - self.signal[start - 1]) / (end - start)
        intercept = self.signal[start - 1]
        approx = slope.reshape(-1, 1) * np.arange(
            1, end - start + 1
        ) + intercept.reshape(-1, 1)
        return np.sum((sub - approx.transpose()) ** 2)
