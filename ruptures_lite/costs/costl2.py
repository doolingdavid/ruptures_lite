r"""CostL2 (least squared deviation). Ported from ruptures (BSD-2-Clause)."""

from __future__ import annotations

from ruptures_lite.base import BaseCost
from ruptures_lite.costs import NotEnoughPoints


class CostL2(BaseCost):
    r"""Least squared deviation."""

    model = "l2"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 1

    def fit(self, signal):
        """Set parameters of the instance.

        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)

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
        return self.signal[start:end].var(axis=0).sum() * (end - start)
