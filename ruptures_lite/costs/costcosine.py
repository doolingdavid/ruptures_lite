r"""CostCosine (kernel change point detection with cosine similarity).

Ported from ruptures (BSD-2-Clause).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures_lite.base import BaseCost
from ruptures_lite.costs import NotEnoughPoints


class CostCosine(BaseCost):
    r"""Kernel change point detection with the cosine similarity."""

    model = "cosine"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 1
        self._gram = None

    @property
    def gram(self):
        """Generate the gram matrix (lazy loading)."""
        if self._gram is None:
            self._gram = squareform(1 - pdist(self.signal, metric="cosine"))
        return self._gram

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
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
